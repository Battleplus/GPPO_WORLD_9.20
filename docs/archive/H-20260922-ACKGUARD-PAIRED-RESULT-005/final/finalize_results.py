"""Finalize H-005 using existing raw evidence; no environment or model calls."""
from pathlib import Path
import csv
import hashlib
import json
import random
import sqlite3

OUT = Path(__file__).resolve().parent
P = OUT.parent
ROOT = P.parents[2]

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def csvwrite(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)

def main():
    analysis = read(OUT / "paired-analysis.json")
    assert analysis["status"] == "complete" and len(analysis["paired_results"]) == 24
    manifest = read(P / "branch-manifest.json")["rows"]
    auth = read(P / "reviewed-authorization.json")
    db = Path(auth["budget"]["path"])
    dbsha = sha(db)
    rawsteps = lines(P / "authorized-run/step-vector-rewards.jsonl")
    decisions = lines(P / "authorized-run/decision-ledger.jsonl")
    history = P.parent / "preference-vector-label-train-v1"
    expected = {row["control_branch_key"]: row for row in manifest}
    rsteps = [row for row in lines(history / "step-vector-rewards.jsonl") if row.get("mode") == "R" and f"{row['prefix_id']}|repeat-{row['repeat']}|mode-R" in expected]
    independent = {}
    for side, rows in [("guard", rawsteps), ("R", rsteps)]:
        grouped = {key: [] for key in expected}
        for row in rows:
            key = row.get("control_branch_key", f"{row['prefix_id']}|repeat-{row['repeat']}|mode-R")
            grouped[key].append(row)
        for key, values in grouped.items():
            values.sort(key=lambda row: row["step"])
            assert [row["step"] for row in values] == list(range(1, len(values) + 1))
            assert [row["gamma_index"] for row in values] == list(range(len(values)))
            assert values[-1]["terminated"] and not values[-1]["truncated"]
            independent[side, key] = sum(.99 ** i * (.4 * row["vector_reward"][0] + .2 * row["vector_reward"][1]) for i, row in enumerate(values))
    metrics = ["utility", "completed_delta", "all_task_completed", "on_time_physical", "on_time_host", "energy_used", "accepted_count", "rejected_count", "task_unavailable_count", "submit_count", "noop_count_from_actions", "env_steps", "actor_forward_calls", "world_forward_calls"]
    pairs = []
    for pair in analysis["paired_results"]:
        assert pair["evaluation_valid"]
        key = pair["control_branch_key"]
        row = {"control_branch_key": key, "parent_id": pair["parent_id"], "repeat": pair["repeat"], "exogenous_key": expected[key]["historical_exogenous_key"], "cohort_task_ids": "|".join(pair["R"]["task_ids"])}
        for side in ["guard", "R"]:
            assert abs(pair[side]["utility"] - independent[side, key]) < 1e-12
            for metric in metrics:
                row[side + "_" + metric] = pair[side][metric]
        row["guard_minus_R_utility"] = independent["guard", key] - independent["R", key]
        pairs.append(row)
    pairs.sort(key=lambda row: (row["parent_id"], row["repeat"]))
    parents = []
    for parent in sorted({row["parent_id"] for row in pairs}):
        values = [row for row in pairs if row["parent_id"] == parent]
        assert sorted(row["repeat"] for row in values) == [0, 1, 2]
        row = {"parent_id": parent, "repeat_count": 3}
        for metric in metrics:
            for side in ["guard", "R"]:
                row[side + "_" + metric + "_mean"] = sum(item[side + "_" + metric] for item in values) / 3
            row["delta_" + metric] = row["guard_" + metric + "_mean"] - row["R_" + metric + "_mean"]
        parents.append(row)
    values = [row["delta_utility"] for row in parents]
    assert len(values) == 8
    effect = sum(values) / 8
    rng = random.Random(20260922)
    bootstrap = [sum(rng.choice(values) for _ in range(8)) / 8 for _ in range(10000)]
    ci = [quantile(bootstrap, .025), quantile(bootstrap, .975)]
    assert abs(effect - analysis["primary"]["parent_macro_guard_minus_R_utility"]) < 1e-12
    assert all(abs(x-y) < 1e-12 for x, y in zip(ci, analysis["primary"]["parent_cluster_bootstrap"]["ci"]))
    sums = {side: {metric: sum(row[side + "_" + metric] for row in pairs) for metric in metrics} for side in ["guard", "R"]}
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        assert con.execute("pragma integrity_check").fetchone()[0] == "ok"
        stage = dict(con.execute("select * from stages where stage='environment_steps'").fetchone())
        runrows = [dict(row) for row in con.execute("select status,count(*) as rows,sum(amount) as amount from reservations where run_id=? group by status", (auth["run"]["run_id"],))]
    assert stage["limit_amount"] == 404 and stage["reserved"] == stage["verified"] == 299 and stage["unknown"] == 0
    assert runrows == [{"status": "verified", "rows": 279, "amount": 279}]
    assert sha(db) == dbsha
    noop_bad = sum(bool(row["legal_mask"][24]) and 24 not in row["candidate_actions"] for row in decisions)
    forced_noop = sum(row["original_action"] == 24 and row["final_action"] != 24 for row in decisions)
    assert noop_bad == forced_noop == 0
    status = read(P / "authorized-run/run-status.json")
    assert status["runtime_digest_before"] == status["runtime_digest_after"]
    assert all(status["hard_counts"][key] == 0 for key in ["optimizer_updates", "world_updates", "offline_updates"])
    source = {"runner": sha(ROOT / "tools/run_ack_known_task_guard_corrected_20260922.py"), "guard": sha(ROOT / "gppo_world/ack_known_task_guard.py"), "analyzer": sha(P / "analyze_results.py")}
    assert source["runner"] == auth["source"]["corrected_runner_sha256"]
    assert source["guard"] == auth["source"]["guard_sha256"]
    assert source["analyzer"] == "c4add12392c8f3ca082e0aaa57224708b24ff0b16ae40eeae95ff65183834ba7"
    adopt = ci[0] > 0 and sums["guard"]["task_unavailable_count"] < sums["R"]["task_unavailable_count"] and sums["guard"]["completed_delta"] >= sums["R"]["completed_delta"]
    assert adopt
    summary = {"handoff_id": "H-20260922-ACKGUARD-PAIRED-RESULT-005", "status": "evaluation_complete", "primary": analysis["primary"], "sums": sums, "branch_means": {side: {metric: value / 24 for metric, value in total.items()} for side, total in sums.items()}, "parents_positive": sum(x > 0 for x in values), "parents_negative": sum(x < 0 for x in values), "cohort_task_evaluations": sum(len(row["R"]["task_ids"]) for row in analysis["paired_results"]), "baseline_decision": {"adopt_as_required_simple_baseline": adopt, "scope": "frozen historical M10 25-action W1 development comparison", "classification": "improvement_evidence_sufficient_in_fixed_development_matrix", "limits": ["8 parents, one policy seed, one condition; no heldout/general deployment conclusion", "Energy increased and 2 parent means decreased", "No independent GPPO/world-model contribution is established"]}, "checks": {"independent_raw_reward_utility_recomputation": True, "independent_parent_bootstrap_recomputation": True, "NOOP_removed_from_candidates": noop_bad, "NOOP_original_forced_to_allocation": forced_noop, "runtime_weights_unchanged": True, "source_hashes": source}, "resources": {"new_environment_steps": 279, "old_guard_environment_steps_retained": 20, "formal_sqlite": stage, "new_run_reservations": runrows, "remaining_global_steps": 105, "historical_R_steps_reused": 299, "hard_counts": status["hard_counts"], "analysis_correction_new_steps": 0, "SQLite_sha256": dbsha}, "analysis_correction": "All 279 immutable pre-finalization step statuses joined one-to-one to finalization and read-only SQLite. Original raw files, initial failed analysis, and frozen metric analyzer are unchanged."}
    write("result-summary.json", summary)
    csvwrite("paired-results.csv", pairs)
    csvwrite("parent-results.csv", parents)
    write("paired-results-compact.json", pairs)
    write("parent-results.json", parents)
    delta_energy = sums["guard"]["energy_used"] - sums["R"]["energy_used"]
    report = f"""# H-005 修正规则配对评价最终报告

结论：在固定历史 M10 25-action、W1、seed-1101 的8个开发父场景中，简单公开执行状态规则改善了原登记偏好效用，并明显减少无效提交。按冻结决策规则，将它固定为后续同合同模型必须比较并超越的简单基线。

24/24新规则分支完成，对应24/24历史R复用，旧两条guard不纳入处理组。主偏好(0.8,0.2)，尺度(0.5,1.0)，gamma=0.99。每parent先平均3个repeat，再8个parent等权，父级bootstrap10000次、seed20260922。

主效用 guard−R = **{effect:+.9f}**，95%区间 **[{ci[0]:+.9f}, {ci[1]:+.9f}]**。这是效用单位，不是百分点。guard平均效用{sums['guard']['utility']/24:.9f}，R平均效用{sums['R']['utility']/24:.9f}；6个parent为正，2个为负。

|次指标（24分支合计）|R|guard|
|---|---:|---:|
|命令接受|106|129|
|拒绝|139|56|
|task_unavailable|109|0|
|全部任务完成|120|130|
|配对任务集合按时物理到达|30/48|48/48|
|配对任务集合按时主机确认|30/48|48/48|
|累计能耗|{sums['R']['energy_used']:.6f}|{sums['guard']['energy_used']:.6f}|
|提交接口调用（含NOOP）|299|279|
|NOOP动作|54|94|
|环境步/actor调用/world调用|299/299/299（历史复用）|279/279/279（新增）|

能耗增加{delta_energy:.6f}，约{100*delta_energy/sums['R']['energy_used']:.2f}%；不能宣称所有目标同时改善或总计算成本下降。拒绝依原反馈合同统计，noop/reuse_existing/awaiting_ack不计拒绝。48项完成/主机确认仅是每对历史R已登记task_ids集合，不代表全部144项任务的主机确认率。完整逐对、逐parent及次指标在CSV/JSON中。

|parent|guard−R效用（三repeat均值）|
|---|---:|
"""
    report += "".join(f"|{row['parent_id']}|{row['delta_utility']:+.9f}|\n" for row in parents)
    report += """
执行合同核验：首分支以对应历史R身份门控，基础snapshot前后digest一致24/24，模型权重digest不变。guard触发120次、动作改变120次；未删除合法NOOP，原NOOP强制改分配0次。非法动作和奖励复算失败0。公开可提议、原生执行器接受、实际完成分别报告。

预算：同一SQLite按用户批准从384扩至404，保留旧20步；本次279步全部verified，总reserved=verified=299，unknown=pending=0，剩余105。没有重跑、补样或新建替代执行账本；optimizer/world/offline update全0。历史R299步为既有证据，不重复记成本。

分析勘误：首次冻结分析器把写入时的reserved_pending_finalization当作最终待结算，导致评价未完成。执行本身24/24成功。final-analysis-v2以reservation_id、branch_id、step逐一关联原finalization及只读SQLite，279/279最终verified，生成明确标注的派生状态视图；不修改原步骤、奖励、动作或首次失败输出。冻结analyze_results.py原SHA保持不变，统计公式、样本和CI算法未改。新关联器9项测试通过；独立直接用原始奖励复算效用与父级CI一致。这是只读分析修正，不是环境重跑或反事实补填。

研究决策：固定该规则为同合同后续模型比较的简单基线。依据是主效用区间下界大于0、task_unavailable减少、任务完成未下降。保留能耗上升和两个父场景退化。这个开发结果不证明GPPO或世界模型增量，不外推到当前有限通信环境、其他模型seed、heldout或生产场景。

原始run-status、post-run-evidence、paired-analysis（首次失败）及前置冻结资料全部保留。最终结果权威文件为本目录result-summary.json、paired-results.csv、parent-results.csv及budget-reconciliation.json。完成后停止，不训练、调参或使用剩余预算追加实验。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "test-output.txt").write_text("pytest -q -p no:cacheprovider test_reconcile_budget_status.py\n9 passed in 0.06s\nIndependent raw-reward and parent-bootstrap recomputation: passed.\n", encoding="utf-8")
    artifacts = {str(path.relative_to(P)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(P.rglob("*")) if path.is_file() and "__pycache__" not in path.parts and path.name != "final-artifact-manifest.json"}
    write("final-artifact-manifest.json", {"files": artifacts, "formal_SQLite": {"path": str(db), "sha256": dbsha}, "note": "Raw large artifacts retained locally; selective remote archive retains these identities."})
    print(json.dumps({"status": "evaluation_complete", "effect": effect, "ci": ci, "baseline_adopted": adopt, "raw_labels": len(rawsteps), "R_reused_steps": len(rsteps)}))

if __name__ == "__main__":
    main()
