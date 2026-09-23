"""Frozen two-arm analysis. JSON plus read-only SQLite; no models/environment.

Both arms use the same guard. Native reward/termination checks are reused from
the pinned H005 analyzer without relabeling either arm as historical R.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
import struct

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/finite-communication-ack-lease-fix-20260920"
H005 = BASE / "ackguard-matrix-registration-20260922"
ARMS = ("normal", "event_features_off")
DEPENDENCIES = {
    "native_analysis": (H005 / "analyze_results.py", "c4add12392c8f3ca082e0aaa57224708b24ff0b16ae40eeae95ff65183834ba7"),
    "reconciliation": (H005 / "final-analysis-v2/reconcile_budget_status.py", "91d2a4b53f6d735c6d8b008960bf316076eac5d52240017d96e78c9d7a71e10b"),
    "public_contract": (ROOT / "tools/analyze_world_increment_preflight_20260923.py", "2f36eec39f0c7c921b8a246a17999870cc31e3203de22606cdc3e3732f3dec64"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dependency(name):
    path, expected = DEPENDENCIES[name]
    if sha(path) != expected:
        raise ValueError("analysis dependency identity changed: " + name)
    spec = importlib.util.spec_from_file_location("paired_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), "nonfinite/nonnumeric value")
    return float(value)


def close(a, b, tol=1e-6):
    return abs(finite(a) - finite(b)) <= tol


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, "duplicate identity: " + str(key))
        result[key] = row
    return result


def validate_manifest(rows):
    require(len(rows) == 48, "fixed manifest needs48 branches")
    index = index_unique(rows, ("branch_id",))
    expected = [(f"parent-{p:02d}", repeat) for p in range(8) for repeat in range(3)]
    for pair_index, (parent, repeat) in enumerate(expected):
        pair = rows[2*pair_index:2*pair_index+2]
        order = list(ARMS) if (int(parent[-2:])+repeat)%2 == 0 else list(reversed(ARMS))
        pair_id = f"{parent}|W1|seed-1101|prefix-0|repeat-{repeat}"
        for offset, (row, arm) in enumerate(zip(pair, order)):
            require(row["order"] == 2*pair_index+offset+1 and row["arm"] == arm, "execution order mismatch")
            require(row["pair_id"] == pair_id and row["branch_id"] == pair_id+"|"+arm, "pair identity mismatch")
            require(row["parent_id"] == parent and row["repeat"] == repeat, "parent/repeat mismatch")
            require(row["condition"] == "W1" and row["model_seed"] == 1101 and row["max_steps"] == 16, "matrix changed")
            require(row["split"] == "train" and row["prefix_index"] == 0, "split/prefix changed")
            require(row["control_branch_key"] == pair_id+"|mode-R", "source cohort identity mismatch")
            require(row["exogenous_key"] == row["historical_exogenous_key"], "randomness alias mismatch")
            require(row["cohort_task_ids"] and len(row["cohort_task_ids"]) == len(set(row["cohort_task_ids"])), "cohort missing/duplicate")
        require(pair[0]["exogenous_key"] == pair[1]["exogenous_key"], "unpaired randomness")
        require(pair[0]["cohort_task_ids"] == pair[1]["cohort_task_ids"], "unpaired cohort")
    return index


def recompute_reward(row):
    require(row["task_capacity"] == 6 and row["uav_count"] == 4, "native dimensions changed")
    completed = row["completed_after"] - row["completed_before"]
    expired = row["expired_after"] - row["expired_before"]
    used = max(0.0, finite(row["energy_before"]) - finite(row["energy_after"]))
    require(completed == row["completed_delta"] and expired == row["expired_delta"], "task increments mismatch")
    require(close(used, row["energy_used_delta"]), "energy increment mismatch")
    initial = finite(row["configured_initial_energy"])
    require(initial > 0, "initial energy invalid")
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    calculated = [f32((completed-expired)/6), f32(-used/(4*initial))]
    require(len(row["vector_reward"]) == 2 and all(close(a,b,1e-8) for a,b in zip(calculated,row["vector_reward"])), "native vector reward mismatch")
    return calculated


def primary_statistics(pairs):
    require(len(pairs) == 24, "incomplete pairs")
    index_unique(pairs, ("parent_id", "repeat"))
    parents = []
    for p in range(8):
        parent = f"parent-{p:02d}"
        rows = sorted([r for r in pairs if r["parent_id"] == parent], key=lambda r:r["repeat"])
        require([r["repeat"] for r in rows] == [0,1,2], "parent repeat incomplete")
        parents.append({"parent_id":parent,"difference":sum(finite(r["normal_minus_off_utility"]) for r in rows)/3})
    values = [r["difference"] for r in parents]
    rng = random.Random(20260923)
    samples = sorted(sum(rng.choice(values) for _ in range(8))/8 for _ in range(10000))
    def quantile(p):
        x = (len(samples)-1)*p
        i = int(x)
        return samples[i] + (x-i)*(samples[min(i+1,len(samples)-1)]-samples[i])
    ci = [quantile(.025),quantile(.975)]
    decision = "normal_advantage_under_intervention" if ci[0]>0 else "normal_degradation_under_intervention" if ci[1]<0 else "inconclusive"
    return {"mean":sum(values)/8,"ci95":ci,"parents":parents,"bootstrap_seed":20260923,
            "bootstrap_replicates":10000,"independent_unit":"parent","decision":decision,
            "claim_limit":"specified event-input sensitivity on observed development data; not net world-model benefit"}


def validate_feature_payload(raw, facing, components, logits, probabilities, mask, arm):
    """Numerical log checks only; never executes a tensor/model operation."""
    require(arm in ARMS and len(raw)==len(facing)==25, "feature shape/arm mismatch")
    require(len(mask)==25 and mask[24], "feature public mask mismatch")
    for action,(old,new) in enumerate(zip(raw,facing)):
        require(len(old)==len(new)==17, "feature width mismatch")
        for value in old+new:
            finite(value)
        expected = old if arm=="normal" else old[:12]+[0.0]*5
        require(new==expected, "intervention affects unexpected columns")
        if not mask[action]:
            require(all(value==0 for value in old+new), "illegal candidate row not zero")
    require(set(components)=={"base","preference","candidate"}, "component logits missing")
    require(all(len(values)==25 for values in components.values()) and len(logits)==len(probabilities)==25, "logit dimensions mismatch")
    for i in range(25):
        require(close(sum(finite(values[i]) for values in components.values()),logits[i],2e-5), "component logit sum mismatch")
    maximum=max(finite(logits[i]) for i in range(25) if mask[i])
    weights=[math.exp(finite(logits[i])-maximum) if mask[i] else 0.0 for i in range(25)]
    total=sum(weights)
    require(all(close(w/total,p,2e-6) for w,p in zip(weights,probabilities)), "probabilities/logits mismatch")


def analyze_core(manifest, branches, raw_steps, decisions, finals, reservations, run_id, attempt_id):
    expected = validate_manifest(manifest)
    branch_index = index_unique(branches, ("branch_id",))
    require(set(branch_index) == set(expected), "incomplete/unexpected branch matrix")
    steps = load_dependency("reconciliation").reconcile(raw_steps, finals, reservations, run_id, attempt_id)
    step_index = index_unique(steps, ("branch_id","step"))
    decision_index = index_unique(decisions, ("branch_id","step"))
    require(set(step_index) == set(decision_index), "decision-step identity mismatch")
    load_dependency("public_contract").validate_rows(decisions, manifest)
    native = load_dependency("native_analysis")
    summaries = {}
    for item in manifest:
        branch_id = item["branch_id"]
        branch = branch_index[(branch_id,)]
        require(branch["arm"] == item["arm"] and branch["pair_id"] == item["pair_id"], "branch arm/pair mismatch")
        require(branch["deepcopy_isolated"] is True, "copy isolation failed")
        require(branch["source_snapshot_digest_before"] == branch["source_snapshot_digest_after"], "snapshot modified")
        require(branch["source_snapshot_semantics_before"] == branch["source_snapshot_semantics_after"], "snapshot semantics modified")
        for field in ("optimizer_updates","world_updates","offline_updates"):
            require(branch["counter_delta"][field] == 0, "forbidden update")
        branch_steps = sorted([r for r in steps if r["branch_id"] == branch_id],key=lambda r:r["step"])
        require(0 < len(branch_steps) <= item["max_steps"], "branch step cap")
        for field in ("actor_forward_calls", "world_forward_calls"):
            require(branch[field] == len(branch_steps), "trajectory probe count mismatch")
        require(branch["guard_trigger_count"] == sum(bool(decision_index[(branch_id,r["step"])]["triggered"]) for r in branch_steps), "guard trigger total mismatch")
        for field in ("completed", "expired"):
            require(branch_steps[0][field+"_before"] == branch["initial_counts"][field], "initial counts mismatch")
            require(branch_steps[-1][field+"_after"] == branch["final_counts"][field], "final counts mismatch")
        require(close(branch_steps[0]["energy_before"], branch["initial_energy"]), "initial energy mismatch")
        require(close(branch_steps[-1]["energy_after"], branch["final_energy"]), "final energy mismatch")
        for field in ("policy_hidden", "world_hidden"):
            require(branch_steps[0][field+"_before_sha256"] == branch["source_snapshot_semantics_before"][field+"_sha256"], "initial hidden mismatch")
        for step in branch_steps:
            require(step["arm"] == item["arm"] and step["pair_id"] == item["pair_id"], "step arm/pair mismatch")
            recompute_reward(step)
            dec = decision_index[(branch_id,step["step"])]
            for field in ("reservation_id","public_observation_sha256","policy_hidden_before_sha256","world_hidden_before_sha256","selected_action_hidden_sha256","legal_mask","original_action","probabilities"):
                require(step[field] == dec[field], "decision/step mismatch: "+field)
            require(step["action"] == dec["final_action"], "decision action mismatch")
            require(step["world_hidden_after_sha256"] == step["selected_action_hidden_sha256"], "world hidden progression mismatch")
        for left,right in zip(branch_steps,branch_steps[1:]):
            for field in ("completed", "expired", "energy"):
                require(close(left[field+"_after"], right[field+"_before"]), "reward increment continuity mismatch")
            for field in ("policy_hidden","world_hidden"):
                require(left[field+"_after_sha256"] == right[field+"_before_sha256"], "hidden chain mismatch")
            require(left["public_next_observation_sha256"] == right["public_observation_sha256"], "observation chain mismatch")
        e = {**item,"pair_key":item["control_branch_key"]}
        summary, issues = native._branch_summary(branch,e,"guard")
        step_summary, step_issues = native._validate_steps(branch_steps,e,"guard",summary)
        require(not issues and not step_issues, "native branch/step contract: "+str(issues+step_issues))
        for key,value in step_summary.items():
            if key not in ("raw_last_step","completion_records"):
                summary[key] = value
        records, errors = native._cohort_records(branch,"guard",branch_steps)
        require(not errors and records is not None, "completion evidence missing")
        require(all(t in records for t in item["cohort_task_ids"]), "cohort outcomes missing")
        summary["cohort_task_ids"] = item["cohort_task_ids"]
        for scope,ids in (("cohort",item["cohort_task_ids"]),("all_recorded",list(records))):
            summary[scope+"_on_time_physical"] = sum(records[t].get("physical_arrival_before_deadline") is True for t in ids)
            summary[scope+"_on_time_host"] = sum(records[t].get("host_confirmation_before_deadline") is True for t in ids)
        summary.update({"arm":item["arm"],"pair_id":item["pair_id"]})
        summary.pop("_decision_rows",None)
        summaries[branch_id] = summary
    pairs=[]
    for pair_id in dict.fromkeys(r["pair_id"] for r in manifest):
        a,b = [branch_index[(pair_id+"|"+arm,)] for arm in ARMS]
        for field in ("initial_env_state_sha256","initial_public_observation_sha256","source_snapshot_semantics_before","exogenous_key"):
            require(a[field] == b[field], "pair initial state mismatch: "+field)
        left,right = [summaries[pair_id+"|"+arm] for arm in ARMS]
        difference = left["utility"]-right["utility"]
        pairs.append({"pair_id":pair_id,"parent_id":left["parent_id"],"repeat":left["repeat"],
                      "normal":left,"event_features_off":right,"normal_minus_off_utility":difference,
                      "descriptive_tie":abs(difference)<=1e-6})
    return {"status":"complete","paired_results":pairs,"primary":primary_statistics(pairs),
            "verified_steps":len(steps),"branch_count":len(branches),"budget_reconciliation":"exact reservation/branch/step/finalization/SQLite join"}


def summarize_secondary(paired_results):
    """Keep all-task and registered cohort outcomes separate."""
    metrics = ("completed_delta", "all_task_completed", "cohort_on_time_physical", "cohort_on_time_host",
               "all_recorded_on_time_physical", "all_recorded_on_time_host", "energy_used",
               "accepted_count", "rejected_count", "task_unavailable_count", "submit_count",
               "noop_count_from_actions", "guard_trigger_count", "action_change_count", "env_steps")
    totals = {arm:{name:sum(finite(pair[arm][name]) for pair in paired_results) for name in metrics} for arm in ARMS}
    differences = {name:totals["normal"][name]-totals["event_features_off"][name] for name in metrics}
    return {"arm_totals":totals,"normal_minus_off_totals":differences,
            "note":"descriptive totals only; primary inference uses8parent macro means, not pooled decisions or tasks"}


def validate_runtime_evidence(features, decisions, steps, manifest, status, costs, gate):
    d_index = index_unique(decisions,("branch_id","step"))
    s_index = index_unique(steps,("branch_id","step"))
    f_index = index_unique(features,("branch_id","step"))
    require(set(f_index)==set(d_index)==set(s_index), "feature/decision/step coverage mismatch")
    by_branch = {r["branch_id"]:r for r in manifest}
    first_pair = manifest[0]["pair_id"]
    sums = {"policy_encode":0,"world_candidate_batch":0,"actor_readout":0}
    timing = defaultdict(float)
    for key, feature in f_index.items():
        dec,step,item = d_index[key],s_index[key],by_branch[key[0]]
        require(feature["arm"]==item["arm"] and feature["pair_id"]==item["pair_id"] and feature["exogenous_key"]==item["exogenous_key"], "feature identity mismatch")
        for field in ("public_observation_sha256","policy_hidden_before_sha256","world_hidden_before_sha256","legal_mask","original_action"):
            require(feature[field]==dec[field],"feature input/decision mismatch: "+field)
        probabilities = [p["probability"] for p in dec["probabilities"]]
        require(feature["probabilities"]==probabilities,"feature output probabilities mismatch")
        validate_feature_payload(feature["candidate_features_raw_25x17"],feature["candidate_features_actor_25x17"],
            {k:feature[k+"_logits"] for k in ("base","preference","candidate")},feature["logits"],probabilities,feature["legal_mask"],item["arm"])
        require(feature["event_features_before_25x5"]==[r[12:] for r in feature["candidate_features_raw_25x17"]], "raw event slice mismatch")
        require(feature["event_features_after_25x5"]==[r[12:] for r in feature["candidate_features_actor_25x17"]], "actor event slice mismatch")
        require(feature["selected_action"]==dec["final_action"] and feature["guard_candidates"]==dec["candidate_actions"],"feature guard mismatch")
        require(set(feature["by_action_hidden_sha256"])=={str(i) for i in range(25)},"candidate hidden missing")
        require(feature["by_action_hidden_sha256"][str(dec["final_action"])]==step["selected_action_hidden_sha256"], "selected world output mismatch")
        require(feature["next_policy_hidden_sha256"]==step["policy_hidden_after_sha256"], "selected policy output mismatch")
        expected_calls = {"policy_encode":1,"world_candidate_batch":1,"actor_readout":2 if item["pair_id"]==first_pair and key[1]==1 else 1}
        require(feature["model_calls"]==expected_calls,"per-probe call count mismatch")
        for name,count in expected_calls.items():
            require(feature["model_call_status"][name]=={"attempted":count,"completed":count},"per-probe incomplete forward")
            sums[name] += count
        for name,value in feature["timing"].items():
            require(finite(value)>=0,"negative/nonfinite timing")
            timing[name]+=value
    first_rows = [f_index[(first_pair+"|"+arm,1)] for arm in ARMS]
    for field in ("public_observation_sha256","policy_hidden_before_sha256","world_hidden_before_sha256",
                  "candidate_features_raw_25x17","by_action_hidden_sha256","next_policy_hidden_sha256"):
        require(first_rows[0][field]==first_rows[1][field],"first input mismatch: "+field)
    require(gate["ok"] is True and gate["pair_id"]==first_pair and gate["checks"] and all(v is True for v in gate["checks"].values()),"first-pair gate incomplete")
    for obj in (status,costs):
        require(obj["status"]=="completed","runtime not completed")
        require(obj["model_call_counts"]["attempted"]==sums==obj["model_call_counts"]["completed"],"global model call discrepancy")
        require(obj["model_call_counts"]["limits"]=={"policy_encode":768,"world_candidate_batch":768,"actor_readout":770},"model limits changed")
    require(sums=={"policy_encode":len(steps),"world_candidate_batch":len(steps),"actor_readout":len(steps)+2},"probe totals mismatch")
    require(len(steps)<=768,"environment cap exceeded")
    require(status["runtime_digest_before"] and status["runtime_digest_before"]==status["runtime_digest_after"],"frozen runtime mutated")
    counters = status["hard_counts"]
    for name in ("env_steps","env_step_calls","successful_env_steps","verified_steps","actor_forward","world_forward","probe_calls"):
        require(counters[name]==len(steps),"runtime step/probe counters mismatch: "+name)
    require(counters["branches_completed"]==48,"incomplete completed branches")
    for name in ("optimizer_updates","world_updates","offline_updates"):
        require(counters[name]==0,"runtime update violation")
    return {"model_call_counts":sums,"world_rows_per_batch":25,"per_probe_timing_totals":dict(timing),
            "wall_seconds":finite(costs["wall_seconds"]),"same_input_gate_pair":first_pair,
            "cost_interpretation":"observed timings include instrumentation; no inference of overall savings from actor call counts"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--sqlite",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--attempt-id",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    require(not args.out.exists(), "analysis output exists; preserve old analysis")
    result = {"status":"incomplete","primary":None}
    try:
        before = sha(args.sqlite)
        with sqlite3.connect(args.sqlite.resolve().as_uri()+"?mode=ro",uri=True) as con:
            con.row_factory = sqlite3.Row
            require(con.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity")
            stages = [dict(r) for r in con.execute("SELECT * FROM stages")]
            require(all(r["unknown"]==0 and r["reserved"]==r["verified"] for r in stages), "unresolved budget")
            reservations = [dict(r) for r in con.execute("SELECT * FROM reservations WHERE run_id=?",(args.run_id,))]
        files = {name:args.run/name for name in ("branch-results.jsonl","step-vector-rewards.jsonl","decision-ledger.jsonl","budget-finalization.jsonl","feature-ledger.jsonl","run-status.json","runtime-costs.json","first-pair-gate.json")}
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = analyze_core(manifest,
            read_jsonl(files["branch-results.jsonl"]),read_jsonl(files["step-vector-rewards.jsonl"]),
            read_jsonl(files["decision-ledger.jsonl"]),read_jsonl(files["budget-finalization.jsonl"]),
            reservations,args.run_id,args.attempt_id)
        result["secondary"] = summarize_secondary(result["paired_results"])
        result["resource_evidence"] = validate_runtime_evidence(read_jsonl(files["feature-ledger.jsonl"]),
            read_jsonl(files["decision-ledger.jsonl"]),read_jsonl(files["step-vector-rewards.jsonl"]),manifest,
            *(json.loads(files[name].read_text(encoding="utf-8")) for name in ("run-status.json","runtime-costs.json","first-pair-gate.json")))
        require(sha(args.sqlite)==before,"SQLite changed during analysis")
        result["input_hashes"]={str(p):sha(p) for p in [args.manifest,args.sqlite,*files.values()]}
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error) as exc:
        result = {"status":"incomplete","primary":None,"technical_stop":str(exc),"algorithm_negative_result":False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":result["status"],"output":str(args.out)},ensure_ascii=False))
    return 0 if result["status"]=="complete" else 2


if __name__=="__main__":
    raise SystemExit(main())
