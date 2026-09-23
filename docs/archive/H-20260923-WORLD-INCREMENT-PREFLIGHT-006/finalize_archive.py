"""Read-only final verification and selective archive; no experiment imports."""
from pathlib import Path
import base64
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import urllib.request

REVIEW = Path(__file__).resolve().parent
ROOT = Path(r"E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt")
OUT = ROOT / "runs/finite-communication-ack-lease-fix-20260920/world-increment-preflight-20260923"
DB = OUT.parent / "ack-known-task-guard-baseline-v1/budget.sqlite3"
EXPECTED_DB = "9ff13f0e18ab8ce56f418f00eac786f08a47e6107a2b8d31f7184b2502c9d420"
HEAD = "8ac988ff1d7dae07c9a599e49917c4dcbbf6705c"
HANDOFF = "H-20260923-WORLD-INCREMENT-PREFLIGHT-006"
STAGE = REVIEW / "selected-archive"
PREFIX = "docs/archive/" + HANDOFF

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert digest(DB) == EXPECTED_DB
connection = sqlite3.connect(DB.as_uri() + "?mode=ro&immutable=1", uri=True)
assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
connection.close()
assert digest(DB) == EXPECTED_DB
hashes = json.loads((OUT / "hashes.json").read_text(encoding="utf-8"))
for name, item in hashes.items():
    assert digest(OUT / name) == item["sha256"], name
identity = json.loads((OUT / "identity-index.json").read_text(encoding="utf-8"))
assert all(item.get("exists") for item in identity["files"].values())
for key, item in identity["files"].items():
    if key != "prefix_snapshots":
        assert digest(Path(item["path"])) == item["sha256"], key
manifest = json.loads((OUT / "paired-manifest-proposal.json").read_text(encoding="utf-8"))
assert len(manifest) == len({r["branch_id"] for r in manifest}) == 48
pairs = {}
for row in manifest:
    pairs.setdefault(row["pair_id"], []).append(row)
assert len(pairs) == 24
assert all({x["arm"] for x in rows} == {"normal", "event_features_off"}
           and len({x["exogenous_key"] for x in rows}) == 1 for rows in pairs.values())
assert sum(rows[0]["arm"] == "normal" for rows in pairs.values()) == 12
assert sum(r["max_steps"] for r in manifest) == 768
review = {"handoff_id":HANDOFF,"decision":"A_event_feature_sensitivity_only",
          "hash_entries_verified":len(hashes),"input_identity_verified_except_reused_snapshot":True,
          "snapshot_identity_source":"H005 prior verified identity; no deserialize/rehash",
          "pure_tests_passed":10,"decision_contract_rows":279,"manifest_pairs":24,
          "future_branches":48,"future_budget_pending":768,"sqlite_sha256":EXPECTED_DB,
          "sqlite_integrity":"ok","sqlite_unchanged":True,
          "env_step":0,"reset_replay":0,"model_forward":0,"all_updates":0,"new_attempt":0,
          "runtime_source_modified":False,"archive_status":"prepared_not_yet_remote_verified"}
(REVIEW / "final-review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding="utf-8")
files=[]
for path in sorted(OUT.iterdir()):
    if path.is_file():
        rel=PREFIX+"/"+path.name
        target=STAGE/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        files.append(rel)
for relsrc in ["tools/analyze_world_increment_preflight_20260923.py","tests/test_world_increment_preflight_20260923.py"]:
    rel=PREFIX+"/source/"+relsrc
    target=STAGE/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/relsrc,target)
    files.append(rel)
for name in ["final-review.json","finalize_archive.py"]:
    rel=PREFIX+"/"+name
    shutil.copyfile(REVIEW/name,STAGE/rel)
    files.append(rel)

auth=subprocess.run(["gh","auth","token"],capture_output=True,text=True,check=False)
if auth.returncode or not auth.stdout.strip():
    raise RuntimeError("GitHub credential unavailable; local review complete, archive not attempted")
token=auth.stdout.strip()
def api(path):
    request=urllib.request.Request("https://api.github.com"+path,
        headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json","User-Agent":"Codex-H006-review"})
    with urllib.request.urlopen(request,timeout=30) as response:
        return json.load(response)
assert api("/repos/Battleplus/GPPO_WORLD_9.20/git/ref/heads/archive/research-progress")["object"]["sha"] == HEAD
obj=api("/repos/Battleplus/GPPO_WORLD_9.20/contents/docs/RESEARCH_PROGRESS.md?ref="+HEAD)
old=base64.b64decode(obj["content"]).decode("utf-8")
header="""# 当前最新状态：H-006 零步可检验性审查完成

最终判定 **A，仅限事件特征关闭敏感性**。原生 M10 25-action world 的17维候选特征进入 actor评分；现有use_events=False仅关闭末5维事件输入。这不是完整no-world对照，未证明实际排名改变、世界模型净收益或辅助训练贡献。置零是接口约定，不能认定为训练分布内中性值。

同guard后的279决策中：NOOP-only94、单非NOOP7、至少两个非NOOP178；全部保留合法NOOP，候选空集合0。279条公开候选/概率/选择合同核对通过；纯测试10passed。多候选不代表未执行动作会被接受或存在收益权衡。

下一步仅提出48新分支（8parent×3repeat×2arm）、最多768环境步的开发性配对方案。两臂normal/event_features_off，固定同一guard、权重、公开mask、外生键、奖励、偏好和hidden推进合同。原SQLite仍299/404，剩105不用；总上限1067（299+768、增加663）仅待批准提案，未改库、未新建替代账本。

本轮env.step、reset/replay、模型前向、全部更新和新增attempt均0。H005简单规则基线结论保留；本轮不执行动态评价、训练、validation/heldout或下一阶段。

[报告](archive/H-20260923-WORLD-INCREMENT-PREFLIGHT-006/report.md) · [协议](archive/H-20260923-WORLD-INCREMENT-PREFLIGHT-006/protocol.json) · [身份](archive/H-20260923-WORLD-INCREMENT-PREFLIGHT-006/identity-index.json) · [待批准申请](archive/H-20260923-WORLD-INCREMENT-PREFLIGHT-006/authorization-request.md) · [审查](archive/H-20260923-WORLD-INCREMENT-PREFLIGHT-006/final-review.json)。远端提交和逐文件核验另存本地receipt，仅成功回读后宣布归档完成。

以下为原样保留的历史进度，历史“当前任务”标题不代表最新状态。

---

"""
(STAGE/"docs/RESEARCH_PROGRESS.md").write_text(header+old,encoding="utf-8")
files.append("docs/RESEARCH_PROGRESS.md")
index={rel:{"sha256":digest(STAGE/rel),"bytes":(STAGE/rel).stat().st_size} for rel in files}
index_rel=PREFIX+"/archive-index.json"
(STAGE/index_rel).write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding="utf-8")
files.append(index_rel)
(REVIEW/"archive-allowlist.json").write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding="utf-8")
receipt=REVIEW/"final-remote-verification.json"
if receipt.exists():
    raise RuntimeError("Archive receipt already exists: inspect and verify existing commit; do not duplicate")
run=subprocess.run([sys.executable,"-X","utf8",str(REVIEW/"archive_selected.py"),"--root",str(STAGE),
    "--files",*files,"--expected-head",HEAD,"--headline","Archive H006 world decision-path preflight and scoped event-feature proposal",
    "--phase","final","--output",str(receipt)],input=token,text=True,capture_output=True)
if run.returncode:
    print("Archive helper failed; inspect persisted receipt before retry. No credential logged.")
    print(run.stderr.replace(token,"[REDACTED]")[-2000:])
    raise SystemExit(run.returncode)
print(run.stdout)
