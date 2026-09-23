"""Apply the one proposed same-ledger extension only after explicit approval.

No environment/model imports. The current task must not invoke --apply.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "runs/finite-communication-ack-lease-fix-20260920/model-rule-stage1-execution-package-v1"
DATABASE = PACKAGE.parent / "ack-known-task-guard-baseline-v1/budget.sqlite3"
OLD_SHA = "ea42090b92b9ecea9ab1c487742dbb059d4f95c2087f90f8d66dbfbb6be38d50"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def logical_rows(con):
    names=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {name:sorted([list(r) for r in con.execute('SELECT * FROM "'+name.replace('"','""')+'"')],key=repr) for name in names}


def validate_approval(approval, package_hash):
    require(approval.get("status")=="approved", "explicit user approval required; pending template cannot apply")
    require(approval.get("experiment_id")=="model-rule-stage1-v1","wrong objective")
    require(approval.get("package_hashes_sha256")==package_hash,"approval package identity mismatch")
    require(approval.get("approved_by")=="user" and approval.get("user_message_reference"),"approval provenance missing")
    require(isinstance(approval.get("approved_at"), str) and approval["approved_at"].strip(),"approval timestamp missing")
    require(approval.get("accepted_model_rule_development_scope") is True,"interpretation not approved")
    require(approval.get("old_limit")==1067 and approval.get("new_limit")==1625 and approval.get("new_step_cap")==768,"approved budget scope mismatch")
    require(approval.get("model_call_caps")=={"policy_encode":384,"world_candidate_batch":384,"actor_readout":384},"model call scope mismatch")
    require(approval.get("rule_selector_cap")==384,"rule selector scope mismatch")
    require(approval.get("all_updates")==0,"updates not authorized")


def extend_same_database(path, expected_sha, backup, receipt, approval_hash):
    """Transaction changes exactly one stage limit; tests use temporary ledgers."""
    require(path.is_file() and sha(path)==expected_sha,"pre-extension SQLite identity mismatch")
    require(not backup.exists() and not receipt.exists(),"migration artifacts already exist; inspect rather than retry")
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as source:
        require(source.execute("PRAGMA integrity_check").fetchone()[0]=="ok","SQLite integrity failure")
        before=logical_rows(source)
        backup.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(backup) as target:
            source.backup(target)
    con=sqlite3.connect(path, isolation_level=None)
    try:
        con.execute("BEGIN IMMEDIATE")
        require(logical_rows(con)==before,"database changed while preparing extension")
        states={r[0]:r[1:] for r in con.execute("SELECT stage,limit_amount,reserved,verified,unknown FROM stages")}
        require(states.get("environment_steps")== (1067,857,857,0),"old stage state mismatch")
        require(all(v==(0,0,0,0) for k,v in states.items() if k!="environment_steps"),"update stage not zero")
        require(con.execute("SELECT COUNT(*) FROM reservations WHERE status IN ('pending','unknown')").fetchone()[0]==0,"unresolved reservations")
        cursor=con.execute("UPDATE stages SET limit_amount=1625 WHERE stage='environment_steps' AND limit_amount=1067 AND reserved=857 AND verified=857 AND unknown=0")
        require(cursor.rowcount==1,"limit update did not match one row")
        after=logical_rows(con)
        expected=json.loads(json.dumps(before))
        for row in expected["stages"]:
            if row[0]=="environment_steps":row[1]=1625
        require(after==expected,"unexpected migration mutation")
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        if con.in_transaction:con.rollback()
        raise
    finally:
        con.close()
    result={"status":"applied","database":str(path),"before_sha256":expected_sha,"after_sha256":sha(path),
            "backup":str(backup),"backup_sha256":sha(backup),"approval_sha256":approval_hash,
            "only_change":"environment_steps limit1067->1625","old_history_preserved":True,
            "historical_reserved_verified":857,"new_attempt":0,"env_step":0,"model_forward":0,"all_updates":0}
    with receipt.open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply",action="store_true",required=True)
    ap.add_argument("--approval",type=Path,required=True)
    ap.add_argument("--authorization-out",type=Path,required=True)
    ap.add_argument("--receipt-dir",type=Path,required=True)
    args=ap.parse_args()
    require(not args.authorization_out.exists(),"authorization output already exists")
    hashes_path=PACKAGE/"hashes.json"
    approval=json.loads(args.approval.read_text(encoding="utf-8"))
    validate_approval(approval,sha(hashes_path))
    hashes=json.loads(hashes_path.read_text(encoding="utf-8"))
    for item in hashes["files"]:
        require(sha(Path(item["path"]))==item["sha256"],"package/source drift: "+item["path"])
    runner_path=ROOT/"tools/run_model_rule_pair_v1.py"
    spec=importlib.util.spec_from_file_location("event_runner_authorization",runner_path)
    runner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=runner
    spec.loader.exec_module(runner)
    check=runner.package_validation()
    require(check["ok"],"static package check failed")
    require(DATABASE.resolve()==runner.BUDGET.resolve(),"wrong ledger path")
    args.receipt_dir.mkdir(parents=True,exist_ok=False)
    receipt=extend_same_database(DATABASE,OLD_SHA,args.receipt_dir/"pre-extension.sqlite3",
                                 args.receipt_dir/"migration-receipt.json",sha(args.approval))
    auth=runner.authorization_template(check)
    auth["status"]="authorized"
    auth["budget"]["sha256"]=receipt["after_sha256"]
    auth["review"]={"reviewed_by":"user-approved research lead","reviewed_at":approval["approved_at"],
                    "accepted_interpretation":True,"user_message_reference":approval["user_message_reference"]}
    auth["migration_receipt"]={"path":str(args.receipt_dir/"migration-receipt.json"),"sha256":sha(args.receipt_dir/"migration-receipt.json")}
    args.authorization_out.parent.mkdir(parents=True,exist_ok=True)
    with args.authorization_out.open("x",encoding="utf-8") as stream:json.dump(auth,stream,ensure_ascii=False,indent=2)
    print(json.dumps({"budget_extension":"applied","authorization":str(args.authorization_out),"execution_started":False}))


if __name__=="__main__":
    main()
