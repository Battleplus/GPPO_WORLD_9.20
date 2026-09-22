"""Join durable step reservations to finalization and read-only SQLite evidence.
Never alter raw run artifacts. Only a derived budget status view is emitted.
"""
from pathlib import Path
import argparse, hashlib, json, sqlite3

def reconcile(steps, finals, reservations, run_id, attempt_id):
    def index(rows, name):
        out = {}
        for row in rows:
            key = row.get("reservation_id")
            if not key or key in out:
                raise ValueError(name + ": missing or duplicate reservation identity")
            out[key] = row
        return out
    a, b, c = index(steps, "steps"), index(finals, "finalization"), index(reservations, "SQLite")
    if set(a) != set(b) or set(a) != set(c):
        raise ValueError("reservation sets differ")
    pairs=set(); derived=[]
    for row in steps:
        key=row["reservation_id"]; f=b[key]; d=c[key]
        pair=(row.get("branch_id"),row.get("step"))
        if None in pair or pair in pairs: raise ValueError("duplicate or missing branch-step")
        pairs.add(pair)
        if (f.get("branch_id"),f.get("step")) != pair: raise ValueError("finalization branch-step mismatch")
        if f.get("status") != "verified" or f.get("actual_reservation_status") != "verified": raise ValueError("unverified finalization")
        if d.get("status") != "verified" or d.get("amount") != 1 or d.get("stage") != "environment_steps": raise ValueError("unverified or wrong SQLite reservation")
        if d.get("run_id") != run_id or d.get("attempt_id") != attempt_id: raise ValueError("SQLite run/attempt mismatch")
        if row.get("budget_status") != "reserved_pending_finalization": raise ValueError("unexpected original logging status")
        derived.append({**row,"original_budget_status":row["budget_status"],"budget_status":"verified","budget_status_source":"reservation_id + branch_id + step joined to finalization and read-only SQLite"})
    return derived

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registration",type=Path,required=True)
    args=ap.parse_args(); p=args.registration; out=p/"final-analysis-v2"
    read=lambda f:json.loads(f.read_text(encoding="utf-8"))
    readlines=lambda f:[json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    auth=read(p/"reviewed-authorization.json");db=Path(auth["budget"]["path"])
    before=sha(db); run=auth["run"]["run_id"]; attempt=auth["run"]["attempt_id"]
    with sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True) as con:
        con.row_factory=sqlite3.Row
        assert con.execute("pragma integrity_check").fetchone()[0]=="ok"
        stages=[dict(r) for r in con.execute("select * from stages")]
        assert all(r["unknown"]==0 and r["reserved"]==r["verified"] for r in stages)
        reservations=[dict(r) for r in con.execute("select * from reservations where run_id=?",(run,))]
    sources=[p/"authorized-run/step-vector-rewards.jsonl",p/"authorized-run/budget-finalization.jsonl",p/"reviewed-authorization.json",db]
    steps=readlines(sources[0]); finals=readlines(sources[1])
    derived=reconcile(steps,finals,reservations,run,attempt)
    assert sha(db)==before
    target=out/"step-rewards-with-verified-budget-view.jsonl"
    with target.open("x",encoding="utf-8") as f:
        for row in derived:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    report={"status":"matched","step_rows":len(steps),"finalization_rows":len(finals),"SQLite_reservations":len(reservations),"run_id":run,"attempt_id":attempt,"sqlite_stages":stages,"raw_files_modified":False,"additional_env_steps":0,"model_forwards":0,"updates":0,"source_hashes":{str(f):sha(f) for f in sources},"derived_view":str(target),"derived_sha256":sha(target),"interpretation":"Original step rows were written before complete(); status is resolved from existing finalization and SQLite. Rewards, actions, identities and metrics are unchanged."}
    (out/"budget-reconciliation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"matched","reservations":len(steps),"database_unchanged":True}))
if __name__=="__main__":main()
