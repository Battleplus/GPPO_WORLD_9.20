"""Stage1 model-vs-rule frozen analysis. JSON and read-only SQLite only.

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
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/finite-communication-ack-lease-fix-20260920"
H005 = BASE / "ackguard-matrix-registration-20260922"
ARMS = ("frozen_model", "public_rule")
MANIFEST = BASE / "model-rule-stage1-execution-package-v1/paired-manifest.json"
MANIFEST_SHA256 = "00bfaeeb5ce75d422ba2d0708483a4237b7ede7100702360aaf783a820d1adad"
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


def public_rule_module():
    """Load only the two pure rule files, never the native environment package."""
    import importlib
    for filename, digest in {
        "public_dispatch_rule_v1.py": "c11f7dbc2f4ae6854a5a4c1eb2b7c82a8cc0eb3ce96a8d0acd36a5995dd5b452",
        "ack_known_task_guard.py": "2bc314ff9e3d4f78ec871c1a2e739a92669b1f9bd573aabce6a8be5b3e9e808a",
    }.items():
        require(sha(ROOT / "gppo_world" / filename) == digest, "public rule source drift: " + filename)
    namespace = "_model_rule_analysis_public"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(ROOT / "gppo_world")]
        sys.modules[namespace] = package
    return importlib.import_module(namespace + ".public_dispatch_rule_v1")


def canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_public_rule_record(public_inputs, rule_evidence, decision):
    """Recompute selection from saved public fields, without learned state."""
    recomputed = public_rule_module().select_public_dispatch_action(public_inputs).to_dict()
    require(rule_evidence == recomputed, "saved rule ranking differs from public-only recomputation")
    require(decision["original_action"] == recomputed["original_argmax"], "rule original action mismatch")
    require(decision["final_action"] == recomputed["selected_action"], "rule final action mismatch")
    require(decision["candidate_actions"] == recomputed["guard"]["candidate_actions"], "rule candidate set mismatch")
    require(decision["excluded_actions"] == recomputed["guard"]["excluded_actions"], "rule exclusion mismatch")
    require([p["probability"] for p in decision["probabilities"]] == recomputed["rank_based_probabilities"],
            "rule rank carrier mismatch")
    return recomputed


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


def validate_manifest_file(path):
    require(Path(path).resolve()==MANIFEST.resolve(), "analysis manifest is not canonical")
    require(sha(path)==MANIFEST_SHA256, "analysis manifest hash mismatch")


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
        parents.append({"parent_id":parent,"difference":sum(finite(r["model_minus_rule_utility"]) for r in rows)/3})
    values = [r["difference"] for r in parents]
    rng = random.Random(20260924)
    samples = sorted(sum(rng.choice(values) for _ in range(8))/8 for _ in range(10000))
    def quantile(p):
        x = (len(samples)-1)*p
        i = int(x)
        return samples[i] + (x-i)*(samples[min(i+1,len(samples)-1)]-samples[i])
    ci = [quantile(.025),quantile(.975)]
    decision = "worthwhile_model_gain" if ci[0]>.01 else "below_minimum_worthwhile_gain" if ci[1]<.01 else "inconclusive_about_worthwhile_gain"
    return {"mean":sum(values)/8,"ci95":ci,"parents":parents,"bootstrap_seed":20260924,"minimum_meaningful_effect":.01,
            "bootstrap_replicates":10000,"independent_unit":"parent","decision":decision,
            "claim_limit":"frozen model combination versus named public rule on observed saved prefixes; no isolated world, preference or generalization claim"}


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
            require(branch[field] == (len(branch_steps) if item["arm"]=="frozen_model" else 0), "actual model forward count mismatch")
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
            if item["arm"] == "public_rule":
                for hidden in ("policy_hidden", "world_hidden"):
                    require(step[hidden+"_before_sha256"] == step[hidden+"_after_sha256"],
                            "rule arm changed inert learned hidden")
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
                      "frozen_model":left,"public_rule":right,"model_minus_rule_utility":difference,
                      "descriptive_tie":abs(difference)<=1e-6})
    primary=primary_statistics(pairs)
    primary["arm_macro_means"]={arm:sum(pair[arm]["utility"] for pair in pairs)/24 for arm in ARMS}
    primary["repeat_differences"]={f"parent-{p:02d}":[row["model_minus_rule_utility"] for row in pairs if row["parent_id"]==f"parent-{p:02d}"] for p in range(8)}
    return {"status":"complete","paired_results":pairs,"primary":primary,
            "verified_steps":len(steps),"branch_count":len(branches),"budget_reconciliation":"exact reservation/branch/step/finalization/SQLite join"}


def summarize_secondary(paired_results):
    """Keep all-task and registered cohort outcomes separate."""
    metrics = ("completed_delta", "all_task_completed", "cohort_on_time_physical", "cohort_on_time_host",
               "all_recorded_on_time_physical", "all_recorded_on_time_host", "energy_used",
               "accepted_count", "rejected_count", "task_unavailable_count", "submit_count",
               "noop_count_from_actions", "guard_trigger_count", "action_change_count", "env_steps")
    totals = {arm:{name:sum(finite(pair[arm][name]) for pair in paired_results) for name in metrics} for arm in ARMS}
    differences = {name:totals["frozen_model"][name]-totals["public_rule"][name] for name in metrics}
    return {"arm_totals":totals,"model_minus_rule_totals":differences,
            "note":"descriptive totals only; primary inference uses8parent macro means, not pooled decisions or tasks"}


def analyze_decision_space(manifest, decisions, steps):
    """Actual observations and selected feedback only, never hypothetical truth."""
    d=index_unique(decisions,("branch_id","step"))
    s=index_unique(steps,("branch_id","step"))
    coverage={arm:{"decision_rows":0,"mask_nonnoop_0":0,"mask_nonnoop_1":0,"mask_nonnoop_2plus":0,
        "guard_nonnoop_0":0,"guard_nonnoop_1":0,"guard_nonnoop_2plus":0,"legal_noop_retained":0} for arm in ARMS}
    for item in manifest:
        arm=item['arm']
        for (branch,step),row in d.items():
            if branch!=item['branch_id']:continue
            count=coverage[arm];count['decision_rows']+=1
            for label,n in [('mask',sum(bool(x) for x in row['legal_mask'][:24])),('guard',sum(a!=24 for a in row['candidate_actions']))]:
                count[label+'_nonnoop_'+('0' if n==0 else '1' if n==1 else '2plus')]+=1
            count['legal_noop_retained']+=int(bool(row['legal_mask'][24]) and 24 in row['candidate_actions'])
    first_divergences=[]
    for pair_id in dict.fromkeys(x['pair_id'] for x in manifest):
        branches=[pair_id+'|'+arm for arm in ARMS]
        shared_steps=sorted({k[1] for k in d if k[0]==branches[0]} & {k[1] for k in d if k[0]==branches[1]})
        found=None
        for k in shared_steps:
            a,b=[d[(branch,k)] for branch in branches]
            sa,sb=[s[(branch,k)] for branch in branches]
            if a['final_action']==b['final_action']:continue
            same_public=a['public_observation_sha256']==b['public_observation_sha256']
            same_env=sa['env_state_before_sha256']==sb['env_state_before_sha256']
            found=dict(pair_id=pair_id,step=k,same_public_input=same_public,same_env_state=same_env,
                model_action=a['final_action'],rule_action=b['final_action'],model_feedback=sa['feedback'],rule_feedback=sb['feedback'],
                both_selected_allocations_accepted=(same_public and same_env and a['final_action']!=24 and b['final_action']!=24 and sa['feedback']=='accepted' and sb['feedback']=='accepted'))
            break
        first_divergences.append(found or dict(pair_id=pair_id,step=None,no_selected_action_divergence_in_shared_step_range=True))
    return dict(coverage=coverage,first_divergences=first_divergences,
        interpretation='Acceptances describe only selected actions. Outcome differences are paired terminal observations, not unique causal attribution of a single action. No acceptance label is inferred for an unexecuted candidate.')


def validate_runtime_evidence(selector_rows, decisions, steps, manifest, status, costs, gate):
    """Validate selector provenance and the runner's hard runtime accounting.

    The primary analysis validates rewards, state transitions, and budget joins.
    This second pass covers evidence that is specific to the two selector arms:
    one selector row per actual probe, public-only rule inputs, zero model calls
    on the rule arm, three model calls on the frozen-model arm, and consistent
    aggregate counters/timings in both runtime artifacts.
    """
    validate_manifest(manifest)
    by_branch = index_unique(manifest, ("branch_id",))
    d_index = index_unique(decisions, ("branch_id", "step"))
    s_index = index_unique(steps, ("branch_id", "step"))
    selector_index = index_unique(selector_rows, ("branch_id", "step"))
    require(set(selector_index) == set(d_index) == set(s_index), "selector/decision/step coverage mismatch")

    call_names = ("policy_encode", "world_candidate_batch", "actor_readout")
    expected_model_calls = {name: 1 for name in call_names}
    expected_rule_calls = {name: 0 for name in call_names}
    expected_totals = {name: 0 for name in call_names}
    arm_steps = {arm: 0 for arm in ARMS}
    arm_selector_seconds = {arm: 0.0 for arm in ARMS}
    timing_totals = defaultdict(float)

    def call_status(values):
        return {name: {"attempted": int(values[name]), "completed": int(values[name])} for name in call_names}

    for key, selector in selector_index.items():
        branch_id, step_number = key
        item = by_branch[branch_id]
        decision = d_index[key]
        step = s_index[key]
        arm = str(item["arm"])
        expected = expected_model_calls if arm == "frozen_model" else expected_rule_calls
        arm_steps[arm] += 1

        require(selector.get("branch_id") == branch_id and selector.get("pair_id") == item["pair_id"],
                "selector branch/pair identity mismatch")
        require(selector.get("exogenous_key") == item["exogenous_key"],
                "selector exogenous identity mismatch")
        require(selector.get("arm") == arm and int(selector.get("step", -1)) == int(step_number),
                "selector arm/step identity mismatch")
        for row, label in ((decision, "decision"), (step, "step")):
            require(row.get("arm") == arm and row.get("pair_id") == item["pair_id"],
                    label + " arm/pair identity mismatch")
            require(row.get("exogenous_key") == item["exogenous_key"], label + " exogenous identity mismatch")

        require(selector.get("public_observation_sha256") == decision.get("public_observation_sha256") == step.get("public_observation_sha256"),
                "selector public observation mismatch")
        require(selector.get("selected_action") is not None, "selector selected action is missing")
        require(selector.get("selected_action") == decision.get("final_action") == step.get("action"),
                "selector selected action mismatch")
        require(selector.get("original_action") == decision.get("original_action") == step.get("original_action"),
                "selector original action mismatch")
        require(selector.get("legal_mask") == decision.get("legal_mask") == step.get("legal_mask"),
                "selector legal mask mismatch")
        require(selector.get("probabilities") == decision.get("probabilities") == step.get("probabilities"),
                "selector probability ledger mismatch")
        require(selector.get("guard_candidates") == decision.get("candidate_actions"),
                "selector guard candidate mismatch")
        require(selector.get("model_call_status") == decision.get("model_call_status") == step.get("model_call_status"),
                "selector model call status mismatch")

        require(selector.get("model_calls") == expected, "per-probe model call declaration mismatch")
        require(selector.get("model_call_status") == call_status(expected), "per-probe model call completion mismatch")
        require(int(selector.get("rule_selector_calls", -1)) == (1 if arm == "public_rule" else 0),
                "per-probe rule selector count mismatch")
        require(selector.get("rule_selector_status") == {
            "attempted": 1 if arm == "public_rule" else 0,
            "completed": 1 if arm == "public_rule" else 0,
        }, "per-probe rule selector status mismatch")
        selector_seconds = finite(selector.get("selector_seconds", 0.0))
        require(selector_seconds >= 0.0, "negative selector timing")
        arm_selector_seconds[arm] += selector_seconds

        public_hash = selector.get("public_rule_input_sha256")
        public_only_hash = selector.get("public_only_input_sha256")
        require(public_hash == public_only_hash and isinstance(public_hash, str) and len(public_hash) == 64,
                "public input digest mismatch")
        if arm == "public_rule":
            require(selector.get("input_kind") == "public_observation_only", "rule selector input kind mismatch")
            require(selector.get("hidden_transition") == "identity_audit_only", "rule hidden transition marker missing")
            public_inputs = selector.get("public_rule_inputs")
            require(isinstance(public_inputs, dict), "public rule inputs are missing")
            require(set(public_inputs) == {"uavs", "tasks", "time", "mask", "public_entity_ids", "continuation_actions"},
                    "public rule input whitelist changed")
            require(canonical_hash(public_inputs) == public_hash, "public rule input digest mismatch")
            recomputed = public_rule_module().select_public_dispatch_action(public_inputs).to_dict()
            validate_public_rule_record(public_inputs, recomputed, decision)
            require(selector.get("rule_scores") == recomputed["rank_based_probabilities"],
                    "saved rule scores differ from recomputation")
            require(selector.get("rule_ranks") == recomputed["order"], "saved rule ranks differ from recomputation")
        else:
            require(selector.get("input_kind") == "model_probe", "model selector input kind mismatch")
            require("public_rule_inputs" not in selector and "rule_scores" not in selector and "rule_ranks" not in selector,
                    "model selector contains public-rule evidence")

        for name, count in expected.items():
            expected_totals[name] += count
        for name, value in selector.get("model_call_status", {}).items():
            require(int(value.get("attempted", -1)) == int(value.get("completed", -1)),
                    "incomplete model call in selector ledger: " + str(name))
        for field in ("public_observation_sha256", "public_rule_input_sha256", "public_only_input_sha256"):
            require(isinstance(selector.get(field), str) and len(selector[field]) == 64,
                    "invalid selector digest: " + field)

    require(len(steps) <= 768, "environment step cap exceeded")
    require(sum(arm_steps.values()) == len(steps), "arm step accounting mismatch")
    require(arm_steps["frozen_model"] + arm_steps["public_rule"] == len(steps), "arm coverage mismatch")
    expected_model_counts = {
        "attempted": dict(expected_totals),
        "completed": dict(expected_totals),
        "limits": {name: 384 for name in call_names},
        "attempted_total": sum(expected_totals.values()),
        "completed_total": sum(expected_totals.values()),
    }
    require(status.get("status") == "completed" and costs.get("status") == "completed", "runtime not completed")
    require(status.get("schema") == "model-rule-pair-run-status/1.0.0", "run status schema mismatch")
    require(costs.get("schema") == "model-rule-pair-runtime-cost/1.0.0", "runtime cost schema mismatch")
    require(status.get("model_call_counts") == expected_model_counts == costs.get("model_call_counts"),
            "global model call discrepancy")
    require(status.get("rule_selector_calls") == costs.get("rule_selector_calls") == arm_steps["public_rule"],
            "global rule selector discrepancy")
    require(status.get("runtime_digest_before") and status.get("runtime_digest_before") == status.get("runtime_digest_after"),
            "frozen runtime mutated")
    require(status.get("no_retry") is True and status.get("failure_is_algorithm_result") is False and
            status.get("result_classification") == "evaluation_completed", "runtime classification changed")
    require(costs.get("failure_is_algorithm_result") is False and costs.get("result_classification") == "evaluation_completed",
            "runtime cost classification changed")

    expected_hard = {
        "probe_calls": len(steps), "env_step_calls": len(steps), "env_steps": len(steps),
        "successful_env_steps": len(steps), "verified_steps": len(steps),
        "model_forward": 3 * arm_steps["frozen_model"],
        "actor_forward": arm_steps["frozen_model"], "world_forward": arm_steps["frozen_model"],
        "optimizer_updates": 0, "world_updates": 0, "offline_updates": 0,
        "branches_completed": len(manifest),
    }
    for obj in (status, costs):
        hard = obj.get("hard_counts") if isinstance(obj.get("hard_counts"), dict) else {}
        require(all(int(hard.get(name, -1)) == value for name, value in expected_hard.items()),
                "runtime hard counter discrepancy")

    expected_limits = {
        "environment_steps": 768, "policy_encode": 384, "world_candidate_batch": 384,
        "actor_readout": 384, "rule_selector_calls": 384, "updates": 0,
    }
    require(costs.get("limits") == expected_limits, "runtime limits changed")
    budget_totals = status.get("budget_totals", {}).get("environment_steps", {})
    require(int(budget_totals.get("reserved", -1)) == len(steps) and int(budget_totals.get("verified", -1)) == len(steps)
            and int(budget_totals.get("unknown", -1)) == 0 and int(budget_totals.get("pending", -1)) == 0,
            "runtime budget totals mismatch")

    for obj in (status, costs):
        per_arm = obj.get("hard_counts_per_arm")
        require(isinstance(per_arm, dict) and set(per_arm) == set(ARMS), "per-arm runtime counters missing")
        for arm in ARMS:
            row = per_arm[arm]
            expected_row = {
                "branches": sum(1 for item in manifest if item["arm"] == arm),
                "steps": arm_steps[arm], "probe_calls": arm_steps[arm],
                "actor_forward": arm_steps[arm] if arm == "frozen_model" else 0,
                "world_forward": arm_steps[arm] if arm == "frozen_model" else 0,
                "rule_selector_calls": arm_steps[arm] if arm == "public_rule" else 0,
            }
            require(all(int(row.get(name, -1)) == value for name, value in expected_row.items()),
                    "per-arm runtime counter discrepancy")
            require(close(row.get("selector_seconds", -1.0), arm_selector_seconds[arm], 1e-5),
                    "per-arm selector timing discrepancy")

    require(gate.get("schema") == "model-rule-pair-first-gate/1.0.0" and gate.get("ok") is True,
            "first-pair gate incomplete")
    first_pair = manifest[0]["pair_id"]
    require(gate.get("pair_id") == first_pair and int(gate.get("step", -1)) == 1,
            "first-pair gate identity mismatch")
    checks = gate.get("checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()),
            "first-pair gate checks failed")

    branch_timing_index = index_unique(costs.get("branch_timings", []), ("branch_id",))
    require(set(key[0] for key in branch_timing_index) == set(by_branch), "branch timing coverage mismatch")
    arm_timings = {arm: {"branch_seconds": 0.0, "selector_seconds": 0.0,
                         "environment_and_runner_overhead_seconds": 0.0} for arm in ARMS}
    for (branch_id,), row in branch_timing_index.items():
        item = by_branch[branch_id]
        require(row.get("pair_id") == item["pair_id"] and row.get("arm") == item["arm"], "branch timing identity mismatch")
        branch_seconds = finite(row.get("branch_seconds"))
        selector_seconds = finite(row.get("selector_seconds"))
        overhead = finite(row.get("environment_and_runner_overhead_seconds"))
        require(branch_seconds >= 0.0 and selector_seconds >= 0.0 and overhead >= 0.0,
                "negative branch timing")
        require(close(branch_seconds, selector_seconds + overhead, 1e-5), "timing decomposition mismatch")
        arm = item["arm"]
        arm_timings[arm]["branch_seconds"] += branch_seconds
        arm_timings[arm]["selector_seconds"] += selector_seconds
        arm_timings[arm]["environment_and_runner_overhead_seconds"] += overhead
    for name in ("setup_seconds", "model_load_seconds", "shared_snapshot_load_seconds", "wall_seconds"):
        value = finite(costs.get(name))
        require(value >= 0.0, "invalid runtime timing: " + name)
        timing_totals[name] = value
    for (branch_id, step_number), selector in selector_index.items():
        timing_totals["selector_seconds"] += finite(selector.get("selector_seconds", 0.0))

    return {
        "model_call_counts": dict(expected_totals),
        "rule_selector_calls": arm_steps["public_rule"],
        "per_probe_timing_totals": dict(timing_totals),
        "setup_seconds": timing_totals["setup_seconds"],
        "model_load_seconds": timing_totals["model_load_seconds"],
        "shared_snapshot_load_seconds": timing_totals["shared_snapshot_load_seconds"],
        "arm_timings": arm_timings,
        "wall_seconds": timing_totals["wall_seconds"],
        "same_input_gate_pair": first_pair,
        "cost_interpretation": "selector timing is recorded per arm; shared prefix deserialization and model load are reported separately",
    }


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
        validate_manifest_file(args.manifest)
        before = sha(args.sqlite)
        with sqlite3.connect(args.sqlite.resolve().as_uri()+"?mode=ro",uri=True) as con:
            con.row_factory = sqlite3.Row
            require(con.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity")
            stages = [dict(r) for r in con.execute("SELECT * FROM stages")]
            require(all(r["unknown"]==0 and r["reserved"]==r["verified"] for r in stages), "unresolved budget")
            reservations = [dict(r) for r in con.execute("SELECT * FROM reservations WHERE run_id=?",(args.run_id,))]
        files = {name:args.run/name for name in ("branch-results.jsonl","step-vector-rewards.jsonl","decision-ledger.jsonl","budget-finalization.jsonl","selector-ledger.jsonl","run-status.json","runtime-costs.json","first-pair-gate.json")}
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = analyze_core(manifest,
            read_jsonl(files["branch-results.jsonl"]),read_jsonl(files["step-vector-rewards.jsonl"]),
            read_jsonl(files["decision-ledger.jsonl"]),read_jsonl(files["budget-finalization.jsonl"]),
            reservations,args.run_id,args.attempt_id)
        result["secondary"] = summarize_secondary(result["paired_results"])
        result["decision_space"] = analyze_decision_space(manifest,read_jsonl(files["decision-ledger.jsonl"]),read_jsonl(files["step-vector-rewards.jsonl"]))
        result["resource_evidence"] = validate_runtime_evidence(read_jsonl(files["selector-ledger.jsonl"]),
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
