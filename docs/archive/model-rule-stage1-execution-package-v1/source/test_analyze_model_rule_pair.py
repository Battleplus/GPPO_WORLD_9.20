from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import struct
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("event_pair_analysis", ROOT/"tools/analyze_model_rule_pair_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    manifest = json.loads((m.BASE/"model-rule-stage1-execution-package-v1/paired-manifest.json").read_text(encoding="utf-8"))
    branches, steps, decisions, finals, reservations = [], [], [], [], []
    f32 = lambda x:struct.unpack("f",struct.pack("f",x))[0]
    for item in manifest:
        reward = int(item["arm"] == "frozen_model")
        ident = {k:item[k] for k in ("branch_id","control_branch_key","pair_id","parent_id","prefix_id","condition","model_seed","repeat","arm","exogenous_key","split")}
        probs=[{"action":a,"legal":a in (0,24),"probability":.2 if a==0 else .8 if a==24 else 0.0,
                "original_argmax":a==24,"selected":a==24} for a in range(25)]
        mask = [int(a in (0,24)) for a in range(25)]
        rid = "reservation-"+str(item["order"])
        shared = {**ident,"step":1,"reservation_id":rid,"legal_mask":mask,"probabilities":probs,
                  "original_action":24,"public_observation_sha256":"obs0","policy_hidden_before_sha256":"ph0",
                  "world_hidden_before_sha256":"wh0","selected_action_hidden_sha256":"wh1"}
        outcomes = {t:{"physical_arrival_before_deadline":bool(reward and t=="task-0"),"host_confirmation_before_deadline":bool(reward and t=="task-0")} for t in item["cohort_task_ids"]}
        decisions.append({**shared,"branch_exogenous_key":item["exogenous_key"],"candidate_actions":[0,24],"excluded_actions":[],
             "action_task_ids":[[0,"task-0"]],"continuation_actions":[],"continuation_task_ids":[],"final_action":24,
             "original_probability":.8,"final_probability":.8,"triggered":False})
        steps.append({**shared,"budget_status":"reserved_pending_finalization","action":24,"gamma_index":0,"gamma":.99,
             "reward_scales":{"task":.5,"energy":1.0},"behavior_preference":[.8,.2],"task_capacity":6,"uav_count":4,
             "completed_before":0,"completed_after":reward,"completed_delta":reward,"expired_before":0,"expired_after":0,"expired_delta":0,
             "energy_before":36.0,"energy_after":35.64,"energy_used_delta":.36,"configured_initial_energy":9.0,
             "vector_reward":[f32(reward/6),f32(-.36/36)],"terminated":True,"truncated":False,"submit_command":True,"feedback":"noop",
             "actual_feedback":{"feedback":"noop","completion_records":outcomes},"world_hidden_after_sha256":"wh1",
             "policy_hidden_after_sha256":"ph1","public_next_observation_sha256":"obs1"})
        branches.append({**ident,"env_steps":1,"rows":1,"verified_steps":1,"successful_env_steps":1,"terminated":True,"truncated":False,
             "end_reason":"terminated","initial_counts":{"completed":0,"expired":0},"final_counts":{"completed":reward,"expired":0},
             "initial_energy":36.0,"final_energy":35.64,"energy_used":.36,"actor_forward_calls":int(item["arm"]=="frozen_model"),"world_forward_calls":int(item["arm"]=="frozen_model"),
             "command_count":1,"task_unavailable_count":0,"guard_trigger_count":0,"action_change_count":0,"noop_count":1,
             "deepcopy_isolated":True,"source_snapshot_digest_before":"snapshot","source_snapshot_digest_after":"snapshot",
             "source_snapshot_semantics_before":{"policy_hidden_sha256":"ph0","world_hidden_sha256":"wh0"},
             "source_snapshot_semantics_after":{"policy_hidden_sha256":"ph0","world_hidden_sha256":"wh0"},
             "initial_env_state_sha256":"env0","initial_public_observation_sha256":"obs0",
             "counter_delta":{"optimizer_updates":0,"world_updates":0,"offline_updates":0},"last_info":{"completion_records":outcomes}})
        finals.append({"reservation_id":rid,"branch_id":item["branch_id"],"step":1,"status":"verified","actual_reservation_status":"verified"})
        reservations.append({"reservation_id":rid,"run_id":"run","attempt_id":"attempt","stage":"environment_steps","amount":1,"status":"verified"})
        if item["arm"] == "public_rule":
            for row in (decisions[-1], steps[-1]):
                row["selected_action_hidden_sha256"] = "wh0"
            steps[-1]["world_hidden_after_sha256"] = "wh0"
            steps[-1]["policy_hidden_after_sha256"] = "ph0"
    return [manifest,branches,steps,decisions,finals,reservations,"run","attempt"]


def test_complete_matrix_uses_both_guard_arms_and_parent_units():
    args=fixture()
    before=copy.deepcopy(args)
    result=m.analyze_core(*args)
    assert result["status"]=="complete"
    assert result["branch_count"]==48 and result["verified_steps"]==48
    assert len(result["primary"]["parents"])==8
    assert result["primary"]["mean"]==pytest.approx(.4*struct.unpack("f",struct.pack("f",1/6))[0])
    assert result["primary"]["ci95"][0]>0
    assert args==before


@pytest.mark.parametrize("fault", ["duplicate","missing","randomness","reward","hidden","finalization","sqlite","mask","update","termination","snapshot","pair_state"])
def test_corrupt_evidence_never_produces_primary(fault):
    args=fixture()
    if fault=="duplicate":args[2].append(copy.deepcopy(args[2][0]))
    elif fault=="missing":args[1].pop()
    elif fault=="randomness":args[0][1]["exogenous_key"]="wrong"
    elif fault=="reward":args[2][0]["vector_reward"][0]=1.0
    elif fault=="hidden":args[2][0]["world_hidden_after_sha256"]="wrong"
    elif fault=="finalization":args[4][0]["branch_id"]="wrong"
    elif fault=="sqlite":args[5][0]["status"]="unknown"
    elif fault=="mask":args[3][0]["candidate_actions"].remove(24)
    elif fault=="update":args[1][0]["counter_delta"]["world_updates"]=1
    elif fault=="termination":args[2][0]["terminated"]=False
    elif fault=="snapshot":args[1][0]["source_snapshot_digest_after"]="wrong"
    else:args[1][0]["initial_env_state_sha256"]="wrong"
    with pytest.raises(ValueError):m.analyze_core(*args)


def test_exact_null_and_negative_have_fixed_decisions():
    pairs=[{"parent_id":f"parent-{p:02d}","repeat":r,"model_minus_rule_utility":0.0} for p in range(8) for r in range(3)]
    assert m.primary_statistics(pairs)["decision"]=="below_minimum_worthwhile_gain"
    for row in pairs:row["model_minus_rule_utility"]=-.02
    assert m.primary_statistics(pairs)["decision"]=="below_minimum_worthwhile_gain"


def test_parent_not_decision_bootstrap_and_repeat_completeness():
    pairs=[{"parent_id":f"parent-{p:02d}","repeat":r,"model_minus_rule_utility":float(p)} for p in range(8) for r in range(3)]
    result=m.primary_statistics(pairs)
    assert result["mean"]==3.5
    assert result["ci95"]==m.primary_statistics(pairs)["ci95"]
    pairs[-1]["repeat"]=1
    with pytest.raises(ValueError):m.primary_statistics(pairs)



def test_practical_threshold_is_not_zero_significance():
    pairs=[dict(parent_id=f"parent-{p:02d}",repeat=r,model_minus_rule_utility=.005) for p in range(8) for r in range(3)]
    assert m.primary_statistics(pairs)["decision"]=="below_minimum_worthwhile_gain"
    for row in pairs:row["model_minus_rule_utility"]=.02
    assert m.primary_statistics(pairs)["decision"]=="worthwhile_model_gain"

def test_rule_arm_model_calls_cannot_be_faked():
    args=fixture()
    next(b for b in args[1] if b["arm"]=="public_rule")["world_forward_calls"]=1
    with pytest.raises(ValueError):m.analyze_core(*args)


def test_candidate_space_does_not_infer_unexecuted_acceptance():
    manifest,_,steps,decisions,*_=fixture()
    for row in steps:row['env_state_before_sha256']='same-env'
    a,b=decisions[:2]
    a['final_action']=0;b['final_action']=1
    steps[0]['feedback']=steps[1]['feedback']='accepted'
    result=m.analyze_decision_space(manifest,decisions,steps)
    assert result['first_divergences'][0]['both_selected_allocations_accepted'] is True
    steps[1]['env_state_before_sha256']='different-env'
    result=m.analyze_decision_space(manifest,decisions,steps)
    assert result['first_divergences'][0]['both_selected_allocations_accepted'] is False
    assert result['coverage']['public_rule']['legal_noop_retained']==24
