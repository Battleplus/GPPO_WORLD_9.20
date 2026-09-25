"""Finalize and verify only the independent decision package (stdlib only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'results-first-20260924-v1'
SRC = Path(r'E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot')
WT = Path(r'E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt')
RUNS = WT / 'runs/finite-communication-ack-lease-fix-20260920'
RUN = RUNS / 'model-rule-stage1-cli-recovery-run-v1'
PACKAGE = RUNS / 'model-rule-stage1-execution-package-v1'
CP = SRC.parent / 'training/seed-1101/P_train/last-recovery.pt'
CONFIG = Path(r'E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final\run\training\seed-1101\WD\resolved-config.json')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

remote_manifest = json.loads((OLD/'remote-read-manifest.json').read_text(encoding='utf-8-sig'))
entries = []
for item in remote_manifest['files']:
    src = OLD/'repo-docs'/item['path']
    assert sha(src) == item['sha256'], str(src)
    dst = ROOT/'remote'/item['path']
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    entries.append({'path': str(dst), 'sha256': sha(dst), 'git_path': item['path'],
                    'git_blob': item['git_blob'], 'role': 'specified_remote_instruction'})
fifth = ROOT/'EVIDENCE_AND_MIGRATION.md'
assert sha(fifth) == 'ca4d9f89728a1dba9f121957f8672983c897baed2ab4e34dc1536799e4e31830'
entries.append({'path':str(fifth),'sha256':sha(fifth),'git_path':'docs/EVIDENCE_AND_MIGRATION.md',
                'git_blob':'cd36df86d9b8e2bd1f36c346ff99e3f4f12f89c2','role':'specified_remote_evidence_index'})

native = json.loads((PACKAGE/'native-source-manifest.json').read_text(encoding='utf-8-sig'))
source_names = ['gppo_world/joint_gppo.py','gppo_world/joint_training.py','gppo_world/m10_environment.py',
                'gppo_world/m10_communication.py','gppo_world/task_policy_view.py']
for name in source_names:
    p=SRC/name
    assert sha(p) == native['files'][name]['sha256'], name
    entries.append({'path':str(p),'sha256':sha(p),'role':'native_source_inspected',
                    'matches_historical_native_manifest':True})

other = [
    (RUN/'report.md','completed_comparison_report'),
    (RUN/'analysis.json','observed_development_only_statistics'),
    (RUN/'run-status.json','completed_runtime_call_evidence'),
    (RUN/'research-decision.json','historical_decision'),
    (PACKAGE/'protocol.json','prior_frozen_contract'),
    (PACKAGE/'runtime-source.py','prior_native_runtime_call_path'),
    (PACKAGE/'native-source-manifest.json','native_identity_reference'),
    (WT/'tools/run_replan_value_experiment.py','actual_strict_model_loader_source'),
    (WT/'gppo_world/ack_known_task_guard.py','prior_public_guard_definition'),
    (CONFIG,'resolved_inherited_config'),
    (CP,'checkpoint_bytes_hash_only_not_deserialized'),
]
for p, role in other: entries.append({'path':str(p),'sha256':sha(p),'role':role})
assert sha(CP) == 'bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a'
config=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
assert config['environment']['completion_notice_mode']=='single_shot'
assert config['environment']['task_completion_mode']=='arrival_to_region'
assert config['environment']['deadline_basis']=='physical_arrival'
assert config['environment']['horizon']==18.0
status=json.loads((RUN/'run-status.json').read_text(encoding='utf-8-sig'))
assert status['status']=='completed' and status['hard_counts']['branches_completed']==48
assert status['model_call_counts']['completed']=={'policy_encode':279,'world_candidate_batch':279,'actor_readout':279}
analysis=json.loads((ROOT/'analysis.json').read_text(encoding='utf-8'))
assert analysis['planning']['proposed_parent_n']==1600
assert analysis['planning']['formal_step_cap']==172800
assert analysis['mask_fixture_applicability']['blocking_for_frozen_plan'] is False
assert analysis['planning']['design_alternative']==.02

write('evidence-index.json',{
    'repository':'Battleplus/GPPO_WORLD_9.20',
    'branch':'docs/results-first-roadmap-20260924',
    'commit':remote_manifest['commit'],
    'remote_branch_head_rechecked_this_turn':True,
    'inputs':entries,
    'model_interface':{
        'policy_class':'JointGraphPreferencePolicy', 'world_class':'ActionConditionedTemporalWorldModel',
        'candidate_shape':[1,25,17],
        'features':{'policy_feature':8,'next_vector_reward':2,'next_completed_expired_increments':2,'public_event_probabilities':5},
        'actor_consumes_candidate_features':True,
        'recursive_rollout_used_in_frozen_probe':False,
        'calibrated_arrival_probability_claim':False,
        'not_the_D02_global_latent_adapter':True,
        'citations':{'joint_gppo.py':[74,103,131,146,167,187,207],
                     'joint_training.py':[368,766,772,781],
                     'runtime-source.py':[2571,2592,2610,2620]},
    },
    'scope':{'no_checkpoint_deserialization':True,'no_env_reset_or_step':True,
             'no_model_forward':True,'no_sqlite_access':True,'no_unpublished_validation_heldout_test_results':True},
})

seed_manifest=[]
for stage,n,base,repeats in [('gate',2,924100000,1),('formal',1600,924200000,3)]:
    for i in range(n):
        parent=f'gate-{i:03d}' if stage=='gate' else f'eval-{i:04d}'
        seed_manifest.append({'stage':stage,'parent_id':parent,'seed':base+i,'repeats':list(range(repeats)),
            'exogenous_keys':[f'frozen-native-public-history-allocation-v1|{parent}|repeat-{r}' for r in range(repeats)]})
assert len({row['seed'] for row in seed_manifest})==1602
write('planned-parent-manifest.json',{
    'status':'planned_seeds_only_no_scenarios_generated_no_attempts',
    'generator':'formal_three_condition_tape', 'native_split':'train', 'condition':'W1','name':'mixed',
    'formal_experiment_label':'independent_development_eval',
    'historical_identity_collision_check':'required_before_any_future_execution_using_permitted_identity_metadata',
    'parents':seed_manifest,
})

resources={
    'technical_gate':{'parents':2,'repeats':1,'arms':2,'episodes':4,'max_steps_per_episode':18,
        'environment_steps':72,'policy_encode':36,'world_candidate_batch':36,'actor_readout':36,
        'wall_hours':1,'artifact_gib':1},
    'formal':{'parents':1600,'repeats':3,'arms':2,'episodes':9600,'max_steps_per_episode':18,
        'environment_steps':172800,'policy_encode':86400,'world_candidate_batch':86400,'actor_readout':86400,
        'wall_hours':36,'artifact_gib':40},
    'all_stages':{'environment_steps':172872,'policy_encode':86436,'world_candidate_batch':86436,
        'actor_readout':86436,'all_three_forward_total':259308,'updates':0,'gpu_hours':0},
    'authorization':'none_for_execution_or_budget_mutation',
    'separate_approval_for_technical_gate_and_formal':True,
    'all_failure_calls_included':True,'no_auto_retry_or_sample_expansion':True,
    'historical_balance_use':False,
}
for s in ['technical_gate','formal']:
    r=resources[s]
    assert r['parents']*r['repeats']*r['arms']==r['episodes']
    assert r['episodes']*r['max_steps_per_episode']==r['environment_steps']
    assert r['parents']*r['repeats']*r['max_steps_per_episode']==r['world_candidate_batch']
assert resources['all_stages']['environment_steps']==sum(resources[s]['environment_steps'] for s in ['technical_gate','formal'])
write('decision.json',{
    'decision':'A','meaning':'Recommend bounded formal validation after explicit stage-specific authorization.',
    'single_hypothesis':'Frozen public-history and action-conditional world features yield worthwhile task utility over a reasonable public-history planner.',
    'why_not_B_or_C':'Model interface matches a falsifiable hypothesis; weak-rule advantage and independent task benefit are questions for the experiment, not required advance proofs. The discovered bounded_retry-only mask fixture is not reachable in frozen single_shot.',
    'unproven':['independent task increment','world-specific contribution','forecast calibration','multi-training-seed generalization','deployment economics'],
    'primary_comparison':'M_minus_R','practical_margin_native_utility':.01,
    'primary_ci':'single_comparison_parent_percentile_bootstrap_95pct_10000_seed20260924',
    'resource_proposal':resources,
    'cost_gate':{'controller_mean_cpu_ms_max':10,'controller_wall_p95_ms_max':50,'process_peak_rss_gib_max':4},
    'adoption_guardrails':'Parent macro on-time physical and on-time host confirmation means must not fall; not a noninferiority proof.',
    'decisions_after_authorized_run':{
        'worthwhile':'CI lower>.01, task and cost gates pass: separately budget world attribution.',
        'insufficient':'CI upper<=.01 or task/cost gate fails: simplify/stop this frozen scheme.',
        'uncertain':'CI straddles .01 or evidence not comparable: state gap and stop without added samples/training.',
    },
    'technical_stop':['identity_or_contract_mismatch','information_leak','illegal_action','reward_or_hidden_mismatch',
                      'budget_pending_unknown_or_duplicate','missing_labels','non_native_truncation','resource_cap','first_gate_failure'],
    'prior_plan_disposition':'v1/v2 preserved, their proof-before-experiment gate and 64-parent formal budget superseded.',
    'unapplied_mask_patch_disposition':'Do not apply; outside the frozen single_shot execution path.',
    'current_turn':analysis['hard_counts'],
    'no_progress_audit':{'classification':'progress','evidence':'Actual interface inspected, power recalculated, scoped defect reachability resolved, one decision and executable protocol created.'},
})

required=['report.md','protocol.md','decision.json','evidence-index.json']
assert all((ROOT/name).is_file() and (ROOT/name).stat().st_size>100 for name in required)
report=(ROOT/'report.md').read_text(encoding='utf-8')
protocol=(ROOT/'protocol.md').read_text(encoding='utf-8')
assert '决定 A' in report and 'single_shot' in protocol
assert '172800' in protocol and '1600' in protocol and '259308' in protocol
assert 'recursive_rollout' in report and '没有调用' in report
assert 'bounded_retry' in report and '不在本实验可达路径' in report
assert 'next_state' in report and 'task_consequence' in report
write('completion-audit.json',{
    'objective':'Problem closure and reviewable frozen experiment design; no experiment execution.',
    'checks':[
        {'requirement':'Five specified remote documents read at current branch commit','status':'verified','evidence':'remote copies, EVIDENCE_AND_MIGRATION.md, evidence-index remote commit and hashes'},
        {'requirement':'One meaningful hypothesis, lawful signals, real choices and evidence for/against','status':'verified','evidence':'report sections hypothesis and evidence; protocol sections 2-5'},
        {'requirement':'Actual native model output, weights identity and actor interface checked','status':'verified','evidence':'evidence-index model_interface, native source hash checks, checkpoint byte hash, historical completed call records'},
        {'requirement':'Strong public-history baseline and same execution/information constraints','status':'specified_not_executed','evidence':'protocol sections 3 and 5, shared resource/task continuation guard'},
        {'requirement':'Independent parents and untouched final blind test','status':'specified_no_data_generated','evidence':'planned-parent-manifest, protocol section 2; no pickle/model/test load'},
        {'requirement':'Primary utility, threshold, full costs, statistical scale and resource caps','status':'verified_plan_and_arithmetic','evidence':'protocol sections 7-9, analysis planning, decision resources'},
        {'requirement':'Technical stops, failure accounting, no automatic retry/extra samples','status':'specified','evidence':'protocol sections 6 and 9; decision technical_stop'},
        {'requirement':'Exactly one A/B/C decision and conditional next action','status':'verified','evidence':'decision.json decision=A; experiment authorization=none'},
        {'requirement':'Five required deliverables','status':'verified_on_finalize','evidence':'report/protocol/decision/evidence-index/hashes'},
        {'requirement':'Authorized zero-step boundary respected','status':'verified_script_scope','evidence':'stdlib-only analysis/finalizer, raw text/JSON/hash reads, no environment/model import or SQLite access; all mutations within this planning directory'},
    ],
    'not_claimed_complete':['runner_implementation','technical_gate','formal_experiment','world_attribution','research_general_goal'],
    'current_planning_goal_complete':True,
})
hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='hashes.json'}
write('hashes.json',hashes)
assert all(sha(ROOT/name)==value for name,value in hashes.items())
assert (ROOT/'hashes.json').is_file()
print(json.dumps({'decision':'A','verified_output_files':len(hashes),'remote_docs':5,'native_sources_matched':len(source_names),
    'parent_plan_count':len(seed_manifest),'env_step_this_turn':0,'model_forward_this_turn':0,
    'formal_step_cap':172800,'report_sha256':sha(ROOT/'report.md')},ensure_ascii=False))
