"""Zero-environment-step source fixture and saved-result budget analysis.

Only stdlib; does not import gppo_world, deserialize checkpoints, or open SQLite.
The executed function is the AST-extracted TaskPolicyView.snapshot method,
with explicit public-value fixtures. These are not new scenario rollouts.
"""
import ast
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot')
RUN = Path(r'E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\model-rule-stage1-cli-recovery-run-v1')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot_function(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskPolicyView')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'snapshot')
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {
        'UAV_FIELDS': ('x','y','energy','alive','connected','idle'),
        'TASK_FIELDS': ('x','y','deadline','remaining_service','priority','pending','region_id','target_id'),
        'TASK_REQUIRED_FIELDS': ('x','y','deadline','remaining_service','priority','pending'),
        'TaskPolicySnapshot': lambda time, version, uavs, tasks, mask: SimpleNamespace(mask=mask),
    }
    exec(compile(module, '<AST-only original snapshot>', 'exec'), scope)
    return scope['snapshot']

class PublicStore:
    def __init__(self, rows): self.rows = rows
    def read(self, entity, field, now, max_age):
        return {'value': self.rows[entity][field], 'known': True, 'valid': True, 'age': 0.0}

def fixture(n, completed, stale_pending):
    tasks = [f'task-{i}' for i in range(n)]
    uavs = tuple(f'uav-{i}' for i in range(4))
    u = {uid: dict(x=0., y=0., energy=9., alive=1., connected=1., idle=1.) for uid in uavs}
    t = {tid: dict(x=1., y=1., deadline=14., remaining_service=1., priority=1.,
                  pending=float(stale_pending or tid not in completed), region_id=0., target_id=0.) for tid in tasks}
    return SimpleNamespace(_advance=lambda now: None, version=1, max_age=2.5, task_capacity=6,
        uav_ids=uavs, _tasks=tasks, _completed_tasks=set(completed), _uavs=PublicStore(u), _task_values=PublicStore(t))

path = SOURCE / 'gppo_world/task_policy_view.py'
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'f5877b6582a02a5bb6c9c39b469dd222e9f5727390fd8080c01e160cd7b23317'
old_bytes = b'            for task in tv:'
new_bytes = b'            for i, task in enumerate(tv):'
assert raw.count(old_bytes) == 1
proposed_raw = raw.replace(old_bytes, new_bytes, 1)
before = raw.decode('utf-8-sig')
after = proposed_raw.decode('utf-8-sig')
old_fn, new_fn = snapshot_function(before), snapshot_function(after)
cases = []
for n, completed, stale in [(6, [], False), (6, ['task-5'], False), (6, ['task-0'], True), (2, ['task-0'], True)]:
    public = fixture(n, completed, stale)
    old_mask = old_fn(public, 2.0).mask
    new_mask = new_fn(public, 2.0).mask
    expected = tuple(j < n and f'task-{j}' not in completed for _ in range(4) for j in range(6)) + (True,)
    assert len(old_mask) == len(new_mask) == 25
    assert new_mask == expected
    cases.append({'task_slots': n, 'public_completed': completed, 'stale_pending': stale,
        'original_nonnoop_actions': [i for i in range(24) if old_mask[i]],
        'expected_nonnoop_actions': [i for i in range(24) if expected[i]],
        'proposed_equals_public_contract': new_mask == expected,
        'original_equals_public_contract': old_mask == expected})
assert len(cases[1]['original_nonnoop_actions']) == 0
assert len(cases[1]['expected_nonnoop_actions']) == 20
patch = ''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                   fromfile='a/gppo_world/task_policy_view.py', tofile='b/gppo_world/task_policy_view.py'))
(ROOT / 'proposed-mask-index.patch').write_text(patch, encoding='utf-8', newline='')

analysis = json.loads((RUN / 'analysis.json').read_text(encoding='utf-8-sig'))
d = [r['difference'] for r in analysis['primary']['parents']]
s = statistics.stdev(d)
zcrit, zpower = statistics.NormalDist().inv_cdf(.975), statistics.NormalDist().inv_cdf(.8)
def n_required(sd, target): return math.ceil(((zcrit + zpower) * sd / (target - .01)) ** 2)
def power(n, sd, target): return statistics.NormalDist().cdf((target-.01)*math.sqrt(n)/sd-zcrit)
result = {
    'scope': 'AST snapshot public fixtures + descriptive statistics; not environment/model execution.',
    'source_sha256': sha(path), 'virtual_proposed_source_sha256': hashlib.sha256(proposed_raw).hexdigest(),
    'proposed_source_written_or_applied': False,
    'snapshot_fixture_results': cases,
    'original_failing_fixture_count': sum(not r['original_equals_public_contract'] for r in cases),
    'no_claim_of_historical_episode_impact_rate': True,
    'historical_parent_n': len(d), 'historical_parent_difference_sd': s,
    'planning': {
        'single_primary_comparison': 'frozen_model minus public_history_planner',
        'practical_margin': .01, 'design_alternative': .02,
        'normal_approx_parent_n_for_80pct_power_at_03': n_required(s,.03),
        'normal_approx_parent_n_at_15x_sd': n_required(1.5*s,.03),
        'normal_approx_parent_n_at_02': n_required(s,.02),
        'previous_n64_expected_95pct_halfwidth': zcrit*s/math.sqrt(64),
        'proposed_parent_n': 1600, 'repeats_per_parent': 3,
        'n400_approx_power_at_03': power(400,s,.03),
        'n400_approx_power_at_03_sd15x': power(400,1.5*s,.03),
        'n400_approx_power_at_02': power(400,s,.02),
        'n400_approx_halfwidth_sd15x': zcrit*1.5*s/math.sqrt(400),
        'n1600_approx_power_at_02_sd15x': power(1600,1.5*s,.02),
        'n1600_approx_probability_upper_ci_below_margin_if_true_zero_sd15x': power(1600,1.5*s,.02),
        'n1600_approx_halfwidth_sd15x': zcrit*1.5*s/math.sqrt(1600),
        'n1600_approx_power_at_02_sd2x': power(1600,2*s,.02),
        'n1600_approx_power_at_03_sd2x': power(1600,2*s,.03),
        'full_episodes': 1600*3*2, 'formal_step_cap': 1600*3*2*18,
        'technical_gate_episodes': 2*2, 'technical_gate_step_cap': 2*2*18,
        'transport_caveat': '8 observed prefixes vs old rule do not establish new full-episode planner variance; power is a sensitivity calculation, not a guarantee.',
    },
    'mask_fixture_applicability': {
        'frozen_completion_notice_mode': 'single_shot',
        'mark_completed_callsite': 'm10_environment.py:588 in _accept_bounded_completion_notice',
        'single_shot_receive': '_accept_message does not call mark_completed',
        'blocking_for_frozen_plan': False,
        'instruction': 'Do not apply the illustrative patch or switch to bounded_retry in this experiment.',
    },
    'hard_counts': {'env_step':0,'model_forward':0,'updates':0,'new_attempt':0,'sqlite_writes':0},
}
(ROOT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'fixture_cases':len(cases),'original_failures':result['original_failing_fixture_count'],
    'proposed_virtual_fixtures_passed':True,'planning':result['planning'],'env_step':0},ensure_ascii=False))
