"""Read existing JSON/source only. Never import an experiment module."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = ROOT / 'gppo-world-runtime-minimal-paired-20260925'
STAGE2 = ROOT / 'gppo-world-stage2-consolidated-20260924'
RECOVERY = STAGE2 / 'local-recovery2-preparation-20260925'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def mean_cpu(batches):
    return 1000 * sum(b['cpu_seconds'] for b in batches) / sum(b['decisions'] for b in batches)

inputs = [
    ROOT / 'gppo-world-decision-freeze-20260924/protocol.md',
    ROOT / 'gppo-world-cost-replay-preparation-20260924/cost_replay.py',
    ROOT / 'gppo-world-stage2-gate-preparation-20260924/runs/technical-gate-v1/decisions.jsonl',
    STAGE2 / 'protocol.md', STAGE2 / 'stage_worker.py', STAGE2 / 'run_pipeline.py',
    STAGE2 / 'decision_only_inference.py', STAGE2 / 'runs/consolidated-v1/A/candidate-result.json',
    RECOVERY / 'stage_worker.py', RECOVERY / 'runtime_support.py',
    RECOVERY / 'runs/consolidated-v1/C/analysis.json',
    CURRENT / 'protocol.md', CURRENT / 'base_worker.py', CURRENT / 'runtime_worker.py',
    CURRENT / 'decision_only_inference.py', CURRENT / 'no_online_world.py',
    CURRENT / 'run_diagnostic.py', CURRENT / 'process_support.py',
    CURRENT / 'execution-manifest.json', CURRENT / 'sample-manifest.json',
    CURRENT / 'runs/runtime-necessity-v1/A/result.json',
    CURRENT / 'runs/runtime-necessity-v1/technical-stop.json',
    CURRENT / 'runs/runtime-necessity-v1/final-review/decision.json',
    CURRENT / 'runs/runtime-necessity-v1/budget.sqlite3',
]
before = {str(p): digest(p) for p in inputs}
current = read(CURRENT / 'runs/runtime-necessity-v1/A/result.json')
stage2 = read(RECOVERY / 'runs/consolidated-v1/C/analysis.json')
old_batches = read(STAGE2 / 'runs/consolidated-v1/A/candidate-result.json')['batches']
assert digest(CURRENT / 'base_worker.py') == digest(STAGE2 / 'stage_worker.py') == digest(RECOVERY / 'stage_worker.py')
assert digest(CURRENT / 'decision_only_inference.py') == digest(STAGE2 / 'decision_only_inference.py')
assert len(current['batches']['candidate']) == 21
assert sum(b['decisions'] for b in current['batches']['candidate']) == 525
assert abs(mean_cpu(current['batches']['candidate']) - current['N_mean_cpu_ms']) < 1e-10
assert current['N_mean_cpu_ms'] > 10
evidence = {
    'audit_type': 'static_code_and_saved_timing_only',
    'new_activity': dict(environment_steps=0, model_forwards=0, model_loads=0, training_updates=0, experiment_attempts=0),
    'threshold': {'cpu_ms': 10, 'wall_p95_ms': 50, 'kind': 'predeclared_research_practical_standard_not_field_SLA', 'source_line': 117, 'changed': False},
    'stage2_formal_controller_cost': stage2['controller_cost'],
    'stage2_formal_cost_guard_pass': stage2['cost_guard_pass'],
    'current_A': {k: current[k] for k in ['N_mean_cpu_ms', 'N_wall_p95_ms', 'paired_replay_cpu_ratio', 'bitwise_reference_candidate_match', 'full_historical_match']},
    'all21_cpu_ms': {k: mean_cpu(v) for k, v in current['batches'].items()},
    'warm20_cpu_ms_descriptive_not_new_gate': {k: mean_cpu(v[1:]) for k, v in current['batches'].items()},
    'old_stage2_A_full_all21_cpu_ms_descriptive': mean_cpu(old_batches),
    'byte_identical': ['current base_worker vs stage2 original/recovery stage_worker', 'decision_only_inference'],
    'configured_threads': {'intra': 4, 'inter': 1},
    'direct_timing_comparability': False,
    'differences': ['closed-loop formal workload vs 25 saved inputs', 'per-decision CPU vs whole-batch CPU', 'extra checks and scalar extraction inside choose', 'worker residency/request scheduling'],
    'hashing_inside_controller_timer': False,
    'overhead_quantified': False,
    'cost_failure_preserved': True,
    'research_decision': 'propose_separate_task_benefit_diagnostic_not_recovery',
    'future_execution_authorized_by_this_audit': False,
    'retry': False,
}
after = {str(p): digest(p) for p in inputs}
assert before == after
evidence['inputs_unchanged_during_audit'] = True
write('input-hashes.json', before)
write('evidence.json', evidence)
outputs = ['report.md', 'audit_static.py', 'input-hashes.json', 'evidence.json']
write('hashes.json', {name: digest(HERE / name) for name in outputs})
print(json.dumps({'input_files': len(inputs), 'inputs_unchanged': True, 'N_mean_cpu_ms': evidence['current_A']['N_mean_cpu_ms'], 'env_steps': 0, 'model_forwards': 0, 'report': str(HERE / 'report.md')}, ensure_ascii=False))
