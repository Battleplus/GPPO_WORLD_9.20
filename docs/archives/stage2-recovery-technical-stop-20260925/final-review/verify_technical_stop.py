"""Read-only terminal inventory. Never imports an environment/model or evaluates effects."""
import collections
import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / 'runs/consolidated-v1'
START = time.perf_counter()
CPU_START = time.process_time()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(name, obj):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')

def key(row):
    return row['parent'], row['repeat'], row['arm']

def rows(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def key_only(k):
    return dict(zip(('parent', 'repeat', 'arm'), k))

status = read(RUN / 'status.json')
assert status['status'] == 'technical_stop'
last_history = status['segment_budget'].pop('history')[-4:]
matrix = read(ROOT / 'frozen-matrix.json')
matrix_keys = [key(r) for r in matrix]
retained = read(ROOT / 'retained-episodes.json')
episodes = rows(RUN / 'C/episodes.jsonl')
starts = rows(RUN / 'C/episode-starts.jsonl')
oldkeys = [key(r) for r in retained]
epkeys = [key(r) for r in episodes]
startkeys = [key(r) for r in starts]
assert len(matrix_keys) == len(set(matrix_keys)) == 9600
assert len(oldkeys) == 4034 and matrix_keys[:4034] == oldkeys
assert len(epkeys) == len(set(epkeys)) and len(startkeys) == len(set(startkeys))
assert epkeys == matrix_keys[4034:4034 + len(epkeys)]
assert startkeys == matrix_keys[4034:4034 + len(startkeys)]
assert not set(oldkeys) & set(startkeys)
epmap = {key(r): r for r in episodes}
startmap = {key(r): r for r in starts}

db = RUN / 'budget.sqlite3'
db_before = sha(db)
conn = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
conn.execute('PRAGMA query_only=ON')
integrity = [r[0] for r in conn.execute('PRAGMA integrity_check')]
assert integrity == ['ok']
stages = {r[0]: dict(zip(('limit', 'reserved', 'verified', 'unknown'), r[1:]))
          for r in conn.execute('SELECT stage,limit_amount,reserved,verified,unknown FROM stages')}
for name, v in stages.items():
    v['pending'] = v['reserved'] - v['verified'] - v['unknown']
    assert all(v[k] == status['segment_budget']['stages'][name][k]
               for k in ('reserved', 'verified', 'unknown', 'pending'))
reservations = {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
    'SELECT reservation_id,stage,amount,status FROM reservations')}
reservation_statuses = dict(conn.execute('SELECT status,COUNT(*) FROM reservations GROUP BY status'))
conn.close()
assert sha(db) == db_before

tapes = read(RUN / 'C/scenario-manifest.json')
assert sha(RUN / 'C/scenario-manifest.json') == sha(ROOT.parent / 'runs/consolidated-v1/C/scenario-manifest.json')
initials = {(r['parent'], r['repeat'], r['arm']): r['initial_public_hash'] for r in retained}
for r in starts:
    k = key(r)
    assert r['random_key'] == f'frozen-native-public-history-allocation-v1|{k[0]}|repeat-{k[1]}'
    assert r['tape_hash'] == tapes['hashes'][int(k[0].split('-')[1])]
    initials[k] = r['reset']['initial_hash']
    if k in epmap:
        assert epmap[k]['initial_public_hash'] == r['reset']['initial_hash']
paired = 0
for p, rep, arm in initials:
    if arm == 'M' and (p, rep, 'R') in initials:
        assert initials[(p, rep, 'M')] == initials[(p, rep, 'R')]
        paired += 1

counts = collections.Counter()
donekeys = []
step_counts = collections.Counter()
last_counts = {}
step_keys = []
lastrow = None
digest = hashlib.sha256()
with gzip.open(RUN / 'C/steps.jsonl.gz', 'rb') as f:
    for line in f:
        digest.update(line)
        row = json.loads(line)
        k = key(row)
        assert k in startmap
        if not step_keys or step_keys[-1] != k:
            assert k not in step_counts
            step_keys.append(k)
        assert row['step'] == step_counts[k]
        step_counts[k] += 1
        counts[k[2]] += 1
        assert row['action'] in row['candidates'] and row['public']['mask'][row['action']]
        expected = {'C/environment_steps', 'C/rule_decisions'} if k[2] == 'R' else {
            'C/environment_steps', 'C/policy_encode', 'C/world_candidate_batch', 'C/actor_readout'}
        rr = [reservations[token['reservation_id']] for token in row['reservations']]
        assert {r[0] for r in rr} == expected
        assert all(r[1:] == (1, 'verified') for r in rr)
        assert len(row['vector_reward']) == 2
        assert all(n in row for n in ('controller', 'environment', 'labels', 'info'))
        last_counts[k[2]] = row['counts']
        if row['done']:
            donekeys.append(k)
            if k in epmap:
                for field, value in row['episode'].items():
                    assert epmap[k][field] == value
                assert epmap[k]['steps'] == step_counts[k]
        lastrow = row

assert step_keys == startkeys[:len(step_keys)]
assert counts['M'] + counts['R'] == stages['C/environment_steps']['verified']
assert counts['M'] == stages['C/actor_readout']['verified'] == stages['C/policy_encode']['verified'] == stages['C/world_candidate_batch']['verified']
assert counts['R'] == stages['C/rule_decisions']['verified']
assert len(starts) == stages['C/environment_resets']['verified']
assert set(epkeys) <= set(donekeys)
missing_summary = [k for k in donekeys if k not in epmap]
partial = [k for k in startkeys if k not in donekeys]
completed = oldkeys + donekeys
not_started = matrix_keys[4034 + len(startkeys):]
assert completed == matrix_keys[:len(completed)]
assert len(completed) + len(partial) + len(not_started) == 9600

bound = read(ROOT / 'recovery-manifest.json')['files']
binding_checks = [{'path': r['path'], 'expected': r['sha256'], 'actual': sha(Path(r['path']))} for r in bound]
assert all(r['expected'] == r['actual'] for r in binding_checks)
write('frozen-input-verification.json', binding_checks)
write('completed-keys.json', [key_only(k) for k in completed])
write('complete-missing-summary-keys.json', [key_only(k) for k in missing_summary])
write('incomplete-episode-keys.json', [{**key_only(k), 'durable_steps': step_counts[k]} for k in partial])
write('not-started-keys.json', [key_only(k) for k in not_started])
write('terminal-budget-and-resources.json', status)

summary = {
    'attempt': 'consolidated-v1-recovery-1-once',
    'status': status['status'], 'error': status['error'],
    'original_completed_retained': len(oldkeys),
    'recovery_complete_with_summary': len(epkeys),
    'recovery_complete_missing_summary': len(missing_summary),
    'complete_total': len(completed), 'planned': 9600,
    'recovery_started': len(starts), 'incomplete_episode_count': len(partial),
    'not_started': len(not_started), 'remaining_not_complete': 9600-len(completed),
    'last_completed': key_only(completed[-1]),
    'last_record': {**key_only(key(lastrow)), 'step': lastrow['step'], 'done': lastrow['done']},
    'paired_initial_hash_checks': paired,
    'new_steps_by_arm': dict(counts), 'new_environment_steps': sum(counts.values()),
    'last_worker_counters': last_counts,
    'gzip_crc_and_complete_stream': 'pass', 'uncompressed_sha256': digest.hexdigest(),
    'sqlite_integrity': integrity, 'reservation_statuses': reservation_statuses,
    'sqlite_sha256_unchanged_by_audit': db_before,
    'stages': stages, 'last_ledger_events': last_history,
    'frozen_bound_files': len(bound), 'bound_identity_mismatches': 0,
    'formal_analysis_exists': (RUN/'C/analysis.json').exists(),
    'partial_effect_computed': False, 'controller_cost_gate_evaluated': False,
    'new_environment_steps_during_audit': 0, 'model_forwards_during_audit': 0,
    'updates': 0, 'retry_authorized': False,
    'verification_wall_seconds': time.perf_counter()-START,
    'verification_cpu_seconds': time.process_time()-CPU_START,
    'verification_cost_scope': 'Separate post-stop stdlib audit process, not included in frozen experiment process snapshot.'
}
write('verification.json', summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('stages','last_ledger_events','last_worker_counters')}, ensure_ascii=False, indent=2))
