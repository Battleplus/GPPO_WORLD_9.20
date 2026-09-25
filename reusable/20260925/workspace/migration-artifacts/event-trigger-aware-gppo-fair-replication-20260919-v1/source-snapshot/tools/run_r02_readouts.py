"""Fixed-recipe development readouts of frozen J-01 models; never reads Test."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from gppo_world.data import STATE_DIM, load_jsonl, state_vector
from gppo_world.jepa import GraphJEPA, encode_batch, select_batch
from tools.run_j01_experiment import prepare, batches, fit_probe, predict_probe
from tools.run_d02_adapter_diagnostics import write, sha, git


def macro(values, rows):
    """Equal scenario weight, equal tape weight within scenario."""
    grouped = defaultdict(lambda: defaultdict(list))
    for value, row in zip(values.tolist(), rows):
        grouped[row['scenario_id']][row['tape_id']].append(value)
    by_scenario = {s: sum(sum(v) / len(v) for v in tapes.values()) / len(tapes)
                   for s, tapes in grouped.items()}
    return {'scenario_macro': sum(by_scenario.values()) / len(by_scenario),
            'transition_mean': values.mean().item(), 'by_scenario': by_scenario}


@torch.no_grad()
def views(model, data, rows):
    result = defaultdict(list)
    for ids in batches(len(rows), 64):
        current, predicted = model(select_batch(data['graph'], ids), data['actions'][ids], data['evidence'][ids])
        # Use the SAME online encoder for current and observed future.
        observed = encode_batch(model.online, select_batch(data['next'], ids))
        action = F.one_hot(data['actions'][ids], model.config.action_dim).float()
        extra = torch.cat([action, data['evidence'][ids]], 1)
        width = model.config.latent_dim + extra.shape[1]
        for name, x in {'current': current, 'current_action_evidence': torch.cat([current, extra], 1),
                        'predicted_future': predicted, 'observed_future_oracle': observed}.items():
            result[name].append(F.pad(x, (0, width - x.shape[1])).double())
        raw = torch.stack([state_vector(rows[i].graph) for i in ids.tolist()])
        result['raw_graph_action_evidence'].append(torch.cat([raw, extra], 1).double())
    return {name: torch.cat(xs) for name, xs in result.items()}


def run(args, output):
    started = time.monotonic()
    protocol_path = ROOT / 'nodes/R-02/protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if git(ROOT, 'status', '--porcelain'):
        raise ValueError('Freeze source with a clean local commit before execution')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inventory = json.loads((ROOT / 'nodes/J-01/evidence/release-inventory.json').read_text(encoding='utf-8'))
    expected = {m['path']: m for a in inventory['archives'] for m in a['members']}
    inputs = []

    def verify(path, key):
        record = expected[key]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Sealed input mismatch: {key}')
        inputs.append({'path': str(path.resolve()), 'sha256': record['sha256']})

    data, transitions, raw = {}, {}, {}
    for split in ('train', 'validation'):
        path = args.data / f'dataset/{split}.jsonl'
        verify(path, f'dataset/{split}.jsonl')
        raw[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        transitions[split] = load_jsonl(path)
        data[split] = prepare(transitions[split])
    for key in ('tape_id', 'episode_id'):
        if {r[key] for r in raw['train']} & {r[key] for r in raw['validation']}:
            raise ValueError(f'Train/Validation overlap: {key}')
    matrix = [(group, seed) for group in protocol['readouts']['groups'] for seed in protocol['readouts']['seeds']]
    for group, seed in matrix:
        key = f'{group}-{seed}/terminal.pt'
        verify(args.models / key, key)
    write(output / 'run-manifest.json', {'protocol': protocol, 'protocol_sha256': sha(protocol_path),
        'source_commit': git(ROOT, 'rev-parse', 'HEAD'), 'inputs': inputs, 'test_loaded': False,
        'python': sys.version, 'torch': torch.__version__, 'device': 'cpu'})
    train_y, val_y = data['train']['y'].double(), data['validation']['y'].double()
    scale = train_y.std(0, unbiased=False).clamp_min(0.05)
    active = train_y[:, :STATE_DIM].std(0, unbiased=False) > 1e-6
    write(output / 'target-normalization.json', {'scale': scale.tolist(), 'active_state_dimensions': active.nonzero().flatten().tolist()})
    all_results = []
    for group, seed in matrix:
        if time.monotonic() - started > protocol['readouts']['max_wall_seconds']:
            raise TimeoutError('Fixed development budget exceeded')
        model, metadata = GraphJEPA.load(args.models / f'{group}-{seed}/terminal.pt')
        state_before = {k: v.clone() for k, v in model.state_dict().items()}
        xs = {split: views(model, data[split], transitions[split]) for split in data}
        for name in xs['train']:
            probe = fit_probe(xs['train'][name], train_y, xs['validation'][name], val_y, scale, active, [1.0])
            prediction = predict_probe(probe, xs['validation'][name])
            squared = ((prediction - val_y) / scale).square()
            scores = {'state_active_nmse': squared[:, :STATE_DIM][:, active].mean(1),
                      'state_all_nmse': squared[:, :STATE_DIM].mean(1), 'reward_nmse': squared[:, STATE_DIM],
                      'costs_nmse': squared[:, STATE_DIM + 1:STATE_DIM + 8].mean(1),
                      'continuation_brier_clipped': (prediction[:, -1].clamp(0, 1) - val_y[:, -1]).square()}
            record = {'group': group, 'seed': seed, 'view': name, 'input_width': xs['train'][name].shape[1],
                      'metrics': {key: macro(value, raw['validation']) for key, value in scores.items()}}
            destination = output / f'{group}-{seed}'
            destination.mkdir(exist_ok=True)
            torch.save(probe, destination / f'{name}-probe.pt')
            write(destination / f'{name}-per-transition.json', [
                {'episode_id': row['episode_id'], 'tape_id': row['tape_id'], 'scenario_id': row['scenario_id'],
                 'step': row['step'], **{k: float(v[i]) for k, v in scores.items()}}
                for i, row in enumerate(raw['validation'])])
            all_results.append(record)
        if any(not torch.equal(v, state_before[k]) for k, v in model.state_dict().items()):
            raise ValueError('Readout changed frozen model')
        print(f'Completed readouts: {group} seed {seed}', flush=True)
    if any(sha(r['path']) != r['sha256'] for r in inputs):
        raise ValueError('Input changed during diagnosis')
    write(output / 'results.json', {'status': 'completed_development_diagnosis', 'test_loaded': False,
        'model_weights_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
        'ridge': 1.0, 'hyperparameter_search': False, 'results': all_results,
        'limitations': ['Posthoc Train/Validation diagnosis, not new Test evidence',
            'Observed future alone may not recover deltas without current state; oracle is a clue, not an upper bound',
            'Four latent views use equal padded input width; raw graph reference has a larger width',
            'JEPA predictor was trained toward EMA space; current/oracle use the identical online encoder',
            'No causal action-branch or policy-benefit claim']})
    write(output / 'inventory.json', [{'path': p.relative_to(output).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                                    for p in sorted(output.rglob('*')) if p.is_file()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, args.output)
    except Exception as error:
        write(args.output / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
