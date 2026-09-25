"""Independent disk audit of three-phase causal and branch data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gppo_world.phase_data import validate_record, history_inputs, model_input, digest
from tools.run_d02_adapter_diagnostics import sha, write


def audit(folder):
    manifest=json.loads((folder/'dataset-manifest.json').read_text())
    for member in json.loads((folder/'inventory.json').read_text()):
        p=folder/member['path']
        assert p.stat().st_size==member['bytes'] and sha(p)==member['sha256']
    seeds=[]
    for tape in manifest['tapes']:seeds.extend([tape['initial_seed'],tape['event_seed']])
    assert len(seeds)==len(set(seeds))
    split_ids={}; reports={}
    for split in ('train','validation'):
        rows=[json.loads(l) for l in (folder/f'{split}.jsonl').read_text().splitlines()]
        branches=[json.loads(l) for l in (folder/f'{split}-branches.jsonl').read_text().splitlines()]
        histories=history_inputs(rows)
        by_key={(r['episode_id'],r['step']):r for r in rows}
        assert len(by_key)==len(rows)
        by_origin=defaultdict(list)
        for branch in branches:
            validate_record(branch)
            key=(branch['episode_id'],branch['step']); factual=by_key[key]
            assert branch['pre']==factual['pre'] and branch['audit']['before']==factual['audit']['before']
            assert branch['commit']['rng']==branch['audit']['pre_rng']
            assert branch['split']==split
            assert branch['history_actions']==[by_key[(key[0],s)]['action'] for s in range(key[1])]
            if branch['action']==factual['action']:
                for field in factual:assert branch[field]==factual[field]
            by_origin[key].append(branch)
        varying_reward=0
        for key, values in by_origin.items():
            factual=by_key[key]
            legal=[i for i,v in enumerate(factual['pre']['graph']['action_mask']) if v]
            assert sorted(v['action'] for v in values)==legal
            varying_reward+=max(v['reward'] for v in values)-min(v['reward'] for v in values)>1e-9
        for row in rows:
            validate_record(row)
            assert row['split']==split
            assert set(model_input(row))=={'graph','time','action_version','evidence','decision_duration','action'}
            if row['step']<manifest['protocol']['branch_origins_per_episode']:assert (row['episode_id'],row['step']) in by_origin
        split_ids[split]={r['tape_id'] for r in rows}
        reports[split]={'rows':len(rows),'episodes':len({r['episode_id'] for r in rows}),'tapes':len(split_ids[split]),
            'branches':len(branches),'branch_origins':len(by_origin),'origins_with_reward_difference':varying_reward,
            'history_valid_lengths':dict(Counter(sum(h['valid']) for h in histories)),
            'profile_rows':dict(Counter(r['scenario_id'] for r in rows)),
            'constraint_violation_positives':sum(r['costs']['constraint_violation']>0 for r in rows),
            'commit_next_graph_differences':sum(r['commit']['graph']!=r['next']['graph'] for r in rows),
            'truncated_rows':sum(r['truncated'] for r in rows)}
    assert not split_ids['train']&split_ids['validation']
    return {'status':'passed','checks':['archive member hashes','split isolation','all legal branch coverage in declared first-four origins',
        'factual branch equality','full history identity','causal input allowlist','commit timing and versions','commit RNG unchanged'],
        'splits':reports,'limitation':'Next-phase branches are not claimed to share external random draws; commit-only causal comparison'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.data);write(args.output,result);print(json.dumps(result,ensure_ascii=False))
