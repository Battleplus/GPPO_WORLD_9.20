"""Frozen J02 development matrix; no Test access or policy training."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from gppo_world.jepa_two_stage import TwoStageModel,MaskedPretrainer,spread_penalty
from gppo_world.phase_learning import load_rows,prepare_phases,model_arguments
from gppo_world.jepa import select_batch
from gppo_world.data import STATE_DIM
from tools.run_d02_adapter_diagnostics import write,sha,git
from tools.audit_j02_phases import audit


def mean(values):return sum(values)/len(values)


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [to_device(v,device) for v in value]
    return value


def macro(values,rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for value,row in zip(values,rows):grouped[row['scenario_id']][row['tape_id']].append(float(value))
    return mean([mean([mean(v) for v in tapes.values()]) for tapes in grouped.values()])


def fit_readout(x,y,scale):
    x=x.double();y=y.double();scale=scale.double()
    xm=x.mean(0);xs=x.std(0,unbiased=False).clamp_min(.05);ym=y.mean(0)
    z=(x-xm)/xs;w=torch.linalg.solve(z.T@z+torch.eye(z.shape[1],dtype=z.dtype),z.T@((y-ym)/scale))
    return {'xmean':xm,'xscale':xs,'ymean':ym,'yscale':scale,'weight':w}


def readout(p,x):return (((x.double()-p['xmean'])/p['xscale'])@p['weight'])*p['yscale']+p['ymean']


@torch.no_grad()
def representations(model,data,target_encoder=None):
    predicted=[];current=[];target=[]
    device=next(model.parameters()).device
    for ids in torch.arange(len(data['rows'])).split(64):
        predicted.append(model(**to_device(model_arguments(data,ids),device)).flatten(1).cpu())
        current.append((target_encoder or model.encoder)(to_device(select_batch(data['graphs'][-1],ids),device)).flatten(1).cpu())
        target.append((target_encoder or model.encoder)(to_device(select_batch(data['commit'],ids),device)).flatten(1).cpu())
    return tuple(torch.cat(xs) for xs in (predicted,current,target))


def task_metrics(pred,y,scale,active,rows):
    error=((pred-y.double())/scale.double()).square()
    parts={'state_active_nmse':error[:,:STATE_DIM][:,active].mean(1),'state_all_nmse':error[:,:STATE_DIM].mean(1),
           'reward_nmse':error[:,STATE_DIM],'cost_nmse':error[:,STATE_DIM+1:].mean(1)}
    return {k:{'scenario_macro':macro(v,rows),'transition_mean':float(v.mean())} for k,v in parts.items()}


def branch_regret(pred,rows,tolerance):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['episode_id'],r['step']].append(i)
    regrets=[];origins=[]
    for ids in groups.values():
        if len(ids)<2:continue
        rewards=[rows[i]['reward'] for i in ids]
        if max(rewards)-min(rewards)<=tolerance:continue
        scores=pred[ids,STATE_DIM]
        choices=[ids[j] for j in range(len(ids)) if float(scores.max()-scores[j])<=tolerance]
        chosen=min(choices,key=lambda i:rows[i]['action'])
        regrets.append(max(rewards)-rows[chosen]['reward']);origins.append(rows[ids[0]])
    return {'immediate_reward_regret':macro(regrets,origins),'origins':len(origins),
            'per_origin':[{'episode_id':r['episode_id'],'tape_id':r['tape_id'],'scenario_id':r['scenario_id'],'step':r['step'],'regret':v}
                          for r,v in zip(origins,regrets)]}


def execute(args):
    protocol_path=args.protocol;p=json.loads(protocol_path.read_text());cfg=p['training']
    if git(ROOT,'status','--porcelain'):raise ValueError('Commit/freeze all source before training')
    import platform,os
    if not p.get('server_only') or platform.system()!='Linux' or not cfg['device'].startswith('cuda'):
        raise ValueError('Current user instruction requires a frozen Linux GPU server protocol; no local/CPU training')
    if p.get('execution_host') and platform.node()!=p['execution_host']:
        raise ValueError('Server hostname does not match frozen preflight')
    if p.get('server_only') and platform.system()=='Windows':
        raise ValueError('User requires server-only training')
    device=torch.device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is required; no CPU fallback')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>cfg['max_wall_seconds']:raise TimeoutError('Frozen wall budget exceeded')
    report=audit(args.data)
    write(args.output/'input-audit.json',report)
    train_rows=load_rows(args.data/'train.jsonl');val_rows=load_rows(args.data/'validation.jsonl')
    branch_rows=load_rows(args.data/'validation-branches.jsonl')
    train=prepare_phases(train_rows);val=prepare_phases(val_rows);branch=prepare_phases(branch_rows,val_rows)
    write(args.output/'run-manifest.json',{'source_commit':git(ROOT,'rev-parse','HEAD'),'protocol':p,
        'protocol_sha256':sha(protocol_path),'files':[{ 'path':str(f),'sha256':sha(f)} for f in sorted(args.data.glob('*.json*'))],
        'torch':torch.__version__,'test_loaded':False,'hostname':platform.node(),'device':str(device),
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
    scale=train['y'].std(0,unbiased=False).clamp_min(.05);ym=train['y'].mean(0)
    active=train['y'][:,:STATE_DIM].std(0,unbiased=False)>1e-6
    write(args.output/'normalization.json',{'scale':scale.tolist(),'mean':ym.tolist(),'active_dimensions':active.nonzero().flatten().tolist()})
    normalized=((train['y']-ym)/scale).to(device)
    active_device=active.to(device)
    raw_probe=fit_readout(train['raw'],train['y'],scale)
    raw_metrics=task_metrics(readout(raw_probe,val['raw']),val['y'],scale,active,val_rows)
    mean_metrics=task_metrics(ym.double().expand_as(val['y']),val['y'],scale,active,val_rows)
    torch.save(raw_probe,args.output/'raw-direct-probe.pt')
    results=[];updates=0
    for seed in p['seeds']:
        budget();torch.manual_seed(seed)
        pre=MaskedPretrainer().to(device);gen=torch.Generator().manual_seed(seed+19)
        opt=torch.optim.AdamW(pre.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        pre_dir=args.output/f'pretraining-{seed}';pre_dir.mkdir()
        with (pre_dir/'history.jsonl').open('x') as log:
            for i in range(cfg['shared_pretraining_updates_per_seed']):
                budget();ids=torch.randint(len(train_rows),(cfg['batch_size'],),generator=gen)
                loss=pre.loss(to_device(select_batch(train['graphs'][-1],ids),device),gen)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pre.parameters(),cfg['gradient_clip']);opt.step();pre.update_target();updates+=1
                if not torch.isfinite(loss):raise ValueError('Nonfinite pretraining loss')
                log.write(json.dumps({'update':i+1,'loss':float(loss.detach())})+'\n')
                log.flush()
        torch.save({'encoder':pre.encoder.state_dict(),'target':pre.target.state_dict(),'optimizer':opt.state_dict(),
                    'seed':seed,'updates':500},pre_dir/'terminal.pt')
        shared=deepcopy(pre.encoder.state_dict())
        for group in p['groups']:
            budget();torch.manual_seed(seed)
            fixed=group.startswith('J-fixed');model=TwoStageModel(with_action=group!='J-fixed-noaction',frozen=fixed).to(device)
            if fixed:model.encoder.load_state_dict(shared)
            target=deepcopy(model.encoder).requires_grad_(False)
            decoder=torch.nn.Linear(11*64,STATE_DIM+8).to(device) if group=='S' else None
            params=[v for v in model.parameters() if v.requires_grad]+([] if decoder is None else list(decoder.parameters()))
            opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            gen=torch.Generator().manual_seed(seed+71)
            total=cfg['fixed_predictor_updates'] if fixed else cfg['supervised_updates']
            folder=args.output/f'{group}-{seed}';folder.mkdir()
            with (folder/'history.jsonl').open('x') as log:
                for i in range(total):
                    budget();ids=torch.randint(len(train_rows),(cfg['batch_size'],),generator=gen)
                    predicted=model(**to_device(model_arguments(train,ids),device))
                    if decoder is not None:
                        error=(decoder(predicted.flatten(1))-normalized[ids]).square()
                        loss=(error[:,:STATE_DIM][:,active_device].mean()+error[:,STATE_DIM].mean()+error[:,STATE_DIM+1:].mean())/3
                    else:
                        with torch.no_grad():truth=target(to_device(select_batch(train['commit'],ids),device))
                        loss=F.mse_loss(predicted,truth)
                        if not fixed:loss=loss+spread_penalty(model.encoder(to_device(select_batch(train['graphs'][-1],ids),device)))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite dynamics loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);opt.step();updates+=1
                    if not fixed:
                        with torch.no_grad():
                            for a,b in zip(target.parameters(),model.encoder.parameters()):a.lerp_(b,.01)
                    log.write(json.dumps({'update':i+1,'loss':float(loss.detach())})+'\n')
                    log.flush()
                    if (i+1)%500==0:
                        torch.save({'model':model.state_dict(),'target':target.state_dict(),'optimizer':opt.state_dict(),
                            'decoder':None if decoder is None else decoder.state_dict(),'group':group,'seed':seed,'updates':i+1,
                            'sampling_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},folder/f'progress-{i+1}.pt')
                    if (i+1)%500==0:print(f'{group} seed {seed}: {i+1}/{total} updates',flush=True)
            if fixed and any(not torch.equal(v,shared[k]) for k,v in model.encoder.state_dict().items()):raise ValueError('Frozen encoder changed')
            torch.save({'model':model.state_dict(),'target':target.state_dict(),'decoder':None if decoder is None else decoder.state_dict(),
                'optimizer':opt.state_dict(),'group':group,'seed':seed,'updates':total,'shared_pretraining_updates':500 if fixed else 0,
                'protocol_sha256':sha(protocol_path)},folder/'terminal.pt')
            model.eval()
            train_x,train_current,_=representations(model,train,target if group=='J-joint' else None)
            val_x,val_current,val_target=representations(model,val,target if group=='J-joint' else None)
            branch_x,_,_=representations(model,branch)
            probe=fit_readout(train_x,train['y'],scale);persist_probe=fit_readout(train_current,train['y'],scale)
            torch.save(probe,folder/'probe.pt');torch.save(persist_probe,folder/'persistence-probe.pt')
            predicted=readout(probe,val_x);bp=readout(probe,branch_x)
            metrics=task_metrics(predicted,val['y'],scale,active,val_rows)
            pred_err=(val_x-val_target).square().mean(1);pers_err=(val_current-val_target).square().mean(1)
            pred_mean=macro(pred_err,val_rows);pers_mean=macro(pers_err,val_rows)
            centered=train_current.double()-train_current.double().mean(0)
            eig=torch.linalg.svdvals(centered).square();prob=eig/eig.sum().clamp_min(1e-20)
            rank=float(torch.exp(-(prob*prob.clamp_min(1e-30).log()).sum()))
            br=branch_regret(bp,branch_rows,p['gates']['branch_tie_tolerance'])
            write(folder/'branch-regrets.json',br)
            write(folder/'validation-predictions.json',{'prediction':predicted.tolist(),'identities':[{k:r[k] for k in ('episode_id','tape_id','scenario_id','step')} for r in val_rows]})
            result={'group':group,'seed':seed,'task_metrics':metrics,'latent_prediction_mse':pred_mean,
                'latent_persistence_mse':pers_mean,'latent_persistence_improvement':1-pred_mean/max(pers_mean,1e-20),
                'latent_mean_std':float(train_current.std(0,unbiased=False).mean()),'effective_rank':rank,
                'immediate_branch_regret':br['immediate_reward_regret'],'branch_origins':br['origins'],
                'persistence_readout_metrics':task_metrics(readout(persist_probe,val_current),val['y'],scale,active,val_rows),
                'model_parameters':sum(v.numel() for v in model.parameters())}
            write(folder/'metrics.json',result);results.append(result)
            print(json.dumps({'finished':group,'seed':seed,'latent_improvement':result['latent_persistence_improvement'],
                              'reward_nmse':metrics['reward_nmse']['scenario_macro'],'regret':result['immediate_branch_regret']}),flush=True)
        # Random model is a reference, never optimized or selected.
        torch.manual_seed(seed);random_model=TwoStageModel().to(device).eval()
        tx,_,_=representations(random_model,train);vx,_,_=representations(random_model,val)
        probe=fit_readout(tx,train['y'],scale)
        torch.save({'model':random_model.state_dict(),'probe':probe,'seed':seed,'updates':0},args.output/f'random-{seed}.pt')
        write(args.output/f'random-{seed}-metrics.json',task_metrics(readout(probe,vx),val['y'],scale,active,val_rows))
    if updates!=cfg['unique_updates_total']:raise ValueError('Update accounting mismatch')
    write(args.output/'results.json',{'status':'development_matrix_completed','updates':updates,'pilot_overhead_updates':100,
        'elapsed_seconds':time.monotonic()-started,'results':results,'raw_direct':raw_metrics,'train_mean':mean_metrics,
        'test_loaded':False,'policy_training_started':False})
    write(args.output/'inventory.json',[{'path':f.relative_to(args.output).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size}
        for f in sorted(args.output.rglob('*')) if f.is_file()])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:execute(args)
    except Exception:
        write(args.output/'failure.json',{'traceback':traceback.format_exc()});raise
