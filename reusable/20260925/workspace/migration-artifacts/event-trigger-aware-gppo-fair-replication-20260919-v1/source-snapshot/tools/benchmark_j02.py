"""Fixed 100-update training-only timing pilot; never scores Validation/Test."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from gppo_world.jepa_two_stage import TwoStageModel
from gppo_world.phase_learning import load_rows,prepare_phases,model_arguments
from gppo_world.data import STATE_DIM
from tools.run_d02_adapter_diagnostics import write,sha,git


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if git(ROOT,'status','--porcelain'):raise ValueError('Freeze benchmark implementation first')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(5100)
    data=prepare_phases(load_rows(args.data/'train.jsonl'))
    model=TwoStageModel();decoder=torch.nn.Linear(11*64,STATE_DIM+8)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(decoder.parameters()),lr=.001)
    scale=data['y'].std(0,unbiased=False).clamp_min(.05)
    mean=data['y'].mean(0);target=(data['y']-mean)/scale
    times=[];losses=[]
    for update in range(100):
        start=time.monotonic();ids=torch.randint(len(target),(32,))
        prediction=decoder(model(**model_arguments(data,ids)).flatten(1))
        # Equal task weights rather than letting static graph dimensions dominate.
        error=(prediction-target[ids]).square()
        loss=(error[:,:STATE_DIM].mean()+error[:,STATE_DIM].mean()+error[:,STATE_DIM+1:].mean())/3
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(decoder.parameters()),5.)
        optimizer.step();times.append(time.monotonic()-start);losses.append(float(loss.detach()))
    torch.save({'model':model.state_dict(),'decoder':decoder.state_dict(),'optimizer':optimizer.state_dict(),
        'updates':100,'seed':5100,'purpose':'timing_only_not_selected_or_reused'},args.output/'timing-terminal.pt')
    write(args.output/'timing.json',{'source_commit':git(ROOT,'rev-parse','HEAD'),'train_sha256':sha(args.data/'train.jsonl'),
        'updates':100,'seconds':sum(times),'seconds_per_update':sum(times)/100,'device':'cpu','threads':1,
        'batch_size':32,'validation_or_test_read':False,'losses':losses,'per_update_seconds':times,
        'checkpoint_sha256':sha(args.output/'timing-terminal.pt'),'training_budget_accounting':'100 pilot overhead updates; no scored candidate; no reuse'})
    print(json.dumps({'updates':100,'seconds':sum(times),'estimated_22500_update_seconds':sum(times)*225}))
