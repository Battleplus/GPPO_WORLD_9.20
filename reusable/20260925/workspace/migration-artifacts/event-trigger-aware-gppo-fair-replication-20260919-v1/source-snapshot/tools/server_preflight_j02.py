"""Read GPU state and freeze server-only protocol; no training or installation."""
import argparse,json,platform,socket,subprocess
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Server Linux execution required')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    print(devices);print(processes)
    chosen=next(line for line in devices.splitlines() if int(line.split(',')[0])==args.gpu)
    parts=[s.strip() for s in chosen.split(',')]
    if int(parts[2])>500 or int(parts[4])>10:raise RuntimeError('Selected GPU currently occupied; choose free GPU, do not stop other work')
    p=json.loads(args.protocol.read_text());p['version']='j02b-server-development/1.1.0'
    p['training']['device']=f'cuda:{args.gpu}';p['server_only']=True;p['execution_host']=socket.gethostname()
    p['migration']='User requested server-only execution. Previous CPU attempt interrupted, archived and excluded from CUDA matrix; no score-based selection.'
    p['server_preflight']={'hostname':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'devices':devices,'processes':processes,'prior_cpu_attempt':'incomplete; preserve separately, do not mix hardware scores'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
