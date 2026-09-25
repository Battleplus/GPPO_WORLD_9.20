"""Pin existing evidence and user-approved limits; no model/environment imports."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'gppo-world-runtime-minimal-paired-20260925'
ATTEMPT='independent-task-benefit-20260925-once'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    if (ROOT/'runs').exists():raise RuntimeError('Do not refreeze an attempted run')
    if (ROOT/'execution-manifest.json').exists():raise RuntimeError('Already frozen')
    files={}
    def add(p):
        p=p.resolve();files[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    old=read(OLD/'execution-manifest.json')
    for row in old['files']:
        p=Path(row['path'])
        if sha(p)!=row['sha256']:raise RuntimeError('Prior frozen identity changed: '+str(p))
        add(p)
    req=read(ROOT/'BUDGET_REQUEST.json')
    for p in map(Path,req['inputs'].values()):add(p)
    for p in [OLD/'execution-manifest.json',OLD/'runs/runtime-necessity-v1/budget.sqlite3',OLD/'runs/runtime-necessity-v1/final-review/decision.json',
              OLD/'runs/runtime-necessity-v1/timing-comparability-audit/report.md']:
        add(p)
    ledgers=read(ROOT.parent/'gppo-world-stage3-attribution-planning-20260925/stage2-budget-readonly.json')
    for row in ledgers['segments']:
        p=Path(row['path'])
        if sha(p)!=row['sha256']:raise RuntimeError('Historical ledger changed: '+str(p))
        add(p)
    diffs={}
    for p in ROOT.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
        prior=OLD/p.name
        if prior.exists() and sha(prior)!=sha(p):
            diffs[p.name]=''.join(difflib.unified_diff(prior.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='prior/'+p.name,tofile='diagnostic/'+p.name))
    (ROOT/'source-differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in {'authorization-approved.json','execution-manifest.json','hashes.json'}:add(p)
    save(ROOT/'execution-manifest.json',{'attempt':ATTEMPT,'files':sorted(files.values(),key=lambda r:r['path']),
         'prior_A_evidence_reused':True,'prior_A_cost_failure_preserved':True,'A_replay_forbidden':True})
    approval_text='批准审计报告提出的独立任务收益诊断，授权必要零步实现、测试、身份冻结、B/C条件执行、完整分析和报告。固定800父场景各唯一repeat，只新增N。上限环境步14436、reset802、encode/actor各14436、world/规则0、加载2、wall3小时10分、CPU12小时40分、RSS4GiB、制品3.5GiB、更新0。遵守B/C分项及关机预留，不借用。B技术通过自动C。保留原A成本门失败及全部消耗，不修改阈值、不重试A；10ms失败不作为本次任务诊断中途停止条件。技术或资源异常即停、不重试；不授权训练、后续实验或Git发布。'
    save(ROOT/'authorization-approved.json',{'approved':True,'attempt':ATTEMPT,'caps':req['caps'],'stage_caps':req['stage_caps'],
         'manifest_sha256':sha(ROOT/'execution-manifest.json'),'budget_sha256':sha(ROOT/'BUDGET_REQUEST.json'),
         'user_approval_text':approval_text,'approval_text_is_faithful_summary':True,
         'user_approval_reference':'Current task: explicit approval immediately following timing-comparability audit; subsequent 继续 confirms continuation',
         'prior_A_failed_not_reauthorized':True})
    print(json.dumps({'attempt':ATTEMPT,'frozen_files':len(files),'manifest_sha256':sha(ROOT/'execution-manifest.json'),
          'env_steps':0,'model_forwards':0,'budget_database_created':False},ensure_ascii=False))

if __name__=='__main__':main()
