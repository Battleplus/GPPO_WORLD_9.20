"""Read-only terminal-run audit and concise local research decision. No experiment imports."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/consolidated-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,data):
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def rows(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def main():
    terminal=read(OUT/'status.json')
    assert terminal['status'] in ('completed','technical_stop','research_stop_A_cost')
    manifest=read(OUT/'execution-manifest.json')
    mismatches=[r['path'] for r in manifest['files'] if sha(Path(r['path']))!=r['sha256']]
    assert not mismatches,mismatches
    con=sqlite3.connect((OUT/'budget.sqlite3').as_uri()+'?mode=ro',uri=True)
    integrity=con.execute('pragma integrity_check').fetchone()[0]
    stages={r[0]:dict(zip(['limit','reserved','verified','unknown'],r[1:])) for r in con.execute('select stage,limit_amount,reserved,verified,unknown from stages order by stage')}
    for v in stages.values():v['pending']=v['reserved']-v['verified']-v['unknown'];assert v['reserved']<=v['limit']
    reservations={r[0]:r[1] for r in con.execute('select status,count(*) from reservations group by status')}
    con.close();assert integrity=='ok'
    totals={k:{stat:sum(v[stat] for s,v in stages.items() if s.endswith('/'+k)) for stat in ['reserved','verified','unknown','pending']} for k in ['environment_steps','policy_encode','world_candidate_batch','actor_readout','rule_decisions','environment_resets','model_loads','optimizer_updates','world_updates','offline_updates']}
    matrix={}
    for stage,expected in [('B',4),('C',9600)]:
        episodes=rows(OUT/stage/'episodes.jsonl');keys=[(r['parent'],r['repeat'],r['arm']) for r in episodes]
        assert len(keys)==len(set(keys))
        matrix[stage]={'episodes':len(episodes),'expected':expected,'episode_environment_steps':sum(r['steps'] for r in episodes)}
        if terminal['status']=='completed':
            assert len(episodes)==expected
            assert sum(r['steps'] for r in episodes)==stages[stage+'/environment_steps']['verified']
            assert sum(r['steps'] for r in episodes if r['arm']=='M')==stages[stage+'/policy_encode']['verified']
            assert sum(r['steps'] for r in episodes if r['arm']=='R')==stages[stage+'/rule_decisions']['verified']
    if terminal['status']=='completed':
        assert not any(v['unknown'] or v['pending'] for v in stages.values())
        analysis=read(OUT/'C/analysis.json')
        assert analysis['parent_count']==1600 and analysis['repeat_count']==3 and analysis['episodes']==9600
    else:analysis=None
    assert all(totals[k]['reserved']==0 for k in ['optimizer_updates','world_updates','offline_updates'])
    resources=terminal['resources'];cap=read(ROOT/'BUDGET_REQUEST.json')['caps']
    resource_pass=resources['wall_seconds']<=cap['wall_seconds'] and resources['total_cpu_seconds']<=cap['total_process_cpu_seconds'] and resources['peak_conservative_rss_sum']<=cap['resident_process_rss_sum_bytes'] and resources['bytes']<=cap['artifact_bytes']
    decision=analysis['decision'] if analysis else 'pause_after_'+terminal['status']
    result={'run_status':terminal['status'],'research_decision':decision,'analysis':analysis,'totals':totals,'matrix':matrix,'sqlite_integrity':integrity,'reservation_rows':reservations,'stages':stages,'resources':resources,'resource_caps_pass':resource_pass,'bound_files_verified':len(manifest['files']),'bound_file_mismatches':mismatches,'scope':'Historical native M10/W1, one frozen model seed; no finite-communication deployment or general model-seed claim','stage3_to5_experiments_authorized':False,'automatic_retry':False}
    final=ROOT/'final-review';final.mkdir(exist_ok=False);save(final/'decision.json',result)
    lines=['# 阶段2条件执行结果','',f"运行终态：{terminal['status']}。去留：{decision}。",'']
    if analysis:
        u=analysis['summary']['utility'];m=analysis['controller_cost']['M'];r=analysis['controller_cost']['R']
        lines += [f"独立父场景1600，3次外生重复，9600个episode。主效用M−R={u['mean_difference']:.8f}，95% CI [{u['ci95'][0]:.8f}, {u['ci95'][1]:.8f}]；预定实用增量0.01（效用单位，不是百分点）。",'',f"任务护栏通过：{analysis['task_guard_pass']}；模型控制成本护栏通过：{analysis['cost_guard_pass']}。M CPU均值{m['mean_cpu_ms']:.6f}ms、wall p95 {m['wall_p95_ms']:.6f}ms；R CPU均值{r['mean_cpu_ms']:.6f}ms。",'']
    else:lines += ['未完成正式比较，不能判定模型具有实用增量。技术失败与算法负结果分开。',str(terminal.get('error','A成本门未通过。')),'']
    lines += [f"实际环境步 verified={totals['environment_steps']['verified']}，unknown={totals['environment_steps']['unknown']}，pending={totals['environment_steps']['pending']}；encode/world/actor各自消耗详见decision.json。训练更新0。",'',f"完整运行wall={resources['wall_seconds']:.3f}s，CPU={resources['total_cpu_seconds']:.3f}s；父调度器与全部驻留worker（含暂停）的保守RSS峰值和={resources['peak_conservative_rss_sum']/1024**2:.3f}MiB。完整资源上限通过：{resource_pass}。CPU/wall为调度器终态采样，外部最终只读报告成本不混入模型决策成本。",'','仅适用于历史原生M10/W1与一个冻结模型seed，不能扩展为当前有限通信环境、训练seed泛化或部署收益。旧成本负结果与本次参考测量分开保留，不把跨次成本差异全归因于简化。阶段3—5、训练、Git发布均未执行。']
    (final/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[f for f in OUT.rglob('*') if f.is_file() and f.suffix not in ('.db-shm','.db-wal') and not f.name.endswith(('-shm','-wal'))]+list(final.iterdir())
    save(final/'hashes.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    print(json.dumps({'decision':decision,'environment_steps':totals['environment_steps'],'formal_complete':analysis is not None,'resource_caps_pass':resource_pass},ensure_ascii=False))
if __name__=='__main__':main()
