"""Previously tested process/monitor primitives, adapted only to this protocol."""

import queue

import subprocess

import threading

from runtime_support import *

from sqlite_diagnostics import call_once

PROGRAM_STARTED=time.perf_counter()

class Monitor:
    def __init__(self,request,out):
        self.request=request;self.out=out;self.children=[];self.started=PROGRAM_STARTED;self.parent_start=0.
        self.parent_parts={};self.peak_sum=0;self.stage=None;self.enter('B')
        self.stage_start=self.started;self.stage_cpu_start=0.;self.stage_parent_cpu_start=0.;self.stage_disk_start=0
    def enter(self,stage):
        if self.stage==stage:return
        self.stage=stage;self.stage_start=time.perf_counter();self.stage_cpu_start=self.total_cpu();self.stage_parent_cpu_start=time.process_time();self.stage_disk_start=self.disk()
    def disk(self):return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    def total_cpu(self):return time.process_time()-self.parent_start+sum(process_cpu(c.p._handle) for c in self.children)
    def measure(self,category,fn,*args,**kwargs):
        value,elapsed=timed(fn,*args,**kwargs)
        part=self.parent_parts.setdefault(self.stage+'/'+category,{'cpu':0.,'wall':0.})
        for k,v in elapsed.items():part[k]+=v
        return value
    def check(self,disk=False):
        cap=self.request['stage_caps'][self.stage];cpu=self.total_cpu();hold=self.request['recovery_shutdown_holdback']
        prep=self.request['preparation_charge_bound'];stage_prep=prep if self.stage=='B' else {'wall_seconds':0,'cpu_seconds':0}
        if time.perf_counter()-self.started+prep['wall_seconds']>self.request['caps']['wall_seconds']-hold['wall_seconds'] or time.perf_counter()-self.stage_start+stage_prep['wall_seconds']>cap['wall_seconds']-hold['wall_seconds']:raise RuntimeError('Wall cap')
        if cpu+prep['cpu_seconds']>self.request['caps']['total_process_cpu_seconds']-hold['cpu_seconds'] or cpu-self.stage_cpu_start+stage_prep['cpu_seconds']>cap['cpu_seconds']-hold['cpu_seconds']:raise RuntimeError('CPU cap')
        # Conservative sum: parent peak plus every resident child peak, including suspended workers.
        rss=peak_rss()
        for c in self.children:
            if c.p.poll() is None:
                try:rss+=process_memory(c.p._handle)[1]
                except OSError:
                    if c.p.poll() is None:raise
        self.peak_sum=max(self.peak_sum,rss)
        if rss>self.request['caps']['resident_process_rss_sum_bytes']:raise RuntimeError('Combined RSS cap')
        if disk:
            size=self.disk();stage_cap=self.request['stage_caps'][self.stage]['artifact_bytes']
            if size>self.request['caps']['artifact_bytes'] or size-self.stage_disk_start>stage_cap:raise RuntimeError('Artifact cap')
    def snapshot(self):
        return {'wall_seconds':time.perf_counter()-self.started,'total_cpu_seconds':self.total_cpu(),
            'preparation_charge_bound':self.request['preparation_charge_bound'],
            'current_stage':self.stage,'stage_wall_seconds':time.perf_counter()-self.stage_start,
            'stage_cpu_seconds':self.total_cpu()-self.stage_cpu_start,'stage_parent_cpu_seconds':time.process_time()-self.stage_parent_cpu_start,
            'parent_cpu_seconds':time.process_time()-self.parent_start,'parent_timed_parts':self.parent_parts,
            'peak_conservative_rss_sum':self.peak_sum,'rss_scope':'parent plus all resident workers, including suspended; sum of individual peaks','bytes':self.disk(),
            'workers':[{'name':c.name,'cpu_seconds':process_cpu(c.p._handle),'exit_code':c.p.poll(),'ready':getattr(c,'ready',None),'final':getattr(c,'final',None)} for c in self.children]}

class OwnedWorker:
    def __init__(self,name,stage,kind,auth,monitor):
        self.name=name;self.monitor=monitor;self.suspended=False;self.responses=queue.Queue();out=monitor.out
        write(out/(name+'-permit.json'),{'stage':stage,'kind':kind})
        self.err=(out/(name+'-stderr.txt')).open('x',encoding='utf-8')
        self.p=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/'runtime_worker.py'),'--authorization',str(auth.resolve()),'--name',name],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
        monitor.children.append(self)
        def reader():
            try:
                for line in self.p.stdout:self.responses.put(json.loads(line))
                self.responses.put({'ok':False,'error':'Worker stdout closed'})
            except BaseException as exc:self.responses.put({'ok':False,'error':str(exc)})
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        self.ready=self.receive();self.pause()
    def receive(self):
        while True:
            self.monitor.check()
            try:response=self.responses.get(timeout=.2)
            except queue.Empty:continue
            if not response.get('ok'):raise RuntimeError('Worker failure: '+str(response))
            return response
    def pause(self):
        if self.p.poll() is None:suspend(self.p._handle);self.suspended=True
    def resume(self):
        if self.suspended:suspend(self.p._handle,resume=True);self.suspended=False
    def call(self,command,**kwargs):
        self.resume();self.p.stdin.write(canonical({'command':command,**kwargs})+'\n');self.p.stdin.flush()
        response=self.receive();self.pause();return response['result']
    def close(self):
        self.resume();self.p.stdin.write(canonical({'command':'shutdown'})+'\n');self.p.stdin.flush()
        self.final=self.receive()['result'];self.p.stdin.close()
        while self.p.poll() is None:
            self.monitor.check()
            try:self.p.wait(timeout=.2)
            except subprocess.TimeoutExpired:continue
        if self.p.returncode!=0:raise RuntimeError('Worker nonzero exit')
        self.err.close()
    def kill(self):
        if self.p.poll() is None:self.p.kill();self.p.wait(timeout=10)
        self.err.close()

class Reservations:
    def __init__(self,budget,monitor):self.budget=budget;self.monitor=monitor;self.pending=[]
    def reserve(self,amounts):
        if self.pending:raise RuntimeError('Unresolved prior reservation')
        for key,amount in amounts.items():
            if amount:self.pending.append(self.monitor.measure('ledger',call_once,'reserve',self.budget.reserve,self.monitor.stage+'/'+key,amount,diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path))
    def complete(self):
        for token in list(self.pending):
            self.monitor.measure('ledger',call_once,'complete',self.budget.complete,token,diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path)
            self.pending.remove(token)
    def unknown(self,error):
        for token in list(self.pending):
            try:call_once('mark_unknown',self.budget.unknown,token,str(error),diagnostic_dir=self.monitor.out/'diagnostics',db_path=self.budget.db_path)
            except Exception as secondary:error.add_note('Unable to mark pending reservation unknown: '+repr(secondary))
