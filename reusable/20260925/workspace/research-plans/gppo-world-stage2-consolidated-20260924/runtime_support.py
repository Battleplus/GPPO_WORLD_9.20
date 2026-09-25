"""Shared authorization, pinned imports and Windows process accounting."""
import ast
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
PREP=ROOT.parent/'gppo-world-stage2-gate-preparation-20260924'
COST=ROOT.parent/'gppo-world-cost-replay-preparation-20260924'
sys.path.insert(0,str(PREP));sys.path.insert(0,str(COST))
from run_gate import primitive,canonical,digest,sha,peak_rss,verify_vector,ending
from public_controller import PublicMemory,PublicPlanner,public_copy

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        f.write(canonical(obj)+'\n');f.flush();os.fsync(f.fileno())
def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(canonical(obj)+'\n');f.flush();os.fsync(f.fileno())
def authorize(path):
    if not path or not Path(path).is_file():raise PermissionError('Explicit consolidated budget approval required')
    a=read(path);b=read(ROOT/'execution-manifest.json');r=read(ROOT/'BUDGET_REQUEST.json')
    if (a.get('approved') is not True or a.get('execution_manifest_sha256')!=sha(ROOT/'execution-manifest.json')
        or a.get('budget_request_sha256')!=sha(ROOT/'BUDGET_REQUEST.json')
        or a.get('caps')!=r['caps'] or a.get('stage_caps')!=r['stage_caps']
        or not a.get('user_approval_reference') or not a.get('user_approval_text')):
        raise PermissionError('Approval does not bind the complete conditional chain')
    for row in b['files']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Bound input changed: '+row['path'])
    return a,b,r

def native_package(binding):
    """Import submodules without eager model re-exports in native __init__.py.

    Both arms use this same transparent package shim. Submodule source is not
    modified; skipping the export-only initializer prevents Torch in R.
    """
    src=Path(binding['native_root'])/'gppo_world'
    if 'gppo_world' in sys.modules:raise RuntimeError('Unexpected preloaded native package')
    tree=ast.parse((src/'__init__.py').read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):continue
        if isinstance(n,ast.ImportFrom):continue
        if isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id=='__all__' for t in n.targets):continue
        raise RuntimeError('Native package initializer no longer only re-exports')
    pkg=types.ModuleType('gppo_world');pkg.__path__=[str(src)];pkg.__file__=str(src/'__init__.py');pkg.__package__='gppo_world'
    sys.modules['gppo_world']=pkg

def verify_imports(binding):
    src=Path(binding['native_root']).resolve();found={}
    for name,m in list(sys.modules.items()):
        if name=='gppo_world' or name.startswith('gppo_world.'):
            path=Path(m.__file__).resolve();relative=path.relative_to(src).as_posix()
            if sha(path)!=binding['native_python_files'].get(relative):raise RuntimeError('Unpinned native import')
            found[name]=str(path)
    return found

def load_reward_function(binding):
    import numpy as np
    source=(Path(binding['native_root'])/'gppo_world/joint_training.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_vector_reward')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ns={'np':np};exec(compile(ast.fix_missing_locations(module),'<pinned native reward function>','exec'),ns)
    return ns['_vector_reward']

def budget_class(binding):
    path=Path(binding['native_root'])/'gppo_world/budget_executor.py'
    spec=importlib.util.spec_from_file_location('pinned_pipeline_budget',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.PersistentBudget

def process_cpu(handle):
    class FT(ctypes.Structure):_fields_=[('low',ctypes.c_ulong),('high',ctypes.c_ulong)]
    times=[FT() for _ in range(4)];fn=ctypes.windll.kernel32.GetProcessTimes
    fn.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(FT)]*4;fn.restype=ctypes.c_int
    if not fn(ctypes.c_void_p(int(handle)),*(ctypes.byref(v) for v in times)):raise OSError('GetProcessTimes failed')
    return sum((v.high<<32)+v.low for v in times[2:])*1e-7

def process_memory(handle):
    class PC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    pc=PC();pc.cb=ctypes.sizeof(pc);fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong];fn.restype=ctypes.c_int
    if not fn(ctypes.c_void_p(int(handle)),ctypes.byref(pc),pc.cb):raise OSError('Memory accounting failed')
    return pc.working,pc.peak

def suspend(handle,resume=False):
    fn=getattr(ctypes.windll.ntdll,'NtResumeProcess' if resume else 'NtSuspendProcess')
    fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_long
    if fn(ctypes.c_void_p(int(handle)))!=0:raise OSError('Owned worker suspend/resume failed')

def timed(fn,*args,**kwargs):
    w=time.perf_counter();c=time.process_time();value=fn(*args,**kwargs)
    return value,{'cpu':time.process_time()-c,'wall':time.perf_counter()-w}

class CompressedRecords:
    def __init__(self,path):
        self.raw=Path(path).open('xb');self.gz=gzip.GzipFile(fileobj=self.raw,mode='wb',compresslevel=1,mtime=0)
        self.digest=hashlib.sha256();self.rows=0
    def add(self,row):
        data=(canonical(row)+'\n').encode('utf-8');self.digest.update(data);self.rows+=1
        self.gz.write(data);self.gz.flush();self.raw.flush();os.fsync(self.raw.fileno())
    def close(self):
        self.gz.close();self.raw.close()

def verify_compressed(path,expected_hash,rows):
    h=hashlib.sha256();count=0
    with gzip.open(path,'rb') as f:
        for line in f:h.update(line);json.loads(line);count+=1
    if h.hexdigest()!=expected_hash or count!=rows:raise RuntimeError('Compressed record round-trip failed')
