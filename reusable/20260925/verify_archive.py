"""Read-only archival checks, with no experimental imports."""
import ast,gzip,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source-manifest.json').read_text(encoding='utf-8'))
count=0
for row in manifest['files']:
    rel=Path(row['archive_path']).relative_to('reusable/20260925')
    blob=(root/rel).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==row['archive_sha256'],str(rel)
    raw=gzip.decompress(blob) if row['encoding']=='gzip' else blob
    assert hashlib.sha256(raw).hexdigest()==row['source_sha256'],str(rel)
    assert len(raw)==row['source_bytes'],str(rel)
    if row['source'].endswith('.py'): ast.parse(raw,filename=row['source'])
    count+=1
print('PASS:',count,'source files; no model loads, forwards, environment steps or updates')
