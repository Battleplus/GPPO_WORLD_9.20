import hashlib,json,zipfile
import pytest
from tools.export_training_run import export,completion_state


def test_export_preserves_every_file_and_checks_hashes(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'model.pt').write_bytes(b'opaque checkpoint fixture')
    (run/'history.jsonl').write_text('{"update":1}\n')
    out=tmp_path/'export.zip';result=export(run,out)
    assert not result['completed'] and result['checkpoint_count']==1
    with zipfile.ZipFile(out) as z:
        manifest=json.loads(z.read('EXPORT-MANIFEST.json'))
        for m in manifest['files']:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    with pytest.raises(FileExistsError):export(run,out)


def test_completion_requires_full_matrix_and_no_failure(tmp_path):
    run=tmp_path
    (run/'results.json').write_text(json.dumps({'status':'development_matrix_completed','updates':2,'results':[{'group':'S','seed':1}]}))
    assert not completion_state(run)
    (run/'run-manifest.json').write_text(json.dumps({'protocol':{'groups':['S'],'seeds':[1],'training':{'unique_updates_total':2}}}))
    assert not completion_state(run)
    (run/'S-1').mkdir();(run/'S-1'/'terminal.pt').write_bytes(b'fixture')
    assert completion_state(run)
    (run/'failure.json').write_text('{}');assert not completion_state(run)
