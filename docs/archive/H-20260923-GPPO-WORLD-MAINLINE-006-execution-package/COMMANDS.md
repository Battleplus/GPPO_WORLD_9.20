# H-006 可复制命令（授权前仅允许静态检查）

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:\Python314\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_world_event_feature_pair_20260923.py tests/test_analyze_world_event_feature_pair_20260923.py tests/test_authorize_world_event_feature_pair_20260923.py
C:\Python314\python.exe -X utf8 tools/run_world_event_feature_pair_20260923.py --check-package
```

正式动态命令必须在用户批准同库扩额、生成 signed authorization JSON 后，由授权器生成的 authorization 文件作为唯一入口；不得绕过授权器、另建 SQLite 或使用现有 105 步。
