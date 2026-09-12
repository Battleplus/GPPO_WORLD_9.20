# 服务器 pilot 阻塞记录（2026-09-12）

## 连接结论

- 目标：`user1@172.17.27.173`。
- UTC 检查时间：`2026-09-12T09:10:17.5756298Z`。
- TCP/22：5 秒限时探测结果为 `timeout_or_unreachable`。
- SSH：一次 `BatchMode=yes`、`ConnectTimeout=10`、`ConnectionAttempts=1` 检查未完成；未获得登录、主机名或用户身份输出。该结果不能归类为密码/密钥错误。
- 未启动远程进程，未停止或修改任何远程任务，未修改本机网络配置，也未购买算力。

当前唯一阻塞是恢复到 `172.17.27.173:22` 的网络/路由可达性（可能需要原校园网/VPN 路由或管理员恢复主机/网络）。可达后再使用已有安全密钥/会话核对认证；本轮没有继承可用认证会话，因此没有写入或索取凭据。

## 已核验本地输入

- 源码提交：`94d41d36f53f29934870261f9085b2633812c2d0`。
- 开发数据：`E:\Z博士\9.2日\m10-consequence-data-dev-h6-20260912`。
- 开发 manifest：`gppo-consequence-dataset/v2`，Graph-5/25-action，`horizon_steps=6`。
- 开发数据仅用于窗口/生成器/审计检查；其中 test/OOD 已被查看，不能作为未触碰盲测。
- 该目录 train/validation/test/OOD 的文件 SHA-256、记录数和 identity SHA-256 已写入 manifest，并由审计入口验证通过。

## 恢复后的直接执行顺序

在服务器独立环境中：

```bash
git checkout 94d41d36f53f29934870261f9085b2633812c2d0
python tools/generate_m10_consequence_dataset.py \
  --out <new-pilot-data> \
  --count-per-split 4 \
  --prefix-steps 2 \
  --horizon-steps 6 \
  --protocol world-gppo-9.11-consequence/0.1.1-dev-window6
python tools/audit_m10_consequence_data.py \
  --data <new-pilot-data> \
  --manifest <new-pilot-data>/manifest.json
python tools/train_m10_consequence_model.py \
  --protocol configs/world-gppo-9.11-consequence-v0.1.1-dev-window6.json \
  --data <new-pilot-data> \
  --manifest <new-pilot-data>/manifest.json \
  --out <unique-server-pilot-run> \
  --device cuda \
  --run-id <unique-server-pilot-run-id>
```

启动前必须记录 GPU、CUDA/Torch/依赖、磁盘、实际随机状态、输入哈希、最大更新数 `10000` 和墙钟上限 `7200s`。pilot 结束后才检查 best/last 加载及同配置中断续跑；不使用本地 smoke 数据冒充 CUDA 证据。

