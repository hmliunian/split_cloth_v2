# split_cloth_v2

基于实测关节和夹爪反馈的叠衣分段：张开候选 → 完整动作 DTW → S5→S4→S3→S2 逆序动态规划，S1 为不计分的整理前缀。6 轴和 7 轴共用 `src/split_cloth_v2/joint/`，通过 `configs/joint/` 选择布局、数据读取和模板。推断不读取图像，不调用模型服务。

先读 [boundary.md](boundary.md)、[architecture.md](architecture.md)、[DEVELOPMENT.md](DEVELOPMENT.md)。算法公式见 [JOINT_REVERSE_SUFFIX.md](docs/JOINT_REVERSE_SUFFIX.md)，数据及展示规则见 [DATA.md](docs/DATA.md)、[VISUALIZATION.md](docs/VISUALIZATION.md)。

## 安装与使用

需要 Python 3.11+、本地 `g++`；MCAP 的 zstd 压缩需要系统 `libzstd`。DTW 源码随包安装，首次调用在临时缓存中本地编译。

```bash
uv pip install --python .venv/bin/python -e '.[test]'

# 6 轴：冻结 NPZ/manifest 输入；prepare 验证并冻结检测结果
.venv/bin/python -m split_cloth_v2.cli prepare \
  --config configs/joint/fold_cloth_v3_6axis.yaml \
  --manifest /path/to/prepared/manifest.json --run runs/new_prepared

.venv/bin/python -m split_cloth_v2.cli infer \
  --config configs/joint/fold_cloth_v3_6axis.yaml \
  --input-run runs/new_prepared --run runs/new_predictions

.venv/bin/python -m split_cloth_v2.cli export-video --run runs/new_predictions --workers 2
.venv/bin/python -m split_cloth_v2.cli report --run runs/new_predictions

# 7 轴：清单列出 source_mcap、source_metadata、key 和来源
.venv/bin/python -m split_cloth_v2.cli prepare \
  --config configs/joint/fold_cloth_v3_pushcar_7axis.yaml \
  --manifest /path/to/selection.json --run runs/new_7axis_prepared

# 先观察并标注十条不同示范，再拟合模板
.venv/bin/python -m split_cloth_v2.cli fit \
  --config configs/joint/fold_cloth_v3_pushcar_7axis.yaml \
  --input-run runs/new_7axis_prepared --references /path/to/references.json \
  --run runs/new_7axis_templates

.venv/bin/python -m split_cloth_v2.cli infer \
  --config configs/joint/fold_cloth_v3_pushcar_7axis.yaml \
  --templates runs/new_7axis_templates/templates.json \
  --input-run runs/new_7axis_prepared --training-replay --run runs/new_7axis_replay
```

所有输出目录必须是新目录，且在只读输入之外。默认视频与报告分别写入运行下的 `exports/video` 和 `exports/report`，再次导出须指定新的 `--output`。相对资源路径按配置文件所在目录解析。安装后的控制台入口为 `split-cloth-v2`，命令相同。

6 轴模板保存在 `configs/joint/templates/`，保持既有双初始化 b1 验收。7 轴配置不附带臆造模板；必须先完成十条符合连续五段语义的参考。示范回放必须显式使用 `--training-replay`，不作为独立评估。

## 示范参考

`references.json` 以轨迹 key 为键。每条记录包含 `source: "Codex"`、`human_confirmed: false`、`selection_reason` 和 `boundaries`。后者固定含 `b1`–`b4`，值为 `null` 或 `{frame, node_id, timestamp_s, reason, evidence}`；`evidence` 为已查看证据的路径列表。边界必须对应冻结候选；缺失或不确定就保留 `null`，仅训练两端有依据的动作段。完整规则见 [boundary.md](boundary.md)。

## 检查与产物

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_joint_reverse.py
```

运行冻结配置、输入摘要、源码摘要和模板。预测包含固定 b1–b4、候选身份、最优两条路径、代价和缺失原因；五段或四段标签缺少所需边界时整条为 `-1`。视频逐帧保留源相对 PTS，导出后重新解码验证。

VLM 专属源码、CLI、配置及依赖已移除；`experiments/` 和历史 `runs/` 保留为只读证据，正式包不依赖它们。实际验证结果与未完成实验见 [DEVELOPMENT.md](DEVELOPMENT.md)。测试通过、结构完整和视频导出成功均不表示边界准确率。
