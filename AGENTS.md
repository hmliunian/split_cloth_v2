# 开发入口

先读 [boundary.md](boundary.md)、[architecture.md](architecture.md) 和 [DEVELOPMENT.md](DEVELOPMENT.md)，再按任务读 `docs/`。[README.md](README.md) 区分当前可运行入口、目标架构与历史证据。用户最新要求优先；业务语义以 `boundary.md` 为唯一规范。

## 主算法与代码位置

- 仓库只保留纯 joint 分段作为主算法：实测关节/夹爪反馈 → 张开候选 → 完整动作 DTW → S5→S4→S3→S2 逆序动态规划；S1 为不计分的自由整理前缀。推断不读取图像、不调用 VLM。公式见 [docs/JOINT_REVERSE_SUFFIX.md](docs/JOINT_REVERSE_SUFFIX.md)。
- 6 轴与 7 轴使用同一算法和配置结构。正式代码目标为 `src/split_cloth_v2/joint/`，配置为 `configs/joint/`，由仓库 CLI 统一调度；不得按机器人复制训练、特征、搜索和输出实现。
- `experiments/joint_reverse_suffix/` 是现有 6 轴基线，`experiments/pushcar_joint/` 是 7 轴草稿，仅作为迁移来源。正式包运行时不得导入实验目录、旧库、历史运行源码或 `/tmp` worktree。
- 用户已要求删除 VLM，不保留兼容功能、入口或自动回退。2026-10-30 用户已授权按现有 Markdown 实施代码，并要求精简、无冗余；已进入正式包迁移与验证阶段。

## 关键边界

- 只接收叠衣服来源：`fold_cloth_v3` 与 `fold_cloth_v3_pushcar_batch_20260923`。原始目录、14/16 维布局、反馈来源和对齐规则见 `boundary.md`；EEF、遥操作输入和控制命令不能代替实测 joint。
- 任意一爪张开即可。b1–b4 必须来自冻结的非末帧候选，边界帧属于前段；缺失保留 `null` 和已支持的后缀，不补位。7 轴草稿的三个自由窗口搜索不作为配置选项保留。
- Codex 可手选、查看并标注 10 条不同的 7 轴示范建立模板；保存来源、不确定性和替换理由。Codex 参考、模型预测与用户确认分别记录，示范不得混入独立评估。
- 原始数据、旧库、历史产物及人工记录只读；新产物写入新的 `runs/<run_id>/`。密钥、视频、MCAP、原始 Parquet 和生成大文件不入版本库。
- 配置、检测、特征、模板或参考变化使用新运行并冻结摘要。模块职责和依赖按 `architecture.md`，读取器不决定阶段，渲染器不重新推断。

## 验证与交付

- 默认离线检查：`.venv/bin/python -m pytest -q`；现有 joint 专项：`.venv/bin/python -m pytest -q tests/test_joint_reverse.py`。命令存在不代表本轮已执行，历史通过数不得当作本轮结果。
- 按 [docs/EVALUATION.md](docs/EVALUATION.md) 验证 6 轴回放兼容、7 轴布局/对齐、共同搜索和缺失语义、源数据保护及安装后的包独立运行。
- 后续 10 条 7 轴示范各交付完整 MP4、参考/预测来源、候选和时间轴验证。布局见 [docs/VISUALIZATION.md](docs/VISUALIZATION.md)，阶段与标签语义以 `boundary.md` 为准。
- 完成说明分别报告文档/软件检查、程序结果、算法质量、Codex 观察和用户确认，给出结果路径和未解决问题；测试通过或阶段完整不代表准确率。

普通改动只更新受影响的契约与文档，无需另建审批体系或重复增加设计文件。旧 VLM 文档与运行记录不作为当前开发规则。
