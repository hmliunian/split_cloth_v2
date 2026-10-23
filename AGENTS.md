# 开发入口

先读 [boundary.md](boundary.md) 和 [DEVELOPMENT.md](DEVELOPMENT.md)，再按任务读 `docs/`。本库已有独立 Python 包及 CLI，实际用法与已执行检查见 [README.md](README.md)。默认离线验证入口为 `.venv/bin/python -m pytest -q`。

## 有效规则

- 用户最新要求优先；业务规则以 `boundary.md` 为准，`docs/PROMPT.md` 是对应的模型接口。
- 核心流程是传统 state 候选、程序倒序逐节点调用 VLM、stage3→stage2→stage1。
- Dataset 只做叠衣服，其他任务不作为输入或负例；不增加通用任务分类和挂衣识别流程。
- 开发第一阶段（P1）交付 10 条不同叠衣轨迹的完整 MP4，使用 [第一阶段清单](docs/phase1_manifest.jsonl)；默认可视化按 [docs/VISUALIZATION.md](docs/VISUALIZATION.md) 仿照旧版画面、阶段面板和时间线。
- 任意一爪张开即可；不要引入“所有参与手必须释放”、让 VLM 自由选帧或整段一次推理。
- 原始数据、旧库和历史产物只读；新产物落在本库 `runs/`。密钥、视频、原始 Parquet 和生成的大文件不入版本库。
- 实现后保持新库独立运行，不在运行时导入旧库或 `/tmp` worktree。迁移代码时只选本任务需要的部分。

## 验证与交付

- 按 [docs/EVALUATION.md](docs/EVALUATION.md) 验证相关边界。默认检查离线，模拟调用不访问真实服务。
- 真实 VLM 调用会发送目标图像并计费，按当次任务已授权的范围执行。已有授权后连续完成，不增加重复确认步骤。
- prompt、模型、采样或检测配置变化应使用新运行并保存请求指纹。`uncertain` 和语义不匹配不能靠重复采样换答案。
- 000543 的 400/485/562 只用于评估，不进 prompt。程序通过、用户确认和模型质量分别报告。
- 完成说明写清改动、实际执行的检查、结果路径和未解决问题。测试、程序状态、模型质量和人工确认分别报告，不从程序通过推断准确率。

普通实现只更新受影响的契约和文档；无需另建审批体系或重复增加设计文件。
