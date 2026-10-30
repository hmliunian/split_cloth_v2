# 开发与验证

2026-10-30 用户授权按既有 Markdown 进入代码实施，要求精简、无冗余。业务边界见 [boundary.md](boundary.md)，依赖方向见 [architecture.md](architecture.md)。

正式代码提供两种布局共用的配置、特征、模板、DTW、搜索、结果、CLI 和视频导出。6 轴冻结模板迁入 `configs/joint/templates/`。7 轴读取器使用实测执行臂反馈，按原视频时间轴最近采样对齐，保留逐帧源行、时间和误差。VLM 专属实现、资产、依赖、命令及测试移除，历史产物继续只读。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_joint_reverse.py
```

实现检查覆盖共同搜索、缺失语义、候选身份、布局/模板错配、两种反馈布局、MCAP 对齐、源文件保护及 CLI。另须执行 6 轴冻结回放，比较边界、路径、分数、b1 验收和逐帧数组，并验证安装后的包从仓库外运行。

7 轴十条示范的观察与候选参考、模板建立及完整视频属于实验交付；既有三窗口标注不能直接转换为新参考。未完成观察时保持缺失，不用自动预测充当 Codex 标注。独立评估使用与示范不重叠的清单，示范回放不报告准确率。

本次实际检查：

- 默认离线测试：47 passed；joint 专项：9 passed；`git diff --check` 与当前文档链接检查通过。
- 6 轴冻结回放：30 条、19,709 帧、403 个候选，边界、原始假设、最优两条路径、代价、b1 验收和逐帧标签全部一致。证据：[replay_verification.json](runs/joint_package_replay_20261030_v2/replay_verification.json)。
- 7 轴原始读取：10 个不同来源、12,226 帧，全部为 16 维反馈；最大反馈对齐误差 41.576 ms，源 MP4 相对 PTS 最大误差 0。逐帧超 20 ms 情况保存在各轨迹对齐记录。证据：[input_verification.json](runs/joint_7axis_prepared_20261030_v1/input_verification.json)。
- 安装独立性：离线构建 wheel、安装到新的临时环境，从 `/tmp` 使用隔离 Python 完成 CLI 实际推断和 DTW 调用；仓库不在导入路径，包内含 C++ 源码且无 VLM 模块。证据：[package_verification.json](runs/joint_package_reader_checks_20261030_v1/package_verification.json)。

7 轴实验尚未完成：[参考状态](runs/joint_7axis_observations_20261030_v1/observation_status.json)。十条来源与输入已冻结，其中四条保存了 b2–b4 的部分 Codex 观察，b1 因准备/抓取与第一折开始无法可靠区分而保持空值，其余参考复核未完成。未使用旧三窗口标注代替新参考，未建立 7 轴模板，也未导出十条预测复核视频。现有源视频转换检查与合成 VFR 导出测试不能替代这项实验交付。所有示范仍未获用户确认，没有独立准确率结论。

旧验证见 [历史记录](docs/experiments/joint_reverse_suffix_20261029/verification.json)，不作为本次测试结果。
