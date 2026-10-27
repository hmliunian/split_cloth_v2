# split_cloth_v2

当前分支 `experiment/main-stage3-prompt-20261027` 从重新 fetch 的 `origin/main`（5306d01）建立，只实验 stage3 prompt。入口为 [reverse_node_v2_stage3_shape_guard.yaml](configs/reverse_node_v2_stage3_shape_guard.yaml)，文本为 [prompt_stage3_shape_guard.txt](configs/prompt_stage3_shape_guard.txt)。保留折回覆盖与长方形短包说明，明确排除成包后整理或搬运再释放；使用 main 的 state 张开候选，耗尽即停止。

同样十条的新运行位于 `runs/p1_main_stage3_prompt_20261027/`。prepare 时将下文配置路径替换为 `configs/reverse_node_v2_stage3_shape_guard.yaml`；模型、检测、单张示例、五帧局部上下文及程序代码均沿用 main。结果完整导出后交由用户查看，程序状态不代表已确认准确率。

本次已完成：**6 complete、4 needs_review**，stage3 找到 6/10，均没有补采样。002768/487 保持通过，006453 从旧 prompt v1 的 488 改回 472，003546 的 stage3 改为 507；但 001061/423 又被拒绝，**000543 回归失败，stage1 从参考 400 错选为 274**（实际 274/485/562）。因此本次没有达到“保留改善且没有误选”的质量目标，不建议直接采用。全部原始响应及缺槽都已保留，未人工确认。

用户要求的 [10 条候选标记视频](runs/p1_main_stage3_prompt_20261027/exports/candidate_marks/videos/)已完成，时间线上包含全部 148 个张开候选，当前候选显示节点 ID、帧号、目标 stage、判断和是否选中；NOW 表示候选原帧，NEAR 表示附近的信息展示。[完整对照报告](runs/p1_main_stage3_prompt_20261027/reports/comparison.html)同时列出 main 历史结果、此前 v1 原候选结果和本次结果。标记版是独立导出，原视频与预测保留。

离线 **197 passed in 9.70s**；原版与标记版各 10 个完整 MP4、6568 帧，最大相对 PTS 误差均为 0。额外核对了 148 个候选的精确帧、相邻 399/400 的独立高亮和全部 6568 帧的 NOW 判定，原产物摘要未变。真实推理为 102 个逻辑请求、104 次 HTTP（两次网络传输重试），记录输入/输出 153045/9806 tokens；合法语义回答没有重问。

候选标记导出入口为 [scripts/export_candidate_marks.py](scripts/export_candidate_marks.py)，只读取已有结果，沿用本库编码器与逐帧验证。示例命令：`PYTHONPATH=src .venv/bin/python scripts/export_candidate_marks.py --run runs/<id> --export-name candidate_marks`；重导出使用新的 export-name。

新库用于实现：**夹爪 state 生成张开候选 → 程序倒序逐点调用 VLM → 找到小方块、双侧长条、单侧叠入三个结束点 → 导出四阶段划分。**

已实现独立 Python 包、离线准备、逐候选倒序搜索、请求持久化与恢复、完整 MP4 导出及逐帧时间验证。日期：2026-10-26。程序验证与模型视觉质量分别记录。

已接入 `reverse_node_v2` 的[边界](boundary.md)、[架构](architecture.md)与[开发计划](DEVELOPMENT.md)：每次提供一个候选及原有局部上下文，附当前阶段的一张 expert example，判断是否“本次折叠刚完成、下一动作尚未开始”，程序倒序搜索 stage3→stage2→stage1。v2 使用 [configs/reverse_node_v2.yaml](configs/reverse_node_v2.yaml)，固定 prompt 唯一入口为 [configs/prompt.txt](configs/prompt.txt)。prepare 冻结原文、示例与摘要，annotate 核验后使用；原 v1 默认 YAML 和历史产物保留。

**开发第一阶段（P1）的首次交付为 10 条不同叠衣轨迹的完整 MP4**，附边界结果、运行用量、导出验证和实验汇总。P1 是开发里程碑，与视频内的 stage1 不同。

## 安装与离线检查

本轮实际使用 Python 3.12.3。依赖固定在 `pyproject.toml`，包括集群 SOCKS 代理需要的 `httpx[socks]`；运行记录另存实际环境版本。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

本轮完整离线检查：**197 passed**。测试包括合成媒体、实际编码后重新解码、模拟 HTTP、崩溃恢复、外置 prompt 原文与示例冻结、防篡改、六图标签、少量候选与缺槽状态，不访问真实 VLM。10 条真实输入的离线准备共核验 6568 帧，生成 148 个候选，检测器配置和结果与 v1 一致；000543 的 562 仍仅右爪张开。

## 运行

```bash
.venv/bin/python -m split_cloth_v2.cli prepare --manifest docs/phase1_manifest.jsonl --config configs/reverse_node_v2.yaml --run runs/<new_run_id>
.venv/bin/python -m split_cloth_v2.cli annotate --run runs/<new_run_id> --episode fold_cloth_v3__episode_000543 --credential-file .secrets/dashscope_api_key.txt
.venv/bin/python -m split_cloth_v2.cli annotate --run runs/<new_run_id> --credential-file .secrets/dashscope_api_key.txt
.venv/bin/python -m split_cloth_v2.cli export-video --run runs/<new_run_id>
.venv/bin/python -m split_cloth_v2.cli report --run runs/<new_run_id>
```

`prepare` 完全离线。`annotate` 发送候选、上下文及当前阶段示例并计费，优先读取 `DASHSCOPE_API_KEY`，或使用显式 `--credential-file`。按用户要求，本地凭据保存在被 Git 忽略的 `.secrets/dashscope_api_key.txt`，目录权限 700、文件权限 600；内容不写入运行记录或日志。第二次 annotate 核对指纹并重放已有判断，合法 skip/uncertain 不重问。模型、prompt、示例、采样、检测、源码或环境变化需要新运行。

需要完整离线模拟时，在 prepare 加 `--mock`；可加 `--mock-decisions <json>` 提供节点判断，否则返回模拟 uncertain。模拟模式明确记录，不能标记真实 P1 完成。

默认输出在 `runs/<id>/videos/`。旧导出存在或导出失败时，使用 `export-video --run ... --export-name retry_01` 生成独立版本，配套 `report --run ... --export-name retry_01`；历史视频、索引和 `.part.mp4` 不覆盖。任意轨迹导出失败，命令返回非零。

## v2 十条实际测试（2026-10-26）

冻结运行：`runs/p1_reverse_node_v2_examples_20261026/`，协议 `reverse_node_v2`、prompt `node_completion_prompt_v1`。同一清单、检测器、局部取样、模型与推理参数，增加三个用户指定示例及完成时刻判据。**10 个完整 MP4 全部导出并重新解码通过，共 6568 帧，最大相对 PTS 误差 0。**

真实执行 93 次逻辑请求 / 93 次 HTTP，无传输重试；输入 116,342、输出 8,434 tokens，记录的 provider 耗时 373.268 秒。程序 5 条 `complete`、5 条 `needs_review`、0 条 `execution_error`。请求图片索引、原始回答、源码副本和逐阶段搜索状态均保留，源文件摘要未变。全部新结果人工状态仍为 `unreviewed`，不从交付完成推断模型准确率。

000543 为 **400 / 485 / 562**，精确参考回归通过。agent 定性复核支持 003546 的 stage1 从 406 改选 371 更接近完成动作；006453 避开旧整理节点，但新 stage1=351 仍有偏晚疑点。001776 晚选未修复，002768 / 001061 / 001372 三槽全空，004150 仅保留 stage3。后续原始 state 核查确认：002962 的末折张开事件、001372 较早的 662 张开事件均被稳定平台规则过滤；001372 的较晚 681 虽被 VLM 拒绝，也不能据此判定纯模型误拒。若干模型依据与画面不符，不能据此宣称整体质量改善。

[远端可阅读的实验归档与最新漏选诊断](docs/experiments/reverse_node_v2_20261026/README.md) 包含小型汇总、候选、93 次判断及验证证据。下列完整运行材料保留在本地被忽略的 `runs/`。

- [十条结果、用量和视频路径](runs/p1_reverse_node_v2_examples_20261026/reports/phase1_report.md)
- [与 v1 逐条对照及未解决问题](runs/p1_reverse_node_v2_examples_20261026/reports/model_comparison.md)
- [十条 agent 定性复核](runs/p1_reverse_node_v2_examples_20261026/reports/visual_review_summary.json)
- [视频目录](runs/p1_reverse_node_v2_examples_20261026/videos/)
- [逐帧视频验证](runs/p1_reverse_node_v2_examples_20261026/reports/video_verification.json)
- [导出画面抽查](runs/p1_reverse_node_v2_examples_20261026/reports/all_video_visual_check.json)
- [程序执行与源保护审查](runs/p1_reverse_node_v2_examples_20261026/reports/execution_audit.json)
- [每次请求的图像索引](runs/p1_reverse_node_v2_examples_20261026/reports/request_images.json)

## v1 历史实验记录

正式冻结运行：`runs/p1_reverse_node_prompt_v3_20261023/`，代码 `c154ca8`，prompt 版本 `node_shape_prompt_v3`。已交付 **10 个完整 MP4**，全部重新解码验证通过，共 6568 帧，相对 PTS 最大误差为 0。请求、响应、trace、候选标签、视频及汇总均在该目录。

正式运行 83 次逻辑请求 / 83 次 HTTP 请求，输入 61,676 tokens、输出 6,128 tokens，记录的 provider 耗时约 257.239 秒，无传输重试。7 条程序 `complete`、3 条 `needs_review`，0 条执行错误；全部人工状态仍为 `unreviewed`。

**模型质量尚有问题：000543 实际为 399 / 485 / 562，参考为 400 / 485 / 562，精确回归未通过。** 另有准备动作误选、短包误判为长条、搬运节点误选和缺阶段问题，不能把程序 complete 计作正确率。逐条 agent 视觉复核已保存，不充当人工真值。

- [P1 十条实验报告](runs/p1_reverse_node_prompt_v3_20261023/reports/phase1_report.md)
- [结构化汇总](runs/p1_reverse_node_prompt_v3_20261023/reports/summary.json)
- [视频逐帧验证](runs/p1_reverse_node_prompt_v3_20261023/reports/video_verification.json)
- [真实控制流与源保护检查](runs/p1_reverse_node_prompt_v3_20261023/reports/execution_audit.json)
- [逐条 agent 视觉复核](runs/p1_reverse_node_prompt_v3_20261023/reports/visual_review_summary.json)

首次连通性运行 `runs/p1_reverse_node_v1_20261023/` 因缺少 SOCKS 依赖在 HTTP 请求前失败，实际 HTTP 次数为 0；记录保留。各次修复后使用新运行，没有用重复语义采样替换 skip/uncertain 结果。

随后 `runs/p1_reverse_node_v1_final_20261023/` 的000543首次目标请求返回HTTP400。只读模型列表检查确认模型存在，独立无目标图像的文本诊断定位到 JSON mode 要求消息正文包含 `json`。v2 补明 JSON 返回格式后新建运行；失败请求没有形成可用语义判断，原记录保留。

`runs/p1_reverse_node_prompt_v2_20261023/` 首个响应HTTP200，但 `target_stage` 返回字符串 `stage3`，被严格类型校验拒绝；1次请求、755 tokens，原始响应保留。v3 将整数类型及四字段约束直接写入实际prompt，不转换旧回答、不改变阶段形态。

## 阅读顺序

1. [boundary.md](boundary.md)：先读。阶段语义、任意一爪规则、搜索与输出边界。
2. [DEVELOPMENT.md](DEVELOPMENT.md)：开发步骤、状态机、产物与计划验证入口。
3. [architecture.md](architecture.md)：外置 prompt 配置、模块职责、数据流与冻结恢复。
4. [docs/DATA.md](docs/DATA.md)：真实输入位置、state 格式、时间轴与候选字段。
5. [docs/PROMPT.md](docs/PROMPT.md)：带 expert example 的单候选输入和 VLM 输出契约；固定文本见 [configs/prompt.txt](configs/prompt.txt)。
6. [docs/EVALUATION.md](docs/EVALUATION.md)：000543 回归、第一阶段 10 条实验、离线测试和交付验收。
7. [docs/LEGACY_REFERENCES.md](docs/LEGACY_REFERENCES.md)：旧代码定位、可复用部分和应删除的旧约束。
8. [docs/VISUALIZATION.md](docs/VISUALIZATION.md)：仿照现有导出器的画面布局、阶段时间线、编码和验证方法。

默认可视化为每条轨迹的完整 MP4：左侧原视频，右侧阶段与边界信息，底部彩色时间线。10 条都要导出，节点图作为辅助；识别不确定也保留待复核视频，不只挑成功样本。

[AGENTS.md](AGENTS.md) 是后续开发 agent 的简短入口。[第一阶段 10 条清单](docs/phase1_manifest.jsonl) 包含已有四条诊断及另外六条 development 轨迹；[选择记录](docs/phase1_selection.json) 保存固定规则、摘要和输入核验结果。[原四条清单](docs/diagnostic_manifest.jsonl) 保留为回归子集。所有清单均不含边界答案或模型预测，Dataset 仅限叠衣服。

用户已确认旧试跑中只有 `fold_cloth_v3__episode_000543` 的划分正确，对应旧产物边界为 **400 / 485 / 562**。这是新实现必须首先检查的回归样例；保留的另外三条叠衣轨迹用于定位错误，尚无完整正确边界。旧试跑中的挂衣轨迹已从新库清单移除。

CLI、模块及验证入口已实现。开发时按业务契约修改相应模块，并运行受影响的离线测试；历史资料只读，不作为运行时依赖。
