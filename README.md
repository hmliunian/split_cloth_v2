# split_cloth_v2

新库用于实现：**夹爪 state 生成张开候选 → 程序倒序逐点调用 VLM → 找到小方块、双侧长条、单侧叠入三个结束点 → 导出四阶段划分。**

已实现独立 Python 包、离线准备、逐候选倒序搜索、请求持久化与恢复、完整 MP4 导出及逐帧时间验证。日期：2026-10-23。程序验证与模型视觉质量分别记录。

**开发第一阶段（P1）的首次交付为 10 条不同叠衣轨迹的完整 MP4**，附边界结果、运行用量、导出验证和实验汇总。P1 是开发里程碑，与视频内的 stage1 不同。

## 安装与离线检查

本轮实际使用 Python 3.12.3。依赖固定在 `pyproject.toml`，包括集群 SOCKS 代理需要的 `httpx[socks]`；运行记录另存实际环境版本。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

本轮完整离线检查：**144 passed**。测试包括合成媒体、实际编码后重新解码、模拟 HTTP 与崩溃恢复，不访问真实 VLM。10 条真实输入的离线准备共核验 6568 帧，生成 148 个候选；000543 候选精确复现旧冻结结果，562 仅右爪张开。

## 运行

```bash
.venv/bin/python -m split_cloth_v2.cli prepare --manifest docs/phase1_manifest.jsonl --config configs/default.yaml --run runs/<new_run_id>
.venv/bin/python -m split_cloth_v2.cli annotate --run runs/<new_run_id> --episode fold_cloth_v3__episode_000543
.venv/bin/python -m split_cloth_v2.cli annotate --run runs/<new_run_id>
.venv/bin/python -m split_cloth_v2.cli export-video --run runs/<new_run_id>
.venv/bin/python -m split_cloth_v2.cli report --run runs/<new_run_id>
```

`prepare` 完全离线。`annotate` 发送候选及上下文图像并计费，读取 `DASHSCOPE_API_KEY`，或使用 `--credential-file /absolute/external/file` 显式指定仓库外凭据。第二次 annotate 会核对指纹并重放已有判断；合法 skip/uncertain 不重问。模型、prompt、采样、检测、源码或环境变化需要新运行。

需要完整离线模拟时，在 prepare 加 `--mock`；可加 `--mock-decisions <json>` 提供节点判断，否则返回模拟 uncertain。模拟模式明确记录，不能标记真实 P1 完成。

默认输出在 `runs/<id>/videos/`。旧导出存在或导出失败时，使用 `export-video --run ... --export-name retry_01` 生成独立版本，配套 `report --run ... --export-name retry_01`；历史视频、索引和 `.part.mp4` 不覆盖。任意轨迹导出失败，命令返回非零。

## 当前实验记录

正式冻结运行：`runs/p1_reverse_node_prompt_v3_20261023/`，prompt 版本 `node_shape_prompt_v3`。实际请求、响应、trace、候选标签、视频及汇总均在该目录；真实模型结论以运行完成后的 `reports/phase1_report.md` 为准。

首次连通性运行 `runs/p1_reverse_node_v1_20261023/` 因缺少 SOCKS 依赖在 HTTP 请求前失败，实际 HTTP 次数为 0；记录保留。修复后新建正式运行，没有用重复语义采样替换结果。

随后 `runs/p1_reverse_node_v1_final_20261023/` 的000543首次目标请求返回HTTP400。只读模型列表检查确认模型存在，独立无目标图像的文本诊断定位到 JSON mode 要求消息正文包含 `json`。v2 补明 JSON 返回格式后新建运行；失败请求没有形成可用语义判断，原记录保留。

`runs/p1_reverse_node_prompt_v2_20261023/` 首个响应HTTP200，但 `target_stage` 返回字符串 `stage3`，被严格类型校验拒绝；1次请求、755 tokens，原始响应保留。v3 将整数类型及四字段约束直接写入实际prompt，不转换旧回答、不改变阶段形态。

## 阅读顺序

1. [boundary.md](boundary.md)：先读。阶段语义、任意一爪规则、搜索与输出边界。
2. [DEVELOPMENT.md](DEVELOPMENT.md)：模块、开发步骤、状态机、产物与计划验证入口。
3. [docs/DATA.md](docs/DATA.md)：真实输入位置、state 格式、时间轴与候选字段。
4. [docs/PROMPT.md](docs/PROMPT.md)：短 prompt、单节点输入和 VLM 输出契约。
5. [docs/EVALUATION.md](docs/EVALUATION.md)：000543 回归、第一阶段 10 条实验、离线测试和交付验收。
6. [docs/LEGACY_REFERENCES.md](docs/LEGACY_REFERENCES.md)：旧代码定位、可复用部分和应删除的旧约束。
7. [docs/VISUALIZATION.md](docs/VISUALIZATION.md)：仿照现有 MP4 导出，规定画面布局、阶段时间线、编码和验证方法。

默认可视化为每条轨迹的完整 MP4：左侧原视频，右侧阶段与边界信息，底部彩色时间线。10 条都要导出，节点图作为辅助；识别不确定也保留待复核视频，不只挑成功样本。

[AGENTS.md](AGENTS.md) 是后续开发 agent 的简短入口。[第一阶段 10 条清单](docs/phase1_manifest.jsonl) 包含已有四条诊断及另外六条 development 轨迹；[选择记录](docs/phase1_selection.json) 保存固定规则、摘要和输入核验结果。[原四条清单](docs/diagnostic_manifest.jsonl) 保留为回归子集。所有清单均不含边界答案或模型预测，Dataset 仅限叠衣服。

用户已确认旧试跑中只有 `fold_cloth_v3__episode_000543` 的划分正确，对应旧产物边界为 **400 / 485 / 562**。这是新实现必须首先检查的回归样例；保留的另外三条叠衣轨迹用于定位错误，尚无完整正确边界。旧试跑中的挂衣轨迹已从新库清单移除。

CLI、模块及验证入口已实现。开发时按业务契约修改相应模块，并运行受影响的离线测试；历史资料只读，不作为运行时依赖。
