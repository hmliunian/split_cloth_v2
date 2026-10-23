# 开发计划

状态：核心模块、CLI、离线验证与视频导出已实现；协议：`reverse_node_v1`。实际检查和实验记录见 [README.md](README.md)。下文保留职责、算法与交付契约，改动先遵守 [boundary.md](boundary.md)。

## 1. 第一阶段交付：10 条 MP4

开发第一阶段（P1）固定处理 [10 条叠衣轨迹](docs/phase1_manifest.jsonl)，交付 10 个不同 episode 的完整 MP4。每条同时输出传统方法生成的候选、每次 VLM 请求与判断、三个边界、候选阶段数组和可视复核材料。10 条使用同一冻结配置；000543 的已确认划分用于回归。输入清单限制在已登记的叠衣来源，不实现其他任务分类或负例识别。

已有四条诊断构成回归子集；另外六条从旧 development 清单按固定规则补齐，选择记录见 [phase1_selection.json](docs/phase1_selection.json)。P1 交付以 10 个有效完整视频、10 份导出记录、统一实验汇总和视频验证记录为准，包含识别不确定的结果。执行失败造成视频缺失时应修复后补交，不能把少于 10 条写成完成。

第一版先做 CLI 和文件产物。可视化按 [docs/VISUALIZATION.md](docs/VISUALIZATION.md) 仿照现有 MP4 导出：每条轨迹一个完整复核视频，源画面外侧展示阶段、边界和时间线，节点图作为辅助。沿用 Python 3.11+；基础依赖按实际使用选 `numpy`、`pyarrow`、`av`、`Pillow`、`httpx`、`PyYAML`，测试用 `pytest`。依赖版本应在实际安装验证后记录。

## 2. 建议目录与职责

```text
src/split_cloth_v2/
  cli.py          命令入口、配置、运行目录
  data.py         manifest、Parquet、视频 PTS 和对齐检查
  detector.py     纯 state 张开事件检测；不调用 VLM
  evidence.py     候选原帧和局部上下文抽取、图像映射
  prompt.py       从固定模板组装单节点请求
  provider.py     API、模拟 provider、重试、用量记录
  search.py       倒序游标、stage3→2→1、断点恢复
  results.py      JSON 校验、节点映射与阶段数组
  video_export.py 原视频逐帧合成、阶段面板、时间线、MP4 与验证记录
configs/          显式配置和固定 prompt 资产
tests/            离线契约测试
runs/             运行产物，加入 .gitignore
```

模块可按规模合并，职责边界保持清楚。`search.py` 通过可替换的 provider 接口取得判断，测试不需要真实密钥。`results.py` 只检查协议和映射，不引入第二套阶段定义。

建立项目时补 `pyproject.toml` 和 `.gitignore`；忽略 `.venv/`、`runs/`、缓存、`.env`、密钥文件及大型媒体。不把旧虚拟环境、完整旧仓库或历史 runs 复制进新库。

## 3. 顺序状态机

```python
slots = {1: None, 2: None, 3: None}
target_stage = 3
has_uncertainty = False

for node in candidates_sorted_by_frame_desc:
    if node.frame == frame_count - 1:
        record_skip(node, reason="no_stage4_frames")
        continue
    response = judge_one_node(target_stage, node, context(node))
    # 请求失败或结构非法：保存现场并报 execution_error，不能当 skip。
    validate_response(response, expected_node=node, expected_stage=target_stage)
    persist_call_and_decision(node, target_stage, response)

    if response.decision == "uncertain":
        has_uncertainty = True
        continue
    if response.decision == "skip":
        continue

    slots[target_stage] = node
    target_stage -= 1
    if target_stage == 0:
        break

# 耗尽时尚未选中的槽位保持 None；没有补位、同候选改问或第二轮搜索。
finalize(slots, has_uncertainty)
```

每个 episode 内是依赖式串行调用。不同 episode 后续可有限并行，但首轮先设 `max_inflight_episodes=1`，让调用顺序容易核查。无需保留不断增长的对话历史；每次用独立短请求，目标阶段由程序填写。

单轨迹逻辑请求数不超过可选候选数 M。命中 stage1 后停止。HTTP 重试另计；第一次不符合或不确定都不重复抽样。

## 4. 起始配置

配置已实现于 `configs/default.yaml`，数值是可复现起点，不是已验证的新最优参数。检测参数来源见 [docs/DATA.md](docs/DATA.md)。

```yaml
spec_version: reverse_node_v1
candidate_policy: any_open
search_order: [3, 2, 1]
context_offsets_s: [-1.0, -0.5, 0.0, 0.5, 1.0]
context_policy: fixed_local_v1
max_inflight_episodes: 1
provider:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model_id: qwen3.8-max-0902
  api_key_env: DASHSCOPE_API_KEY
  temperature: 0
  max_output_tokens: 512
  timeout_s: 180
  max_attempts: 3
```

模型标识与地址来自旧运行记录，服务模型列表已确认该标识存在；实际推理返回的模型名写入每次调用记录。上下文取样与输出上限是本版工程默认，模型效果以正式运行记录为准；调整后使用新运行 ID。

## 5. 运行记录、缓存与恢复

```text
runs/<run_id>/
  run.json                     配置、协议、代码版本/源码摘要、创建时间
  manifest.jsonl               本次冻结输入及源摘要
  prompt.txt                   本次最终展开的固定模板
  episodes/<key>/
    input.json                 视频/state 元数据与时间轴检查
    video_axis.json            全部源原帧对应的相对 PTS
    candidates.json            候选和检测依据
    evidence/index.json        节点、原帧、上下文图片及摘要
    calls/<node_id>/
      request.json             目标阶段、文本、图像路径/摘要，不存鉴权头
      response.json            原始模型回答与返回模型名
      attempts.json            HTTP 次数、耗时、状态与用量
      decision.json            通过结构校验的判断
    trace.jsonl                实际遍历和目标切换顺序
    result.json                固定槽位、状态、原因和调用统计
    stage_candidates.npy       可视复核用候选标签
    review.json                人工复核，存在时独立保存
    stage_reviewed.npy         人工确认后生成
  videos/
    <episode_key>__candidates.mp4 完整轨迹的默认复核视频
    <episode_key>__candidates.json 导出来源、边界与编码记录
    exports.json               每条视频位置与导出状态
  reports/summary.json
  reports/phase1_report.md      覆盖全部 10 条的第一阶段实验汇总
  reports/video_verification.json 重新解码后的帧数与 PTS 检查
```

请求指纹至少覆盖：视频与 state 摘要、候选帧及检测配置、上下文帧/图像摘要、目标阶段、完整 prompt、模型、推理参数和协议版本。缓存命中必须比较指纹，不能只看 episode/node 名称。

先原子保存请求、响应和判断，再推进状态。恢复时按已保存的合法判断重放状态机，核对目标阶段，避免掉电后重复付费请求或跳过候选。配置变化拒绝沿用旧运行，要求新 run ID。

网络错误、限流、服务故障、截断与非法 JSON 可按固定次数重试并记录全部尝试；耗尽后报执行错误。合法的 `skip` / `uncertain` 不触发重试。结构错误不得改写为 `skip`；首版可直接结束并等待修复，不要求自动纠错工作流。

## 6. 分步实施与完成证据

| 步骤 | 实现内容 | 完成证据 |
| --- | --- | --- |
| A：输入与候选 | 包结构、数据对齐、传统检测、候选可视化；复用必要旧纯函数 | 离线测试；000543 候选包含 400/485/562，并核查 562 是右爪事件。 |
| B：单点与状态机 | 短 prompt、固定局部上下文、模拟 provider、顺序选择 | 模拟判断的调用顺序、阶段切换、耗尽与未知测试通过。 |
| C：结果与恢复 | 原帧映射、标签、持久化、缓存失效和恢复 | 边界帧归属、非法返回、不同配置不复用缓存等测试通过。 |
| D：P1 真实实验 | 同一配置运行清单内 10 条叠衣，保留所有请求与候选图 | 核查 000543 回归；逐条报告全部 10 条的状态、视觉结论、调用量与失败原因。 |
| E：P1 MP4 交付 | 仿照现有导出器，为 10 条轨迹逐帧导出完整 MP4，配阶段面板和时间线 | 10 个视频及同名记录；重新解码帧数/PTS 检查；边界切换与可读性抽查通过。 |

P1 先验证 000543，再在同一冻结配置下完成另外三条既有诊断和六条补充样本。若修改 prompt 或检测配置，新建运行；用于最终 10 条交付的运行必须使用同一版本，不能把不同配置的最好结果拼成一次成功实验。结果是 `needs_review` 也要交付带说明的视频，不能挑选成功案例替换原清单。

## 7. 已实现的 CLI

以下命令已实现；用法、环境、模拟模式和命名导出版本见 README：

```text
python -m split_cloth_v2.cli prepare --manifest docs/phase1_manifest.jsonl --config configs/default.yaml --run runs/<id>
python -m split_cloth_v2.cli annotate --run runs/<id>
python -m split_cloth_v2.cli report --run runs/<id>
python -m split_cloth_v2.cli export-video --run runs/<id>
```

`prepare` 离线完成对齐、检测、上下文和请求预算估计；`annotate` 才访问真实 provider。`annotate --episode <key>` 支持先运行单条回归，随后全量 annotate 重放已有判断并继续。默认测试入口为 `.venv/bin/python -m pytest -q`，普通测试不访问网络。已有授权的真实实验按其范围完成；不要把在线调用混入安装、测试或文档检查。

## 8. 需如实保留的限制

局部图像可能不足以区分铺衣与第一折，或第二折长条与被提起的衣物；此时允许 `uncertain`。从后向前首次匹配仍可能误选晚期修正/搬运，必须靠诊断验证，不能由状态机正确性推出视觉正确性。

P1 的 10 条属于开发实验样本，不能充当独立 heldout。当前唯一用户确认的完整划分是 000543；另三条旧结果已知有错，新增六条尚未做本算法推理或人工边界验收。先实现并观察错误，再决定是否需要扩大上下文；通用任务分类不属于本库范围。
