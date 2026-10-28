# 开发实现与验证

目标：实现 [boundary.md](boundary.md) 的 `reverse_node_v2`，使用单个张开候选、原有前后局部上下文和当前阶段的 expert example，选择“本次折叠刚完成、下一动作尚未开始”的节点。日期：2026-10-26。状态：v2 已接入独立包与 CLI，[外置 prompt](configs/prompt.txt) 和示例通过 [configs/reverse_node_v2.yaml](configs/reverse_node_v2.yaml) 加载，完整离线检查 197 passed。

`configs/default.yaml`、旧包内 prompt 和历史十条实验保留 `reverse_node_v1`、`node_shape_prompt_v3`。本轮 v2 在新运行 `runs/p1_reverse_node_v2_examples_20261026/` 使用同一十条清单完成离线 prepare，并按用户本轮授权从 000543 开始真实调用。实际推理、视频与视觉复核记录以 [README.md](README.md) 和运行报告为准；实现、离线检查、模型质量与人工确认分别报告。

## 1. 本轮目标与十条复核范围

2026-10-23 已交付第一阶段（P1）的十条完整 MP4。新一轮沿用同一清单，在新运行中比较三类问题：001776、003546、001061 的 stage1 时刻偏晚；002962、001372、004150 的缺阶段；006453、002768 将初始整理误选为 stage1。用户反馈与旧程序记录分开列在 [EVALUATION.md](docs/EVALUATION.md)，尚无这些轨迹的替代正确帧。

开发第一阶段（P1）固定处理 [10 条叠衣轨迹](docs/phase1_manifest.jsonl)，交付 10 个不同 episode 的完整 MP4。每条同时输出传统方法生成的候选、每次 VLM 请求与判断、三个边界、候选阶段数组和可视复核材料。10 条使用同一冻结配置；000543 的已确认划分用于回归。输入清单限制在已登记的叠衣来源，不实现其他任务分类或负例识别。

已有四条诊断构成回归子集；另外六条从旧 development 清单按固定规则补齐，选择记录见 [phase1_selection.json](docs/phase1_selection.json)。P1 交付以 10 个有效完整视频、10 份导出记录、统一实验汇总和视频验证记录为准，包含识别不确定的结果。执行失败造成视频缺失时应修复后补交，不能把少于 10 条写成完成。

继续通过独立 CLI 和文件产物交付。可视化按 [docs/VISUALIZATION.md](docs/VISUALIZATION.md) 导出：每条轨迹一个完整复核视频，源画面外侧展示阶段、边界和时间线。沿用现有运行环境及依赖，实际版本写入新运行。

第一轮对照固定现有检测器、局部上下文取样、模型和推理参数，集中改进 expert example 和完成时刻判据，并接入外置 prompt 配置。先核查正确时刻是否有候选；确认漏检时另开检测器变更实验，不按已知错误编号调阈值或移动节点。

## 2. 本轮改动落点

模块职责与数据流统一写在 [architecture.md](architecture.md)。已实现：`config.py` / `cli.py` 加载并冻结外置 prompt；新增 `expert_examples.py` 校验、按原字节冻结和核验三阶段示例；复用 `evidence.py` 的候选原帧与固定时间上下文；`prompt.py` / `provider.py` 在单候选请求中加入目标阶段示例及独立标签；`search.py` / `results.py` 核验冻结示例与缓存恢复，并输出逐阶段 `search_status`。`search.py` 保留可替换 provider，普通测试使用模拟调用。

`cli.py` 在 prepare 时读取本库配置及示例，冻结到本次运行；运行时只依赖新库与该运行的冻结输入。示例是三个小型固定输入资产，保留用户原文件；所有生成媒体和请求记录仍落在被忽略的 `runs/`。

## 3. 顺序状态机

下面的伪代码概括已实现的 v2 状态机，不对应逐字函数接口。单候选及局部上下文的输入规则以 [boundary.md](boundary.md) 第4节为准。

```python
slots = {1: None, 2: None, 3: None}
target_stage = 3
has_uncertainty = False

for node in candidates_sorted_by_frame_desc:
    if node.frame == frame_count - 1:
        record_skip(node, reason="no_stage4_frames")
        continue
    context = local_context(node)
    example = frozen_expert_example(target_stage)
    response = judge_one_node(target_stage, node, context, example)
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

每个 episode 内是依赖式串行调用。2026-10-28 已接入不同 episode 的有界线程池，`max_inflight_episodes` 接受 1–16；首轮旧配置保留 1，新入口 `configs/reverse_node_v2_parallel16.yaml` 使用 16。每条在 worker 内核验准备摘要、独立建立 provider 并执行整个搜索；主线程在全部任务结束后汇总，保留清单顺序、失败原因和执行时间。无需保留不断增长的对话历史；每次用独立短请求，目标阶段由程序填写。

视频 render 按轨迹使用 spawn 进程池，`video_export.max_workers` 接受 1–16，也可由导出命令的 `--workers` 指定。每条视频内部仍按源 PTS 正序逐帧处理并重新解码验证；普通版和全部候选标记版共用此调度。导出索引只由主进程写入。推理恢复仅重放指纹相同的已保存回答；错误轨迹不会取消其他轨迹。

单轨迹逻辑请求数不超过可选候选数 M。命中 stage1 后停止。HTTP 重试另计；第一次不符合或不确定都不重复抽样。

上下文帧只帮助解释当前节点；即使碰巧对应另一个候选，也不能在本次被选中或判定。只接受当前 `node_id` 的四字段响应，证据短句需说明衣物与夹爪的可见状态，程序不通过关键词重写模型的 `decision`。候选不足以形成完整三阶段时仍正常遍历现有候选，保留部分结果。

## 4. 配置与 prompt 版本

`configs/default.yaml` 保留 v1：`fixed_local_v1`、`[-1.0, -0.5, 0.0, 0.5, 1.0]` 秒抽帧和 `node_shape_prompt_v3`。[configs/reverse_node_v2.yaml](configs/reverse_node_v2.yaml) 已接入独立存放的 [configs/prompt.txt](configs/prompt.txt)，版本标识为 `node_completion_prompt_v1`。CLI 显式选择 v2 配置即可使用下列字段。

v2 配置关键字段如下；完整 YAML 同时保留本轮冻结的检测与 provider 参数：

```yaml
spec_version: reverse_node_v2
prompt_file: prompt.txt
prompt_version: node_completion_prompt_v1
candidate_policy: any_open
search_order: [3, 2, 1]
context_policy: fixed_local_v1
context_offsets_s: [-1.0, -0.5, 0.0, 0.5, 1.0]
expert_examples:
  stage1: images/s1.png
  stage2: images/s2.png
  stage3: images/s3.png
max_inflight_episodes: 1
provider:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model_id: qwen3.8-max-0902
  api_key_env: DASHSCOPE_API_KEY
  temperature: 0
  max_output_tokens: 512
  timeout_s: 180
  max_attempts: 3
  max_images: 6
```

示例路径相对配置文件解析，prepare 校验三个阶段齐全且图像可解码；缺图、映射错误或图片预算不足应在发送请求前报告配置错误。每次只附当前阶段的一张示例；候选与前后上下文截断去重后至多五张，加示例至多六张。provider 的示例标签不伪造本轨迹时间戳，本轨迹图片保留 `BEFORE` / `CANDIDATE` / `AFTER` 角色与相对时间。

模型标识与地址沿用旧冻结配置，实际返回模型名另行记录。完整固定 prompt 见 [configs/prompt.txt](configs/prompt.txt)，四字段接口见 [docs/PROMPT.md](docs/PROMPT.md)。切换协议、prompt、图像策略或推理参数均使用新运行，v2 不复用 v1 缓存。

## 5. 运行记录、缓存与恢复

```text
runs/<run_id>/
  run.json                     配置、协议、代码版本/源码摘要、创建时间
  manifest.jsonl               本次冻结输入及源摘要
  prompt.txt                   外置固定文本原文，保留原换行
  expert_examples/
    index.json                 阶段映射、源图和实际发送图的摘要与处理参数
    s1.png / s2.png / s3.png    按原字节冻结的示例，当前发送不转换
  episodes/<key>/
    input.json                 视频/state 元数据与时间轴检查
    video_axis.json            全部源原帧对应的相对 PTS
    candidates.json            候选和检测依据
    evidence/index.json        当前节点、局部上下文原帧、角色、时间与图像摘要
    calls/<node_id>/
      request.json             目标阶段、文本、图像路径/摘要，不存鉴权头
      response.json            原始模型回答与返回模型名
      attempts.json            HTTP 次数、耗时、状态与用量
      decision.json            通过结构校验的判断
    trace.jsonl                实际遍历和目标切换顺序
    result.json                固定槽位、状态、search_status、原因和调用统计
    stage_candidates.npy       可视复核用候选标签
    review.json                人工复核，存在时独立保存
    stage_reviewed.npy         人工确认后生成
  videos/
    <episode_key>__candidates.mp4 完整轨迹的默认复核视频
    <episode_key>__candidates.json 导出来源、边界与编码记录
    exports.json               每条视频位置与导出状态
  reports/summary.json
  reports/request_images.json  全部请求的示例、候选和上下文图像索引
  reports/phase1_report.md      覆盖全部 10 条的第一阶段实验汇总
  reports/video_verification.json 重新解码后的帧数与 PTS 检查
```

请求指纹至少覆盖：视频与 state 摘要、检测配置、当前节点 ID、候选及上下文的原帧/时间/图像摘要/角色/顺序、目标阶段、示例阶段映射及图像摘要、上下文取样配置、完整 prompt、模型、推理参数和协议版本。缓存命中必须比较指纹，不能只看 episode/node 名称。

先原子保存请求、响应和判断，再推进状态。恢复时按已保存的合法判断重放状态机，核对目标阶段，避免掉电后重复付费请求或跳过候选。配置变化拒绝沿用旧运行，要求新 run ID。

prepare 后的请求从该运行内的冻结示例构造，不临时读取可变的 `configs/images/`。冻结副本被改动时拒绝恢复；采用更新后的源示例应重新 prepare 到新运行。复核记录需能直接定位每次实际发送的候选、局部上下文和一张示例。

网络错误、限流、服务故障、截断与非法 JSON 可按固定次数重试并记录全部尝试；耗尽后报执行错误。合法的 `skip` / `uncertain` 不触发重试。结构错误不得改写为 `skip`；首版可直接结束并等待修复，不要求自动纠错工作流。

## 6. 分步实施与完成证据

A 至 D 的实现与离线行为验证已完成：`.venv/bin/python -m pytest -q` 为 **197 passed**；新运行离线准备核验 10 条、6568 帧，生成 148 个候选，000543 候选仍含 400/485/562。八条问题轨迹的旧结果及候选诊断单独记录，已确认正确帧仍仅来自 000543。E、F 的真实模型效果、视频交付与人工复核，以运行报告的实际证据为准。

| 步骤 | 实现内容 | 完成证据 |
| --- | --- | --- |
| A：离线核查与配置 | 核对八条问题轨迹的候选与旧调用；实现 `prompt_file` 加载、新配置和示例映射 | 区分候选覆盖不足、模型判错及未进入阶段搜索；从 `configs/prompt.txt` 加载并冻结原文及摘要；000543 候选仍包含 400/485/562。 |
| B：证据与请求 | 复用单候选局部上下文，接入目标阶段示例、图片标签与新 prompt | 每次当前节点唯一，上下文不能被选中；默认至多六图，示例与本轨迹时间轴分离。 |
| C：状态与持久化 | 接入冻结示例、指纹和断点恢复 | 倒序目标切换正确；变化后的取样配置/示例不误用缓存；缺槽报告区分未找到与未搜索，保留部分结果。 |
| D：离线验证 | 按 EVALUATION 的新增场景验证，并运行完整离线测试 | `.venv/bin/python -m pytest -q`：197 passed；模拟调用不联网，不据此宣称视觉问题修复。 |
| E：真实十条对照 | 按本轮用户授权，用同一新配置先检查 000543，再完成其余九条 | 保存全部原始回答、候选与上下文、示例和用量；逐条对照三类问题，精确参考帧只用于评估。 |
| F：复核交付 | 为同一运行的十条分别导出完整 MP4 和请求图索引 | 十个视频及验证记录；逐条写明模型结论、人工确认状态、未解决问题和结果路径。 |

P1 先验证 000543，再在同一冻结配置下完成另外三条既有诊断和六条补充样本。若修改 prompt 或检测配置，新建运行；用于最终 10 条交付的运行必须使用同一版本，不能把不同配置的最好结果拼成一次成功实验。结果是 `needs_review` 也要交付带说明的视频，不能挑选成功案例替换原清单。

## 7. 已实现的 CLI

v2 沿用独立 CLI 命令并扩展配置、冻结与请求图像索引。用法、环境、模拟模式和命名导出版本见 README；v2 准备时须显式选择新配置：

```text
python -m split_cloth_v2.cli prepare --manifest docs/phase1_manifest.jsonl --config configs/reverse_node_v2.yaml --run runs/<id>
python -m split_cloth_v2.cli annotate --run runs/<id> --credential-file .secrets/dashscope_api_key.txt
python -m split_cloth_v2.cli report --run runs/<id>
python -m split_cloth_v2.cli export-video --run runs/<id>
```

`prepare` 离线完成对齐、检测、上下文和请求预算估计；`annotate` 才访问真实 provider。`annotate --episode <key>` 支持先运行单条回归，随后全量 annotate 重放已有判断并继续。默认测试入口为 `.venv/bin/python -m pytest -q`，普通测试不访问网络。已有授权的真实实验按其范围完成；不要把在线调用混入安装、测试或文档检查。

密钥也可从 `DASHSCOPE_API_KEY` 读取。用户本轮明确要求的本库凭据文件放在 `.secrets/`，该目录已被 `.gitignore` 排除，目录权限 700、文件权限 600；凭据内容不进入配置、运行产物或日志。

## 8. 需如实保留的限制

固定局部上下文不保证覆盖一次动作的全过程，示例与目标视角、衣物外观也可能不同。不能确认“刚完成且未开始下一动作”时保留 `uncertain`，不隐式扩图、移动候选或重复抽样。倒序首次匹配仍可能判错，需靠实际图像和完整视频复核。

十条均为已见开发样本，不构成独立 heldout。已确认的完整参考仍只有 000543 的旧展示划分；本轮用户对八条提供了错误类型，未提供替代帧。v1 的十条推理与视频导出已完成，v2 的实现和离线行为验证已完成；真实视觉效果和十条视频交付另看新运行的实际报告，不从测试或程序 `complete` 推断新方案准确率。
