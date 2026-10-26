# 数据来源与输入契约

日期：2026-10-26。业务范围以 [boundary.md](../boundary.md) 的 dataset 边界为准：**只做叠衣服**。

## 1. 第一阶段的 10 条叠衣数据

数据根目录：`/gpfs/jiuquyun/datasets/shenzhen/lerobot_eef/`。第一阶段（P1）使用 `fold_cloth_v3` 和 `fold_cloth_v4` 各五条完整轨迹，均为主视角 `base_0_rgb`、20 FPS。

| dataset_id | episode | chunk | 原始帧数 | state 行数 |
| --- | --- | --- | ---: | ---: |
| fold_cloth_v3 | episode_000543 | chunk-000 | 681 | 681 |
| fold_cloth_v3 | episode_006453 | chunk-006 | 573 | 573 |
| fold_cloth_v4 | episode_002768 | chunk-002 | 595 | 595 |
| fold_cloth_v4 | episode_004150 | chunk-004 | 483 | 483 |
| fold_cloth_v3 | episode_001776 | chunk-001 | 786 | 786 |
| fold_cloth_v3 | episode_002962 | chunk-002 | 544 | 544 |
| fold_cloth_v3 | episode_003546 | chunk-003 | 620 | 620 |
| fold_cloth_v4 | episode_001061 | chunk-001 | 540 | 540 |
| fold_cloth_v4 | episode_001372 | chunk-001 | 781 | 781 |
| fold_cloth_v4 | episode_001395 | chunk-001 | 965 | 965 |

前四条为既有诊断子集，后六条为补充 development 样本；总计 10 条、6568 帧。

路径模式：

```text
<root>/<dataset_id>/videos/<chunk>/base_0_rgb/<episode>.mp4
<root>/<dataset_id>/data/<chunk>/<episode>.parquet
```

使用 [phase1_manifest.jsonl](phase1_manifest.jsonl) 作为 P1 的显式输入。[diagnostic_manifest.jsonl](diagnostic_manifest.jsonl) 保留原四条回归子集，不能把它当作 10 条交付的完整输入。

补充选择规则：在旧冻结 development 清单中分别取 `fold_cloth_v3`、`fold_cloth_v4`，排除已选 key，按 `episode_index` 升序各取前三条。选择时未读取模型输出或按效果筛样；已核对 key 与视频 SHA-256 均不重叠旧 reference/heldout。完整来源摘要、规则与核验结果见 [phase1_selection.json](phase1_selection.json)。

六条补充样本的任务元数据均为 `Fold the shirts.`，来源符合叠衣范围；本轮继续使用原清单，不因模型划分好坏替换样本。用户对八条的已有反馈与程序结果分别保存在 EVALUATION；这不构成其替代边界的人工验收。新增准备如发现源任务不符或文件不可用，记录原因、按同一规则取下一条并重新冻结清单。

挂衣、叠裤子和混合任务不进入本库清单，不作为负例或效果评估分母。不遍历整个数据根目录自动纳入所有任务。新增来源时先登记确认的叠衣数据目录和对应 state 格式；不靠 VLM 自动扩大任务范围。

## 2. state 与视频时间轴

已读取 10 个真实 Parquet 的完整 state 与时间轴，确认 state 列名均为 `state`、14 维，`frame_index=0`、`timestamp=0.0` 起始；行数与冻结视频帧数一致。10 个视频及 10 个 Parquet 均重新计算 SHA-256 并与冻结清单一致。v2 运行 `runs/p1_reverse_node_v2_examples_20261026/` 的离线 prepare 已完整解码全部 6568 帧并核对视频/state 的逐帧 PTS 对齐，生成 148 个候选。

| 字段 | 含义 |
| --- | --- |
| `state` / `observation.state` | 实测反馈；首批显式指定 `state`。 |
| `state[6]` | 左夹爪反馈，0 基索引。 |
| `state[13]` | 右夹爪反馈，0 基索引。 |
| `frame_index` | 对应源视频原始帧号。 |
| `timestamp` | 秒，与解码视频 PTS 对齐。 |

左右索引来自旧库的 14D EEF 适配，不可推广到未核验的 16D joint 或其他 state 布局。开合方向首轮沿用左右均 `higher`（值增大表示张开），由节点图和原始反馈检查支撑，不把数值单位猜成米或归一化夹爪开度。

读取器要求字段明确、数值有限、帧号连续、时间严格递增，并对照每一帧解码 PTS。若 `state` 与 `observation.state` 同时存在而配置未指明，应报歧义；`actions` 不能替代反馈。不得直接把 Parquet 行号当作原始帧号。

## 3. 起始候选检测配置

为首先隔离 VLM 输入与搜索流程的变化，首轮沿用旧 `any_open` 稳定平台检测。以下是旧运行的实际参数，不是新数据的通用物理标定：

```yaml
node_policy: any_open
opening_directions: {left: higher, right: higher}
candidate_levels: [[0.25, 0.10], [0.50, 0.25], [0.75, 0.45], [0.90, 0.65], [0.98, 0.90]]
grouping_window_s: 0.25
settled_samples: 3
settled_duration_s: 0.10
settled_tolerance_fraction: 0.02
settled_reversal_fraction: 0.10
settled_min_excursion_fraction: 0.10
settled_search_window_s: 2.0
```

`candidate_levels` 是旧实现按单轨迹反馈范围计算的阈值对；应复用并验证原算法含义，不把这些比例当原始 state 的固定阈值。`grouping_window_s` 用于同臂阈值交叉的归组，左右不同原始帧事件仍分别保留。

记录每个候选的 `node_id`、`source_frame_id`、`timestamp_s`、`opening_arms`、各臂检测依据、平台首帧和确认帧，以及本轮检测参数摘要。另一只手的状态可保留在本地诊断元数据，不参与 VLM 资格校验。

## 4. 清单与产物字段

交付清单每行包含：轨迹 key、dataset_id、dataset_root、episode_id、episode_index、视频与 Parquet 绝对路径、帧数、FPS、相机、实测列名、时间列名、帧号列名和两个源文件 SHA-256。`task_family=fold_cloth` 是来源登记信息，不作为模型答案字段发送。

第一阶段清单的摘要已对 10 条真实源文件重新核对。v2 `prepare` 已完成路径、Parquet schema/全部行、视频完整解码及 state/PTS 对齐检查，实测结果保存在各 episode 的 `input.json`、`video_axis.json`，汇总见运行内 `reports/prepare_summary.json`。

最终 `result.json` 至少记录：

- `spec_version`、轨迹 key、帧数、候选摘要与运行指纹。
- 固定 stage1/2/3 槽位；每槽保存 `node_id`、`end_frame`、对应调用记录，未选时为 `null`。
- `status`、`has_uncertainty`、未找到/执行错误原因、每次判断的 trace 路径。
- 逐阶段 `search_status`：`selected`、`searched_not_found`、`not_searched` 或 `execution_error`，包含原因、逻辑调用数及 uncertain 数量。
- 逻辑调用数、HTTP 尝试数、输入/输出 tokens、缓存命中数与耗时。
- `review_status=unreviewed`、`training_ready=false`；人工确认另存，不覆盖自动结果。

此接口不要求 `task_type` 多任务分类、`participant_arms` 或全视频逐区间活动表。stage4 终态未评估时明确标为 `not_evaluated`，不得暗示搬运成功。

## 5. 文件读写

原始数据目录只读。新库通过绝对路径引用输入，不复制视频和 Parquet。新运行将输入清单、配置和摘要冻结到自身 `runs/`；生成的节点图、请求记录、MP4 与标签写入该运行。v2 另将固定 prompt 与三张示例的原字节、阶段映射及摘要冻结到运行内；`reports/request_images.json` 汇集每次实际请求的图像路径、角色、时间和摘要，示例不带轨迹时间。

新库可以读取旧结果用于离线比较，但推理模块不读取旧 annotation、人工边界或回归期望。密钥使用环境变量 `DASHSCOPE_API_KEY` 或显式 `--credential-file`。按用户最新要求，本库凭据可放在 `.secrets/dashscope_api_key.txt`，目录权限 700、文件权限 600，`.secrets/` 已由 `.gitignore` 排除。密钥内容不写入运行配置、图像请求索引、日志、文档或版本库。
