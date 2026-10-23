# 旧库资料与迁移边界

以下为只读参考。新协议以 [boundary.md](../boundary.md) 为准，不继承旧文档中与当前用户要求冲突的规则。

## 1. 实际位置

```text
OLD=/gpfs/jiuquyun/home/xuran/xuran_projects/split_cloth/split_cloth
WORKTREE=/tmp/split-cloth-stage-text-implementation
RUN=$OLD/runs/stage3_anchor_diag5_20261023
NEW=/gpfs/jiuquyun/home/xuran/xuran_projects/split_cloth/split_cloth_v2
```

这些是路径缩写，后文路径相对相应根目录。

| 位置 | 2026-10-23 核对状态 |
| --- | --- |
| OLD | 分支 `experiment/human-boundary-prompt-v3`，HEAD `58c9e7097d42b637af046452546c572730242234`。 |
| WORKTREE | 分支 `experiment/stage-text-implementation`，HEAD `3d68aaacb386d3592320c52068bbf57cb99dc937`。 |
| RUN | 旧五条 stage3-anchor 试跑的冻结配置、代码、模型请求、结果与视频。 |

WORKTREE 中 stage3-anchor 配置、实验目录和视频导出器在核对时属于未提交改动。仅 checkout 上述 commit 不能恢复这些文件；优先使用 RUN 中的冻结配置/代码快照定位实际运行版本。`/tmp` worktree 可能消失，新库不能运行时依赖它。

## 2. 建议复用的实现

| 旧位置 | 可复用内容 | 迁移时注意 |
| --- | --- | --- |
| `RUN/code_snapshot/gripper_events.py` | state 读取、14D 索引、时间轴校验、多阈值交叉 | 只支持已核验 state 布局；原始 action 不能代替 state。 |
| `RUN/code_snapshot/gripper_nodes.py` | any_open、去抖、稳定平台与原帧映射 | 保留任一爪规则；不同帧事件不合并。 |
| `RUN/code_snapshot/event_visuals.py` | 按原帧解码、时间戳、图像抽取 | 改为单节点局部上下文，不再要求全区间拼图。 |
| `RUN/code_snapshot/provider.py` | API 适配、凭据读取、重试和用量 | 新增完整请求指纹与目标阶段校验；不能照搬仅凭目录缓存响应的行为。 |
| `WORKTREE/scripts/export_stage_videos.py` | 默认可视化的直接参考：左侧原视频、右侧阶段面板、底部时间线、H.264 编码 | 依 [VISUALIZATION.md](VISUALIZATION.md) 迁移至独立 `video_export.py`；适配新 result，删除 task_type、参与手、interval_observations 依赖，按新契约处理缺边界。 |

先审阅并提取所需纯函数与对应测试，再形成新库独立模块。不要整份复制旧 interval pipeline 作为新库主体，也不要通过 PYTHONPATH 指向旧库来长期运行。

## 3. 明确停用的旧接口/规则

- `interval_probe.py` 的整条轨迹一次逻辑请求，以及全区间 `interval_observations` 输出。
- `interval_contract.py` 中“所有 participant_arms 必须包含于 open_arms”的阶段资格检查。
- “VLM 先阅读完整视频，再自己选择三个节点”的搜索职责。
- `boundary.md` 旧粗标注协议的相邻 1 FPS 样本 L/R 推断和自由粗边界区间。
- 挂衣、裤子、混合任务分类及负例评估。新库输入已限定叠衣。
- 以 `accepted` 当人工正确，或用视频结束自动证明第四阶段完成。

000543 在旧运行被标 `needs_review`，主要因为模型证据引用到了所选边界之后，且 stage3 填双手参与、候选仅右爪张开。用户确认其展示划分正确；新库依当前单爪与单节点接口实现，不为消除旧检查错误移动其边界。

## 4. 可查阅的文件

| 相对路径 | 用途 |
| --- | --- |
| `RUN/run.json` | 实际模型、检测配置、协议、代码摘要。 |
| `RUN/manifest.jsonl` | 旧五条输入；新库保留其中四条叠衣作为回归子集，再补六条 development 组成 P1 的 10 条清单。 |
| `WORKTREE/experiments/stage_text_protocol_v1_20261023/splits/development_manifest.jsonl` | P1 补充六条的来源；按每个叠衣 dataset 的 episode_index 升序各取前三个未选样本，源摘要保存在 phase1_selection.json。 |
| `RUN/episodes/fold_cloth_v3/episode_000543/selection/request.json` | 旧实际请求，便于确认旧版输入负担。 |
| `RUN/episodes/fold_cloth_v3/episode_000543/gripper_nodes.json` | 候选帧、左右状态、平台依据。 |
| `RUN/episodes/fold_cloth_v3/episode_000543/annotation.json` | 候选 400/485/562 与旧检查原因。 |
| `RUN/reports/diag5_report.json` | 旧辅助视觉结果与用量。 |
| `RUN/videos/` | 用户查看的旧带划分视频。 |
| `WORKTREE/configs/fold_cloth_prompt_stage3_anchor.txt` | 旧 v2 加反向回查规则，尚未逐节点调用。 |
| `OLD/configs/fold_cloth_prompt_v3.txt` | human-boundary v3 的文字澄清。 |
| `OLD/configs/fold_cloth_text_criteria_v3.json` | v3 判据及离线来源记录。 |

human-boundary v3 的有用经验是允许不整齐的折叠、区分正常重抓和返工、缺候选不借搬运节点补位。新短 prompt 已吸收相关直观含义，不把整份冗长判据或双手条件迁入。

## 5. 旧实验事实的适用范围

旧模型是 `qwen3.8-max-0902`，temperature 0，经 DashScope compatible API 调用。旧五条一次一条整段推理合计 5 次逻辑请求、70 张目标图、65,498 tokens；这些是旧五条样本和旧输入方式的统计，不能作为新 10 条逐节点调用的成本预算。

新库实际调用量由候选数和首次匹配位置决定，应先离线列出每条最多请求数。当前文档未检查服务最新型号与计价，也未使用密钥调用外部服务。

旧人工边界窗口来自另一个叠衣数据来源的开发材料，不是新库 P1 的独立真值。000543 之外的九条均没有本轮确认的正确边界，不要从旧汇总自动生成“正确答案”。
