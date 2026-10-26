# 带 expert example 的单候选 VLM 接口

prompt：`node_completion_prompt_v1`；协议：`reverse_node_v2`；日期：2026-10-26。已通过 [configs/reverse_node_v2.yaml](../configs/reverse_node_v2.yaml) 接入 [configs/prompt.txt](../configs/prompt.txt)、固定局部上下文与当前阶段示例。对应 [边界契约](../boundary.md)和[运行架构](../architecture.md)。`configs/default.yaml` 及包内 `node_shape_prompt_v3.txt` 保留 v1。离线接口与恢复验证已通过；真实模型结果和人工确认另见 README 及运行报告。

## 1. 固定 prompt 的配置入口

固定 system 文本只维护在 [configs/prompt.txt](../configs/prompt.txt)，本文不复制全文。v2 YAML 通过 `prompt_file: prompt.txt` 指定该文件，路径相对 YAML 所在目录解析；版本标识为 `node_completion_prompt_v1`。文件包含阶段形态、expert example 的使用方式、“刚完成且未开始下一动作”的判据和四字段返回要求。

prepare 将原文（保留原换行）及 SHA-256 冻结为运行内的 `prompt.txt`，annotate 使用冻结副本并精确核对文本和摘要。动态的目标阶段、当前节点和图片标签由请求构造器补充；加载、校验和恢复职责见 [architecture.md](../architecture.md)。

## 2. 每次 user 内容与图像

```text
当前候选：{node_id}
当前目标：stage{target_stage}
返回JSON中的target_stage字段必须是整数{target_stage}。
先提供一张当前阶段的 EXPERT_EXAMPLE，再提供本轨迹当前候选及其前后局部上下文。
本轨迹图像按时间正序排列，只有 CANDIDATE 是本次待判断节点。
请参照示例的衣物和夹爪状态，判断当前节点是否刚完成该次折叠且尚未开始下一动作。
```

示例与本轨迹图像分别标记；本轨迹部分沿用原有固定时间上下文，默认整次请求至多六张图片：

| 输入 | 角色及标识 | 时间信息 |
| --- | --- | --- |
| 当前目标示例 | `EXPERT_EXAMPLE`、`stage{target_stage}` | 无本轨迹帧号、PTS、相对时间或候选 ID。 |
| 候选之前的局部上下文 | `BEFORE` | 相对当前候选的负时间偏移。 |
| 当前候选 | `CANDIDATE`、当前 `node_id` | 相对时间为 0。 |
| 候选之后的局部上下文 | `AFTER` | 相对当前候选的正时间偏移。 |

按实际视频 PTS 取候选相对时间 `[-1.0, -0.5, 0.0, 0.5, 1.0]` 秒附近的原始帧，越界截断后去重；0 必须精确为候选原帧，不能用附近采样代替。本轨迹图片始终按时间正序排列，原帧号、PTS、角色和摘要保存在本地请求记录。上下文不要求是张开事件，碰巧对应其他候选时也不改变本次唯一待判节点。没有“至少三个候选才调用”的门槛，按既有倒序搜索处理可用候选并保留部分结果。

示例映射固定为 stage1 → [s1.png](../configs/images/s1.png)、stage2 → [s2.png](../configs/images/s2.png)、stage3 → [s3.png](../configs/images/s3.png)。每次只附当前目标对应的一张。prepare 冻结全部示例和映射，请求使用运行内的副本；缺失示例属于配置错误，不能静默降级。

不向 VLM 发送原始夹爪数组、另一爪的结构化开合状态、人工边界帧号、旧模型回答或数据集任务标签。程序负责张开事件资格，VLM 依据图像判断完成状态及动作时刻。图片预算与处理参数写入 run 配置及证据索引。

每个候选只作一次正式语义判断，上下文不接受单独判断。不做 `uncertain` 后的隐式扩图、换示例或重问；改变请求策略需要新版本和新运行。

## 3. 返回格式

以下是结构示例，不是实际模型结果：

```json
{
  "node_id": "n0007",
  "target_stage": 3,
  "decision": "match",
  "evidence": "衣物已横折成短包，夹爪仍在包体附近的释放位置，结合局部上下文可见此次折叠刚结束，当前尚未转入搬运。"
}
```

严格限定四个字段：`node_id` 必须等于当前候选，`target_stage` 必须等于当前目标且为整数 1/2/3，`decision` 只能是 `match`、`skip`、`uncertain`，`evidence` 为非空短句。返回其他候选 ID 即使其原帧恰在局部上下文中也必须拒绝；同时拒绝自由帧号、时间戳、示例标识、`participant_arms` 和额外阶段槽位。

格式检查失败属于执行/接口错误，不等同于衣物不匹配。`evidence` 留给人工核查，程序不使用关键字或第二个模型去偷偷改写 decision。

## 4. prompt 管理

`prompt_file` 已接入，将 `configs/prompt.txt` 作为 v2 固定指令的唯一来源，不在 Python 或包内资产中维护另一份 v2 文本。运行冻结实际文本、版本和摘要；源配置或源示例在 prepare 后变化不会影响已冻结请求。冻结副本被改动则拒绝继续，采用新文本或示例时新建运行。

`reports/request_images.json` 记录各请求的节点、目标阶段、请求指纹及按实际发送顺序排列的图像元数据；每条请求的 `request.json` 同时保存完整文本、图像路径与摘要。`EXPERT_EXAMPLE` 无轨迹时间信息，本轨迹图像保留原帧号、PTS、相对时间和角色。

请求指纹包括当前候选 ID、候选与上下文的原帧、图像摘要、角色、时间和顺序，上下文取样配置、目标阶段，示例阶段映射与源图/发送图摘要，以及完整 prompt、模型、推理参数和协议版本。上下文、示例或策略变化都不能误用旧缓存。程序负责倒序与目标切换，模型只判断当前节点；完成时刻判据和示例输入使用新协议并重新运行回归。
