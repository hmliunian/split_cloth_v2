# split_cloth_v2

新库用于实现：**夹爪 state 生成张开候选 → 程序倒序逐点调用 VLM → 找到小方块、双侧长条、单侧叠入三个结束点 → 导出四阶段划分。**

当前仅交付开发文档和诊断输入索引，尚无算法实现、可运行 CLI 或本库测试结果。文档日期：2026-10-23。

**开发第一阶段（P1）的首次交付为 10 条不同叠衣轨迹的完整 MP4**，附边界结果、运行用量、导出验证和实验汇总。P1 是开发里程碑，与视频内的 stage1 不同。

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

从开发文档的第一步开始建立 Python 包和离线验证，随后实现逐节点状态机。文档中的 CLI 名称均为待实现接口，不能作为已存在命令运行。
