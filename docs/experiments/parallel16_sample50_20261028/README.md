# 50 条叠衣轨迹：16 路推理与 render

日期：2026-10-28。实际运行已完成，全部模型结果仍为 `unreviewed`，没有人工准确率。

代码提交：`5a86b66d48e19a578b4d015dbb01a1d75c901b2b`。配置为 [reverse_node_v2_parallel16.yaml](../../../configs/reverse_node_v2_parallel16.yaml)，沿用 main 的 stage3 shape guard prompt、Qwen 模型、state 检测器、五帧上下文及单张阶段示例。

从 `fold_cloth_v3` 和 `fold_cloth_v4` 各无放回抽取 25 条，独立种子为 `SHA256(20261028:dataset_id)`，交错排列；没有按模型结果筛选或替换。清单与源 SHA-256 见 [manifest.jsonl](manifest.jsonl)，总体、元数据摘要及抽样索引见 [selection.json](selection.json)。与 P1 十条清单没有重叠；这不构成独立 heldout 或人工质量验证。

| 指标 | 实测 |
| --- | --- |
| 程序状态 | 27 complete / 23 needs_review / 0 execution_error |
| 找到阶段边界数 S1/S2/S3 | 27 / 38 / 39 |
| 候选与逻辑请求 | 664 个候选 / 485 次判断 |
| 正式运行 HTTP 尝试 | 485，全部 HTTP 200；无重试 |
| 输入 / 输出 tokens | 738819 / 46805 |
| 推理墙钟耗时 | 192.111 秒 |
| 推理轨迹 / HTTP 同时在途峰值 | 16 / 16 |
| render 墙钟耗时 | 210.761 秒 |
| render 进程数 / 并行峰值 | 16 / 16 |
| 完整视频与逐帧验证 | 50 个 / 33893 帧 |
| 最大相对 PTS 误差 | 0.0 秒 |
| 离线检查 | 212 passed in 17.46s；模拟调用无网络 |

推理在线程池中按轨迹调度，每条依然 stage3→stage2→stage1 串行倒序；合法 skip/uncertain 均不重问。render 使用 16 个 spawn 进程，每条视频内部按原始 PTS 正序全帧导出。候选标记包含全部 state 张开候选，未查询节点不伪造回答，缺槽仍显示未知。

程序 `complete` 只表示找齐合法边界。`needs_review` 包含不确定或缺槽，不能由缺槽推断视频没有该动作。准确率和机器人任务成功率均未评估。

最初在沙箱中的运行因网络限制产生 150 次 transport_error，未保存模型响应；原记录完整保留。获得网络执行许可后，在新的 `..._online` 运行中使用相同冻结配置和准备材料完成正式实验，没有复制旧调用、响应或判断。详见 [transport_recovery.json](transport_recovery.json)。

本地完整运行：`runs/parallel16_sample50_20261028_online/`。视频位于 `exports/candidate_marks/videos/`，完整报告位于 `reports/experiment_report.md`；源代码快照、全部请求/响应、图像索引及渲染器副本都保留在该运行。视频、图片、密钥和原始 Parquet 不入版本库。

本目录保留 [完整执行审计](execution_audit.json)、[逐条模型判断及请求指纹](decisions.jsonl)、[并行任务时间记录](parallel_execution.json)、[视频逐帧验证](video_verification.json) 和 [离线检查记录](offline_checks.json)。审计逐一核验候选与帧、阶段顺序、请求中的冻结 prompt/图像摘要、未重问、标签和源输入未变，以及 render 前后所有 6094 个标注文件的摘要。

| 轨迹 | S1 / S2 / S3 原帧 | 程序状态 | 调用数 |
| --- | --- | --- | --- |
| fold_cloth_v3__episode_001291 | None / 432 / 500 | needs_review | 13 |
| fold_cloth_v4__episode_002970 | None / None / None | needs_review | 14 |
| fold_cloth_v3__episode_002528 | None / 387 / 436 | needs_review | 15 |
| fold_cloth_v4__episode_001058 | None / 680 / 745 | needs_review | 15 |
| fold_cloth_v3__episode_003844 | 296 / 334 / 386 | complete | 5 |
| fold_cloth_v4__episode_000628 | 414 / 527 / 593 | complete | 7 |
| fold_cloth_v3__episode_001189 | None / 425 / 556 | needs_review | 10 |
| fold_cloth_v4__episode_004014 | None / None / None | needs_review | 14 |
| fold_cloth_v3__episode_003573 | 242 / 331 / 377 | complete | 6 |
| fold_cloth_v4__episode_000273 | None / 420 / 510 | needs_review | 8 |
| fold_cloth_v3__episode_005951 | 473 / 542 / 644 | complete | 7 |
| fold_cloth_v4__episode_003434 | 719 / 1015 / 1092 | complete | 9 |
| fold_cloth_v3__episode_006069 | 410 / 456 / 518 | complete | 4 |
| fold_cloth_v4__episode_003849 | None / None / None | needs_review | 12 |
| fold_cloth_v3__episode_001134 | 412 / 413 / 522 | complete | 7 |
| fold_cloth_v4__episode_003263 | 338 / 652 / 717 | complete | 7 |
| fold_cloth_v3__episode_003120 | 320 / 367 / 434 | complete | 6 |
| fold_cloth_v4__episode_000996 | 343 / 415 / 459 | complete | 7 |
| fold_cloth_v3__episode_002628 | None / None / None | needs_review | 13 |
| fold_cloth_v4__episode_003910 | None / None / None | needs_review | 24 |
| fold_cloth_v3__episode_000153 | None / 392 / 474 | needs_review | 10 |
| fold_cloth_v4__episode_000834 | 423 / 535 / 591 | complete | 6 |
| fold_cloth_v3__episode_002293 | 368 / 456 / 524 | complete | 5 |
| fold_cloth_v4__episode_001074 | 453 / 458 / 519 | complete | 5 |
| fold_cloth_v3__episode_001940 | None / None / None | needs_review | 12 |
| fold_cloth_v4__episode_003030 | 266 / 510 / 601 | complete | 9 |
| fold_cloth_v3__episode_003429 | 321 / 369 / 418 | complete | 6 |
| fold_cloth_v4__episode_001280 | None / None / None | needs_review | 14 |
| fold_cloth_v3__episode_006210 | 316 / 384 / 550 | complete | 7 |
| fold_cloth_v4__episode_000298 | 277 / 337 / 417 | complete | 6 |
| fold_cloth_v3__episode_001429 | None / 478 / 598 | needs_review | 14 |
| fold_cloth_v4__episode_000365 | None / None / None | needs_review | 12 |
| fold_cloth_v3__episode_000963 | None / None / None | needs_review | 13 |
| fold_cloth_v4__episode_003040 | None / 480 / 557 | needs_review | 16 |
| fold_cloth_v3__episode_001780 | 544 / 553 / 617 | complete | 5 |
| fold_cloth_v4__episode_001168 | 320 / 387 / 438 | complete | 6 |
| fold_cloth_v3__episode_000267 | 255 / 468 / 549 | complete | 5 |
| fold_cloth_v4__episode_004159 | None / None / 612 | needs_review | 15 |
| fold_cloth_v3__episode_001007 | 286 / 564 / 654 | complete | 9 |
| fold_cloth_v4__episode_003199 | 699 / 780 / 843 | complete | 8 |
| fold_cloth_v3__episode_005944 | 327 / 337 / 405 | complete | 3 |
| fold_cloth_v4__episode_000432 | None / 326 / 402 | needs_review | 9 |
| fold_cloth_v3__episode_000294 | 410 / 498 / 577 | complete | 4 |
| fold_cloth_v4__episode_003644 | None / 550 / 619 | needs_review | 15 |
| fold_cloth_v3__episode_004032 | 299 / 377 / 428 | complete | 7 |
| fold_cloth_v4__episode_002180 | None / 679 / 735 | needs_review | 19 |
| fold_cloth_v3__episode_003519 | 272 / 352 / 410 | complete | 5 |
| fold_cloth_v4__episode_000327 | 338 / 385 / 441 | complete | 6 |
| fold_cloth_v3__episode_003382 | None / None / None | needs_review | 18 |
| fold_cloth_v4__episode_003142 | None / None / None | needs_review | 13 |
