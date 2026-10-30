# 仓库主算法架构

业务语义以 [boundary.md](boundary.md) 为准。6 轴和 7 轴共享正式包中的数值算法，差异由同一配置结构声明。

```text
src/split_cloth_v2/
  cli.py                 prepare / fit / infer / report / export-video
  detector.py            实测夹爪张开候选，不判断阶段
  io_utils.py            JSON、摘要与原子写入
  joint/
    config.py            来源、布局、时间、检测、模板配置
    data.py              标准输入校验与准备运行
    adapters/            MCAP/FlatBuffer 解析、反馈对齐与视频转换
    features.py          按布局展开的共用运动特征和动作区间
    dtw.py / dtw.cpp      完整动作 DTW，随安装包分发
    templates.py         冻结模板加载、显式参考拟合和兼容校验
    search.py            S5→S2 最优两条后缀动态规划
    results.py           验收策略、固定边界槽位和逐帧标签
    pipeline.py          拟合/推断调度、输入保护和运行摘要
    render.py            已有结果的完整 MP4、报告及 PTS 验证
configs/joint/           两份 YAML 与轻量冻结模板
runs/<run_id>/           新运行；原数据和历史运行只读
```

配置及源反馈 → 标准 joint/原帧时间轴 → 检测候选与共用特征 → 完整动作评分 → 逆序搜索 → 结果与标签 → 视频/报告。示范参考仅进入模板拟合，不进入目标推断。渲染器只读已有结果，不重新训练、推断或移动边界。

配置显式声明实测来源、关节名称和顺序、单位、夹爪列与张开方向、相机、对齐策略、检测/特征参数、模板和验收策略。14/16 维分别产生 100/116 维特征；不裁维、补零或按机器人名称切换算法。模板加载先核对布局、特征和检测参数。

6 轴冻结模板的双初始化 b1 一致性与新示范的显式边界验收均由共用结果层执行。验收只能保留或拒绝候选，不能移动或补足边界。

正式包不导入实验目录、旧库或历史运行源码，不修改 `sys.path`，不需要服务凭据。实验目录仅保留迁移来源与历史证据；安装后的包可从仓库外运行。验证入口与证据见 [DEVELOPMENT.md](DEVELOPMENT.md)。
