---
title: >-
  [论文解读] WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?
description: >-
  [CVPR 2026][3D视觉][立体匹配] 系统研究合成立体数据集的设计空间——在 Infinigen 过程化生成器上逐一变换六大参数（浮动物体密度/背景物体/物体类型/材质/相机基线/光照增强），量化其对零样本立体匹配的影响；发现 **"真实室内场景 + 浮动物体"** 的组合最有效，据此构建 WMGStereo-150k 数据集，仅用此单一数据集训练即超越 SceneFlow+CREStereo+TartanAir+IRS 四合一（Middlebury 降 28%，Booster 降 25%），与 FoundationStereo 竞争力相当。
tags:
  - CVPR 2026
  - 3D视觉
  - 立体匹配
  - 合成数据
  - procedural generation
  - 零样本
  - dataset design
---

# WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?

**会议**: CVPR 2026  
**arXiv**: [2504.16930](https://arxiv.org/abs/2504.16930)  
**代码**: [GitHub](https://github.com/princeton-vl/InfinigenStereo)  
**领域**: 3D视觉 / 立体匹配  
**关键词**: 立体匹配, 合成数据, procedural generation, 零样本, dataset design

## 一句话总结

系统研究合成立体数据集的设计空间——在 Infinigen 过程化生成器上逐一变换六大参数（浮动物体密度/背景物体/物体类型/材质/相机基线/光照增强），量化其对零样本立体匹配的影响；发现 **"真实室内场景 + 浮动物体"** 的组合最有效，据此构建 WMGStereo-150k 数据集，仅用此单一数据集训练即超越 SceneFlow+CREStereo+TartanAir+IRS 四合一（Middlebury 降 28%，Booster 降 25%），与 FoundationStereo 竞争力相当。

## 研究背景与动机

- **现有问题**: 合成数据是训练立体匹配网络的基石，但 **"什么使一个数据集有效？"** 尚无系统回答。SceneFlow 用飞行物体、TartanAir 用写实室内、FoundationStereo 用混合方案，每次同时改变多个因素，无法分离单因素贡献。
- **研究空白**: 大多数数据集不开源生成代码，无法复现、修改或消融。唯一相关的参数研究（Mayer et al. 2018）针对 2D warp 光学流，结论"realism is overrated"，是否适用于 3D 立体匹配未知。
- **核心思路**: 以 Infinigen 过程化生成器为平台，对每个参数单独控制变量生成 5000 对 stereo 对 → 训练 RAFT-Stereo → 评估 7 个 benchmark → 找出最优参数组合 → 生成大规模数据集。
- **关键价值**: 不仅提供一个新数据集，更提供 **可解释的参数分析** 和 **开源生成代码**，使得社区可按需定制数据。

## 方法详解

### 整体框架

基于 Infinigen（室内+自然场景过程化生成器）和 Blender Python API 构建可控 stereo 数据生成系统。支持三种场景类型：

1. **室内 + 浮动物体**（Indoor Floating）：Infinigen Indoors 生成真实房间布局，再在室内随机放置浮动物体
2. **密集浮动物体**（Dense Floating）：空天空背景中密集放置 200 个浮动物体
3. **自然场景**（Nature）：Infinigen Nature 生成自然环境

对每个参数设定生成独立数据集 → 训练 RAFT-Stereo 75k 步 → 在 Middlebury/ETH3D/KITTI/Booster 上评估零样本性能 → 确定最优配置 → 用最优配置生成 WMGStereo-150k（163,396 对）。

### 关键设计

#### 1. 浮动物体放置系统

- **功能**: 在给定场景中随机生成并布置浮动物体，支持基于光线投射的相机视野内放置和边界框内放置
- **核心思路**: 浮动物体提供极大的几何多样性——不同形状的物体在空间中随机分布，创造丰富的遮挡、深度不连续和复杂几何结构
- **设计动机**: SceneFlow 的 FlyingThings3D 已经证明飞行物体有效，但从未在真实场景背景上测试过这一策略的增益；本文将浮动物体与真实室内布局结合
- **消融验证**: 无浮动物体 → 0-10 个 → 10-30 个，Middlebury(H) 从 12.52 → 7.78 → 6.60 持续下降，确认浮动物体对零样本性能极其关键

#### 2. 背景真实性保留策略

- **功能**: 保留 Infinigen Indoors 生成的家具、桌椅等真实排列的背景物体
- **核心思路**: 与 optical flow 领域"真实性无关紧要"的结论相反，真实背景几何提供了有用的训练信号——规则的平面、合理的空间布局帮助网络学习真实世界的深度分布
- **设计动机**: FlyingThings3D 等数据集完全没有背景，而 TartanAir 完全依赖真实场景但缺乏浮动物体。本文结合两者的优势
- **消融验证**: 有背景 vs 无背景，所有 benchmark 一致提升（Middlebury(H) 8.35 → 6.60）

#### 3. 物体类型多样性与材质筛选

- **功能**: 使用 Infinigen 全部物体生成器（椅子、架子、灌木等），同时移除高 error 物体（仙人掌、海胆等针状结构）和极端材质（全透明玻璃、全反射金属）
- **核心思路**: 物体类型多样性保证跨 benchmark 的鲁棒性；材质多样性重要但需过滤极端情况——当前网络架构无法在不损害漫反射区域性能的前提下学习极端非朗伯表面
- **设计动机**: 只用椅子在 Middlebury 表现好（5.29）但 KITTI-15 差（7.02）；只用灌木在 ETH3D 好（3.13）但 Booster 差（12.19）。偏向任何单一类型都会引入 benchmark 偏差
- **筛选方法**: 通过 per-pixel EPE 按物体/材质分组聚合（占比 ≥0.1% 的像素），识别并移除 top-error 物体和材质

#### 4. 相机基线随机化

- **功能**: 将左右相机间距均匀采样于 [0.04, 0.4] m 的大范围
- **核心思路**: 基线决定视差分布——窄基线 [0.04, 0.1] m 产生小视差，大基线 [0.2, 0.3] m 产生大视差。大范围采样覆盖所有下游场景
- **设计动机**: 窄基线模型在 Middlebury(H) 达 9.60 但 Middlebury(F) 爆到 32.47；大基线在 ETH3D 上 14.05 惨败。基线多样性是"免费"但至关重要的改进
- **消融验证**: [0.04, 0.1] → [0.2, 0.3] → [0.04, 0.4]，宽范围在所有 benchmark 上最鲁棒

#### 5. 生成成本优化

- **功能**: 在固定计算预算下最大化数据量，通过三个手段降低成本约 6 倍
- **核心思路**:
    - **降低室内求解器步数**（550 → 60 步）：约束求解器只贪心添加物体，不移动/删除，生成时间从 50.85 分钟降至 13 分钟
    - **降低光线追踪采样**（8192 → 1024 样本 + OptiX 去噪）：渲染时间降至 27 秒/帧
    - **场景复用**：每个室内场景放 20 个相机位，每个密集浮动场景随机化 200 次位置/朝向/光照/基线
- **设计动机**: 在固定计算预算下，低质量但多数量（30k 对）反而优于高质量少数量（5k 对）——Middlebury(H) 从 6.60 降至 5.63

## 实验关键数据

### 主实验：WMGStereo-150k vs 现有数据集（DLNR, 200k 步训练）

| 训练数据 | Midd-14(F) | Midd-14(H) | Midd-21 | ETH3D | KITTI-12 | KITTI-15 | Booster(Q) |
|---------|-----------|-----------|---------|-------|---------|---------|-----------|
| SceneFlow | 10.96 | 6.20 | 8.44 | 23.12 | 9.45 | 15.74 | 18.17 |
| CREStereo | 14.45 | 11.53 | 10.60 | 5.18 | 4.95 | 5.90 | 14.61 |
| TartanAir | 12.56 | 7.27 | 14.47 | 4.35 | 3.98 | 5.33 | 18.14 |
| IRS | 7.81 | 6.13 | 8.49 | 3.91 | 4.56 | 5.60 | 10.32 |
| FSD | 5.80 | 3.27 | 6.93 | 2.13 | 3.56 | 4.18 | 7.51 |
| **WMGStereo-150k** | **5.10** | 3.76 | **6.72** | 2.50 | **3.30** | 4.54 | 9.09 |
| FSD+WMGStereo | 5.24 | **3.24** | 6.88 | **2.08** | 3.59 | **4.26** | **7.42** |

### 跨架构验证：单数据集 vs 四合一 Mixed（SF+CRE+Tartan+IRS, 600k 对）

| 模型-数据 | Midd-14(H) | Midd-21 | ETH3D | KITTI-12 | KITTI-15 | Booster(Q) |
|----------|-----------|---------|-------|---------|---------|-----------|
| RAFT-Mixed | 5.50 | 8.97 | 2.58 | 3.64 | 4.95 | 11.46 |
| **RAFT-WMGStereo** | **4.48** | **8.17** | 2.93 | **3.25** | **4.25** | **9.17** |
| Sel-IGEV-Mixed | 5.24 | 8.24 | 2.37 | 3.97 | 5.31 | 11.00 |
| **Sel-IGEV-WMGStereo** | **3.61** | **7.62** | 2.47 | **3.26** | **4.55** | **8.84** |
| DLNR-Mixed | 5.21 | 9.30 | 2.50 | 3.68 | 4.95 | 12.17 |
| **DLNR-WMGStereo** | **3.76** | **6.72** | 2.50 | **3.30** | **4.54** | **9.09** |

### 消融实验：参数研究关键设定（RAFT-Stereo, 5k 对, 75k 步）

| 参数 | 设定 | Midd-14(H) | ETH3D | KITTI-15 | Booster(Q) |
|------|------|-----------|-------|---------|-----------|
| 浮动密度 | 无浮动物体 | 12.52 | 4.47 | 6.19 | 16.40 |
| | 0-10 个 | 7.78 | 3.62 | 6.09 | 12.21 |
| | 10-30 个 | **6.60** | 3.92 | **5.11** | **10.60** |
| 背景物体 | 无背景 | 8.35 | 4.39 | 6.28 | 12.72 |
| | 有背景 | **6.60** | **3.92** | **5.11** | **10.60** |
| 材质 | 无材质 | 9.02 | 3.48 | 6.07 | 14.07 |
| | 单一漫反射 | 7.21 | **2.77** | 5.41 | 12.73 |
| | 仅金属+玻璃 | 8.37 | 4.95 | **4.97** | **9.80** |
| | 全材质 | **6.60** | 3.92 | 5.11 | 10.60 |
| 基线范围 | [0.04, 0.1] m | 9.60 | **2.89** | 6.64 | 17.03 |
| | [0.2, 0.3] m | 7.01 | 14.05 | 5.37 | **8.96** |
| | [0.04, 0.4] m | **6.60** | 3.92 | **5.11** | 10.60 |

### 关键发现

- **采样效率惊人**: 仅 500 个 WMGStereo 样本在 Middlebury 上的 EPE 低于 100K 个 CREStereo 样本，数据集设计比数据量更重要
- **场景混合最佳比例**: 33% Indoor + 33% Dense Floating + 33% Nature 在所有 benchmark 上最鲁棒，Indoor Floating 是最佳单一类型
- **固定计算下量 > 质**: 降低渲染精度 6 倍但增加数据量 6 倍（5k → 30k），Middlebury(H) 从 6.60 降至 5.63
- **泛化到未见 benchmark**: 在未参与参数调优的 DrivingStereo 上，WMGStereo 的 3px error 为 1.89，比 FSD 的 2.59 低 27%

## 亮点与洞察

- **"真实场景 + 随机物体"组合超越两者单独使用**：推翻了 optical flow 领域"realism is overrated"的经典结论——对 stereo 来说，背景的真实几何确实提供有用的训练信号，但仅靠真实性不够，还需要浮动物体带来的几何多样性
- **相机基线多样性是被严重低估的关键因素**：简单地扩大基线采样范围就能获得巨大收益，但此前从未被系统研究
- **数据设计 >> 数据规模**：500 样本 WMGStereo > 100K CREStereo，说明参数选择的质量远比堆数据量重要
- **开源生成代码的独特价值**：与 FSD（静态数据集）不同，用户可按需调整参数——如针对非朗伯场景只生成玻璃材质数据，实现数据与架构的联合设计

## 局限性

- 非朗伯表面仍是瓶颈：当前只能移除极端透明/反射材质作为折衷，而非真正解决网络对这类表面的学习困难
- 自然场景（Nature）单独表现最差（Midd-14(H) 12.27），可能需要更好的自然物体生成和相机放置策略
- 未涉及时序/视频 stereo 数据生成
- 与 FoundationStereo 的差距部分来自架构（FS 有新的 architecture），不完全是数据问题
- 参数研究固定使用 RAFT-Stereo，最优参数是否架构依赖未完全验证（仅最终配置做了跨架构验证）

## 相关工作与启发

- **vs FoundationStereo 数据集（FSD）**: FSD 同时引入多个新特性 + 新架构，本文分离并分析每个因素的贡献。两者互补——合用效果最好（ETH3D 2.08，Booster 7.42）
- **vs SceneFlow/FlyingThings3D**: 经典飞行物体数据集但缺乏场景真实性和材质多样性。WMGStereo 用约 1/4 的数据量大幅超越
- **vs Mayer et al. (2018) 的 flow 数据研究**: 那个研究聚焦 2D warp + 光学流，结论"realism overrated"。本文在 3D stereo 中发现背景真实性确实有帮助——这是一个重要的领域差异
- **vs Infinigen 原始 stereo 尝试**: Raistrick et al. 用 Infinigen Nature 做过 stereo 但未达到竞争力。本文通过加入浮动物体、材质筛选、基线扩展等数据工程才实现突破

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 创新性 | 7 | 并非新方法/新架构，而是对数据集设计空间的系统研究，角度新颖但技术门槛不高 |
| 实验充分性 | 9 | 参数研究覆盖六大因素，跨三个架构验证，含成本分析和采样效率曲线，非常详实 |
| 实用价值 | 9 | 开源生成代码 + 参数指南，社区可直接使用或定制，WMGStereo-150k 本身就是高价值数据集 |
| 写作质量 | 8 | 结构清晰，Tab.1 的控制变量实验设计教科书级别，图表丰富 |
| **总分** | **8.0** | 数据驱动的系统研究，实验设计严谨，对 stereo 数据生成社区有长期指导价值 |

<!-- RELATED:START -->

## 相关论文

- [What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)
- [EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)

<!-- RELATED:END -->
