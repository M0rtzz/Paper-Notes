---
title: >-
  [论文解读] I Can't Believe It's Not Scene Flow!
description: >-
  [ECCV 2024][场景流估计] 揭示现有场景流方法在行人等小目标上的灾难性失败被现有评估指标所掩盖，提出类别感知且速度归一化的Bucket Normalized EPE评估协议，以及一个简单但SOTA的TrackFlow基线（检测器+跟踪器生成场景流），在行人运动描述上实现1.5倍提升。
tags:
  - ECCV 2024
  - 场景流估计
  - 评估协议
  - 类别感知
  - 3D目标跟踪
  - LiDAR
---

# I Can't Believe It's Not Scene Flow!

**会议**: ECCV 2024  
**arXiv**: [2403.04739](https://arxiv.org/abs/2403.04739)  
**代码**: https://github.com/kylevedder/BucketedSceneFlowEval  
**领域**: 自动驾驶 / 3D视觉  
**关键词**: 场景流估计, 评估协议, 类别感知, 3D目标跟踪, LiDAR

## 一句话总结
揭示现有场景流方法在行人等小目标上的灾难性失败被现有评估指标所掩盖，提出类别感知且速度归一化的Bucket Normalized EPE评估协议，以及一个简单但SOTA的TrackFlow基线（检测器+跟踪器生成场景流），在行人运动描述上实现1.5倍提升。

## 研究背景与动机

**领域现状**：场景流估计是自动驾驶中的核心任务，目标是描述连续两帧点云之间的3D运动场。当前SOTA方法（如ZeroFlow XL 5x）在标准Threeway EPE指标上达到了约4.9cm的平均误差，看似已经达到厘米级精度。主流方法分为监督方法（FastFlow3D、DeFlow）和无监督方法（NSFP、ZeroFlow）。

**现有痛点**：然而，这些看似优秀的数字掩盖了一个严重问题——所有现有方法在行人、自行车等小目标上的场景流估计几乎完全失败。作者通过可视化发现，即使在行人LiDAR回波密度异常高的"最简单"情况下，所有先前方法都无法描述行人的运动。但标准的Threeway EPE指标完全没有暴露这个问题，因为行人点数占总动态点的比例不到1%，被大量的汽车点所淹没。

**核心矛盾**：现有评估协议存在双重缺陷：（1）类别不感知——小目标的少量点被大目标的大量点淹没在平均值中；（2）速度不归一化——0.5m/s的误差对20m/s的汽车微不足道（<2.5%），但对0.5m/s的行人意味着100%运动未被描述。这使得方法在安全关键类别上的灾难性失败被隐藏。

**本文目标** （1）设计能揭示小目标场景流失败的评估指标；（2）证明利用类别重平衡技术可以显著提升小目标场景流质量。

**切入角度**：受目标检测中mAP对每个类别等权重评估的启发，场景流评估也应该类别感知。同时观察到场景流的ground truth本身就来自3D bbox跟踪，因此用高质量检测器+跟踪器直接生成场景流是一个自然的baseline。

**核心 idea**：用类别感知+速度归一化的评估指标揭示现有方法的失败，并用"检测+跟踪→场景流"这个"令人尴尬地简单"的pipeline超越所有先前方法。

## 方法详解

### 整体框架
本文贡献分两部分：（1）新评估指标Bucket Normalized EPE；（2）新基线方法TrackFlow。TrackFlow的pipeline极其简单：输入两帧LiDAR点云→SOTA 3D目标检测器（LE3DE2E）→Kalman滤波跟踪器（AB3DMOT）→基于跟踪bbox的刚性变换生成逐点场景流。

### 关键设计

1. **Bucket Normalized EPE评估协议**:

    - 功能：提供类别感知且速度归一化的场景流评估，公平对比不同类别间的性能
    - 核心思路：将所有点按其ground truth的类别和速度分配到一个类别-速度矩阵中。对每个桶（bucket）计算Average EPE和平均速度。报告两个数字：Static EPE（静态桶的误差）和Dynamic Normalized EPE（各非空速度桶的归一化EPE均值，即 $\text{Average EPE} / \text{average speed}$）。Dynamic Normalized EPE度量的是"未被描述的运动比例"——0表示完美，1.0表示只预测了ego-motion补偿后的零流。最终通过对各类别取均值得到mean Dynamic Normalized EPE（类似mAP）
    - 设计动机：解决Threeway EPE的两个盲点。类别分桶确保行人等稀有类别不被汽车淹没；速度归一化使得不同速度目标间的误差可以直接比较。这让方法在安全关键类别上的真实表现无处隐藏

2. **TrackFlow场景流基线**:

    - 功能：通过3D目标跟踪的刚性变换生成场景流估计
    - 核心思路：运行SOTA 3D检测器LE3DE2E（使用低置信度阈值0.2以最大化recall），然后用Kalman滤波跟踪器AB3DMOT关联检测框生成轨迹。对于每个被检测到的目标，其内部的点云使用跟踪框之间的刚性变换来描述运动。未被检测到的点使用ego-motion补偿后的零流。该方法之所以有效是因为它直接模拟了ground truth的生成过程——ground truth流就是从bbox跟踪的刚性变换得来的
    - 设计动机：关键洞察是现代3D检测器使用了成熟的类别重平衡技术（copy-paste增强、focal loss等），因此在行人等稀有类别上有很好的检测能力。而现有场景流方法完全没有利用这些类别重平衡技术，导致在小目标上表现极差

3. **检测器选择与置信度调优**:

    - 功能：为TrackFlow选择最优的检测器配置
    - 核心思路：与常规目标检测使用高置信度阈值（0.7-0.9）不同，TrackFlow使用低阈值（0.2）以最大化recall。因为漏检的代价极大——每个false negative意味着该目标的所有点都只能得到零流，100%运动未被描述。而false positive可以被跟踪器的关联逻辑过滤掉。实验表明LE3DE2E在低阈值下的recall远优于BEVFusion，这直接导致TrackFlow显著优于TrackFlowBEVF
    - 设计动机：揭示了"适合TrackFlow的好检测器"不是mAP最高的，而是低阈值下recall最高且朝向估计准确的检测器——两个mAP相似的检测器可能产生截然不同的场景流质量

### 损失函数 / 训练策略
TrackFlow本身不需要训练场景流模型——它直接使用预训练的检测器和跟踪器。检测器的训练使用了标准的类别重平衡技术。评估时使用Argoverse 2的test split。

## 实验关键数据

### 主实验

| 指标 | TrackFlow | DeFlow | ZeroFlow XL 5x | NSFP |
|------|-----------|--------|----------------|------|
| Threeway EPE | **SOTA** (↓1.5mm) | 次优 | 第三 | 第四 |
| mean Dyn. Norm. EPE | **0.287** | ~0.39 | ~0.45 | ~0.50 |
| 行人 Dyn. Norm. EPE | **~0.40** | ~0.60 | ~0.80 | ~0.70 |
| 行人运动描述率 | **>50%** | ~30% | ~20% | ~30% |

### 消融实验

| 配置 | mean Dyn. Norm. EPE | 说明 |
|------|---------------------|------|
| TrackFlow (LE3DE2E) | **0.287** | 完整模型，recall高 |
| TrackFlowBEVF (BEVFusion) | +10-22%退化 | mAP仅低2%但recall差距大 |
| BEVFusion阈值0.1 | 0.4816 | 低阈值也无法弥补recall差 |
| BEVFusion阈值0.4 | 0.8176 | 高阈值下严重退化 |

### 关键发现
- TrackFlow在行人上描述了超过50%的运动，比DeFlow多20%（1.5倍提升），这是一个量级的差异
- Threeway EPE上TrackFlow仅领先1.5mm，但Bucket Normalized EPE揭示了巨大的性能差距——这完美证明了旧指标的失灵
- DeFlow在汽车类别上实际优于TrackFlow，但在行人上远远落后
- 检测器的recall比mAP更重要：BEVFusion的mAP只比LE3DE2E低2%，但TrackFlowBEVF的性能退化10-22%
- 后续Argoverse 2 2024 Scene Flow Challenge上，Flow4D通过架构改进将TrackFlow的动态误差减半，但未使用任何类别感知loss

## 亮点与洞察
- **用最简单的方法打败所有先前工作**：TrackFlow本质上就是"检测+跟踪"，没有任何场景流特定的设计，却是SOTA。这不是TrackFlow多强，而是现有方法在小目标上多差——一记响亮的耳光。巧妙之处在于利用了检测领域已经成熟的类别重平衡技术
- **评估指标的深刻反思**：论文的核心价值不在方法本身，而在于暴露了整个领域的评估盲区。这种"先修指标再提方法"的研究路径值得学习——有时候发现问题比解决问题更重要
- **recall vs precision的权衡在流估计中完全不同**：在检测任务中两者需要平衡，但在场景流中漏检的代价是灾难性的。这个洞察可以迁移到其他需要稠密逐点预测的任务

## 局限与展望
- TrackFlow只能预测刚性流（基于bbox），无法处理非刚性运动（如行人步态的关节运动）
- 依赖于闭集检测器的固定分类体系，无法处理开放世界中的未知目标类别（但作者指出可以替换为class-agnostic检测器）
- 分桶策略依赖语义标注，作者也展示了按体积分桶的替代方案，但需要进一步验证
- 未探索将类别重平衡技术直接注入到端到端场景流方法中的可能性——这可能是更elegant的方向

## 相关工作与启发
- **vs FastFlow3D**: 基于PointPillars的监督方法，是很多后续工作的基础架构，但缺乏类别重平衡导致行人失败
- **vs ZeroFlow**: 通过蒸馏的无监督方法，在Threeway EPE上接近SOTA但在行人上只描述了不到20%的运动
- **vs NSFP**: 基于在线优化的无监督方法，单帧优化MLP最小化Chamfer距离，计算昂贵且小目标表现差
- **vs Flow4D**: 挑战赛中的后续工作，通过4D体素架构改进将误差减半，证明架构创新也能帮助小目标

## 评分
- 新颖性: ⭐⭐⭐⭐ 评估指标的贡献大于方法本身，揭示领域盲区的工作很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 指标分析极其透彻，检测器消融、类别分析、可视化都很到位
- 写作质量: ⭐⭐⭐⭐⭐ 标题吸引人，论述逻辑清晰，FAQ部分坦诚回应质疑
- 价值: ⭐⭐⭐⭐⭐ 改变了整个场景流领域的评估标准，已被Argoverse 2挑战赛采用

<!-- RELATED:START -->

## 相关论文

- [What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](../../ACL2025/others/its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](../../AAAI2026/others/how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](../../NeurIPS2025/others/metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)
- [Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](../../AAAI2026/others/rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)

<!-- RELATED:END -->
