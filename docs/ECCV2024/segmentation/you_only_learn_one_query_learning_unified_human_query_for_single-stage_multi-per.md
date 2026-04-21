---
title: >-
  [论文解读] You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception
description: >-
  [ECCV 2024][图像分割][人体感知] 提出 HQNet 框架，通过学习统一的 Human Query 表示，在单阶段单模型中同时完成行人检测、实例分割、2D 姿态估计、3D Mesh 恢复、属性识别等多种以人为中心的感知任务，并构建了首个全面的多任务人体感知基准 COCO-UniHuman。
tags:
  - ECCV 2024
  - 图像分割
  - 人体感知
  - 统一查询
  - 多任务学习
  - DETR
  - 实例分割
---

# You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception

**会议**: ECCV 2024  
**arXiv**: [2312.05525](https://arxiv.org/abs/2312.05525)  
**代码**: [https://github.com/lishuhuai527/COCO-UniHuman](https://github.com/lishuhuai527/COCO-UniHuman)  
**领域**: 图像分割  
**关键词**: 人体感知, 统一查询, 多任务学习, DETR, 实例分割

## 一句话总结

提出 HQNet 框架，通过学习统一的 Human Query 表示，在单阶段单模型中同时完成行人检测、实例分割、2D 姿态估计、3D Mesh 恢复、属性识别等多种以人为中心的感知任务，并构建了首个全面的多任务人体感知基准 COCO-UniHuman。

## 研究背景与动机

**领域现状**: 以人为中心的感知（Human-Centric Perception, HCP）涵盖行人检测、人体分割、2D 关键点估计、3D 人体重建、属性识别等多种任务。这些任务各自有成熟的方法，但大多是单任务模型或多阶段 pipeline（先检测再逐人分析）。

**现有痛点**:
   - **缺乏统一基准**: 不同 HCP 任务使用各自独立的数据集（如 COCO 只有检测/姿态/分割，CelebA 只有人脸属性），规模、视角、场景各异，无法在统一环境下评估多任务方法
   - **多阶段方法的缺陷**: (1) 早期决策问题——整个 pipeline 高度依赖检测器，检测失败无法恢复；(2) 运行时间与人数成正比，多人场景效率低；(3) 忽略了不同 HCP 任务之间的内在关联
   - **数据集偏差**: 不同任务数据集存在固有的尺度差异（场景图 vs 裁剪图）和域偏差（实验室 vs 监控视角），朴素混合训练会引入偏差

**核心矛盾**: 不同 HCP 任务需要不同粒度的特征——检测需要全局语义、属性需要全局+局部语义、分割需要细粒度语义、姿态需要细粒度定位，如何用统一表示兼顾多粒度需求。

**本文目标** 设计单阶段单模型框架同时处理所有代表性 HCP 任务，并构建对应的统一评估基准。

**切入角度**: 借鉴 DETR 系列的 query 学习思想，将每个人体实例编码为一个统一的 Human Query，该 query 同时承载多粒度信息供各任务使用。

**核心 idea**: 学习一个 all-in-one 的 Human Query 表示，编码每个人体实例的全局/局部外观特征和粗糙/精细定位特征，通过共享 Transformer 解码器和轻量级任务专用头完成多任务统一推理。

## 方法详解

### 整体框架

HQNet 由四个关键组件构成：
- **Backbone 网络**（如 ResNet-50 / Swin-L / ViT-L）：提取多尺度图像特征
- **共享 Transformer Encoder**：增强特征表示，输出增强的多尺度特征和位置编码
- **任务共享 Transformer Decoder**：使用 Deformable Attention 和 Mixed Query Selection 初始化 positional query，通过多层解码逐步精炼 content query（即 Human Query），所有任务共享同一个解码器
- **轻量级任务专用头**：各任务自有的 MLP/FC 预测头，从共享的 Human Query 出发做最终预测

核心理念：最大限度地权重共享（backbone + encoder + decoder），仅在最后的预测头处分支，确保可扩展性。

### 关键设计

1. **Human Query（统一人体查询表示）**:

    - **功能**: 将 DETR 中的 object query 扩展为承载多粒度人体信息的统一表示
    - **核心思路**: query 由 positional query（4D anchor box 编码中心坐标和宽高）和 content query 组成。Content query 即 Human Query，通过共享解码器的多层 deformable attention 与图像特征交互，编码实例级的全局/局部外观特征和粗糙/精细定位特征
    - **设计动机**: 不同 HCP 任务关注不同粒度的特征，一个足够丰富的统一表示可以同时服务于所有任务，同时利用任务间的协同效应

2. **HumanQuery-Instance Matching (HQ-Ins Matching)**:

    - **功能**: 在训练时综合多个任务的损失进行 query-GT 匹配
    - **核心思路**: 匹配代价为 $\lambda_{cls}L_{cls} + \lambda_{det}L_{det} + \lambda_{seg}L_{seg} + \lambda_{pose}L_{pose}$，综合考虑分类、检测、分割和姿态损失
    - **设计动机**: 传统 DETR 仅用检测损失匹配，可能出现一个人的姿态被匹配到另一个人的情况。HQ-Ins Matching 通过多任务联合约束确保每个 query 对所有任务都一致地映射到同一个 GT 实例

3. **Gender-aided human Model Selection (GaMS)**:

    - **功能**: 利用性别预测结果选择对应的 SMPL 人体模型（男性/女性/中性）进行 3D Mesh 恢复
    - **核心思路**: 训练和推理时根据性别标签/预测选择不同版本的 SMPL 模型
    - **设计动机**: 以往工作因缺乏性别标注只能用中性模型，COCO-UniHuman 提供了性别标注，可利用任务间协同提升 3D 重建精度

4. **任务专用头设计（三种范式）**:

    - **坐标预测任务**（检测、姿态）: 共享 reference point，MLP 回归归一化偏移量
    - **密集预测任务**（分割）: Human Query 与高分辨率 pixel embedding map 做点积，生成实例感知的逐像素分类
    - **分类任务**（性别、年龄）: Human Query 直接映射到分类预测结果

### 损失函数 / 训练策略

- **检测头**: 类别分类 + bbox 回归（L1 + GIoU）
- **分割头**: Dice loss + Focal loss（参考 Mask DINO）
- **姿态头**: 关键点坐标回归 + 置信度预测 + 辅助 heatmap loss（训练时用）
- **属性头**: 性别—二分类BCE；年龄—85类分类 + softmax expected value 估计
- **3D Mesh 头**: SMPL pose/shape 参数回归
- **Contrastive DeNoising (CDN)**: 仅对检测任务应用去噪训练加速收敛
- 训练配置：100 epoch，COCO-UniHuman train，DINO 数据增强策略

### COCO-UniHuman 数据集

基于 COCO 数据集扩展：
- 200K 图像，273K 人体实例
- 新增标注：**性别**（body-based 标注）、**表观年龄**（两阶段策略：先粗粒度年龄段→再精细表观年龄，10 名标注员投票取平均）、**3D Mesh**（EFT 方法生成 SMPL 伪 GT）
- 首个同时覆盖分类（性别/年龄）、检测（人体/人脸）、分割、姿态（2D/3D）的多人多任务基准

## 实验关键数据

### 主实验（COCO-UniHuman val）

| 模型 | Backbone | Det. AP | Seg. AP | Pose AP | Gender AP | Age AP |
|------|----------|---------|---------|---------|-----------|--------|
| Faster R-CNN | R-50 | 65.3 | — | — | — | — |
| DINO | R-50 | 73.3 | — | — | — | — |
| Mask DINO | R-50 | 72.3 | 64.8 | — | — | — |
| Mask R-CNN | R-50 | 66.7 | 58.4 | — | — | — |
| PETR | R-50 | — | — | 68.8 | — | — |
| ViTPose† (top-down) | ViT-L | — | — | 78.2 | — | — |
| **HQNet (D+S+P+C)** | **R-50** | **74.9** | **65.8** | **69.3** | **56.0** | **53.8** |
| **HQNet** | **Swin-L** | **77.3** | **68.1** | **72.6** | **57.9** | **56.2** |
| **HQNet** | **ViT-L** | **78.0** | **68.6** | **75.3** | **58.0** | **58.0** |

### OCHuman 数据集（拥挤遮挡场景）

| 模型 | Backbone | Det. AP | Seg. AP | Pose AP | 说明 |
|------|----------|---------|---------|---------|------|
| Mask R-CNN | R-50-FPN | — | 16.9 | — | 两阶段方法 |
| CondInst | R-50-FPN | — | 20.1 | — | 单阶段分割 |
| SBL† (top-down) | R-50 | — | — | 30.4 | 自上而下姿态 |
| HrHRNet† (bottom-up) | HRNet-w32 | — | — | 39.4 | 自下而上姿态 |
| CID† | HRNet-w32 | — | — | 44.0 | 单阶段姿态 |
| **HQNet** | **R-50** | **29.5** | **31.1** | **40.0** | 大幅超越同量级 |
| **HQNet** | **ViT-L** | **35.8** | **38.8** | **45.6** | 全面 SOTA |

### 3D Mesh 恢复消融

| 方法 | Backbone | MPJPE↓ | PA-MPJPE↓ | 说明 |
|------|----------|--------|-----------|------|
| HMR | R-50 (GT bbox) | 109.62 | 72.03 | 使用 GT 框 |
| HMR+ | R-50 (GT bbox) | 78.06 | 50.36 | 使用 GT 框 |
| ROMP | R-50 | 119.52 | 72.27 | 单阶段无 GT 框 |
| HQNet w/o GaMS | R-50 | 87.00 | 54.92 | 无性别辅助 |
| **HQNet w/ GaMS** | **R-50** | **84.74** | **50.80** | 性别辅助选模型 |
| **HQNet** | **ViT-L** | **76.31** | **48.26** | 最强配置 |

### 泛化性验证

| 新任务 | 方法 | 关键指标 | 说明 |
|--------|------|----------|------|
| 人脸检测 (finetune) | Faster R-CNN | AP 43.9 | 从头训练 |
| 人脸检测 (finetune) | ZoomNet | AP 58.2 | 专用模型 |
| 人脸检测 (finetune) | **HQNet (R-50)** | **AP 68.4** | 冻结 backbone 微调 |
| MOT (zero-shot) | FairMOT (finetune) | IDF1 63.2 | 需在 MOT 数据上微调 |
| MOT (zero-shot) | **HQNet (R-50)** | **IDF1 64.6** | 无需 MOT 训练数据 |
| MOT (zero-shot) | **HQNet (ViT-L)** | **IDF1 69.1** | 无需 MOT 训练数据 |

### 关键发现

- **多任务协同有效**: HQNet (D+S+P+C) 全任务联合训练比单任务子集（D+S、D+S+P）在各指标上均有提升，证明任务间存在正向协同
- **HQ-Ins Matching 关键**: 使用多任务联合匹配避免了检测与姿态匹配到不同人的错误
- **GaMS 有效**: 利用性别预测选择 SMPL 模型使 MPJPE 从 87.00 降到 84.74
- **强泛化能力**: Human Query 在未见过的人脸检测和 MOT 任务上表现出色，证明学到的表示具有通用性
- **高效可扩展**: 多任务共享 backbone、encoder 和 decoder，每个新任务仅增加轻量头，参数开销极小
- **R-50 的 HQNet (COCO 训练) 在 OCHuman 上大幅超越同 backbone 的专用方法**: Seg AP 31.1 vs CondInst 20.1，Pose AP 40.0 vs SBL 30.4

## 亮点与洞察

- **"一个 query 搞定一切"** 的设计思想简洁而强大——Human Query 同时编码外观、位置、结构信息，各任务按需解码
- **COCO-UniHuman 数据集的长期价值**: 首个覆盖四大类七种 HCP 任务的统一多人基准，填补了社区空白。body-based 年龄标注和两阶段标注策略本身也是有价值的贡献
- **共享解码器的设计选择**: 不同于之前工作用任务专用解码器，HQNet 用完全共享的解码器，最大化知识共享的同时保持可扩展性
- **零样本 MOT 能力**: Human Query 未经 MOT 训练却能作为 Re-ID 特征实现竞争性跟踪性能，说明学到的表示具有深层次的实例判别能力
- **实用性**: 单阶段推理效率不随人数增长，适合实际部署

## 局限与展望

- 数据集的性别分布不均衡（男:女 ≈ 65:35），可能引入偏差
- 年龄分布偏向 25-35 岁，对儿童和老年人的估计能力可能不足
- 3D Mesh GT 使用 EFT 伪标注而非真实 GT，质量受限
- 与 top-down 方法（尤其是 ViTPose）相比，姿态估计仍有差距（single-stage 的固有局限）
- 属性预测（性别/年龄）的绝对 AP 值偏低（56-58%），room for improvement 较大
- Small category person 被排除在评估之外，远距离小目标人体感知未被充分评估
- HQ-Ins Matching 增加了匹配的计算复杂度

## 相关工作与启发

- **DINO** [ICLR 2023]: HQNet 的主要技术基础，Mixed Query Selection + CDN + Deformable Attention
- **Mask DINO** [CVPR 2023]: 统一检测与分割的 DETR 框架，HQNet 的分割头设计参考了此工作
- **UniHCP** [CVPR 2023]: 大规模多任务人体预训练，但仍是单任务推理；HQNet 实现了真正的单阶段多任务推理
- **PETR** [CVPR 2022]: 单阶段多人姿态估计，HQNet 的姿态头参考其 joint decoder layer 设计
- **启发**: 共享 query 是实现多任务统一的有效范式，可扩展到更多以实体为中心的感知任务（如手部姿态、动作识别等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 统一 query 思想虽然延续 DETR 路线但在 HCP 多任务统一上首次系统实现，COCO-UniHuman 基准有开创性
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个任务、3 个数据集（COCO-UniHuman/OCHuman/PoseTrack21/Human-Art）、与大量 baseline 对比、泛化性验证全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，Table 1 的数据集对比和 Table 2 的全面实验排版专业，但部分细节需参考 supplementary
- **价值**: ⭐⭐⭐⭐⭐ 数据集 + 方法 + 基准三位一体，对统一人体感知领域有重要推动作用，Human Query 的泛化能力展示了其作为通用人体表示的潜力

<!-- RELATED:START -->

## 相关论文

- [OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](../../NeurIPS2025/segmentation/hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)
- [Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)
- [UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](../../NeurIPS2025/segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)

<!-- RELATED:END -->
