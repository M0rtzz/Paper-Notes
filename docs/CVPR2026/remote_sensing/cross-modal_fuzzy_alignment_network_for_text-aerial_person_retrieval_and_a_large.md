---
title: >-
  [论文解读] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark
description: >-
  [CVPR 2026][遥感][文本-航拍行人检索] 提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。
tags:
  - CVPR 2026
  - 遥感
  - 文本-航拍行人检索
  - 模糊逻辑
  - 跨模态对齐
  - 无人机
  - Chain-of-Thought 标注
---

# Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark

**会议**: CVPR 2026  
**arXiv**: [2603.20721](https://arxiv.org/abs/2603.20721)  
**代码**: [https://github.com/Yifei-AHU/AERI-PEDES](https://github.com/Yifei-AHU/AERI-PEDES) (有)  
**领域**: 遥感 / 行人检索  
**关键词**: 文本-航拍行人检索, 模糊逻辑, 跨模态对齐, 无人机, Chain-of-Thought 标注

## 一句话总结
提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。

## 研究背景与动机

**领域现状**：文本-图像行人检索（TIPR）已取得显著进展，但所有现有工作均基于固定地面摄像头数据。无人机（UAV）提供了动态多角度监控的独特优势，将 TIPR 扩展到航拍场景具有重大研究价值。

**现有痛点**：(1) 航拍图像因拍摄角度和高度剧烈变化导致人物外观非线性畸变；(2) 航拍视图中行人视觉线索稀疏或部分缺失（如只能看到头顶），文本描述包含的完整属性无法与航拍图像完全对应；(3) 在做 token 级细粒度对齐时，不可观察的 token 引入错误跨模态对齐。

**核心矛盾**：目击者描述通常详细完整，但航拍图像只能覆盖部分语义区域——这种可见性不一致导致细粒度对齐时产生大量噪声匹配。

**本文要解决**：如何在航拍图像视觉线索不完整的情况下实现鲁棒的文本-航拍行人跨模态检索？

**切入角度**：(1) 用模糊逻辑量化每个 token 的可靠程度，抑制不可观察/噪声 token 的影响；(2) 用地面视图作为中间桥梁，自适应平衡直接对齐和桥接对齐。

**核心 idea**：模糊隶属度建模 token 可靠性 + 上下文感知动态对齐 = 鲁棒的文本-航拍对齐。

## 方法详解

### 整体框架
CFAN 包含两个核心模块：Context-Aware Dynamic Alignment (CDA) 用于样本级自适应对齐，Fuzzy Token Alignment (FTA) 用于 token 级细粒度对齐。共享 CLIP 图像编码器提取航拍/地面特征，CLIP 文本编码器提取描述特征。

### 关键设计

1. **Context-Aware Dynamic Alignment (CDA)**：

    - **功能**：计算文本-航拍和文本-地面的余弦相似度差异 $\Delta_i = \text{sim}(T_i^C, A_i^C) - \text{sim}(T_i^C, G_i^C)$，通过 sigmoid 映射为软决策门 $\alpha_i$，自适应加权直接对齐和桥接对齐。
    - **核心公式**：$\alpha_i = \frac{1}{1 + \exp[-k \cdot \Delta_i]}$
    - **损失**：$\mathcal{L}_{\text{CDA}} = \frac{1}{B} \sum_{i=1}^B [\alpha_i \cdot \mathcal{L}_{\text{direct}} + (1-\alpha_i) \cdot \mathcal{L}_{\text{bridge}}]$
    - **设计动机**：不同航拍图像的对齐难度不同——低空近拍的直接对齐即可，高空远拍的需要地面桥接。$\alpha_i$ 自动估计"对齐难度"并分配对齐策略。桥接对齐中对地面特征做 stop-gradient 防止干扰地面表征。

2. **Fuzzy Token Alignment (FTA)**：

    - **功能**：用共享可学习 query $\mathbf{Q} \in \mathbb{R}^{K \times D}$ 分别与两个模态做 cross-attention，得到模态感知查询表征。然后用高斯模糊隶属函数计算每个 token 的可靠性：
    - **核心公式**：$\mu_j^a = \exp(-\frac{(1-r_j)^2}{2\sigma^2})$，其中 $r_j$ 是 query token 与全局 class token 的余弦相似度。
    - 融合两个模态的隶属度：$\mu_j^{\text{joint}} = \mu_j^a \cdot \mu_j^t$（模糊 AND 运算）
    - 加权相似度：$\text{sim}(Q_a, Q_t) = \frac{1}{K} \sum_{j=1}^K \mu_j^{\text{joint}} s_j$
    - **设计动机**：只有在两个模态中都高可靠的 token 才对齐——即双模态可见且语义一致的部分贡献最大，不可观察/噪声 token 被自然抑制。高斯尺度 $\sigma$ 从全局 class token 自适应预测，使模型能根据图像内容调整可靠性阈值。

3. **AERI-PEDES 数据集构建**：

    - 用 Chain-of-Thought 分解文本生成：属性解析→初始标注→审核精化
    - 训练集用 VLM 生成标注，测试集人工标注确保评估可靠性

### 损失函数 / 训练策略
- CDA 直接对齐和桥接对齐均用 SDM（Similarity Distribution Matching）损失
- FTA 用 KL 散度实现 similarity distribution matching
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{CDA}} + \mathcal{L}_{\text{FTA}}$
- Adam 优化器，初始学习率 $5 \times 10^{-6}$，cosine decay，60 epochs

## 实验关键数据

### 主实验

| 方法 | AERI-PEDES R1↑ | AERI-PEDES mAP↑ | TBAPR R1↑ | TBAPR mAP↑ |
|------|---------------|-----------------|-----------|------------|
| IRRA (CVPR23) | 35.14 | 33.42 | 39.63 | 35.31 |
| HAM (CVPR25) | 44.58 | 42.45 | 47.81 | 41.86 |
| CFAN (无地面) | 45.06 | 43.27 | 49.15 | 42.89 |
| **CFAN (有地面)** | **47.16** | **44.79** | **49.47** | **43.96** |

### 消融实验

| 配置 | R1 | mAP | RSum | 说明 |
|------|-----|------|------|------|
| Baseline (仅桥接) | 43.88 | 41.58 | 174.84 | 基线 |
| + CDA | 46.18 | 43.98 | 183.04 | RSum +8.2% |
| + FTA | 44.55 | 41.89 | 176.64 | R1 +0.67% |
| + CDA + FTA | **47.16** | **44.79** | **186.65** | 全部合计 |

桥接模态对比：

| 桥接方式 | R1 | mAP | 说明 |
|---------|-----|------|------|
| None（无桥接） | 45.06 | 43.27 | 仅 FTA |
| Aerial（低空航拍桥接） | 46.08 | 44.20 | 有效但有限 |
| Ground（地面视图桥接） | **47.16** | **44.79** | 最优 |

### 关键发现
- CDA 贡献最大（RSum 提升 8.2%），说明自适应平衡直接/桥接对齐是核心
- FTA 提供了补充的细粒度对齐改进
- 即使无地面图像，仅用 FTA 的 CFAN 也已超越所有竞品

## 亮点与洞察
- **模糊逻辑与深度学习的结合**：用模糊隶属函数量化 token 可靠性是一个优雅的设计，比简单的注意力权重更有理论基础
- **大规模数据集贡献**：AERI-PEDES（112K+ 图像）填补了文本-航拍行人检索的数据空白
- **CoT 标注流水线**：结构化推理步骤的标注方法可推广到其他数据集构建

## 局限与展望
- 需要同一行人的航拍和地面配对图像，实际部署中可能难以获取
- 模糊隶属函数为高斯形式，更灵活的参数化可能更好
- 仅使用 CLIP 编码器，更大更强的 VLM 可能带来进一步提升

## 相关工作与启发
- 模糊深度学习在医学图像分析中已有应用，本文将其推广到跨模态检索
- CDA 中的动态对齐难度估计思路可推广到其他不完整视觉信息的检索场景（如雾天、遮挡）

## 评分
- 新颖性: ⭐⭐⭐⭐ 模糊逻辑+动态桥接对齐的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集+完整消融+参数敏感性分析
- 写作质量: ⭐⭐⭐⭐ 公式清晰，结构完整
- 价值: ⭐⭐⭐⭐ 数据集和方法对文本-航拍行人检索领域有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[ECCV 2024\] Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach](../../ECCV2024/remote_sensing/cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach.md)
- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)
- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](../../AAAI2026/remote_sensing/asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)

<!-- RELATED:END -->
