---
title: >-
  [论文解读] Fast Data Attribution for Text-to-Image Models
description: >-
  [NeurIPS 2025][图像生成][数据归因] 将精确但缓慢的 Attribution by Unlearning 方法蒸馏到一个轻量特征嵌入空间中，通过 learning-to-rank 训练使得简单的余弦相似度检索就能近似昂贵的归因排序，首次在 Stable Diffusion + LAION-400M 规模上实现毫秒级数据归因。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "数据归因"
  - "文生图模型"
  - "learning-to-rank"
  - "特征蒸馏"
  - "高效检索"
---

# Fast Data Attribution for Text-to-Image Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.10721](https://arxiv.org/abs/2511.10721)  
**代码**: [https://peterwang512.github.io/FastGDA](https://peterwang512.github.io/FastGDA)  
**领域**: 图像生成  
**关键词**: 数据归因, 文生图模型, learning-to-rank, 特征蒸馏, 高效检索

## 一句话总结

将精确但缓慢的 Attribution by Unlearning 方法蒸馏到一个轻量特征嵌入空间中，通过 learning-to-rank 训练使得简单的余弦相似度检索就能近似昂贵的归因排序，首次在 Stable Diffusion + LAION-400M 规模上实现毫秒级数据归因。

## 研究背景与动机

数据归因（Data Attribution）的目标是：给定文生图模型生成的一张图像，找出在训练集中对该生成结果影响最大的训练图像。这个问题对创作者补偿模型、版权追溯等实际应用至关重要。

现有方法面临一个核心困境——效率与精确度的权衡：

- **影响函数方法**（TRAK、D-TRAK）：需要预存所有训练样本的梯度信息，存储开销巨大（30-290 GB），单次查询需数十秒到数分钟，且降维投影会损害精度
- **Unlearning 方法**（AbU）：通过对生成图像做"遗忘"操作然后检测哪些训练图像被影响来计算归因，精度高但速度极慢——单次查询需 2 小时以上
- **现成特征检索**（DINO、CLIP）：毫秒级检索速度，但仅基于视觉/语义相似度，无法真正反映因果层面的数据影响力
- **经济不可行**：文生图平台每张图收费 5-10 美分，而现有归因方法的计算成本可能高出几个量级

核心洞察在于：是否可以将慢方法的精确归因能力"教"给一个快速的特征检索系统？

## 方法详解

### 整体框架

整个方法分为离线训练和在线部署两个阶段。离线阶段利用 AbU+ 作为教师生成大量归因排名数据，通过 learning-to-rank 训练一个归因专用的特征嵌入网络。在线阶段仅需计算查询图像的特征嵌入并使用 FAISS 等索引进行快速相似度检索。

### 关键设计

1. **Attribution by Unlearning+ (AbU+)**：对预训练模型执行 certified unlearning——通过一步 Newton 更新最大化生成图像的损失（相当于"遗忘"该图像），然后检测哪些训练图像的重建损失因此增大。归因分数定义为 $\tau(\hat{\mathbf{z}}, \mathbf{z}) = \mathcal{L}(\mathbf{z}, \theta_{-\hat{\mathbf{z}}}) - \mathcal{L}(\mathbf{z}, \theta_0)$。相比原始 AbU，用 EK-FAC（Eigenvalue-corrected Kronecker Factorization）替代对角 Fisher 近似来逆转 Fisher 信息矩阵，显著提升归因质量。然而 AbU+ 需要对每个训练样本执行前向传播，10 万训练图像需 2 小时。

2. **两阶段数据收集策略**：直接对整个训练集计算归因分数计算量过大。观察到大多数训练样本对给定生成图像没有实质性贡献，因此先用现成 DINO 特征检索 K=10000 个最近邻，再只对这些近邻计算 AbU+ 归因分数。这将每个查询的数据收集成本从 O(N) 降至 O(K)，只要近邻集包含真正有影响的图像即可。

3. **归因专用特征学习**：特征嵌入 $f_\psi = g_\psi \circ \phi$，其中 $\phi$ 是预训练编码器（DINO 图像编码器 + CLIP 文本编码器拼接），$g_\psi$ 是三层 MLP。使用余弦相似度预测归因排名 $r_\psi(\hat{\mathbf{z}}, \mathbf{z}_i) = \cos(f_\psi(\hat{\mathbf{z}}), f_\psi(\mathbf{z}_i))$，通过 BCE 损失训练：归因排名归一化为 $[1/K, 1]$ 区间的标签，经带可学习仿射缩放的 sigmoid 变换后计算交叉熵。选择 BCE 而非 MSE 回归（不收敛）或 ordinal loss（不支持快速余弦检索）。

### 损失函数 / 训练策略

- **BCE 排名损失**：$\mathcal{L}(\psi, \alpha, \beta) = \mathbb{E} [\ell_{\text{BCE}}(\pi_{\hat{\mathbf{z}}}^i, \sigma_{\alpha,\beta}(r_\psi(\hat{\mathbf{z}}, \mathbf{z}_i)))]$，其中 $\sigma_{\alpha,\beta}(x) = 1/(1+e^{-(\alpha x + \beta)})$ 带两个可学习的仿射参数
- **负样本注入**：以概率 0.1 从非近邻集合随机采样训练样本并赋予最差排名（rank=1），帮助模型区分无关图像——0.1 比例下 mAP 从 0.709 提升至 0.724，过高则损害细粒度排名学习
- **近邻子采样**：每次迭代仅用 $M \approx 0.1K$ 个候选训练，大幅减少数据收集计算量而几乎不损失排名精度
- **数据规模研究**：增加查询数量比增加每查询的候选数更有效——在固定 245 万归因对的预算下，更多查询 + 更少候选的配置表现更好

## 实验关键数据

### 主实验：MSCOCO 反事实评估

在 10 万 MSCOCO 训练集上，对 110 个生成图像查询进行 leave-K-out 反事实测试（移除 K 个归因最高的训练图像后重训模型）：

| 方法 | 延迟 | 存储 | ΔL(k=500)↑ | ΔL(k=4000)↑ | MSE(k=500)↑ | CLIP(k=500)↓ |
|------|------|------|-----------|-----------|------------|------------|
| Random | — | — | 3.51 | 3.47 | 4.09 | 7.86 |
| D-TRAK | 46.7s | 30GB | 5.44 | 9.59 | **5.86** | 7.31 |
| AbU+ | 2.28hr | 1.9GB | **5.83** | **10.70** | 5.64 | **7.15** |
| DINO | 11.6ms | 354MB | 4.76 | 8.06 | 4.51 | 7.41 |
| **Ours** | **18.7ms** | **354MB** | **5.28** | **9.35** | 4.78 | 7.37 |

在快速方法（延迟 < 生成时间 21.5s）中归因性能最优；比 D-TRAK 快 2500×，比 AbU+ 快 400000×。

### 消融实验

**特征空间选择**：

| 特征 | Tuning 前 mAP | Tuning 后 mAP |
|------|-------------|-------------|
| CLIP-Text | 较高 | 中等 |
| DINO | 中等 | 较高 |
| DINO + CLIP-Text | — | **最高** |

Tuning 前文本特征更好，Tuning 后图像特征反超——说明视觉信息对归因更本质，但需要归因专用训练来激活。最终采用 DINO + CLIP-Text 拼接。

**损失函数对比**：MSE 回归无法收敛；ordinal loss 排名精度与 BCE 相当但不支持快速余弦检索；BCE 在精度和效率间最优。

**数据规模**：性能随查询数量增加快速提升后趋于饱和，数千个查询即可捕获大部分排名信号。

### Stable Diffusion 规模验证

在 Stable Diffusion v1.4 + LAION-400M 上验证：

- 对每个查询检索 10 万个近邻候选，总计收集 1.01 亿归因对用于训练
- Tuning 后的 DINO+CLIP-Text 特征在所有 mAP 阈值上显著提升
- 与 MSCOCO 不同，文本特征在 SD 模型上对归因更关键——可能因为 AbU+ 的归因分数与文本相似度的相关性更强
- 数据仍未饱和，更多计算预算可进一步提升

### 关键发现

- 归因专用特征可以从慢方法中有效蒸馏——18.7ms 的检索保持了数小时级别方法的大部分归因精度
- 特征空间的选择和组合对结果影响巨大——单纯视觉或文本特征都不够，融合后效果最好
- 两阶段收集+子采样的组合将数据收集效率提升了数十倍，使大规模数据收集实际可行

## 亮点与洞察

- **蒸馏思路的优雅性**：不改变检索机制本身（仍是余弦相似度），只优化特征空间使其对齐归因排名——部署零开销
- **首次大规模验证**：在 LAION-400M 级别训练的 Stable Diffusion 上成功应用数据归因，证明方法可扩展
- **AbU+ 本身的贡献**：EK-FAC 替代对角 Fisher 近似是独立有价值的改进
- **系统性设计研究**：对特征选择、损失函数、数据规模、采样策略的全面消融为后续工作提供了清晰的设计指导

## 局限与展望

1. 蒸馏只保留排名信息，丢失了影响力的绝对大小和集中/分散程度——无法区分"恰好被一张训练图强烈影响"和"被多张训练图温和影响"
2. 教师方法 AbU+ 的系统性偏差会被继承到学生模型中
3. 离线数据收集仍需大量 GPU 时间——MSCOCO 需约 1470 GPU 小时，SD 需约 17250 GPU 小时
4. 仅在 diffusion model 上验证，对 flow matching、one-step 模型等新架构的适用性未探索
5. 训练模型的 MLP 很轻量（3 层），但性能可能受限于预训练特征的瓶颈

## 相关工作与启发

- **TRAK/D-TRAK**：基于梯度投影的影响函数方法，速度适中但存储大且精度受限于投影维度
- **AbU**：本文教师方法的前身，精确但 2 小时/查询不实用
- **AbC（Wang et al. 2023）**：在 customization 场景下调整特征用于归因，本文将此思路推广到通用归因
- **FAISS**：高效近似最近邻检索库，使本方法的数亿级训练集检索成为可能
- **启发**：蒸馏+检索的范式可推广到其他需要"慢准 vs 快粗"权衡的场景，如语义搜索、推荐系统

## 评分

⭐⭐⭐⭐ (4/5)

理由：首次将数据归因扩展到 Stable Diffusion + LAION-400M 规模并实现毫秒级部署，蒸馏+检索的技术方案优雅且实用。系统性的设计消融研究为后续工作提供了清晰指导。主要扣分点在于数据收集阶段仍需大量 GPU 投入，且蒸馏损失了归因的定量信息。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)

</div>

<!-- RELATED:END -->
