---
title: >-
  [论文解读] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective
description: >-
  [CVPR 2026][图像生成][dataset condensation] 提出 D2C（Diffusion Dataset Condensation）——首个面向扩散模型的数据集压缩框架，通过"Select + Attach"两阶段流水线，在仅使用 ImageNet 0.8%–8% 数据的条件下实现 100–233× 的训练加速，同时保持高质量图像生成能力。
tags:
  - CVPR 2026
  - 图像生成
  - dataset condensation
  - 扩散模型
  - training acceleration
  - data-centric
  - efficient training
---

# Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective

**会议**: CVPR 2026  
**arXiv**: [2507.05914](https://arxiv.org/abs/2507.05914)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: dataset condensation, diffusion model, training acceleration, data-centric, efficient training

## 一句话总结

提出 D2C（Diffusion Dataset Condensation）——首个面向扩散模型的数据集压缩框架，通过"Select + Attach"两阶段流水线，在仅使用 ImageNet 0.8%–8% 数据的条件下实现 100–233× 的训练加速，同时保持高质量图像生成能力。

## 研究背景与动机

**扩散模型训练代价高昂**：当前扩散模型（DiT、SiT 等）通常需要数百万张图像和数百万迭代步才能训练出高质量模型，SiT-XL/2 在 ImageNet 上需要 7M 步、REPA 也需 4M 步，训练耗时数百 GPU·hours。

**数据集压缩聚焦于判别任务**：已有数据集蒸馏/压缩方法（如 SRe2L、RDED、Herding、K-Center）几乎全部面向分类等判别式任务设计，直接迁移到扩散模型训练时效果极差（RDED 在 DiT-L/2 上 FID 高达 166.2）。

**判别特征≠生成特征**：像素级蒸馏方法合成的图像偏向类别判别性特征，缺少对数据分布多样性和语义结构的保留，导致生成质量崩塌、收敛不稳定。

**简单剪枝策略次优**：随机采样或几何方法（K-Center、Herding）无法针对扩散模型的去噪特性进行难度感知选择，性能提升有限。

**数据侧高效训练缺乏系统研究**：已有高效训练工作（架构优化、注意力加速、表征对齐等）多在模型侧发力，从数据角度系统构建压缩子集用于扩散训练的路线尚属空白。

**类别嵌入语义贫乏**：C2I（类别到图像）合成中常用的从零学习类别嵌入缺少固有语义信息，限制了条件生成的表达力，特别是在数据极度受限时更为明显。

## 方法详解

### 整体框架

D2C（Diffusion Dataset Condensation）采用两阶段流水线：**Select** 阶段从大规模训练集中筛选出紧凑、多样且可学习的子集；**Attach** 阶段为子集中每张图像附加丰富的语义与视觉表征信息。最终在压缩后的增强数据集上从头训练扩散模型，结合去噪损失和表征对齐损失进行联合优化。

### 关键设计 1：Diffusion Difficulty Score + Interval Sampling（Select 阶段）

- **功能**：为每张训练图像计算"扩散难度分数"，然后在排序后的列表中进行等间隔采样。
- **核心思路**：利用预训练扩散模型的类条件后验概率 $p_\theta(\mathbf{c}|\mathbf{x})$ 对样本难度进行排序。通过贝叶斯公式简化，难度分数定义为负条件似然（即扩散去噪损失）：$s_{\text{diff}}(\mathbf{x}) = -\mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})\|_2^2]$。分数越高，样本对模型去噪越困难。在每个类别内按难度升序排列后，以固定间隔 $k$ 采样，兼顾简单样本的可学习性和困难样本的多样性。
- **设计动机**：仅选最简单样本（Min）虽收敛快但多样性不足，仅选最难样本（Max）噪声大难优化；等间隔采样实现难度分布的均匀覆盖，在 0.8% 数据预算下 $k=96$、4% 预算下 $k=16$ 取得最优。

### 关键设计 2：Dual Conditional Embedding（DC-Embedding，Attach 阶段·语义信息）

- **功能**：将预训练文本编码器（T5-encoder）生成的类别文本描述嵌入与可学习类别嵌入融合，作为扩散模型的条件输入。
- **核心思路**：对每个类别生成描述性 prompt（如"a photo of a cat"），经文本编码器提取文本嵌入 $t_c$ 和文本掩码 $t_{\text{mask}}$，再通过 1D 卷积 + 残差 MLP 与可学习类别嵌入 $e_c$ 融合：$y_{\text{text}} = \text{MLP}(\tilde{t}_c) + \tilde{t}_c + e_c$。文本嵌入预计算存盘，训练时直接加载。
- **设计动机**：从零学习的类别嵌入在数据受限时语义信息不足；引入预训练文本编码器的丰富语义（尤其类间区分能力）能显著改善条件生成质量，同时保留可学习嵌入的灵活性。

### 关键设计 3：Visual Information Injection（Attach 阶段·视觉信息）

- **功能**：用预训练视觉编码器（DINOv2）为每张选中图像提取实例级视觉表征，存盘后在训练时通过表征对齐损失注入扩散模型。
- **核心思路**：DINOv2 提取每张图像的 patch 级语义特征 $y_{\text{vis}} \in \mathbb{R}^{N \times d}$，截取前 $h$ 个 token 作为紧凑表征。训练时，从扩散模型中间层取出 token 特征 $\{h_i\}$，经投影头映射后与视觉表征计算余弦对齐损失：$\mathcal{L}_{\text{proj}} = -\frac{1}{h}\sum_i \langle \frac{\phi(h_i)}{\|\phi(h_i)\|}, \frac{v_i}{\|v_i\|} \rangle$。
- **设计动机**：语义嵌入主要提供类间结构区分，但类内多样性（纹理、姿态等）需要实例级视觉信息才能捕获；借鉴 REPA 的表征对齐策略，为扩散模型注入空间一致性先验，在极小数据集上尤其关键。

## 损失函数

总训练目标由两部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \cdot \mathcal{L}_{\text{proj}}$$

- **去噪损失** $\mathcal{L}_{\text{diff}}$：标准扩散模型噪声预测 MSE，条件为类别标签 $y$ 和文本信息 $y_{\text{text}}$
- **表征对齐损失** $\mathcal{L}_{\text{proj}}$：扩散模型中间层 token 与 DINOv2 视觉表征的余弦相似度对齐
- 平衡权重 $\lambda = 0.5$

## 实验

### 表 1：ImageNet 256×256 不同数据预算下 gFID-50K 对比（CFG=1.5）

| 数据预算 | 迭代步数 | DiT-L/2 Random | DiT-L/2 D2C | SiT-L/2 Random | SiT-L/2 D2C |
|---------|---------|----------------|-------------|----------------|-------------|
| 0.8% (10K) | 100k | 35.86 | **4.20** | 4.35 | **3.98** |
| 0.8% (10K) | 300k | 4.19 | **4.13** | 4.33 | **3.98** |
| 4.0% (50K) | 100k | 36.78 | **14.81** | 31.13 | **11.21** |
| 4.0% (50K) | 300k | 11.55 | **5.99** | 14.18 | **5.66** |
| 8.0% (100K) | 100k | 41.02 | **22.55** | 36.64 | **15.01** |
| 8.0% (100K) | 300k | 11.49 | **6.49** | 12.56 | **5.65** |

D2C 在所有预算和架构设置下均大幅优于 Random、K-Center、Herding 等基线。尤其在 0.8% 预算下，D2C 仅需 100k 步即可达到 Random 300k 步的水平。

### 表 2：与 SRe2L / RDED 的对比（0.8% 数据，DiT-L/2）

| 方法 | gFID↓ | sFID↓ | IS↑ | Precision↑ |
|------|-------|-------|-----|-----------|
| RDED | 166.2 | 60.1 | 10.8 | 0.09 |
| SRe2L | 104.2 | 20.2 | 14.1 | 0.20 |
| **D2C** | **4.2** | **11.0** | **283.6** | **0.72** |

面向判别任务设计的压缩方法在扩散训练中彻底失效，FID 比 D2C 高出 1–2 个数量级。

### 核心加速结果

使用 SiT-XL/2，D2C 在仅 0.8% 数据（10K 张）、40k 步时即达 FID 4.3，对比 REPA（4M 步）加速 **100×**、对比 vanilla SiT（7M 步）加速 **233×**。在 4% 数据（50K 张）+ CFG=1.5 下，180k 步达到 FID 2.78。

### 消融实验亮点

- **Select 阶段**：单独使用将 gFID 从 37.07 降至 14.96
- **Attach 阶段**：DC-Embedding 单独加入降至 9.01，视觉表征单独加入降至 10.37，两者组合降至 **7.62**
- **间隔 $k$ 的选择**：最优 $k$ 与数据预算大致成反比（10K→$k$=96，50K→$k$=16）
- **Wall-clock**：Attach-only 模式仅需 7.4h（REPA 的 0.99%），完整流水线 9.5h（REPA 的 1.27%）

## 亮点

1. **首创面向扩散模型的数据集压缩**：填补了生成式任务数据集压缩的空白，揭示了判别式压缩方法不可直接迁移的关键发现。
2. **极端压缩比下的强性能**：0.8% 数据即可训练出 FID 3.98 的 SiT-L/2，证明扩散模型训练中存在巨大的数据冗余。
3. **模块化设计清晰**：Select 和 Attach 两阶段可独立使用，即使去掉 Select（仅 Attach）也能超越 REPA，灵活性强。
4. **跨架构跨分辨率泛化**：DiT/SiT × L/XL × 256/512 全面验证，结果一致。
5. **实际加速显著**：Wall-clock 端到端仅为 REPA 的 ~1%，具备真实落地意义。

## 局限性

1. **依赖预训练模型**：需要预训练扩散模型计算难度分数 + T5 编码器 + DINOv2 编码器，方法的独立性不够强，冷启动成本被隐藏。
2. **仅验证 C2I 设置**：主实验均为类别条件 ImageNet 生成，T2I（文本到图像）仅在附录简要提及，大规模 T2I 场景下的效果未充分验证。
3. **间隔超参需调优**：最优间隔 $k$ 与数据预算相关且非平凡，需要额外实验确定。
4. **分辨率上限**：实验最高 512×512，对于当前主流的 1024+ 分辨率生成未做验证。
5. **数据扩展性存疑**：压缩比从 0.8% 到 8% 时性能差异缩小，更大数据量下方法的边际收益是否递减尚不明确。

## 相关工作

- **REPA**（Yu et al.）：通过对齐扩散模型中间表征与预训练视觉编码器加速训练，D2C 的视觉注入模块受此启发但进一步结合了数据筛选。
- **SRe2L / RDED**：像素级/图像级数据集蒸馏的代表方法，面向分类任务，本文实验证明其不适用于扩散训练。
- **InfoBatch / Patch-based methods**：数据侧高效训练的另一条路线，通过重采样/分块加速但不构建压缩子集。
- **Li et al. (2025)**：从 coreset 选择角度研究扩散训练数据剪枝，但不附加额外信息且仅在较小规模验证。
- **DiT / SiT**：本文的主要实验骨干架构，D2C 作为正交的数据侧策略可与之自由组合。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统研究面向扩散模型的数据集压缩问题，Select+Attach 框架设计合理
- **实验充分度**: ⭐⭐⭐⭐ — 多架构、多分辨率、多预算全面对比+详细消融，wall-clock 分析透明
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、公式严谨、图表信息量大，结构易读
- **价值**: ⭐⭐⭐⭐ — 实际加速 100×+ 极具工程意义，为数据-模型协同优化开辟新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](d2c_diffusion_dataset_condensation.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[CVPR 2026\] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_f.md)

<!-- RELATED:END -->
