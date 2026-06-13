---
title: >-
  [论文解读] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers
description: >-
  [ICCV 2025][图像生成][CLIP增强] 发现"完美的图像重建并不总带来最佳视觉表征"，提出 GenHancer——一种仅用轻量级随机初始化去噪器（约预训练重型去噪器 1/10 参数）和全局 [CLS] token 条件的两阶段后训练方法，通过自监督重建任务增强 CLIP 的细粒度视觉感知能力，在 MMVP-VLM 上比 DIVA 提升 6.0%。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "CLIP增强"
  - "生成模型"
  - "细粒度视觉"
  - "扩散反馈"
  - "轻量去噪器"
---

# GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers

**会议**: ICCV 2025  
**arXiv**: [2503.19480](https://arxiv.org/abs/2503.19480)  
**代码**: [github (mashijie1028/GenHancer)](https://mashijie1028.github.io/GenHancer)  
**领域**: 视觉表征增强/图像生成  
**关键词**: CLIP增强, 生成模型, 细粒度视觉, 扩散反馈, 轻量去噪器

## 一句话总结

发现"完美的图像重建并不总带来最佳视觉表征"，提出 GenHancer——一种仅用轻量级随机初始化去噪器（约预训练重型去噪器 1/10 参数）和全局 [CLS] token 条件的两阶段后训练方法，通过自监督重建任务增强 CLIP 的细粒度视觉感知能力，在 MMVP-VLM 上比 DIVA 提升 6.0%。

## 研究背景与动机

### 问题定义

CLIP 等判别模型在高层语义理解上表现优异，但在细粒度视觉感知（方向、颜色、数量、视角等）上存在系统性缺陷。这些缺陷会传导到以 CLIP 为视觉编码器的多模态大语言模型（MLLMs），限制其视觉中心任务的表现。

### 已有方法的不足

**视觉专家集成方案（Cambrian 等）**：拼接多个视觉编码器，增加了推理成本和架构复杂性

**DIVA 等扩散反馈方案**：利用预训练的 Stable Diffusion 重型去噪器作为反馈信号增强 CLIP，但：
   - 依赖预训练重型去噪器（SD 的 UNet/DiT），参数量大
   - 未深入探索为何以及何种生成模型能有效增强表征
   - 端到端训练引入了不相关信息，可能降低增强效果

### 核心动机

**反直觉发现**：通过系统实验发现四个关键事实：
1. 增加训练迭代次数→重建变好，但表征可能变差
2. 增大去噪器规模→重建变好，但表征未必更好
3. 加入少量局部 token→重建显著改善，但表征急剧退化
4. 使用预训练去噪器→重建更好，但表征更弱

**一般性哲学**：生成模型同时包含"有用知识"（视觉模式、细节）和"无关信息"（特征空间差距）。有效增强的关键是最大化有用知识的互信息 $I(V;G_1)$，同时最小化无关信息的互信息 $I(V;G_2)$。

## 方法详解

### 整体框架

两阶段后训练管线：Stage-1 冻结 CLIP ViT，训练投影器和去噪器（消除特征空间差距）；Stage-2 用 LoRA 微调 CLIP ViT（学习细粒度视觉知识）。只使用 [CLS] token 作为去噪器的条件输入。

### 关键设计

#### 1. **条件视觉 Token 选择——仅用 [CLS]**

- **功能**：限制条件输入仅为 CLIP 的全局 class token，丢弃所有局部 patch token
- **核心思路**：基于互信息框架，自监督重建等价于最大化 $I(V;G)$。但如果条件中包含局部 token，这些 token 直接对应图像局部区域，导致重建任务过于简单（信息泄漏），$I(V;G_1)$ 变小，ViT 无法从去噪器学到有用信息。
  $$\max_V I(V;G_1) - \lambda I(V;G_2) \Rightarrow \max_V I(V;G_1) + \lambda d(V;V_0)$$
  实验证明即使只加入 10% 的局部 token，增强性能也会急剧下降。
- **设计动机**：仅用 [CLS] token 迫使 ViT 将所有细粒度视觉信息压缩到全局表征中，最大化互信息传递效率。这一发现对连续和离散去噪器均成立。

#### 2. **两阶段训练策略**

- **功能**：将训练分为两个阶段，先消除无关信息再学习有用知识
- **核心思路**：
    - **Stage-1**：冻结 CLIP ViT $\mathbf{v}_\theta$，训练投影器 $\mathbf{h}_\omega$ 和去噪器 $\mathbf{g}_\phi$。去噪器获得基本生成能力，投影器学习桥接特征空间差距，减少 $I(V;G_2)$。
    - **Stage-2**：用 LoRA（rank=16）微调 CLIP ViT，放大 $I(V;G_1)$，增强细粒度表征。实验证明只要 Stage-1 训练充分，Stage-2 中是否继续训练去噪器和投影器影响可忽略。
    - 端到端训练（无 Stage-1）在所有设置上性能下降超过 5%。
- **设计动机**：去噪器随机初始化且轻量，直接端到端训练会在早期给 ViT 引入噪声梯度（$G_2$），损害原有表征。两阶段训练是对 $\min I(V;G_2)$ 的优雅实现。

#### 3. **轻量去噪器与生成范式**

- **功能**：证明轻量级随机初始化的去噪器即可实现优异的增强效果，适用于连续和离散两种生成范式
- **核心思路**：
    - **连续去噪器（Rectified Flow）**：采用 FLUX 风格 DiT 架构但仅用 2 个 MM-DiT + 4 个 Single-DiT blocks（约 FLUX 原版 1/10 参数），通过 adaptive layernorm 注入 [CLS] 条件。损失为流匹配回归：
    $$\mathcal{L}_c = \mathbb{E}_{t,\mathbf{x}} \|(\widetilde{\mathbf{x}_1} - \widetilde{\mathbf{x}_0}) - \mathbf{g}_\phi(\widetilde{\mathbf{x}_t}, t, \mathbf{h}_\omega \circ \mathbf{v}_\theta(\mathbf{x}))\|_2^2$$
    - **离散去噪器（Perceiver）**：使用 6 层 Perceiver 在 VQ-GAN codebook 上预测 masked token，通过交叉注意力注入 [CLS] 条件。损失为交叉熵：
    $$\mathcal{L}_d = \mathbb{E}_{\mathbf{x}} -\log \prod_{i=1}^L \mathbf{g}_\phi(s_i | s_{<i}, \mathbf{h}_\omega \circ \mathbf{v}_\theta(\mathbf{x}))$$
    - **时间步采样**：提出 scaled Logit-Normal 采样 $t = \text{sigmoid}(s \cdot \varepsilon)$，$\varepsilon \sim \mathcal{N}(0,1)$，$s=1$ 时集中采样中间时间步，增加重建难度以放大 $I(V;G_1)$。
- **设计动机**：轻量去噪器的成功证明：增强无需完美重建能力，只需让 ViT 学习到足够的视觉模式。过大的去噪器反而可能引入更多 $G_2$。

### 损失函数 / 训练策略

- CC3M 数据集训练，每阶段 1 epoch
- AdamW 优化器：Stage-1 lr=1e-4，Stage-2 lr=1e-5
- LoRA rank=16 微调 CLIP ViT
- 全局 batch size 256

## 实验关键数据

### 主实验

**MMVP-VLM 基准（细粒度视觉感知，9 种视觉模式）**：

| CLIP 骨干 | 方法 | 方向 | 特征 | 状态 | 数量 | 位置 | 颜色 | 结构 | 文字 | 视角 | 平均 |
|-----------|------|------|------|------|------|------|------|------|------|------|------|
| OpenAI L@224 | 原始 | 13.3 | 13.3 | 20.0 | 20.0 | 13.3 | 53.3 | 20.0 | 6.7 | 13.3 | 19.3 |
| | DIVA | 13.3 | 20.0 | 40.0 | 6.7 | 20.0 | 53.3 | 46.7 | 20.0 | 13.3 | 25.9 |
| | **Ours** | 13.3 | 33.3 | 33.3 | 20.0 | 6.7 | 73.3 | 46.7 | 20.0 | 40.0 | **31.9 (+6.0)** |
| MetaCLIP H@224 | 原始 | 6.7 | 13.3 | 60.0 | 13.3 | 6.7 | 53.3 | 26.7 | 13.3 | 33.3 | 25.2 |
| | **Ours** | 20.0 | 20.0 | 66.7 | 26.7 | 26.7 | 66.7 | 33.3 | 20.0 | 53.3 | **37.0 (+5.1)** |
| SigLIP SO@224 | 原始 | 26.7 | 20.0 | 53.3 | 40.0 | 20.0 | 66.7 | 40.0 | 20.0 | 53.3 | 37.8 |
| | **Ours** | 20.0 | 20.0 | 66.7 | 60.0 | 20.0 | 86.7 | 40.0 | 13.0 | 53.3 | **42.2 (+1.5)** |

### 消融实验

**条件 Token 比例消融（MMVP-VLM，连续去噪器）**：

| 条件 | [CLS]+0% | [CLS]+10% | [CLS]+50% | [CLS]+100% |
|------|----------|-----------|-----------|------------|
| 重建质量 | 最低 | ↑ | ↑↑ | 最高 |
| 增强效果 | **最好** | 急剧下降 | 极差 | 极差 |

**时间步采样对比（s=scale factor, MMVP-VLM）**：

| 分布 | Scale | OpenAI@224 | OpenAI@336 | MetaCLIP@224 |
|------|-------|------------|------------|--------------|
| Uniform | N/A | 21.5 | 22.2 | 23.7 |
| **Logit-Normal** | **1.0** | **31.9** | **29.6** | **31.9** |

**端到端 vs 两阶段**：端到端训练在所有设置上性能下降 >5%。

### 关键发现

1. **完美重建不等于好表征**：增强 CLIP 的关键不在于生成质量，而在于有效传递细粒度知识
2. **轻量去噪器足够**：2+4 blocks 的 FLUX-lite DiT（~10% 参数）即超越使用完整 SD 的 DIVA
3. **局部 token 是毒药**：即使仅 10% 的局部 token 也导致信息泄漏和训练崩溃
4. **两阶段训练是刚需**：端到端训练因特征空间差距引入过多无关信息
5. **不影响 CLIP 原有能力**：零样本分类和检索性能变化 <0.3%

## 亮点与洞察

1. **"不完美才是完美"**：论文的核心发现是反直觉的——不需要完美的生成模型来增强判别模型，反而过强的生成能力可能有害
2. **互信息理论框架**：用 $I(V;G_1)$ vs $I(V;G_2)$ 的分解优雅地解释了所有实验现象
3. **方法的极致简洁性**：相比 DIVA 需要预训练 SD 整个 UNet，GenHancer 只需一个随机初始化的小模型+1 epoch 训练
4. **连续/离散统一**：三个 Key Points 同时适用于 Rectified Flow 和 Perceiver，验证了原理的普适性
5. **即插即用 MLLM 增强**：增强后的 CLIP 可直接替换 LLaVA 中的视觉编码器

## 局限与展望

1. **仅评估了 CLIP 系列模型**：是否适用于 DINOv2、EVA-CLIP 等其他视觉编码器未知
2. **CC3M 训练集较小**：使用更大规模数据（如 DataComp）是否能进一步提升
3. **LoRA rank=16 是否最优**：未提供 rank 消融
4. **离散去噪器需要 VQ-GAN codebook**：对 codebook 质量有隐式依赖
5. **MMVP-VLM 基准较小**（135 对）：在更大规模基准上的稳定性有待验证

## 相关工作与启发

- 与 DIVA 的核心区别：DIVA 使用预训练的 SD UNet + 端到端训练；GenHancer 使用随机初始化的轻量去噪器 + 两阶段训练
- 与 ROSS 的区别：ROSS 在 MLLM 训练时加入重建损失；GenHancer 独立增强 CLIP，更灵活
- 启发：这一发现可能推广到音频、3D 等其他模态——利用轻量生成模型反馈增强判别模型的细粒度感知

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — "不完美生成模型更好"的发现是深刻且反直觉的，互信息框架具有理论优雅性
- **实验充分度**: ⭐⭐⭐⭐ — 多 CLIP 骨干、连续/离散双范式、MLLM 集成、零样本保持，但基准偏小
- **写作质量**: ⭐⭐⭐⭐⭐ — 论述逻辑清晰，三个 Key Points 层层递进，图示精美
- **价值**: ⭐⭐⭐⭐⭐ — 为 MLLM 的视觉缺陷提供了成本极低的解决方案，启发性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)
- [\[ICCV 2025\] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference](dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)

</div>

<!-- RELATED:END -->
