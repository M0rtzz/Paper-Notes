---
title: >-
  [论文解读] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings
description: >-
  [ICLR 2026][可解释性][模态间隙] 提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "模态间隙"
  - "稀疏自编码器"
  - "跨模态冗余"
  - "等能量假设"
  - "VLM可解释性"
---

# Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings

**会议**: ICLR 2026  
**arXiv**: [2602.06218](https://arxiv.org/abs/2602.06218)  
**代码**: [https://github.com/Parabrele/IsoEnergy](https://github.com/Parabrele/IsoEnergy)  
**领域**: 可解释性  
**关键词**: 模态间隙, 稀疏自编码器, 跨模态冗余, 等能量假设, VLM可解释性

## 一句话总结
提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。

## 研究背景与动机

**领域现状**：CLIP/SigLIP 等视觉-语言模型通过对比学习将图像和文本映射到共享嵌入空间，实现了跨模态对齐。但其嵌入空间的内部几何结构仍不清楚。

**现有痛点**：已知存在"模态间隙"（modality gap）现象——图像和文本嵌入位于隐空间中不相交的锥体中。之前的工作尝试通过移除均值差异或投影某些坐标方向来消除间隙，但这些干预都会损害跨模态性能。用稀疏自编码器（SAE）提取概念字典时，发现概念往往按模态分离，难以找到真正的双模态概念。

**核心矛盾**：VLM 明明是为跨模态对齐而训练的，但提取出的概念字典却大量按模态分离——这是因为概念恢复本身是一个欠定问题（非线性ICA不可辨识），缺乏额外归纳偏置时标准 SAE 无法正确区分双模态和单模态原子。

**本文目标** (a) 如何从 VLM 嵌入中准确恢复双模态 vs 单模态概念？ (b) 模态间隙的本质是什么？ (c) 能否在不损害性能的情况下消除模态间隙？

**切入角度**：从数据生成过程出发——如果多模态数据由共享的潜在概念向量通过各模态生成器产生，那么真正共享的概念应该在两个模态中留下"冗余的统计痕迹"，特别是相同的平均激活能量。

**核心 idea**：利用跨模态冗余作为归纳偏置，通过等能量约束引导 SAE 学到正确的双模态/单模态概念分解，从而揭示和操控 VLM 嵌入的几何结构。

## 方法详解

### 整体框架

这篇论文要回答的核心问题是：VLM 明明为跨模态对齐而训练，为什么用 SAE 提取出的概念字典却大量按模态分离、找不到真正的双模态概念？作者把它归结为一个可辨识性问题，并用「跨模态冗余」这个统计信号来破解。

整条思路建立在一个假想的多模态概念生成过程上：潜在概念向量 $\mathbf{c}$ 被稀疏采样，再经由模态特定生成器 $\mathbf{g}^{(d)}$ 投影成各模态的观测。VLM 编码器 $\mathbf{f}$ 可以看作生成器的近似逆映射，把观测拉回到嵌入空间；SAE $\phi$ 则在嵌入之上再做一次提升，试图还原到底层概念。理想情况下 $\phi \circ \mathbf{f}$ 应当准确恢复 $\mathbf{c}$，但非线性 ICA 本身不可辨识，缺乏额外约束时标准 SAE 会把双模态概念错拆成两个单模态原子。后文的设计就是给这个欠定问题补上一个来自数据生成结构的归纳偏置。

### 关键设计

**1. Iso-Energy 假设：用「能量相等」当作双模态概念的可检验判据**

可辨识性问题的根子在于：标准 SAE 没有任何理由相信某个原子是跨模态共享的。作者从生成过程出发给出一个判据——如果概念 $k$ 真的由同一份潜在代码在两个模态里生成，那它在两个模态中留下的统计痕迹应当对得上，最直接的就是平均激活强度相同。形式化为等能量约束：

$$\mathbb{E}_{X \in \mathcal{X}^{(d)}}[\psi(X)_k^2] = \mathbb{E}_{X \in \mathcal{X}^{(d')}}[\psi(X)_k^2]$$

即概念 $k$ 在模态 $d$ 与 $d'$ 上的平均平方激活相等。这个量简单到只是一个二阶统计，却恰好为不可辨识的非线性 ICA 提供了一个有方向的额外约束，把「哪些原子该被当成双模态」从猜测变成可检验的命题。

**2. Aligned SAE（SAE-A）：把等能量假设落成一个轻量正则项**

有了判据还需要让 SAE 在训练时真的去满足它。SAE-A 在标准 SAE 目标上加一项对齐正则：

$$\mathcal{L}_{\text{SAE-A}} = \mathcal{L}_{\text{SAE}} + \beta \cdot \mathcal{L}_{\text{align}}, \qquad \mathcal{L}_{\text{align}} = -\frac{1}{b}\text{Tr}(\mathbf{Z}^{(d)} \mathbf{Z}^{(d')^\top})$$

其中 $\mathbf{Z}$ 是 $\ell_2$ 归一化后的编码，这一项实际是在最大化配对 image-text 样本编码的余弦相似度，等价于鼓励两模态在共享原子上对齐能量。稀疏性用 Matching Pursuit 实现 $\ell_0$ 约束而非 ReLU/TopK，更贴近稀疏编码的理论假设。关键在于权重 $\beta \approx 10^{-4}$ 极小：它足以纠正双模态/单模态的分解，却几乎不动重建质量（$R^2 \geq 0.99$）——这正是它区别于「移除均值、投影坐标」等粗暴干预的地方，后者一压间隙就掉性能。

**3. 合成数据验证：先证明这个正则项不会无中生有**

正则化最怕的质疑是「你不过是把双模态原子硬塞了进去」。为排除这点，作者构造带 ground-truth 双模态/单模态原子的合成数据，用 $\tau_1$ 控制原子层面的跨模态对齐度、$\tau_2$ 控制嵌入层面的对齐度，于是可以分别考察假设成立与否两种情形。当 Iso-Energy 真的成立（$\tau_1=1$）时，标准 SAE 恢复失败（Wasserstein $=0.396$，mma $=0.29$），SAE-A 明显更准（$0.184$，$0.52$）；而当假设不成立时，两者表现相当。也就是说，正则项只有在数据里确实存在等能量结构时才起作用，不成立时保持中性、不会幻觉出不存在的双模态原子。

**4. 几何分解与干预：把概念字典翻译成嵌入空间的子空间结构**

最后一步是把恢复出的字典用回 VLM 嵌入的几何上。作者对每个原子算一个 modality score $\mu$，据此把字典分成双模态与单模态两类，对应地把嵌入空间切成双模态子空间 $\Gamma$ 与单模态子空间 $\Omega_I \oplus \Omega_T$。双模态原子张成 $\Gamma$，承载真正的跨模态对齐信号；单模态原子张成 $\Omega_{I/T}$，装的是各模态特有的信息，也正是模态间隙的来源。这个分解的价值在于它让干预变得有的放矢——移除单模态原子就能抹平模态间隙而不碰跨模态性能，把向量运算限制在 $\Gamma$ 内则能得到更 in-distribution 的编辑，这是此前纯几何视角下做不到的精确操控。

### 损失函数 / 训练策略

基础 SAE 用 Matching Pursuit 做 $\ell_0$ 稀疏，通过序贯残差更新逐个挑选被激活的原子。对齐项 $\mathcal{L}_{\text{align}}$ 最大化配对样本编码的余弦相似度，权重 $\beta \approx 10^{-4}$ 极小，对重建几乎无影响。

## 实验关键数据

### 主实验

在 6 个 VLM（CLIP, CLIP-L, OpenCLIP, OpenCLIP-L, SigLIP, SigLIP2）上训练 SAE 和 SAE-A：

| 模型 | MSE (SAE/SAE-A) | R² (SAE/SAE-A) | 分类准确率 $p_{\text{acc}}$ (SAE/SAE-A) |
|------|------|----------|------|
| CLIP | 0.141/0.163 | 0.859/0.837 | 0.847/**0.915** |
| SigLIP2 | 0.115/0.115 | 0.884/0.885 | **0.897/0.899** |

- SAE-A 在重建质量几乎不变的情况下，显著提高了双模态原子的激活模式分类准确率

### 消融实验

| 实验 | 关键指标 | 说明 |
|------|---------|------|
| 合成数据 (Iso-Energy成立) | SAE: W=0.396, mma=0.29; SAE-A: W=0.184, mma=0.52 | SAE-A 恢复双模态原子显著更好 |
| 合成数据 (Iso-Energy不成立) | 两者: W≈0.19, mma≈0.82 | 正则化器不会强行创造双模态原子 |
| 移除单模态原子 | 模态间隙消失 + 跨模态性能不降 | 验证了单模态原子=模态间隙的解释 |
| 仅在双模态子空间做向量运算 | 检索性能提升 + 编辑更 in-distribution | 双模态子空间是跨模态操作的正确空间 |

### 关键发现
- 稀疏双模态原子承载了全部跨模态对齐信号——数量少但信息集中
- 少数高能量单模态原子充当"模态偏置"，完全解释了模态间隙
- 移除单模态原子可以在不损害下游性能的情况下消除模态间隙（之前所有方法做不到）
- 将向量运算限制在双模态子空间内可以产生 in-distribution 编辑，改善检索效果
- 与 Papadimitriou et al. (2025) 的发现相反：跨模态信息由共享原子而非特异性原子承载

## 亮点与洞察
- **等能量假设**的简洁与深刻：一个如此简单的统计量（各模态的平均平方激活相等）就足以作为双模态概念的判别标准，且有坚实的生成模型支撑。这个思想可迁移到任何多视角/多模态的概念提取任务
- **"不伤害就是最好的验证"策略**：在合成数据上证明当假设不成立时正则化器是"中性"的（不会fabricate双模态概念），这种验证方式非常巧妙，避免了人为引入偏差的质疑
- **模态间隙的概念级解释**：将之前纯几何的描述（锥体、椭球壳）提升到概念层面（单模态原子=模态偏置），使得间隙不再是需要"消除"的bug，而是模型正确保留模态特定信息的feature
- **Matching Pursuit SAE**：使用 $\ell_0$ 稀疏而非 ReLU/TopK，更符合稀疏编码的理论假设，可迁移到其他SAE应用场景

## 局限与展望
- Iso-Energy 假设要求概念在两个模态中有完全相同的能量，但现实中某些概念可能天然在视觉中更丰富（如颜色、纹理），这种不对称性未被讨论
- 实验仅在双编码器（dual-encoder）VLM 上验证，未扩展到单编码器或编码器-解码器架构（如 LLaVA、Flamingo）
- SAE-A 需要配对的 image-text 数据进行训练，限制了其在未配对数据上的应用
- 对齐正则化的权重 $\beta$ 虽然很小，但仍需要调节，不是完全无超参数的
- 双模态/单模态的二元划分可能过于粗糙，实际中可能存在"部分双模态"的概念

## 相关工作与启发
- **vs Liang et al. (2022) 模态间隙**: 他们描述了间隙的几何现象（锥体结构），但尝试消除间隙会损害性能。本文解释了为什么——间隙来自单模态原子，承载必要的模态特定信息，但可以在概念层面精确移除
- **vs Schrodi et al. (2025)**: 他们尝试通过投影少数canonical方向来消除间隙，但"误伤"了双模态信息。本文的SAE-A能正确分离，避免误伤
- **vs Papadimitriou et al. (2025)**: 他们认为跨模态信息由特异性（idiosyncratic）概念承载，本文发现恰恰相反——由共享原子承载。差异来自标准SAE的可辨识性问题
- **vs 柏拉图表示假设 (Huh et al. 2024)**: 本文的等能量假设可以看作是这一假设的可操作化版本——如果不同模型/模态收敛到相同特征，那么这些特征的统计量应跨模态一致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Iso-Energy假设简洁优雅，首次在概念层面完整解释模态间隙
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据验证充分，但缺少非dual-encoder结构
- 写作质量: ⭐⭐⭐⭐⭐ 理论动机清晰，实验逻辑严密，图表设计精美
- 价值: ⭐⭐⭐⭐⭐ 对VLM可解释性有重要推动，Aligned SAE有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[CVPR 2026\] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](../../CVPR2026/interpretability/missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)

</div>

<!-- RELATED:END -->
