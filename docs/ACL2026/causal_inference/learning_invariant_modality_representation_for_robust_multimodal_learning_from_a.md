---
title: >-
  [论文解读] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective
description: >-
  [ACL 2026][因果推理][因果不变表示] 本文提出 CmIR（因果模态不变表示学习），基于因果推理理论将每种模态显式解纠缠为因果不变表示和环境特定虚假表示，通过不变性约束+互信息约束+重建约束的优雅目标函数确保不变表示具有跨环境的稳定预测关系，在多模态情感/幽默/讽刺检测上取得 SOTA…
tags:
  - "ACL 2026"
  - "因果推理"
  - "因果不变表示"
  - "多模态情感分析"
  - "分布外泛化"
  - "特征解纠缠"
  - "虚拟环境"
---

# Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective

**会议**: ACL 2026  
**arXiv**: [2604.18460](https://arxiv.org/abs/2604.18460)  
**代码**: [GitHub](https://github.com/TmacMai/CmIR)  
**领域**: 音频语音  
**关键词**: 因果不变表示, 多模态情感分析, 分布外泛化, 特征解纠缠, 虚拟环境

## 一句话总结

本文提出 CmIR（因果模态不变表示学习），基于因果推理理论将每种模态显式解纠缠为因果不变表示和环境特定虚假表示，通过不变性约束+互信息约束+重建约束的优雅目标函数确保不变表示具有跨环境的稳定预测关系，在多模态情感/幽默/讽刺检测上取得 SOTA，尤其在 OOD 和噪声场景下表现突出。

## 研究背景与动机

**领域现状**：多模态情感计算通过整合语言/声学/视觉模态预测情感。现有方法在同分布测试上表现良好，但往往学习了训练数据中的虚假跨模态相关性。

**现有痛点**：(1) 模型可能过度依赖说话者一贯的微笑（虚假视觉特征）而非语义内容；(2) 噪声模态（如背景噪声/低分辨率视频）进一步破坏虚假相关性，加剧泛化差距；(3) 现有因果方法要么缺乏理论保证，要么只针对特定偏差（如说话者偏差），不通用。

**核心矛盾**：需要一种通用的框架来区分因果特征和虚假特征——不依赖对偏差类型的先验假设，不需要预定义的偏差标签。

**本文目标**：基于因果推理建立有理论保证的通用框架，将每种模态解纠缠为因果不变和环境虚假两个组分。

**切入角度**：因果不变表示的核心性质是跨环境的预测稳定性——如果 $P(Y|Z_m^{\text{inv}}, E=e_1) = P(Y|Z_m^{\text{inv}}, E=e_2)$，则 $Z_m^{\text{inv}}$ 只包含因果特征。

**核心 idea**：通过三约束优化学习解纠缠：不变性约束确保跨环境预测一致，互信息约束确保两组分独立，重建约束确保无信息丢失。在缺乏显式环境标签时，通过向原始特征注入不同强度的噪声模拟虚拟环境。

## 方法详解

### 整体框架

每种模态 $X_m$ 通过编码器 $g_m$ 解纠缠为 $(Z_m^{\text{inv}}, Z_m^{\text{spu}})$。仅使用不变表示 $\{Z_m^{\text{inv}}\}_{m=1}^M$ 的拼接进行预测。训练时同时优化预测损失和三个约束项。解码器 $r_m$ 从两个组分重建原始输入以防止信息丢失。

### 关键设计

1. **虚拟环境构造 + 不变性约束**:

    - 功能：在缺乏显式环境标签时实现跨环境不变性训练
    - 核心思路：对每个样本随机分配虚拟环境标签 $e \in \{1,...,K\}$，注入强度为 $\alpha^{(e)} = \alpha^{(1)} \cdot e$ 的加性高斯噪声。对扰动特征提取不变表示，用 L1 范数约束不同环境的不变表示一致：$\mathcal{R}_{\text{inv}}^{(m)} = \sum_{e_1 \neq e_2} \|Z_m^{\text{inv},(e_1)} - Z_m^{\text{inv},(e_2)}\|_1$
    - 设计动机：比 KL 散度约束更强（输入相同则输出分布必然相同），且适用于分类和回归任务，无需额外的单模态预测器

2. **正交性近似的互信息最小化**:

    - 功能：确保不变和虚假组分统计独立
    - 核心思路：计算批内的归一化相关矩阵 $\bm{C}^m = \text{Nor}(\bm{Z}_m^{\text{inv}}) \cdot \text{Nor}(\bm{Z}_m^{\text{spu}})^\top$，用加权 Frobenius 范数惩罚：对角项（同样本正交性）权重为 1，非对角项权重为 $\alpha < 1$
    - 设计动机：直接计算互信息不可行，正交性是独立性的必要条件。配合不变性和重建约束一起使用可确保语义分离

3. **重建约束防止退化**:

    - 功能：确保解纠缠后的两个组分保留原始输入的全部信息
    - 核心思路：解码器 $r_m$ 从 $(Z_m^{\text{inv}}, Z_m^{\text{spu}})$ 重建原始特征：$\mathcal{R}_{\text{rec}}^{(m)} = \|X_m - r_m(Z_m^{\text{inv}}, Z_m^{\text{spu}})\|_2^2$
    - 设计动机：无重建约束，模型可能学到退化解——不变组分包含所有信息而虚假组分为空，或反之

### 损失函数 / 训练策略

总目标：$\mathcal{L} = \mathcal{L}_{\text{pred}} + \sum_{m=1}^{M} \lambda_1 \mathcal{R}_{\text{inv}}^{(m)} + \lambda_2 \mathcal{R}_{\text{dec}}^{(m)} + \lambda_3 \mathcal{R}_{\text{rec}}^{(m)}$。提供三个定理的完整证明：不变表示的存在性、可提取性、以及 OOD 风险优势。

## 实验关键数据

### 主实验

在 CMU-MOSI/MOSEI/CH-SIMS-v2（情感）+ UR-FUNNY（幽默）+ MUStARD（讽刺）上评估。CmIR 在标准和 OOD 设置下均取得 SOTA。

### 关键发现

- OOD 设置下（CMU-MOSI OOD），CmIR 的优势更加明显——证实了因果不变表示的泛化优势
- 噪声模态测试中，CmIR 的退化幅度远小于基线——虚假组分的隔离使模型对噪声更鲁棒
- 消融证明三个约束都不可或缺——去掉任一约束都导致性能下降

## 亮点与洞察

- 三约束框架的设计非常优雅——不变性确保"因果性"，正交性确保"纯净性"，重建确保"完整性"，三者缺一不可
- 虚拟环境构造是一个实用的折中——虽然不如真实环境标签精确，但在多数数据集没有环境标签的现实下提供了可行方案
- 理论保证（三个定理）为框架提供了坚实的理论基础

## 局限与展望

- 虚拟环境的构造依赖于加性高斯噪声假设，可能不完全反映真实的分布偏移
- 超参数（环境数K、噪声系数α、三个λ）需要调优
- 编码器/解码器均为简单MLP，更强的架构可能进一步提升

## 相关工作与启发

- **vs IRM**: 针对单模态的不变风险最小化，CmIR 将其扩展到多模态解纠缠
- **vs 现有多模态因果方法**: 针对特定偏差（说话者/模态），CmIR 是通用的不依赖偏差假设的框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在MAC中系统地将因果不变表示学习与特征解纠缠结合
- 实验充分度: ⭐⭐⭐⭐⭐ 6数据集+标准/OOD/噪声设置+完整消融+理论证明
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验全面
- 价值: ⭐⭐⭐⭐⭐ 对多模态鲁棒性研究有范式级贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ECCV 2024\] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](../../ECCV2024/causal_inference/integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)
- [\[ACL 2026\] Function Words as Statistical Cues for Language Learning](function_words_as_statistical_cues_for_language_learning.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](../../ECCV2024/causal_inference/learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)

</div>

<!-- RELATED:END -->
