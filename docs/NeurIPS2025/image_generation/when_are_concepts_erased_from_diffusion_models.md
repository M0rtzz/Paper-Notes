---
title: >-
  [论文解读] When Are Concepts Erased From Diffusion Models?
description: >-
  [NeurIPS 2025][图像生成][概念擦除] 本文提出了两种概念擦除的机制模型（引导式回避 vs. 破坏式移除），并设计了涵盖优化搜索、上下文探测、噪声轨迹探测、分类器引导和动态追踪的五种独立探测方法，系统性地揭示了现有擦除方法大多只是"绕开"概念而非真正"消除"知识。
tags:
  - NeurIPS 2025
  - 图像生成
  - 概念擦除
  - 扩散模型
  - 知识残留
  - 引导式回避
  - 破坏式移除
---

# When Are Concepts Erased From Diffusion Models?

**会议**: NeurIPS 2025  
**arXiv**: [2505.17013](https://arxiv.org/abs/2505.17013)  
**代码**: https://github.com/kevinlu4588/WhenAreConceptsErased  
**领域**: AI安全 / 扩散模型概念擦除  
**关键词**: 概念擦除, 扩散模型, 知识残留, 引导式回避, 破坏式移除

## 一句话总结

本文提出了两种概念擦除的机制模型（引导式回避 vs. 破坏式移除），并设计了涵盖优化搜索、上下文探测、噪声轨迹探测、分类器引导和动态追踪的五种独立探测方法，系统性地揭示了现有擦除方法大多只是"绕开"概念而非真正"消除"知识。

## 研究背景与动机

概念擦除（Concept Erasure）旨在修改扩散模型以阻止其生成特定概念（如特定艺术风格、物体等）。尽管已有大量擦除方法被提出，但一个根本问题始终悬而未决：**被"擦除"的概念是否真的从模型中移除了？还是模型只是学会了回避？**

先前的对抗攻击研究已表明，通过寻找合适的输入（如 Textual Inversion、对抗性提示），可以让擦除的概念重新出现。但这些发现仅涉及文本输入层面的攻击，留下一个开放问题：**被擦除的知识是否还能通过其他途径被发掘？**

本文的核心贡献在于提出两个概念化机制框架和一套覆盖多视角的全面评估工具包，来系统性地回答"概念到底被擦除到什么程度"这一问题。

## 方法详解

### 整体框架

作者提出两种概念化的擦除机制模型：

- **引导式回避（Guidance-Based Avoidance）**：模型修改了条件引导过程，使生成偏离目标概念，但底层知识可能仍然完整。模型的无条件概率 $P(X)$ 未被显著改变。
- **破坏式移除（Destruction-Based Removal）**：模型降低了目标概念的无条件似然 $P(X)$，从根本上抑制甚至消除底层特征。

然后设计五种独立探测方法来检测擦除的彻底程度：

### 关键设计

1. **优化搜索探测（Optimization-based Probing）**：采用 Textual Inversion 和 UnlearnDiffAtk 两种方法，通过优化文本嵌入或 token 来搜索能触发已擦除概念的输入。这一探测直接沿用了先前工作的方法。

2. **上下文探测（In-context Probing）**：

    - **Inpainting 探测**：给模型提供包含目标概念但部分遮挡的图像，观察模型是否能正确补全。如果模型真的不具备该概念的知识，应无法正确修复。
    - **扩散补全探测（Diffusion Completion）**：用原始未擦除模型运行 5 或 10 个去噪步，保存中间结果，然后交给擦除模型完成剩余去噪。如果擦除模型能从部分生成的痕迹中恢复概念，说明知识仍在。

3. **噪声轨迹探测（Noise-Based Probing）**：在每个去噪步骤中注入额外的高斯噪声 $\tilde{x}_{t-1} = (\tilde{x}_t - \alpha\epsilon_D) + \eta\epsilon$，通过控制 $\eta$ 值（搜索 $[1.0, 1.85]$ 范围内的 6 个值），让模型在更大的潜空间中探索。这是一种免训练的方法，不需要优化任何输入。

4. **分类器引导探测（Steered Latent Probing）**：在潜空间中训练一个轻量级的时间步感知二分类器 $f_{c^*}(\mathbf{x}_t, t)$，检测目标概念是否存在。在推理时通过梯度引导将扩散轨迹朝向概念残留区域。搜索 24 个引导强度 $s_{\text{clf}}$ 值。

5. **动态概念追踪（Dynamic Concept Tracing）**：在不同擦除强度下，跟踪生成图像在 CLIP 嵌入空间中的轨迹变化，观察概念表示如何随擦除过程演变。

### 损失函数 / 训练策略

分类器使用 BCEWithLogits 损失，带正类权重平衡。每个 mini-batch 用 7 个噪声视图增强，时间步按幂律分布采样以偏向更高噪声级别。训练 70 个 epoch，选验证损失最低的 checkpoint。

## 实验关键数据

### 主实验（优化搜索探测）

| 擦除方法 | 擦除后 CLIP↓ | Textual Inversion 分类准确率↓ | UnlearnDiffAtk 分类准确率↓ | 无关概念准确率↑ |
|---------|------------|---------------------------|--------------------------|--------------|
| GA | 24.3 | 0.6% | 6.5% | 52.2% |
| UCE | 22.4 | 71.2% | 26.8% | 75.0% |
| ESD-x | 21.1 | 65.9% | 21.0% | 71.3% |
| ESD-u | 20.9 | 31.8% | 16.6% | 70.4% |
| TaskVec | 23.1 | 6.2% | 10.3% | 60.4% |
| STEREO | 19.6 | 6.3% | 3.7% | 52.8% |
| RECE | 21.2 | 58.2% | 7.2% | 71.7% |

### 消融实验（多探测方法对比）

| 擦除方法 | Inpainting 分类准确率↓ | 扩散补全 t=5 准确率↓ | 噪声探测 准确率↓ | 分类器引导 准确率↓ | 分类器+噪声 准确率↓ |
|---------|---------------------|-------------------|----------------|----------------|-------------------|
| GA | 61.7% | 1.1% | 2.7% | 3.7% | 4.1% |
| UCE | 69.1% | 42.7% | 21.9% | 45.6% | 75.6% |
| ESD-x | 69.1% | 37.8% | 30.7% | 47.8% | 73.3% |
| TaskVec | 66.8% | 2.4% | 11.0% | 30.2% | 35.1% |
| STEREO | 63.8% | 3.2% | 1.1% | 5.8% | 20.3% |
| RECE | 68.2% | 36.5% | 13.0% | 33.3% | 36.7% |

### 关键发现

- **UCE 和 ESD-x** 在所有探测下都高度脆弱，属于典型的"引导式回避"方法——概念知识完好无损
- **GA 和 STEREO** 在大多数探测下最为鲁棒，更接近"破坏式移除"
- **TaskVec** 表现出有趣的矛盾：对 Textual Inversion 鲁棒，但在 Inpainting 探测中仍能正确补全被擦除的概念
- **RECE 和 STEREO** 对传统对抗攻击鲁棒，但在扩散补全探测中意外暴露了残留知识
- 噪声探测（无需任何优化或训练）有时比复杂的优化方法更有效：在 UCE/ESD-x 上，仅增加随机性就能恢复概念
- 分类器引导与噪声探测的组合最为强大，恢复准确率比单独使用任一方法高约 1.5 倍

## 亮点与洞察

- **理论框架的价值**：引导式回避 vs. 破坏式移除的二分法为理解各类擦除方法的本质提供了清晰的概念工具
- **多视角评估的必要性**：单一评估方法（如仅用对抗攻击）会给出误导性结论——在一种探测下鲁棒的方法在另一种下可能完全失效
- **噪声探测的简洁力量**：一个如此简单的方法（只是增大推理噪声）竟能比精心设计的优化攻击更有效地揭示残留知识，这本身就是深刻的洞察
- **鲁棒性与通用性的权衡**：擦除更彻底的方法（GA、STEREO）往往对无关概念的生成质量损害更大

## 局限与展望

- 实验覆盖 10 个物体概念和 3 个艺术风格，但未涉及动词、关系或抽象概念（如"暴力"）
- 优化类探测存在固有的因果模糊性：恢复的概念可能来自优化过程本身，而非模型的残留知识
- 分类器引导探测中分类器本身可能引入偏置
- 仅在 Stable Diffusion 1.4 上实验，新一代模型（SDXL、SD3等）的情况可能不同
- 对擦除方法的归类（引导式 vs. 破坏式）仍是初步的，缺乏严格的理论证明

## 相关工作与启发

- UCE 是引导式方法的代表，通过闭式解修改注意力投影矩阵
- STEREO 结合对抗性提示搜索和组合微调，是当前较强的鲁棒擦除方法
- 本文的评估框架可以直接推广到任何新提出的擦除方法
- 对 AI 安全领域的启示：**"看起来被擦除"与"真正被擦除"之间可能存在巨大鸿沟**

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （概念框架和多视角评估体系都是原创性贡献）
- 实验充分度: ⭐⭐⭐⭐⭐ （7种擦除方法 × 5种探测手段，覆盖全面）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐⭐ （对理解和改进概念擦除方法具有重要意义）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[CVPR 2025\] Memories of Forgotten Concepts](../../CVPR2025/image_generation/memories_of_forgotten_concepts.md)
- [\[CVPR 2025\] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](../../CVPR2025/image_generation/concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)

<!-- RELATED:END -->
