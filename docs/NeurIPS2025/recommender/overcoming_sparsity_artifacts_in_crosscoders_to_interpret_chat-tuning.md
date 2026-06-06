---
title: >-
  [论文解读] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
description: >-
  [NeurIPS 2025][推荐系统][Crosscoder] 识别并解决 Crosscoder 中 L1 训练损失引入的两类稀疏性伪影（导致虚假的模型特定潜变量归因），提出 Latent Scaling 诊断方法和 BatchTopK 损失替代方案…
tags:
  - "NeurIPS 2025"
  - "推荐系统"
  - "Crosscoder"
  - "稀疏性伪影"
  - "BatchTopK"
  - "Latent Scaling"
  - "chat-tuning"
---

# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2504.02922](https://arxiv.org/abs/2504.02922)  
**作者**: Julian Minder, Clément Dumas, Caden Juang, Bilal Chughtai, Neel Nanda  
**机构**: EPFL, ETHZ, ENS Paris-Saclay, Northeastern University  
**代码**: [GitHub](https://github.com/jkminder/dictionary_learning)  
**领域**: 推荐系统  

## 一句话总结

识别 Crosscoder 中 L1 损失引入的两类稀疏性伪影（Complete Shrinkage 将弱共享概念错误归零、Latent Decoupling 将共享概念拆分为虚假模型特定潜变量），提出 Latent Scaling 诊断方法和 BatchTopK Crosscoder 替代方案，显著提升 chat-tuning 概念发现的可靠性。

## 研究背景与动机

**领域现状**: Model Diffing 是可解释性研究的新兴方向，旨在理解微调如何改变模型的内部表示和算法。Crosscoder（Lindsey et al., 2024）作为一种 model diffing 工具，在基础模型和微调模型之间学习共享的可解释概念字典，每个概念用一对潜变量方向表示（分别对应 base 和 chat 模型），从而追踪概念在微调中的变化或涌现。

**现有痛点**: 先前工作通过检查 base 模型 decoder 向量的范数是否为零，来判断某个概念是否为 chat-only（即仅在微调模型中存在）。这一方法看似合理，但实际上可能被 L1 稀疏化训练损失引入的系统性偏差严重污染。

**核心矛盾**: L1 正则化在鼓励稀疏表示的同时，会产生两类伪影：(1) **Complete Shrinkage**——当某个概念在 base 模型中贡献较弱但在 chat 模型中贡献较强时，L1 会将 base decoder 范数直接压至零，虚假地标记为 chat-only；(2) **Latent Decoupling**——一个本应为共享的概念被 L1 损失等价地拆分为一个 chat-only + 一个 base-only 潜变量对，因为两种表示的 L1 损失完全相同。

**本文目标**: 如何诊断和消除 Crosscoder 中由 L1 损失引发的虚假 chat-only 潜变量归因，并找到真正由 chat-tuning 引入的可解释概念。

**切入角度**: 从训练损失函数的理论分析出发，揭示 L1 损失的内在缺陷，然后设计诊断工具（Latent Scaling）和替代训练方案（BatchTopK）。

**核心 idea**: 用 BatchTopK 损失替代 L1 损失训练 Crosscoder，从根源上消除缩减偏差，使 decoder 范数差异成为 chat 特异性的可靠度量。

## 方法详解

### 整体框架

本文的方法分为三个递进步骤：(1) **诊断**——理论分析 L1 Crosscoder 中的 Complete Shrinkage 和 Latent Decoupling 两类伪影机制；(2) **量化**——提出 Latent Scaling 方法，通过为每个 chat-only 潜变量学习缩放因子 β，精确度量其在各模型中的真实存在程度；(3) **替代**——训练 BatchTopK Crosscoder，通过直接限制每个 batch 中激活的潜变量数量来实现稀疏性，从根本上避免 L1 的缩减偏差。在 Gemma 2 2B base/chat 模型对的第 13 层残差流上训练，扩展因子 32，字典大小 73728。

### 关键设计

1. **Latent Scaling 诊断工具**
    - 功能：为每个 chat-only 潜变量计算两对缩放因子 β，精确度量其在 base/chat 模型中解释重构误差和激活重构的能力
    - 核心思路：对每个 chat-only 潜变量 j，用其 chat decoder 方向 $\mathbf{d}_j^{\text{chat}}$ 去拟合 base 模型的误差和重构，通过最小二乘求解缩放因子 β，然后计算 base/chat 比率 $\nu_\varepsilon = \beta_\varepsilon^{\text{base}} / \beta_\varepsilon^{\text{chat}}$ 和 $\nu_r = \beta_r^{\text{base}} / \beta_r^{\text{chat}}$。高 $\nu_\varepsilon$ 表示 Complete Shrinkage（该潜变量其实能解释 base 误差），高 $\nu_r$ 表示 Latent Decoupling（该潜变量的信息已存在于 base 重构中）
    - 设计动机：标准 Δnorm 指标在 L1 Crosscoder 中因系统性偏差而不可靠，需要独立的诊断工具来区分真正 chat-specific 的潜变量和因训练损失产生的伪影

2. **BatchTopK Crosscoder 训练方案**
    - 功能：用 BatchTopK 损失替代 L1 损失，直接控制每个 batch 中激活的潜变量数量（k=100），而非通过 L1 间接鼓励稀疏
    - 核心思路：BatchTopK 选择每个样本中激活最强的 top-k 个潜变量，不对 decoder 范数施加显式正则化惩罚，因此不会产生 Complete Shrinkage；同时 top-k 选择创造了潜变量间的竞争，使得 Latent Decoupling 这种冗余表示变得低效（两个潜变量表示同一概念会占据 L0=2 而非 L0=1 的稀疏预算）
    - 设计动机：从 SAE 文献中借鉴 BatchTopK（Bussmann et al., 2024），首次引入 Crosscoder 场景；BatchTopK 的 L0 稀疏性优化天然偏好三潜变量解（shared + chat-only + base-only）而非两潜变量解（chat-only + base-only），从而在 shared 概念的子集上仅需一个潜变量

3. **因果干预验证框架**
    - 功能：通过在 base 模型激活上替换特定 chat-specific 潜变量的表示来验证它们的因果有效性
    - 核心思路：构造混合激活 $\mathbf{h}_S(x) = \mathbf{h}_{\text{base}}(x) + \sum_{j \in S} f_j(x)(\mathbf{d}_j^{\text{chat}} - \mathbf{d}_j^{\text{base}})$，送入 chat 模型后续层，计算输出与原始 chat 模型的 KL 散度。KL 越低说明替换的潜变量集合 S 越能还原 chat 行为
    - 设计动机：仅靠可解释性不够，需要因果证据证明识别出的 chat-specific 潜变量确实驱动了 chat 模型的行为差异

### 损失函数 / 训练策略

**L1 Crosscoder 损失**:

$$\mathcal{L}_{L1}(x) = \frac{1}{2}\|\varepsilon_{\text{base}}\|^2 + \frac{1}{2}\|\varepsilon_{\text{chat}}\|^2 + \mu \sum_j f_j(x)(\|\mathbf{d}_j^{\text{base}}\|_2 + \|\mathbf{d}_j^{\text{chat}}\|_2)$$

**BatchTopK Crosscoder 损失**: 仅包含重构误差 + 辅助死潜变量回收损失，不含显式稀疏正则：

$$\mathcal{L}_{\text{BatchTopK}}(\mathcal{X}) = \frac{1}{n}\sum_i \left[\frac{1}{2}\|\varepsilon_{\text{base}}\|^2 + \frac{1}{2}\|\varepsilon_{\text{chat}}\|^2\right] + \alpha \cdot \mathcal{L}_{\text{aux}}$$

训练细节：base/chat 模型为 Gemma 2 2B / Gemma 2 2B-it，第 13 层残差流，100M tokens（Fineweb + LMSYS-CHAT），L0 ≈ 100，总计约 60 GPU·h（H100）。

## 实验关键数据

### 主实验

| 指标 | L1 Crosscoder | BatchTopK Crosscoder |
|------|--------------|---------------------|
| chat-only 潜变量数 (Δnorm>0.9) | 3,176 | 134 |
| base-only 潜变量数 (Δnorm<0.1) | 1,437 | 5 |
| shared 潜变量数 (Δnorm 0.4-0.6) | 53,569 | 62,373 |
| 真正 chat-specific 比例（ν_ε<0.2 且 ν_r<0.5） | 极低（大部分受伪影影响） | 绝大多数 |
| 全量替换 KL 降低（所有 token） | ~59% | ~59% |
| 全量替换 KL 降低（前 9 token） | ~78% | ~78% |
| Top-50% Δnorm vs Bottom-50% KL（所有 token） | 0.241 vs 0.242（失败） | 0.230 vs 0.267（成功区分） |
| Top-50% Δnorm vs Bottom-50% KL（前 9 token） | 0.619 vs 0.740（反向！） | 50% 降低 vs 6% 降低 |
| Validation FVE | 84.6% | 87.6% |

### 消融实验

| 消融项 | 结果 |
|--------|------|
| Latent Scaling 诊断 L1 → top-50% ν vs bottom-50% ν | 成功识别因果重要潜变量，前 9 token 效果接近 BatchTopK |
| 仅替换 template token 位置激活 | KL 0.239 / 0.507，≈ top-50% chat-specific 潜变量 |
| 独立训练的 L1 Crosscoder (Kissane et al.) 验证 | 17.7% chat-only 落入 shared 分布的 95% 区间，结果一致 |
| Latent Decoupling 余弦相似度 | 109 对 cosim>0.9 的 (chat-only, base-only) 潜变量对，60% 在不同上下文激活 |
| BatchTopK 中高余弦相似度耦合对 | 0 对（Δnorm<0.6 区间无 cosim>0.9 的配对） |

### 关键发现

- **Δnorm 在 L1 Crosscoder 中是训练伪影**: L1 的 chat-only 潜变量中，18% 落入 shared 分布的 95% 中心区间（ν_r），ν_ε 值高达 ~0.5，说明大量假阳性
- **BatchTopK 几乎无伪影**: ν_r 完全不与 shared 重叠，ν_ε 均接近 0；Δnorm 与 ν 指标的 Pearson 相关高达 0.73 / 0.87
- **chat 行为集中在 template token**: BatchTopK 的 40% chat-only 潜变量主要在 template token 上激活，67% 至少 1/3 激活在 template token
- **BatchTopK 发现的可解释 chat 概念**: 有害指令请求检测、敏感内容检测、种族/性别歧视内容检测、拒绝后行为、个人问题识别、虚假信息检测、缺失信息检测、改写请求、笑话检测、回复长度度量、摘要请求、知识边界识别等
- **前 9 token 行为差异最大**: base-chat KL 散度在前 9 token 为 1.69，全 response 仅 0.482，差异集中在回复开头

## 亮点与洞察

- **从工具缺陷到方法论贡献**: 不仅发现了 L1 Crosscoder 的根本性缺陷，还同时提供了诊断工具（Latent Scaling）和根本性解决方案（BatchTopK），形成了完整的方法论闭环
- **理论与实证高度统一**: L1 损失的两类伪影机制有清晰的数学解释（L1 的 $\sqrt{x^2+y^2}$ vs SAE 的 $\sqrt{x^2+y^2}$ 梯度差异），实验数据完美验证理论预测
- **因果验证提升说服力**: 不满足于找到可解释特征，而是通过激活替换实验证明它们确实在因果上驱动 chat 行为
- **Template token 的关键角色**: 揭示了 chat 模型的独特行为很大程度上通过 template token 编码，与并行研究（Leong et al., 2025）相互验证
- **两种 Crosscoder 捕获等量信息但组织方式不同**: L1 和 BatchTopK 在全量替换时达到几乎相同的 KL 降低，但 BatchTopK 将 chat-specific 信息清晰地组织在 Δnorm 高的潜变量中，而 L1 将其混杂在所有潜变量中

## 局限与展望

- **单一模型规模**: 仅在 Gemma 2 2B 上实验，更大模型（如 7B、70B）是否面临相同问题待验证
- **仅关注 chat-only 潜变量**: base-only 和 shared 潜变量（尤其是 base/chat decoder 余弦相似度较低的潜变量）未深入分析，这些可能编码了更微妙的微调效果
- **重构误差中仍含大量信息**: BatchTopK 的 error term 在前 9 token 的 KL 中仍占约 45%，说明字典未能完全捕获 chat 行为差异
- **无法区分"新概念"与"激活偏移"**: Crosscoder 架构无法区分真正在 chat-tuning 中新学到的概念和已有概念仅改变了激活模式的情况
- **缺乏与其他稀疏方法的系统对比**: 未比较 JumpReLU SAE、Gated SAE 等近期稀疏化方法

## 相关工作与启发

- **Sparse Autoencoders (SAE)**: Crosscoder 的基础架构，BatchTopK SAE（Bussmann et al., 2024）直接被本文引入 Crosscoder 场景
- **Anthropic Crosscoder 原始工作**: Lindsey et al. (2024) 提出 Crosscoder 并首次发现 chat-only 潜变量，本文揭示其部分发现可能是伪影
- **Fine-tuning 表示稳定性**: 多项研究表明微调主要调制而非创造新能力（Jain et al., 2024; Wu et al., 2024; Merchant et al., 2020），本文的 chat-specific 潜变量发现为这一论点提供了新的精确工具
- **Template token 的安全角色**: Leong et al. (2025) 同期发现安全机制主要依赖 template token 的聚合信息，与本文发现一致

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统性识别 Crosscoder 中的训练伪影，提出 Latent Scaling 和 BatchTopK Crosscoder 两个新工具 |
| 技术深度 | ⭐⭐⭐⭐⭐ | 从 L1 损失的梯度分析到 Latent Scaling 的闭式解，再到因果干预实验，理论与实验结合极为紧密 |
| 实验充分性 | ⭐⭐⭐⭐ | 包含伪影诊断、因果验证、可解释性评估、独立复现验证等多维度实验，但仅限单一模型规模 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 直接改进了 Crosscoder 方法论的最佳实践，代码开源，对可解释性和 AI Safety 社区有即时价值 |
---
title: >-
  [论文解读] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
description: >-
  [NeurIPS 2025][Crosscoder] 识别并解决 Crosscoder 中 L1 训练损失引入的两类稀疏性伪影（导致虚假的模型特定潜变量归因），提出 Latent Scaling 诊断方法和 BatchTopK 损失替代方案，成功发现 Gemma 2 2B chat 模型中真正由 chat-tuning 引入的可解释概念。
tags:
  - NeurIPS 2025
  - Crosscoder
  - 稀疏性伪影
  - BatchTopK
  - Latent Scaling
  - chat-tuning
---

# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2504.02922](https://arxiv.org/abs/2504.02922)  
**代码**: 无  
**领域**: 可解释性 / 模型微调分析  
**关键词**: Crosscoder, 稀疏性伪影, BatchTopK, Latent Scaling, chat-tuning

## 一句话总结

识别并解决 Crosscoder 中 L1 训练损失引入的两类稀疏性伪影（导致虚假的模型特定潜变量归因），提出 Latent Scaling 诊断方法和 BatchTopK 损失替代方案，成功发现 Gemma 2 2B chat 模型中真正由 chat-tuning 引入的可解释概念。

## 研究背景与动机

**模型微调如何改变模型的内部表示和算法**是可解释性研究的核心问题之一。Crosscoder 是一种新兴的"模型差异分析"（model diffing）方法：它学习一个在基础模型和微调模型之间共享的可解释概念字典，每个概念被表示为潜在方向（latent direction），从而可以追踪概念在微调过程中如何变化或涌现。

先前研究观察到某些概念仅在微调模型中有方向而在基础模型中无方向，并假设这些"模型特定潜变量"代表微调引入的新概念。然而，本文发现这一结论可能是 **L1 稀疏化训练损失引入的伪影**——L1 惩罚会系统性地将某些本应在两个模型中都存在的概念错误地归因为仅存在于微调模型中。

核心矛盾在于：L1 损失在鼓励稀疏性的同时，会产生"缩减偏差"（shrinkage bias），使得在某个模型中激活较弱的概念被完全压缩为零，造成虚假的"模型特定"标签。

## 方法详解

### 整体框架

本文的工作分为三步：(1) 诊断——识别 L1 Crosscoder 中的两类稀疏性伪影；(2) 量化——提出 Latent Scaling 方法精确度量每个潜变量在各模型中的存在程度；(3) 解决——训练 BatchTopK Crosscoder 以实质性缓解这些问题。

### 关键设计

1. **稀疏性伪影的识别与分类**:
    - 功能：系统性地找出 L1 Crosscoder 中两类导致虚假模型特定潜变量的机制
    - 核心思路：第一类伪影是 L1 的缩减偏差将在两个模型中都存在但强度不同的概念错误归为单模型特有；第二类是 L1 在稀疏-密集权衡中选择的分解方式导致某些概念被分为一个共享潜变量+一个虚假的模型特定潜变量
    - 设计动机：如果不解决这些伪影，基于 Crosscoder 的模型差异分析结论将不可信

2. **Latent Scaling 诊断工具**:
    - 功能：为每个潜变量精确度量其在基础模型和微调模型中的"存在程度"
    - 核心思路：通过重新缩放潜变量的激活来消除 L1 缩减偏差的影响，获得更准确的跨模型存在性度量
    - 设计动机：标准 Crosscoder 的激活值被 L1 系统性偏移，直接使用会导致错误的归因

3. **BatchTopK 损失替代方案**:
    - 功能：用 BatchTopK 损失替代 L1 损失训练 Crosscoder
    - 核心思路：BatchTopK 直接限制每个 batch 中激活的潜变量数量（而非用 L1 间接鼓励稀疏），从根本上避免缩减偏差
    - 设计动机：BatchTopK 在稀疏自编码器文献中已被证明能产生更干净的特征分解，本文首次将其引入 Crosscoder

### 因果效应验证

对识别出的 chat 特定潜变量进行因果干预实验，验证这些潜变量确实能因果性地影响模型行为。

## 实验关键数据

### 主实验

| 对比维度 | L1 Crosscoder | BatchTopK Crosscoder |
|---------|--------------|---------------------|
| 虚假模型特定潜变量 | 大量（严重受伪影影响） | 显著减少 |
| 可解释性 | 混杂伪影 | 高度可解释 |
| chat 特定概念 | 含大量假阳性 | 真正 chat 特有 |

### 消融实验

- Latent Scaling 有效标记了 L1 Crosscoder 中的伪影潜变量
- BatchTopK Crosscoder 找到的 chat 特定潜变量在因果测试中表现出强因果效应

### 关键发现

- 使用 BatchTopK Crosscoder 成功识别了多个有意义的 chat 特定概念：
    - **"虚假信息"概念**：模型学会在 chat-tuning 后识别和标记虚假信息
    - **"个人问题"概念**：模型对涉及个人隐私的问题产生特定表示
    - **多个拒绝相关潜变量**：展现出对不同拒绝触发条件的细致区分
- 标准 L1 Crosscoder 将这些概念与大量伪影混在一起，难以区分

## 亮点与洞察

- **方法论贡献深远**：不仅诊断了问题，还提供了切实可行的解决方案，为整个 Crosscoder 研究社区提供了最佳实践建议
- **从工具缺陷到科学发现**：通过修复工具问题，获得了关于 chat-tuning 如何修改模型行为的具体洞察
- **因果验证提升可信度**：不仅找到可解释的特征，还验证了它们的因果效力

## 局限与展望

- 实验仅在 Gemma 2 2B base/chat 单个模型对上进行，结论是否泛化到更大模型或其他微调方式需验证
- BatchTopK 虽然缓解了两类伪影，但可能仍存在未被识别的其他类型偏差
- 缺乏与其他稀疏字典学习方法（如 top-k SAE、JumpReLU SAE）的系统对比
- Latent Scaling 的理论保证尚不完整，在极端情况下可能失效

## 相关工作与启发

- **Sparse Autoencoders (SAE)**: Crosscoder 的基础，本文的改进可能对 SAE 研究也有启发
- **Anthropic 的 Crosscoder 原始工作**: 本文直接在其基础上改进
- **BatchTopK SAE**: 来自 DeepMind 等团队的改进稀疏训练方法
- **对 AI 安全的启示**: 理解 chat-tuning 引入的拒绝和安全行为的内部机制

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统识别 Crosscoder 稀疏伪影并提出解决方案
- 实验充分度: ⭐⭐⭐ 单模型对但分析深入
- 写作质量: ⭐⭐⭐⭐ 问题-诊断-解决的叙事清晰
- 价值: ⭐⭐⭐⭐ 推进可解释性方法论的最佳实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[ACL 2025\] RecLM: Recommendation Instruction Tuning](../../ACL2025/recommender/reclm_recommendation_instruction_tuning.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](../../ACL2025/recommender/bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](../../ACL2026/recommender/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)

</div>

<!-- RELATED:END -->
