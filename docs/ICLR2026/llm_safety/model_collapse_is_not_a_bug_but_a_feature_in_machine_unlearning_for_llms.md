---
title: >-
  [论文解读] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs
description: >-
  [ICLR 2026][LLM安全][machine unlearning] 将通常被视为负面现象的"模型坍缩"（model collapse）重新定位为机器遗忘的工具，提出PMC方法——通过在保留数据和模型自身生成数据上迭代微调来实现针对性信息删除，无需在遗忘目标上直接优化，从理论和实验两方面证明了其有效性…
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "machine unlearning"
  - "model collapse"
  - "partial model collapse"
  - "LLM privacy"
  - "iterative relearning"
---

# Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs

**会议**: ICLR 2026  
**arXiv**: [2507.04219](https://arxiv.org/abs/2507.04219)  
**代码**: [TUM DAML - Partial Model Collapse](https://www.cs.cit.tum.de/daml/partial-model-collapse/)  
**领域**: 图像生成  
**关键词**: machine unlearning, model collapse, partial model collapse, LLM privacy, iterative relearning

## 一句话总结

将通常被视为负面现象的"模型坍缩"（model collapse）重新定位为机器遗忘的工具，提出PMC方法——通过在保留数据和模型自身生成数据上迭代微调来实现针对性信息删除，无需在遗忘目标上直接优化，从理论和实验两方面证明了其有效性。

## 研究背景与动机

隐私法规（如GDPR）要求能够从机器学习模型中选择性删除特定数据的影响。对于LLM而言，完全重训是计算上不可行的，因此需要高效的**机器遗忘**（machine unlearning）技术。

现有LLM遗忘方法存在一个根本性问题：它们**反直觉地依赖要删除的数据本身**来进行遗忘优化。例如梯度上升法（GA）对遗忘目标进行反向训练，NPO（负偏好优化）将遗忘目标作为负面样本。这种做法有两个严重问题：

**违反最小化使用原则**：遗忘过程仍然在使用敏感数据，增加了数据暴露风险

**副作用不明**：可能导致对抗者通过概率探测来推断被遗忘的信息（信息泄漏）

本文的核心洞察来自于**模型坍缩**现象——当生成模型在自己生成的数据上迭代训练时，输出分布会逐渐坍缩，有效地丢失信息。如果能**部分地、可控地**触发这种坍缩，就能实现不接触敏感数据的遗忘。

## 方法详解

### 整体框架

**Partial Model Collapse（PMC）** 把"模型坍缩"反过来用：每一轮迭代里，对保留问题用真实答案做标准微调当作"锚"，对遗忘问题则不碰任何 ground truth，而是让模型自己采样若干回复、用 Bradley-Terry 偏好模型挑出"遗忘得最干净"的那条来微调。如此反复迭代，被遗忘的信息在保留数据的锚定下逐步、可控地坍缩消失，而模型其余能力保持不变。

### 关键设计

**1. 从完全坍缩到部分坍缩：用保留数据当锚，让坍缩只发生在该忘的地方。**

纯粹的迭代自训练会把整个输出分布拖向完全坍缩，所有类别信息一并丢失，这显然不能直接用来做遗忘。PMC 的破局点（Lemma 1）在于引入保留数据作为"锚定"项：当训练目标同时包含保留分布时，迭代不动点不再是退化的单点分布，而是一个**部分坍缩**态——保留类别的概率质量被牢牢钉住不变，只有非保留类别的概率被持续推向零。这正好对应机器遗忘的诉求：留下的要留住，要忘的要忘干净，且整个过程从不直接对遗忘目标做优化。

**2. 连续分布下的指数收敛保证：证明部分坍缩确实收敛到保留分布。**

把上面的直觉推广到连续分布，PMC 将每一步迭代写成一个加权的最大似然问题：

$$p_{t+1} = \arg\min_{p \in \mathcal{P}} \frac{\alpha}{1+\alpha} \mathbb{E}_{x \sim p_r}[-\log p(x)] + \frac{1}{1+\alpha} \mathbb{E}_{x \sim p_t}[-\log p(x)]$$

其中 $p_r$ 是保留分布，$\alpha$ 控制锚定强度。Theorem 2 证明，在无统计误差的理想假设下，$p_t$ 会以速率 $\frac{1}{1+\alpha}$ 指数收敛到 $p_r$，并给出闭式解 $p_t(x) = [1 - (\frac{1}{1+\alpha})^t] p_r(x) + (\frac{1}{1+\alpha})^t p_0(x)$。这个闭式解直观地说明了机制：初始分布 $p_0$ 的成分按几何级数被稀释，最终只剩保留分布——遗忘是收敛过程的自然产物，而非外加的压制。

**3. 基于 Bradley-Terry 偏好的 Q&A 遗忘：用偏好模型为坍缩"导航"。**

在真实的问答场景里，"非保留类别概率趋零"需要落地成具体的训练样本。PMC 为每个遗忘问题采样 $n$ 个候选回复，用 Bradley-Terry 偏好模型 $\mathcal{BT}$ 从中选出遗忘质量最高的那条 $\hat{x}$ 加入训练：

$$\theta_{t+1} = \arg\min_\theta \lambda \mathbb{E}_{(q,x) \in D_r}[-\log f_\theta(x|q)] + \mathbb{E}_{q \in D_f, \hat{x} \sim \mathcal{BT}}[-\log f_\theta(\hat{x}|q)]$$

第一项是保留集 $D_r$ 的锚定，第二项让模型逐步把概率质量转移到"更该说"的遗忘回复上。Corollary 3 给出理论保证：期望奖励收敛到最大值 $e^{r^*}$，方差趋向零，意味着随着迭代推进，被选中的回复会越来越稳定地朝最优遗忘方向靠拢。

**4. 奖励函数与 argmax 近似：用与 ground truth 的距离衡量"忘得多干净"。**

偏好模型需要一个奖励信号来定义"遗忘质量"，PMC 取 $r(x) = 1 - \text{ROUGE-L}(\hat{x}, y)$——回复 $\hat{x}$ 与遗忘问题真实答案 $y$ 的 ROUGE-L 越低，奖励越高，即越偏离正确答案就越"忘得干净"。注意这里 $y$ 只用于打分排序、从不进入梯度，因此仍不算"在遗忘目标上优化"。实现上，完整的 BT 采样可用 argmax 近似替代，即直接选取奖励最高的回复；消融中 BT 温度 $\tau \to 0$ 时遗忘质量最优，验证了这一近似的合理性。

## 实验关键数据

实验在 TOFU 数据集的"forget10"分割（400 个遗忘样本）上、用 Phi-1.5 和 Llama-3.2-3B-Instruct 两个模型进行，每种方法跑 100 种超参配置 × 5 个随机种子共 500 次。核心指标为 ROUGE-L recall：遗忘质量 UQ = 最大分数 − 实际分数（越大越好），效用 = 保留集 + 世界知识 + 真实作者问答三部分的 ROUGE-L 总分。

### 主实验：Pareto前沿

| 方法 | 遗忘质量(UQ) | 效用 | Pareto表现 |
|------|------------|------|-----------|
| GA/GD/NPO/SimNPO | 低-中 | 中-高 | NPO最佳baseline |
| IDK | 中 | 中 | 简单但有效 |
| **PMC (Ours)** | **最高** | **最高** | **显著扩展Pareto前沿** |

PMC在Phi-1.5上实现了同时超越所有baseline的效用-遗忘质量权衡；在Llama-3.2-3B-Instruct上，遗忘后模型能精准拒绝仅遗忘问题的回答，同时保持其他能力。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 迭代轮数(2→20) | UQ持续提升，效用几乎不变 | PMC优势在于更多轮次不牺牲效用 |
| 样本数(1→20) | 更多样本=更好UQ，前6轮效用无影响 | 但大样本量增加方差 |
| λ(0.5→1.5) | 大λ提升效用但降低UQ | 需仔细平衡 |
| 温度(1.0→1.5) | 高温提升UQ | >1.5开始损失效用 |
| BT温度τ | τ→0时UQ最优 | 验证了argmax近似的合理性 |

### 现有方法的副作用分析

| 副作用 | NPO | PMC |
|--------|-----|-----|
| 无关数据集token概率偏移 | 严重偏左(均值-0.12) | 零均值高斯(无偏) |
| 多选题最低概率选项准确率 | 低量化区间高准确率(泄漏) | 无规律(无泄漏) |
| 最小概率分布 | 聚集在零附近 | 正常分布 |

### 关键发现

- **PMC不直接优化遗忘目标**是其核心优势：模型在遗忘问题上生成"I don't know"或相关但无害的回答，而非简单压制正确答案概率
- **信息泄漏问题**：NPO等target-dependent方法创建了对抗者可利用的"信号"——通过选择概率最低的选项就能还原被遗忘的知识
- **生成连贯性**：NPO在wikitext等无关数据集上大幅降低遗忘token的生成概率，影响正常文本生成；PMC无此问题
- **Llama-3.2效果更好**：指令对齐模型能学会精准拒绝遗忘问题同时保持其他回答不变

## 亮点与洞察

- **范式创新**：将model collapse从"bug"重新定义为"feature"，开创了基于坍缩的遗忘新范式
- **不接触敏感数据**的遗忘方法在隐私约束更严格的场景（如GDPR）中具有独特优势
- **理论完整性**：从离散类别分布（Lemma 1）→ 连续分布（Theorem 2）→ Q&A场景（Corollary 3）的渐进推导逻辑清晰
- **揭示现有方法的隐藏风险**：概率探测攻击和跨上下文token概率偏移是重要但被忽视的评估维度

## 局限与展望

- **计算开销较大**：每个遗忘问题需要采样n个回复，对大模型的推理成本显著（Phi-1.5平均约5小时/配置）
- **奖励函数设计依赖场景**：当前使用ROUGE-L，实际应用中可能需要更精细的奖励设计
- **评估局限**：遗忘评估仍是开放问题，本文仅关注抑制特定输出+保持效用，未涉及对重学习（relearning）的鲁棒性
- **仍需接触遗忘问题的prompt**：虽然不使用答案，但仍需要知道哪些问题需要遗忘
- **GPT-4评估的可靠性**：部分评估依赖GPT-4打分，可能引入偏差

## 相关工作与启发

本文巧妙结合了两个看似无关的领域：**模型坍缩**研究（Shumailov et al. 2023; Bertrand et al. 2024）和**机器遗忘**研究（NPO, SimNPO等）。Ferbach et al. (2024)关于curated self-generated data导致坍缩的理论工作为PMC提供了关键理论基础。方法论上，将Bradley-Terry偏好模型作为"导航工具"引导坍缩方向，与DPO系列工作形成有趣的对比——DPO优化偏好以增强能力，PMC利用偏好实现遗忘。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （将model collapse变为unlearning工具，范式突破）
- 实验充分度: ⭐⭐⭐⭐ （500次实验×2模型，但仅一个数据集TOFU）
- 写作质量: ⭐⭐⭐⭐⭐ （理论推导清晰，问题动机强，副作用分析亮眼）
- 价值: ⭐⭐⭐⭐ （隐私合规场景的实用价值高，但计算成本待解决）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[NeurIPS 2025\] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](../../NeurIPS2025/llm_safety/unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[CVPR 2025\] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](../../CVPR2025/llm_safety/forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)

</div>

<!-- RELATED:END -->
