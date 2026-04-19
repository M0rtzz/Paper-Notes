---
title: >-
  [论文解读] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs
description: >-
  [NeurIPS 2025][AI安全][对抗训练] 提出MixAT方法，将离散对抗攻击（PAP改写）与连续嵌入空间扰动相结合进行LLM对抗训练，在保持高效用的同时实现对多种攻击的鲁棒性（ALO-ASR从50%+降至20%以下），且训练成本仅与纯连续方法相当。
tags:
  - NeurIPS 2025
  - AI安全
  - 对抗训练
  - LLM安全
  - 鲁棒性
  - 越狱攻击
  - 连续扰动
---

# MixAT: Combining Continuous and Discrete Adversarial Training for LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2505.16947](https://arxiv.org/abs/2505.16947)  
**代码**: [GitHub](https://github.com/insait-institute/MixAT)  
**领域**: AI安全  
**关键词**: 对抗训练, LLM安全, 鲁棒性, 越狱攻击, 连续扰动

## 一句话总结

提出MixAT方法，将离散对抗攻击（PAP改写）与连续嵌入空间扰动相结合进行LLM对抗训练，在保持高效用的同时实现对多种攻击的鲁棒性（ALO-ASR从50%+降至20%以下），且训练成本仅与纯连续方法相当。

## 研究背景与动机

尽管LLM经过安全对齐训练，当前的对抗攻击仍能持续迫使前沿模型生成有害内容。对抗训练（Adversarial Training, AT）是提升传统ML模型鲁棒性的有效手段，但在LLM场景下面临独特挑战：

**离散攻击训练的困境**：GCG等基于token的对抗攻击能有效产生有害内容，但在训练中使用代价极高。R2D2方法仅训练7B模型就需要100+ GPU小时，因为每次迭代都需运行完整的对抗搜索。

**连续攻击训练的局限**：CAT等方法在嵌入空间进行连续扰动，计算效率高，但扰动后的嵌入不对应任何真实文本序列。这导致连续方法无法涵盖离散攻击所利用的全部漏洞——以CAT训练的模型在面对PAP越狱攻击时ASR仍高达40%。

核心洞察：连续扰动以原始良性输入为中心构建 $\epsilon$-球，而真正的离散对抗样本可能在这个球之外。如果将连续扰动的中心点转移到离散对抗样本上，就能覆盖更广的对抗嵌入空间（如论文Fig.1(a)所示），同时兼顾效率。

## 方法详解

### 整体框架

MixAT将对抗扰动空间定义为离散改写与连续扰动的叠加：

$$\mathcal{N}_{\text{MixAT}}(\mathbf{x}) = \underbrace{\mathcal{R}(\mathbf{x})}_{\text{离散改写}} + \underbrace{\mathcal{B}^2(0, \epsilon)}_{\text{连续}\epsilon\text{-球}}$$

即先对有害prompt进行离散改写（如PAP策略生成的重述），再在改写后的嵌入上叠加连续扰动。这使得训练样本覆盖的对抗区域从"良性输入的邻域"扩展到"对抗样本的邻域"。

### 关键设计

1. **离散种子生成（PAP-AT）**：使用PAP（Persuasion-based Adversarial Prompts）的对抗训练变体生成离散对抗种子。PAP利用40种预定义的说服策略对有害请求进行改写，优势是生成成本低（仅需API调用）、攻击多样且强度高。每轮训练动态生成新的PAP样本（而非静态预生成），因为消融实验证明动态生成对鲁棒性至关重要（静态版ALO-ASR从12.5%升至25%）。

2. **连续扰动叠加（基于CAT）**：在离散种子的token嵌入上施加L2范数约束的连续扰动 $\delta$，通过投影梯度下降（PGD）优化。与标准CAT不同，扰动中心不再是原始良性输入而是离散对抗样本，使得 $\epsilon$-球被推入"更对抗"的嵌入区域。经验分析（Fig.6）证实，PAP+连续扰动产生的prompt与原始恶意请求的余弦相似度最低，同时更接近GCG样本——解释了MixAT对未见攻击的泛化能力。

3. **批次级采样策略**：使用混合参数 $\alpha \in [0,1]$ 控制两种扰动的比例。每个训练batch中，以概率 $P_{C+D} = \alpha$ 在离散种子上叠加连续扰动，以概率 $P_C = 1-\alpha$ 在原始输入上施加连续扰动。默认 $\alpha=0.5$ 作为平衡直接恶意请求与改写攻击的最优折衷。

### 损失函数 / 训练策略

采用Mazeika等人的三部分损失函数：

$$\mathcal{L}_{\text{adv}} = \underbrace{\mathbb{E}[\log P_\theta(\hat{\mathbf{y}}|\hat{\mathbf{x}})]}_{\mathcal{L}_{\text{away}}：降低有害响应概率} \underbrace{- \mathbb{E}[\log P_\theta(\mathbf{y}_s|\hat{\mathbf{x}})]}_{\mathcal{L}_{\text{toward}}：提高安全响应概率} \underbrace{- \mathbb{E}[\log P_\theta(\mathbf{y}|\mathbf{x})]}_{\mathcal{L}_{\text{util}}：维持通用能力}$$

其中 $\hat{\mathbf{x}}$ 为对抗输入，$\hat{\mathbf{y}}$ 为有害响应，$\mathbf{y}_s$ 为安全响应。$\mathcal{L}_{\text{util}}$ 使用额外的效用数据集 $\mathcal{D}_u$ 防止模型忘记通用能力。训练采用LoRA适配器以降低内存需求。

**ALO-ASR指标**：提出"至少一种攻击成功率"（At Least One ASR），衡量元对手使用所有攻击方法的最坏情况成功率，比单一攻击ASR更能反映真实安全风险。

## 实验关键数据

### 主实验（Zephyr-7B）

| 方法 | ARCe↑ | MMLU↑ | 直接请求↓ | PAP↓ | GCG↓ | ALO-ASR↓ |
|------|-------|-------|----------|------|------|---------|
| 无防御 | 81.0 | 56.2 | 85.0 | 87.5 | 85.0 | 100.0 |
| R2D2 | 80.1 | 56.1 | 7.5 | 65.0 | 0.0 | 77.5 |
| CAT | 78.2 | 54.8 | 2.5 | 40.0 | 5.0 | 70.0 |
| LAT SFT | 31.7 | 22.9 | 5.0 | 30.0 | 20.0 | 52.5 |
| DualAT | 81.8 | 56.1 | 2.5 | 2.5 | 10.0 | 22.5 |
| **MixAT** | **81.4** | **55.8** | **0.0** | **0.0** | 12.5 | **15.0** |
| MixAT+GCG | 81.6 | 55.9 | 2.5 | 0.0 | 2.5 | 7.5 |

### 跨模型泛化

| 模型 | 无防御ALO↓ | CAT ALO↓ | MixAT ALO↓ | MixAT效用保持 |
|------|-----------|---------|-----------|-------------|
| Zephyr-7B | 100.0 | 70.0 | 15.0 | 效用仅降1-2点 |
| Llama3-8B | 90.0 | 82.5 | 25.0 | 效用仅降0.5-1.5点 |
| Qwen2.5-14B | 100.0 | 92.5 | 15.0 | 效用轻微提升 |
| Qwen2.5-32B | 100.0 | 82.5 | 7.5 | 效用无显著变化 |

### 关键发现

- **MixAT vs DualAT**：MixAT在对抗样本上叠加连续扰动（组合式攻击）显著优于分别训练两种攻击（DualAT），证实了将连续扰动中心移至离散对抗样本的核心假设
- **动态 vs 静态训练**：动态生成PAP样本的MixAT（ALO 12.5%）远优于静态版本（ALO 25%），说明模型需持续遇到新的对抗模式
- **LoRA缩放分析**：通过调节LoRA权重 $\lambda \in [0,1.5]$，可连续调控鲁棒性-效用权衡，MixAT在所有 $\lambda$ 下均优于CAT
- **温度效应**：随温度升高，平均ASR变化不大，但"至少产生一次有害响应"的概率显著上升，提示采样多次会大幅降低表观安全性
- **训练成本**：MixAT仅比CAT多不到1美元的API调用费用，远低于R2D2的100+ GPU小时

## 亮点与洞察

- **理论直觉+实证验证的闭环**：从嵌入空间的直觉出发，通过余弦相似度分析验证，最终在6类攻击上实证确认
- **ALO-ASR指标的实用意义**：揭示了许多看似鲁棒的防御在面对多种攻击组合时的脆弱性（如R2D2抗GCG强但抗PAP弱）
- **审计现有方法**：系统性地揭示了chat template、量化、非零温度等实际部署因素对防御评估的影响，指出社区评测中的盲点

## 局限与展望

- 对GCG攻击的防御仍是相对弱点（12.5% ASR），加入GCG样本可降至2.5%但训练成本增加5倍
- 仅使用PAP作为离散种子，更多元化的离散攻击方法（如AutoDAN、TAP）可能进一步提升防御覆盖
- 对抗训练可能导致过度拒绝（over-refusal），虽然XSTest测试有所评估但在更细粒度的场景中可能仍存问题
- 未考虑模型篡改攻击（如权重修改），这属于更强的威胁模型

## 相关工作与启发

- R2D2和CAT分别代表离散和连续对抗训练的两极，MixAT的贡献在于找到有效的组合方式
- Constitutional AI和Constitutional Classifiers提供正交的防御路径（外部过滤 vs 模型内在鲁棒性）
- 本工作启发了一个重要思路：对于LLM安全，单一类型的防御难以覆盖所有攻击面，组合式防御可能是必经之路

## 评分

- **新颖性**: ⭐⭐⭐⭐ 组合连续+离散对抗训练的思路自然但有效，核心贡献在于验证了"在对抗样本上做连续扰动"优于"分别训练"
- **实验充分度**: ⭐⭐⭐⭐⭐ 4个模型规模、7种攻击方法、多维度消融、成本分析，非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ 可视化出色，ALO-ASR指标设计合理，审计部分值得社区关注
- **价值**: ⭐⭐⭐⭐⭐ 为LLM对抗训练设立了新的强基线，对安全研究社区有直接影响

<!-- RELATED:START -->

## 相关论文

- [Distributional Adversarial Attacks and Training in Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)
- [On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)
- [Identifying and Understanding Cross-Class Features in Adversarial Training](../../ICML2025/ai_safety/identifying_and_understanding_cross-class_features_in_adversarial_training.md)
- [CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](../../ACL2025/ai_safety/cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)

<!-- RELATED:END -->
