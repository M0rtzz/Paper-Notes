---
title: >-
  [论文解读] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization
description: >-
  [CVPR 2025][LLM对齐][模态偏差] NaPO 针对MLLM的模态偏差问题（过度依赖语言先验或视觉细节），通过mask模态信息构造偏差数据集RLAIF-V-Bias，并提出基于负Box-Cox变换的噪声感知偏好优化算法，在自动构造的含噪数据上实现鲁棒训练，在去偏和减幻觉上均取得显著效果。
tags:
  - "CVPR 2025"
  - "LLM对齐"
  - "模态偏差"
  - "噪声感知优化"
  - "偏好学习"
  - "MLLM去偏"
  - "幻觉缓解"
---

# Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization

**会议**: CVPR 2025  
**arXiv**: [2503.17928](https://arxiv.org/abs/2503.17928)  
**代码**: [https://github.com/zhangzef/NaPO](https://github.com/zhangzef/NaPO)  
**领域**: 对齐RLHF  
**关键词**: 模态偏差, 噪声感知优化, 偏好学习, MLLM去偏, 幻觉缓解

## 一句话总结

NaPO 针对MLLM的模态偏差问题（过度依赖语言先验或视觉细节），通过mask模态信息构造偏差数据集RLAIF-V-Bias，并提出基于负Box-Cox变换的噪声感知偏好优化算法，在自动构造的含噪数据上实现鲁棒训练，在去偏和减幻觉上均取得显著效果。

## 研究背景与动机

**领域现状**：MLLM在多种任务上表现出色，但普遍存在模态偏差（modality bias）问题——模型倾向于过度依赖某一模态的信息而忽略其他模态。

**现有痛点**：模态偏差分为两种：(1) **语言偏差**——模型依赖语言先验知识而忽视视觉输入（如看到北极熊图片仍回答"熊是棕色"）；(2) **视觉偏差**——模型过度关注视觉细节，生成与问题无关的内容（如被问"房子在左边吗？"却描述图片中大量无关视觉细节）。现有方法要么需要平衡数据集分布，要么依赖大规模有监督微调，后者有丢失已有知识的风险。

**核心矛盾**：将去偏formulate为偏好优化问题，自动构造偏差数据相对容易，但自动数据不可避免带有噪声（某些"偏差"回复实际上质量不差），标准DPO在含噪数据上容易过拟合。

**本文目标** (1) 如何自动构造有效的去偏偏好数据？(2) 如何在含噪的自动数据上进行鲁棒的偏好优化？

**切入角度**：通过mask控制信息流来生成偏差回复——遮蔽视觉信息产生语言偏差回复，遮蔽文本信息产生视觉偏差回复，然后用噪声感知的损失函数DPO来处理数据中不可避免的噪声。

**核心 idea**：用模态mask构造偏差数据，用负Box-Cox变换将BCE和MAE平滑过渡，根据数据噪声水平动态调节优化的鲁棒性。

## 方法详解

### 整体框架

方法分两步：(1) **数据构造**——在RLAIF-V数据基础上，通过遮蔽视觉/文本模态信息，让模型生成语言偏差和视觉偏差回复，构成RLAIF-V-Bias数据集；(2) **训练算法**——用NaPO算法训练，对原始偏好数据用标准DPO损失，对偏差对比数据用噪声感知的NaPO损失，并用动态权重平衡三种损失。

### 关键设计

1. **模态偏差回复生成（RLAIF-V-Bias数据集）**:

    - 功能：自动构造针对语言和视觉偏差的偏好训练数据
    - 核心思路：语言偏差回复通过 $y_{lb} = \text{MLLM}([\text{MASK}]; t)$ 生成（遮蔽所有视觉信息，模型只能依赖语言先验）；视觉偏差回复通过 $y_{vb} = \text{MLLM}(v; [\text{MASK}])$ 生成（遮蔽所有文本信息，模型只能依赖视觉内容）。最终数据集 = 原始无偏回复（正样本） + 语言偏差回复 + 视觉偏差回复（负样本）
    - 设计动机：控制信息流而非人工标注，低成本地获取偏差样本；不做显式过滤，通过后续NaPO的soft selection处理噪声

2. **噪声感知偏好优化（NaPO）**:

    - 功能：在含噪的自动构造数据上实现鲁棒的偏好优化
    - 核心思路：通过负Box-Cox变换将DPO中的BCE损失和噪声鲁棒的MAE损失统一起来。NaPO损失为 $\mathcal{L}_{\text{NaPO}} = \frac{1}{q}(1 - \sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})^q)$，其中 $q \in (0,1]$ 控制噪声鲁棒性：$q \to 0$ 趋近BCE（快速收敛但不耐噪），$q \to 1$ 趋近MAE（耐噪但收敛慢）
    - 设计动机：MAE满足对称损失性质（噪声鲁棒），BCE不满足但收敛快。通过动态调节q实现两者的最佳平衡

3. **自适应噪声系数与动态权重**:

    - 功能：根据每个样本的噪声程度自动调整q值和损失权重
    - 核心思路：关键观察是——噪声样本（被误标为偏差但实际无偏的回复）的reward margin较小，而真正有偏的样本margin较大。因此 $q = 1 - \sigma(\alpha \cdot \psi(x, y_w, y_l))$，margin大则q小（更靠近BCE，因为数据可信），margin小则q大（更靠近MAE，因为数据可能有噪声）。同时用margin值计算每种损失的权重 $\gamma_i$
    - 设计动机：语言偏差和视觉偏差数据的噪声特征不同——语言偏差用平均log概率（$\psi_\mu$, $\alpha=0.5$）区分噪声，视觉偏差用log概率之和（$\psi_\Sigma$, $\alpha=0.01$）区分

### 损失函数 / 训练策略

最终优化目标：$\mathcal{L}_\gamma = \gamma_{y_l} \cdot \mathcal{L}_{\text{DPO}}(x, y_w, y_l) + \gamma_{y_{lb}} \cdot \mathcal{L}_{\text{NaPO}}(x, y_w, y_{lb}) + \gamma_{y_{vb}} \cdot \mathcal{L}_{\text{NaPO}}(x, y_w, y_{vb})$。原始数据用DPO（高质量），偏差数据用NaPO（处理噪声），权重$\gamma$基于margin动态计算。训练配置：LLaVA-v1.5-7B，β=0.1，lr=5e-7，4 epochs，batch=4，8×A100 80GB，训练7小时。

## 实验关键数据

### 主实验

| 模型配置 | VLind CB↑ | VLind LP↑ | ObjHal CHAIR_s↓ | ObjHal CHAIR_i↓ | AMBER HalRate↓ | MMHal Score↑ |
|---------|-----------|-----------|-----------------|-----------------|----------------|-------------|
| LLaVA-v1.5-7B (基线) | 0.0 | 0.0 | 53.6 | 25.2 | 36.4 | 2.11 |
| + RLAIF-V (标准DPO) | 39.4 | 25.4 | 32.0 | 8.5 | 23.4 | 3.23 |
| + RLAIF-V-Bias (DPO) | 0.3 | 0.4 | 35.3 | 10.5 | 22.4 | 3.28 |
| **+ RLAIF-V-Bias (NaPO)** | **58.9** | **44.0** | **25.7** | **6.2** | **20.7** | **3.31** |

### 消融实验

| 配置 | VLind CB↑ | VLind LP↑ | CHAIR_s↓ | CHAIR_i↓ |
|------|-----------|-----------|----------|----------|
| Full (NaPO + 动态权重) | 58.9 | 44.0 | 25.7 | 6.2 |
| w/o 动态权重 | 50.0 | 38.2 | 27.7 | 8.0 |
| NaPO → DPO | 43.4 | 32.2 | 29.0 | 8.3 |
| 仅语言偏差数据 | 40.4 | 36.4 | 28.0 | 6.4 |
| 仅视觉偏差数据 | 62.3 | 31.4 | 26.3 | 7.6 |

### 关键发现

- **DPO在偏差数据上严重失效**：用标准DPO训练RLAIF-V-Bias数据，VLind CB仅0.3（几乎无效），甚至比原始RLAIF-V更差，说明DPO无法处理自动数据中的噪声
- **语言偏差和视觉偏差数据互补**：语言偏差数据更擅长缓解语言先验（LP +36.4），视觉偏差数据更擅长缓解常识偏差（CB +62.3）和幻觉（CHAIR_s 26.3），组合使用效果最佳
- **噪声metric选择至关重要**：对语言偏差用$\psi_\Sigma$替代$\psi_\mu$会导致CB从58.9暴跌至21.9，完全破坏效果
- **方法泛化到13B**：LLaVA-v1.5-13B上同样有效，CB从31.5提升至42.1

## 亮点与洞察

- **将去偏问题formulate为偏好优化**非常自然——偏差回复本质上就是非期望行为，DPO天然适合。核心创新在于解决了自动数据含噪的实际问题
- **负Box-Cox变换的理论分析优雅**：将MAE和BCE统一为一个连续损失族，q值提供了噪声鲁棒性的显式旋钮，理论推导严谨
- **语言/视觉偏差需要不同噪声metric**的发现很有实用价值——不同类型噪声的分布特征不同

## 局限与展望

- 仅在LLaVA-v1.5上验证，未测试更先进的MLLM（如Qwen-VL、InternVL）
- 将所有NaPO替换DPO会导致性能崩溃（表6），说明NaPO对数据质量仍有假设
- 偏差是否总是有害值得讨论——某些特定场景中适度偏差可能有益
- 噪声系数α的选择依赖手动调参，自动化程度有限

## 相关工作与启发

- **vs RLAIF-V**: RLAIF-V做通用偏好优化，本文专门针对模态偏差构造数据。同样数据量下RLAIF-V-Bias + NaPO全面超越RLAIF-V（CB +19.5, CHAIR_s -6.3）
- **vs 标准DPO**: DPO在自动构造的含噪数据上完全失效（CB仅0.3），NaPO通过动态噪声鲁棒性解决了这一问题
- **vs 数据过滤方法**: 本文不做显式数据过滤，而是通过损失函数层面的soft selection处理噪声，更优雅且不丢信息

## 评分

- 新颖性: ⭐⭐⭐⭐ 负Box-Cox变换统一BCE/MAE的理论框架有创新，但模态mask构造偏差数据较直接
- 实验充分度: ⭐⭐⭐⭐ 4个benchmark、详细消融、噪声metric分析充分，但模型覆盖度有限
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号系统较复杂
- 价值: ⭐⭐⭐⭐ NaPO算法可泛化到其他含噪偏好优化场景，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ICML 2025\] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](../../ICML2025/llm_alignment/mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [\[CVPR 2025\] SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)

</div>

<!-- RELATED:END -->
