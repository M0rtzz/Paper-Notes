---
title: >-
  [论文解读] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders
description: >-
  [ICCV 2025][多模态VLM][概念遗忘] SAUCE 利用稀疏自编码器（SAE）在 VLM 的中间表征中识别并选择性抑制与目标概念相关的特征，实现了无需权重更新的细粒度概念遗忘，在 60 个概念的测试中遗忘质量超越 SOTA 18%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "概念遗忘"
  - "稀疏自编码器"
  - "视觉语言模型"
  - "机器遗忘"
  - "细粒度控制"
---

# SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders

**会议**: ICCV 2025  
**arXiv**: [2503.14530](https://arxiv.org/abs/2503.14530)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 概念遗忘、稀疏自编码器、视觉语言模型、机器遗忘、细粒度控制

## 一句话总结

SAUCE 利用稀疏自编码器（SAE）在 VLM 的中间表征中识别并选择性抑制与目标概念相关的特征，实现了无需权重更新的细粒度概念遗忘，在 60 个概念的测试中遗忘质量超越 SOTA 18%。

## 研究背景与动机

**领域现状**：随着 VLM（如 LLaVA、LLaMA-Vision）的广泛部署，如何让模型"忘记"特定概念（如有害内容、版权内容）变得日益重要。现有的 VLM 遗忘方法大多直接继承自 LLM 领域的技术路线。

**现有痛点**：当前遗忘方法主要依赖权重更新策略，存在两个核心问题：一是需要大量标注的"遗忘集"数据，获取成本高；二是粒度过粗，在遗忘目标概念的同时往往导致过度遗忘，严重损害模型在无关任务上的效用。

**核心矛盾**：遗忘的精准性与模型效用之间存在根本性的 trade-off。基于权重更新的方法无法精确定位哪些神经元或特征编码了目标概念，只能在整个参数空间中进行粗粒度的调整。

**本文目标**：设计一种不需要修改模型权重、不需要大量标注数据、能在推理时按需执行的细粒度概念遗忘方法。

**切入角度**：稀疏自编码器（SAE）能将神经网络的高维密集表征解耦为高维稀疏的、语义可解释的特征。如果能用 SAE 识别出与目标概念最相关的特征维度，就可以在推理时选择性压制这些维度。

**核心 idea**：用 SAE 将 VLM 的中间层表征分解为可解释的稀疏特征，通过相关性分析定位目标概念的关键特征，在推理时选择性修改这些特征来实现遗忘——不改模型权重，只改推理时的特征传播。

## 方法详解

### 整体框架

SAUCE 的整体流程分为三个阶段：（1）训练 SAE 学习 VLM 中间层的稀疏表征；（2）给定目标遗忘概念，通过特征相关性分析识别最关键的稀疏特征维度；（3）在推理时，通过抑制这些特征维度来阻止模型输出与目标概念相关的内容。输入是正常的图文对，输出是经过"概念遗忘"处理的模型响应。

### 关键设计

1. **稀疏自编码器训练**:

    - 功能：将 VLM 中间层的密集激活分解为高维稀疏表征
    - 核心思路：在 VLM 的关键中间层（如注意力层输出）上训练 SAE，编码器将 $d$ 维密集向量映射到 $D$ 维（$D \gg d$）的稀疏空间，解码器将其还原。通过 L1 正则化确保稀疏性，使得每个特征维度趋向于编码独立的语义概念
    - 设计动机：密集表征中概念信息是纠缠的，无法精确定位；SAE 通过过度完备（overcomplete）表征和稀疏约束，自然实现了概念级别的解耦

2. **概念相关特征识别**:

    - 功能：在数千个稀疏特征中找出与目标遗忘概念最相关的少量特征
    - 核心思路：收集包含目标概念的少量样本（无需大规模标注集），将其通过 VLM + SAE 得到稀疏激活，统计每个特征维度在目标概念样本上的平均激活值。激活值最高的 top-k 特征即为概念相关特征
    - 设计动机：SAE 的稀疏性保证了大部分特征维度在特定概念输入下不会激活，因此高激活的特征具有强概念选择性，这种统计方法简单有效

3. **推理时特征抑制**:

    - 功能：在不修改模型权重的情况下实现概念遗忘
    - 核心思路：推理时，将中间层激活通过 SAE 编码为稀疏表征，将已识别的概念相关特征维度的激活值置零或缩放（clamping），再通过 SAE 解码器还原为密集表征并传回模型。这相当于在信号传播路径中"切断"了概念相关的信息流
    - 设计动机：相比重新训练或微调模型，推理时干预零成本、可即时启用/禁用、支持同时处理多个遗忘请求，且不影响模型权重

### 损失函数 / 训练策略

SAE 训练使用标准的重建损失加 L1 稀疏惩罚：$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$，其中 $x$ 是原始激活，$\hat{x}$ 是重建激活，$z$ 是稀疏编码。概念遗忘阶段本身不涉及训练。

## 实验关键数据

### 主实验

实验覆盖两个 VLM（LLaVA-v1.5-7B、LLaMA-3.2-11B-Vision-Instruct）和两类遗忘任务（具象概念如物体/运动场景、抽象概念如情绪/颜色/材质），共 60 个概念。

| 方法 | 遗忘质量 (UQ↑) | 模型效用 (MU↑) | 综合得分 |
|------|---------------|---------------|---------|
| Gradient Ascent | 基线 | 显著下降 | 较低 |
| Fine-tuning w/ Forget Set | 中等 | 中等下降 | 中等 |
| **SAUCE** | **+18.04%** | **可比** | **最优** |

### 消融实验

| 配置 | 遗忘质量 | 模型效用 | 说明 |
|------|---------|---------|------|
| Full SAUCE | 最优 | 保持 | 完整方法 |
| 随机特征抑制 | 低 | 下降 | 证明特征识别的必要性 |
| 全部特征抑制 | 高遗忘 | 严重下降 | 过度遗忘 |
| 不同 top-k 值 | 随 k 变化 | 随 k 变化 | k 过大导致效用下降 |

### 关键发现

- SAUCE 在遗忘质量上超越 SOTA 方法 18.04%，同时保持了可比的模型效用
- 具象概念（如"猫"、"篮球"）比抽象概念（如"悲伤"、"金属质感"）更容易遗忘，因为具象概念在 SAE 特征空间中更集中
- SAUCE 对对抗攻击（试图恢复已遗忘概念的提示攻击）表现出较好的鲁棒性
- 在一个 VLM 上训练的 SAE 发现的概念相关特征具有一定的跨模型迁移能力
- 支持同时处理多个概念的遗忘请求，且性能退化很小

## 亮点与洞察

- **推理时干预范式**：与主流的权重修改方法完全不同，SAUCE 在推理时通过特征抑制实现遗忘，这意味着可以动态开启/关闭遗忘功能，对部署场景非常友好
- **SAE 作为可解释性工具**：该工作展示了 SAE 不仅可以用于模型解释，还可以作为精确控制模型行为的工具，这种思路可以迁移到其他需要细粒度模型控制的任务
- **低数据需求**：只需少量目标概念样本就能识别相关特征，避免了传统遗忘方法对大规模标注遗忘集的依赖

## 局限与展望

- 论文已被作者撤回，注释称"需要更多对比实验"，说明实验覆盖可能不够全面
- SAE 需要额外的训练成本，且 SAE 的质量直接影响遗忘效果
- 推理时加入 SAE 编解码步骤会增加延迟开销
- 对于深度纠缠的概念（如"人脸"与"年龄/性别"），选择性遗忘的效果可能受限
- 未来可以探索：自适应 top-k 选择策略、多层 SAE 协同遗忘、与 RLHF 安全对齐的结合

## 相关工作与启发

- **vs Gradient Ascent 系列**：传统方法通过在遗忘数据上最大化损失来"忘记"，但这会破坏模型权重的整体结构。SAUCE 完全不修改权重，从根本上避免了这个问题
- **vs EraseDiff / Concept Erasing**：这些方法针对扩散模型设计，不直接适用于自回归 VLM。SAUCE 是第一个专门为 VLM 设计的推理时遗忘方法
- **vs 机械可解释性（Mechanistic Interpretability）**：SAE 在 LLM 可解释性中已被广泛使用，SAUCE 将其从"理解模型"延伸到"控制模型"，是一个有意义的应用方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 推理时用 SAE 做概念遗忘是新颖的角度，但 SAE 本身是已有工具
- 实验充分度: ⭐⭐⭐ 论文已撤回，作者自认实验不够充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述较完整
- 价值: ⭐⭐⭐⭐ 推理时干预的范式对实际部署有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](../../ACL2026/llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](../../NeurIPS2025/llm_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](../../NeurIPS2025/llm_safety/approximate_domain_unlearning_for_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
