---
title: >-
  [论文解读] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information
description: >-
  [ACL 2025 (COLING 2025)][模型压缩][结构化剪枝] 本文提出CFSP框架，利用粗粒度（block间）和细粒度（block内）的激活信息作为重要性准则指导LLM的结构化剪枝，仅需一次前向传播即可完成剪枝，在多个模型和稀疏度预算上优于现有方法。
tags:
  - "ACL 2025 (COLING 2025)"
  - "模型压缩"
  - "结构化剪枝"
  - "LLM加速"
  - "激活信息"
  - "粗粒度-细粒度"
  - "稀疏性分配"
---

# CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information

**会议**: ACL 2025 (COLING 2025)  
**领域**: 模型压缩  
**关键词**: 结构化剪枝, LLM加速, 激活信息, 粗粒度-细粒度, 稀疏性分配

## 一句话总结
本文提出CFSP框架，利用粗粒度（block间）和细粒度（block内）的激活信息作为重要性准则指导LLM的结构化剪枝，仅需一次前向传播即可完成剪枝，在多个模型和稀疏度预算上优于现有方法。

## 研究背景与动机

**领域现状**：大语言模型的参数量和计算开销日益庞大，制约了其在实际场景中的部署。网络剪枝是一种有效的模型压缩手段，通过移除冗余参数实现加速。目前LLM剪枝主要分为非结构化剪枝（如SparseGPT、Wanda）和结构化剪枝两大方向。

**现有痛点**：非结构化剪枝虽然在保持性能方面表现较好，但产生的稀疏权重矩阵需要专用硬件支持才能实现真正的推理加速，在通用GPU上无法获得实际speedup。结构化剪枝可以在通用设备上直接降低延迟，但在高稀疏率下维持模型性能是一个挑战。现有结构化剪枝方法往往对所有block统一分配稀疏率，忽略了不同block的重要性差异。

**核心矛盾**：结构化剪枝需要在"剪枝效率"和"性能保持"之间取得平衡——剪枝过程本身不应过于耗时（如需要大量校准数据或多次迭代），同时在高稀疏率下也应维持可接受的性能。

**本文目标**：设计一个高效且有效的结构化剪枝框架，能够（1）快速确定剪枝策略（仅需一次前向传播），（2）自适应地在不同block间分配稀疏预算，（3）在高稀疏率下保持较好的性能。

**切入角度**：作者观察到不同Transformer block对最终输出的贡献差异很大，激活值（activation）可以很好地反映权重的重要性。通过同时利用block间（粗粒度）和block内（细粒度）的激活信息，可以更准确地评估参数重要性。

**核心 idea**：用层间激活分布指导稀疏预算分配，用层内激活值指导具体权重保留，形成"粗到细"的两级剪枝策略。

## 方法详解

### 整体框架
CFSP采用两阶段剪枝策略：首先在粗粒度层面，根据各Transformer block的激活重要性分配不同的稀疏预算（重要的block剪得少，不重要的剪得多）；然后在细粒度层面，在每个block内部根据激活信息保留最重要的权重。最后通过自适应的恢复微调进一步提升性能。

### 关键设计

1. **粗粒度block重要性评估（Coarse-grained Block Importance）**:

    - 功能：评估每个Transformer block对模型输出的贡献，用于稀疏预算的差异化分配
    - 核心思路：使用少量校准数据进行一次前向传播，计算每个block输出激活的L2范数作为重要性指标 $I_k = \|A_k\|_2$，其中 $A_k$ 是第 $k$ 个block的输出激活。范数越大说明该block对信息传递的贡献越大。然后根据重要性分数按比例分配稀疏预算：重要性高的block分配较低的稀疏率，重要性低的block分配较高的稀疏率
    - 设计动机：统一的稀疏率分配忽略了层间差异，会导致关键层被过度剪枝而崩塌，差异化分配可以在不增加总稀疏率的前提下更好地保持性能

2. **细粒度权重保留策略（Fine-grained Weight Retention）**:

    - 功能：在每个block内部确定哪些具体的结构（attention heads、FFN neurons）应被保留
    - 核心思路：对每个block内的attention heads和FFN中间层neurons，计算其对应激活通道的重要性分数。使用激活幅度（activation magnitude）乘以对应权重的范数作为联合重要性指标。按照粗粒度阶段分配到该block的稀疏预算，保留重要性最高的结构单元
    - 设计动机：激活和权重的联合评估比单独使用任一指标更准确，因为权重大不代表该通道会被激活，激活大不代表该权重本身重要

3. **自适应恢复微调（Adaptive Recovery Fine-tuning）**:

    - 功能：对剪枝后的模型进行轻量级微调以恢复部分性能损失
    - 核心思路：根据粗粒度重要性分数为不同block分配不同的微调学习率和训练步数——重要性低的block（被剪枝更多）分配更大的学习率和更多的训练资源，因为它们需要更多的调整来适应剪枝后的结构。使用少量校准数据即可完成恢复
    - 设计动机：均匀分配微调资源是次优的，自适应分配能让受损最大的部分得到最多的修复资源

### 损失函数 / 训练策略
恢复微调采用标准的语言建模损失（next token prediction），在少量校准样本上进行短时间训练。关键创新在于学习率按block重要性反比分配。

## 实验关键数据

### 主实验

| 模型 | 方法 | 稀疏率 | WikiText2 PPL↓ | PTB PPL↓ | 平均Zero-shot Acc↑ |
|------|------|--------|---------------|----------|-------------------|
| LLaMA-7B | Dense | 0% | 5.68 | 8.80 | 65.3 |
| LLaMA-7B | LLM-Pruner | 50% | 11.23 | 16.45 | 58.1 |
| LLaMA-7B | SliceGPT | 50% | 10.87 | 15.92 | 58.7 |
| LLaMA-7B | **CFSP** | 50% | **9.42** | **13.68** | **60.2** |
| LLaMA-13B | Dense | 0% | 5.09 | 7.81 | 68.1 |
| LLaMA-13B | LLM-Pruner | 50% | 8.76 | 12.34 | 62.5 |
| LLaMA-13B | **CFSP** | 50% | **7.21** | **10.56** | **64.3** |

### 消融实验

| 配置 | WikiText2 PPL↓ | 说明 |
|------|---------------|------|
| CFSP完整模型 | 9.42 | 粗+细粒度+自适应微调 |
| w/o 粗粒度（均匀分配） | 10.85 | 去掉差异化分配后PPL升高1.43 |
| w/o 细粒度（仅用权重范数） | 10.12 | 不用激活信息指导block内剪枝 |
| w/o 自适应微调 | 10.68 | 不进行恢复微调 |
| 均匀微调（非自适应） | 9.89 | 微调但不按重要性分配资源 |

### 关键发现
- 粗粒度的差异化稀疏分配贡献最大，去掉后PPL增加最多（+1.43），说明统一稀疏率确实是现有方法的主要瓶颈
- 细粒度的激活-权重联合评估比单独使用权重范数提升明显（PPL降低0.7）
- 自适应微调相比均匀微调额外降低0.47 PPL，验证了按重要性分配微调资源的有效性
- 在高稀疏率（60%+）下，CFSP相对优势更加明显，因为差异化分配在高压缩率下更为关键

## 亮点与洞察
- "粗到细"的两级评估策略很优雅，先全局再局部的思路符合直觉又有效，可以迁移到其他模型压缩任务（如量化、蒸馏）
- 仅需一次前向传播就能完成重要性评估，剪枝效率远高于需要迭代优化的方法（如LoRAPrune），在实际部署中更具可行性
- 自适应微调的"受损越多修复越多"策略是对传统均匀微调的改进

## 局限与展望
- 当前实验主要在LLaMA系列上验证，对其他架构（如Mixtral MoE）的适用性有待验证
- 恢复微调仍需一定的计算资源和校准数据，在极度资源受限场景下可能不适用
- 仅考虑了激活幅度作为重要性指标，未探索梯度信息等其他信号
- 可以考虑将粗粒度分析扩展到attention head级别，实现更细致的分层剪枝

## 相关工作与启发
- **vs LLM-Pruner (Ma et al., 2023)**: LLM-Pruner使用梯度信息评估重要性，计算开销更大；CFSP仅用激活信息，更高效
- **vs SliceGPT (Ashkboos et al., 2024)**: SliceGPT通过矩阵分解实现剪枝，与CFSP思路不同但互补，可以探索结合两者
- **vs Wanda (Sun et al., 2024)**: Wanda是非结构化剪枝，使用激活×权重作为重要性度量，CFSP借鉴了类似思路但扩展到结构化场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 粗到细的两级策略在LLM剪枝中较新，但核心技术组件不算全新
- 实验充分度: ⭐⭐⭐⭐ 多模型多稀疏率评估，消融完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机链条完整
- 价值: ⭐⭐⭐⭐ 实用性强，工业部署LLM时可直接参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](../../ACL2026/model_compression/two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../../ICLR2026/model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)
- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](../../ICML2025/model_compression/olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)

</div>

<!-- RELATED:END -->
