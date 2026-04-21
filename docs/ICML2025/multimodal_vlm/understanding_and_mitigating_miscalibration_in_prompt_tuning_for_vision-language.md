---
title: >-
  [论文解读] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models
description: >-
  [ICML 2025][多模态][视觉语言模型] 通过分析提示调优导致VLM校准失败的根因（文本特征偏移），提出动态异常值正则化（DOR）方法，利用WordNet中高语义相似度名词作为文本异常值来约束微调过程中的特征漂移，显著降低校准误差。
tags:
  - ICML 2025
  - 多模态
  - 视觉语言模型
  - 提示调优
  - 校准
  - CLIP
  - 异常值正则化
---

# Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models

**会议**: ICML 2025  
**arXiv**: [2410.02681](https://arxiv.org/abs/2410.02681)  
**代码**: [ml-stat-Sustech/Outlier-Calibration](https://github.com/ml-stat-Sustech/Outlier-Calibration)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 提示调优, 校准, CLIP, 异常值正则化

## 一句话总结

通过分析提示调优导致VLM校准失败的根因（文本特征偏移），提出动态异常值正则化（DOR）方法，利用WordNet中高语义相似度名词作为文本异常值来约束微调过程中的特征漂移，显著降低校准误差。

## 研究背景与动机

**领域现状**：CLIP等大规模视觉语言模型通过提示调优（prompt tuning）可以在下游任务上取得优异的分类性能。CoOp、MaPLe、KgCoOp等方法已经在base-to-new泛化设置中展现了不错的准确率。

**现有痛点**：尽管准确率不错，但这些方法在**校准性（calibration）**方面存在严重问题。具体来说，模型输出的置信度与真实准确率之间存在显著偏差，这在安全关键应用（如医疗诊断、自动驾驶）中是不可接受的。

**核心矛盾**：标准提示调优方法（如CoOp）在新类上表现出过度自信（overconfidence），而带正则化的方法（如KgCoOp）虽然缓解了新类过度自信问题，却导致在基类上出现欠自信（underconfidence）。两种策略无法同时兼顾基类和新类的校准。

**本文目标** 找到提示调优导致校准失败的根本原因，并提出一种能同时降低基类和新类校准误差的统一方法。

**切入角度**：作者深入分析了特征空间的变化，发现**文本特征的偏移**是校准失败的根本原因——微调后的文本编码器会产生与零样本CLIP不同的特征分布，导致分类决策边界的偏移。

**核心 idea**：用WordNet中与目标类语义相近但不属于任何已知类的名词作为"异常值锚点"，通过正则化约束这些异常值的文本特征在微调前后保持一致，从而抑制文本特征漂移。

## 方法详解

### 整体框架

DOR方法在标准提示调优（如CoOp）的基础上增加了一个异常值正则化项。训练时，每个epoch从WordNet中动态采样一批与当前任务类别语义相近的名词作为"文本异常值"，然后通过损失函数约束这些异常值在微调后的文本编码器中的特征表示尽量接近其在原始零样本CLIP中的特征。总损失为 $L_{total} = L_{ce} + \lambda \cdot L_{dor}$。

### 关键设计

1. **文本异常值选择策略**:

    - 功能：从大规模名词库中筛选出与目标任务类别语义接近但不重叠的词汇作为正则化锚点
    - 核心思路：利用WordNet中约8万个名词，计算每个名词与目标类别在零样本CLIP文本特征空间中的余弦相似度，选取相似度最高的Top-K个名词（排除与目标类完全重合的词）作为near-OOD异常值
    - 设计动机：语义接近的异常值处于分类决策边界附近，它们的特征漂移对校准的影响最大。相比随机采样或远OOD采样，near-OOD异常值能更有效地约束决策边界区域的特征稳定性

2. **动态异常值正则化损失（DOR Loss）**:

    - 功能：约束微调过程中文本特征的漂移幅度
    - 核心思路：$L_{dor} = 1 - \frac{1}{B}\sum_{b=1}^{B} \text{sim}(\psi(t'_{o_b}), \psi(t_{o_b}))$，其中 $\psi(t'_{o_b})$ 和 $\psi(t_{o_b})$ 分别为微调后和零样本CLIP对异常值词汇 $o_b$ 的文本特征表示，$\text{sim}$ 为余弦相似度
    - 设计动机：直接约束目标类的文本特征会限制模型的适应能力；而通过异常值间接约束，既允许目标类特征适度调整以提升准确率，又防止整体特征空间发生过大的结构性漂移

3. **动态采样机制**:

    - 功能：每个训练epoch重新采样异常值集合，避免过拟合到固定的异常值
    - 核心思路：在每个epoch开始时，从候选异常值池中按语义相似度加权采样新的一批异常值，确保正则化信号的多样性
    - 设计动机：固定异常值可能导致模型仅在这些特定词汇上保持特征一致，而忽略其他区域的漂移；动态采样提供了更广泛的覆盖

### 损失函数 / 训练策略

总损失函数为标准交叉熵损失与DOR正则化项的加权和：

$$L_{total} = L_{ce}(x, y) + \lambda \cdot L_{dor}$$

其中 $\lambda$ 控制正则化强度。DOR损失基于余弦相似度，计算微调前后异常值文本特征的差异。训练超参数方面，$\lambda$ 通过验证集调节，异常值数量 $B$ 设为一个适中的值（如50-100），每个epoch更新一次异常值采样。DOR是一个即插即用的模块，可以直接添加到CoOp、MaPLe、KgCoOp、TCP等任意提示调优方法中。

## 实验关键数据

### 主实验

在11个数据集上的base-to-new泛化校准结果（ECE %，越低越好）：

| 方法 | Base ECE | New ECE | HM ECE |
|:---|:---:|:---:|:---:|
| Zero-shot CLIP | 4.30 | 5.79 | 5.05 |
| CoOp | 3.07 | 14.58 | 8.82 |
| CoOp + DOR | **2.67** | **6.49** | **4.58** |
| MaPLe | 3.10 | 8.98 | 6.04 |
| MaPLe + DOR | **2.73** | **6.33** | **4.53** |
| KgCoOp | 4.86 | 6.56 | 5.71 |
| KgCoOp + DOR | **3.31** | **5.47** | **4.39** |
| TCP | 2.92 | 7.82 | 5.37 |
| TCP + DOR | **2.57** | **5.95** | **4.26** |

DOR在所有4种方法上均实现了显著的ECE降低，平均降低8.09%。

域泛化实验（ImageNet → ImageNet变体）：

| 方法 | Source ECE | Target ECE |
|:---|:---:|:---:|
| CoOp | 2.43 | 7.18 |
| CoOp + DOR | **2.28** | **4.89** |

### 消融实验

异常值选择策略对比（CoOp方法，11数据集平均，ECE %）：

| 异常值类型 | Base ECE | New ECE | HM ECE |
|:---|:---:|:---:|:---:|
| 无正则化（CoOp原版） | 3.07 | 14.58 | 8.83 |
| Near-OOD（本文方法） | **2.68** | 7.09 | **4.89** |
| Far-OOD | 2.95 | 7.72 | 5.34 |
| Random | 2.80 | 7.33 | 5.07 |
| Oracle（已知新类） | 3.13 | **4.34** | 3.74 |

视觉提示调优验证（VPT方法，DTD数据集）：

| 方法 | New ECE |
|:---|:---:|
| VPT | 13.04 |
| VPT + DOR | **8.40** |

### 关键发现

- DOR作为即插即用模块，在CoOp、MaPLe、KgCoOp、TCP四种提示调优方法上均能显著降低校准误差，平均降幅达8.09%
- Near-OOD异常值的效果显著优于Far-OOD和Random策略，验证了语义接近性对校准修正的重要性
- DOR在降低校准误差的同时几乎不损失分类准确率：基类准确率83.20% vs 82.97%，新类准确率也基本持平
- 文本特征偏移是校准失败的根因这一发现具有普遍性，DOR甚至对视觉提示调优（VPT）也有效
- 域泛化场景中DOR同样有效：ImageNet变体上ECE从7.18%降至4.89%

## 亮点与洞察

- **根因分析深入**：不是简单地套用校准方法（如温度缩放），而是从特征空间角度找到了文本特征偏移这个根本原因，这为后续工作提供了重要的理论指导
- **方法设计优雅**：DOR作为一个轻量级即插即用模块，不修改原有提示调优的训练流程，仅增加一个正则化项，实用性极强
- **WordNet作为异常值来源**：巧妙利用了语言知识库提供的丰富语义结构，无需额外标注数据
- **全面的实验验证**：覆盖4种提示调优方法、11个数据集、域泛化、视觉调优等多种设置，可信度高

## 局限与展望

- 异常值采样依赖WordNet，对于非英文场景或专业领域（如医学术语）可能需要替换为领域特定的词汇库
- 文本异常值选择的Top-K参数和 $\lambda$ 需要在验证集上调节，增加了超参数调优成本
- 当前分析集中在文本侧特征漂移，视觉侧特征的校准影响尚未深入探讨
- 仅在分类任务上验证，检测、分割等密集预测任务中的校准问题未涉及

## 相关工作与启发

- **vs CoOp**: CoOp的可学习提示在基类上校准尚可（3.07%），但在新类上严重过度自信（14.58%）；DOR将新类ECE降至6.49%
- **vs KgCoOp**: KgCoOp通过约束提示不偏离手工提示来缓解新类过自信，但导致基类欠自信（4.86%）；DOR从特征漂移角度出发，同时改善两端
- **vs 温度缩放（Temperature Scaling）**: 后处理校准方法需要额外验证集且无法改善特征表示本身；DOR在训练时直接优化特征空间的稳定性
- **vs ProDA**: ProDA通过分布对齐提升泛化，但未专门关注校准；DOR明确以校准为目标

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统分析提示调优的校准问题并定位到文本特征偏移，异常值正则化的设计思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11个数据集、4种方法、域泛化、视觉调优、消融实验，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，动机推导自然，实验组织有条理
- 价值: ⭐⭐⭐⭐ 校准是VLM部署中的重要问题，DOR的即插即用特性使其具有很高的实用价值

<!-- RELATED:START -->

## 相关论文

- [Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](../../ICLR2026/multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](../../CVPR2025/multimodal_vlm/dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](../../ICCV2025/multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance](omnibal_towards_fast_instruction-tuning_for_vision-language_models_via_omniverse.md)

<!-- RELATED:END -->
