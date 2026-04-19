---
title: >-
  [论文解读] OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][本体驱动] 提出 OD-RASE 框架，通过构建道路交通专家知识本体(ontology)来过滤 LVLM 生成的道路基础设施改善方案，实现对事故风险道路结构的前瞻性识别与改善建议生成。
tags:
  - ICCV 2025
  - 自动驾驶
  - 本体驱动
  - 风险评估
  - 基础设施改善
  - 大规模视觉语言模型
  - 扩散模型
---

# OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2603.05936](https://arxiv.org/abs/2603.05936)  
**代码**: [https://kotashimomura.github.io/odrase/](https://kotashimomura.github.io/odrase/)  
**领域**: 自动驾驶安全  
**关键词**: 本体驱动, 风险评估, 基础设施改善, 大规模视觉语言模型, 扩散模型

## 一句话总结

提出 OD-RASE 框架，通过构建道路交通专家知识本体(ontology)来过滤 LVLM 生成的道路基础设施改善方案，实现对事故风险道路结构的前瞻性识别与改善建议生成。

## 研究背景与动机

- 当前自动驾驶系统虽有高感知性能，但在罕见场景或复杂路况下仍有局限
- 传统道路基础设施改善是**被动的**：通常在交通事故发生后，专家才分析原因并提出改善方案
- 对于自动驾驶系统而言，需要**主动**识别潜在风险道路结构，在事故发生前进行改善
- 现有数据集主要关注事故预测和高风险物体描述，忽略了导致事故的**底层道路结构**
- 利用 LVLM 自动生成的标注数据质量难以保证，且验证成本高

## 方法详解

### 整体框架

OD-RASE 框架分为两个核心部分：(1) 基于专家知识的本体驱动数据集构建；(2) 多模态 OD-RASE 模型（图像+文本→基础设施改善方案预测+扩散模型生成改善后道路图像）。

### 关键设计

1. **本体构建（Ontology Construction）**: 基于道路交通系统专家知识，将390+案例中的事故道路结构归纳为**11种事故诱因道路结构**和**10种改善方案**。专家先将原始30种道路结构和26种改善方案去除时间依赖因素后精简合并。本体表示为这些结构与改善方案之间的映射关系。

2. **基于图的思维链提示（G2CoT Prompt）**: 利用 GPT-4o 模拟专家推理过程，通过多阶段链式推理（CoT）为道路图像生成基础设施改善方案。每个阶段的文本输出被转化为图结构提示传递给下一阶段。生成结果对应本体中定义的10种改善方案和11种道路结构类型。

3. **本体驱动数据过滤（Ontology-Driven Data Filtering）**: 将专家知识本体作为有向参考图 $G_A=(V_A, E_A)$，将 GPT-4o 生成的改善方案作为生成图 $G_B=(V_B, E_B)$。通过图匹配进行过滤：

    - 计算节点和边的交集：$V'_B = V_B \cap V_A$，$E'_B = E_B \cap E_A$
    - 移除孤立节点：$V''_B = V'_B \setminus \text{Iso}(G'_B)$
    - 保留端点均在 $V''_B$ 中的边，得到最终过滤图 $G''_B$
    - 若任意两个模块间的所有边被移除，则该数据视为不可信并排除

4. **OD-RASE 多模态模型**: 由视觉编码器（ResNet-50/ViT-B/CLIP/Long-CLIP）、文本编码器（RoBERTa-Base/Flan-T5-xl/Long-CLIP）和 Grounding Block 组成。Grounding Block 用图像嵌入作为 query、文本嵌入作为 key/value 做交叉注意力，输出经全连接层预测10类改善方案。

5. **扩散模型布局控制**: 采用 Instruct Pix2Pix，根据 OD-RASE 输出的改善方案生成文本提示，对原始道路图像进行编辑，生成改善后的道路环境可视化图像，作为决策支持工具。

### 损失函数 / 训练策略

- 任务建模为**多标签分类**（一张道路图像可能对应多个改善方案）
- 损失函数为二元交叉熵：$\mathcal{L} = -\sum_{c=1}^{C}[y_c \log p_c + (1-y_c)\log(1-p_c)]$
- 视觉和文本编码器参数在训练时**冻结**
- batch size 16，训练 25 epochs
- 使用 Mapillary Vistas 和 BDD100K 数据集

## 实验关键数据

### 主实验 (表格)

不同视觉-文本编码器组合在基础设施改善方案预测上的表现：

| 视觉编码器 | 文本编码器 | Mapillary F1 | Mapillary Acc | BDD100K F1 | BDD100K Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-50 | RoBERTa-Base | 64.98 | 37.12 | 77.18 | 45.94 |
| ViT-B | RoBERTa-Base | 67.79 | 40.71 | 78.14 | 47.98 |
| CLIP | RoBERTa-Base | 69.76 | 40.71 | 78.30 | 48.69 |
| **Long-CLIP** | **RoBERTa-Base** | **70.26** | **42.14** | **78.79** | **49.48** |
| Long-CLIP | Flan-T5-xl | 24.39 | 0.00 | 28.22 | 0.00 |

### 消融实验 (表格)

**模态消融**（Long-CLIP + RoBERTa-Base，Mapillary）：

| 输入模态 | Precision | Recall | F1 | Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 仅图像 | 57.42 | 72.37 | 64.03 | 34.50 |
| 仅文本 | 60.02 | 79.76 | 68.50 | 40.63 |
| **图像+文本** | **64.54** | **77.09** | **70.26** | **42.14** |

**本体过滤消融**：

| 数据过滤 | Precision | Recall | F1 | Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 无过滤 | 33.59 | 64.85 | 44.26 | 0.00 |
| **有过滤** | **64.54** | **77.09** | **70.26** | **42.14** |

### 关键发现

- **本体过滤效果显著**：无过滤时 Accuracy 为 0%，启用后提升至 42.14%，F1 从 44.26 提升至 70.26
- **Flan-T5-xl 作为文本编码器表现极差**：recall 高但 precision 极低，可能因 8-bit 量化导致过多误报
- **零样本泛化**：模型在 BDD100K 上训练、Mapillary 上评估时仍表现稳健（F1=68.32），但通用 LVLM（如 GPT-4o/LLaVA）在此任务上远逊于专用模型
- **扩散模型可视化**：Instruct Pix2Pix 生成的改善后道路图像 FID=8.5，专家评估 54.23% 为完全符合

## 亮点与洞察

- 开创性地将**道路基础设施改善**与**自动驾驶安全**相结合，提出前瞻性风险识别框架
- 利用本体作为知识过滤器，有效提升 LVLM 生成数据的可靠性，这种"专家知识+AI生成+图匹配验证"的模式具有通用性
- 扩散模型生成改善后可视化图像的思路增强了可解释性和实用价值

## 局限与展望

- 仅使用前视图像，未考虑视频时序信息或多视角输入
- 排除了交通量等时间依赖因素
- 无法量化评估改善方案实际的事故减少率（需要交通仿真器）
- 改善方案的优先级和紧迫性缺乏量化指标
- 类别粒度较粗（11种道路结构+10种改善方案），实际应用可能需要更细分类

## 相关工作与启发

- 与道路安全研究中的车道缩减（减少50%事故）和环形交叉口改造（减少38%事故）等成熟做法相呼应
- 可以为城市规划和智慧交通提供自动化的道路风险评估工具
- 本体驱动的数据过滤思路可推广到其他需要领域专家验证 AI 生成数据的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将道路基础设施改善与自动驾驶安全相结合，本体驱动数据过滤的思路新颖
- **实验充分度**: ⭐⭐⭐⭐ 在两个数据集上进行了详尽实验，包括零样本、模态消融、过滤消融以及与通用 LVLM 对比
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，框架完整，图例说服力强
- **价值**: ⭐⭐⭐⭐ 实际应用价值高，为自动驾驶安全提供了新视角

<!-- RELATED:START -->

## 相关论文

- [Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](../../AAAI2026/autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)
- [DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)
- [Language Driven Occupancy Prediction (LOcc)](language_driven_occupancy_prediction.md)

<!-- RELATED:END -->
