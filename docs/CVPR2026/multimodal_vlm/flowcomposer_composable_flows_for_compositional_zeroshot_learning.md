---
title: >-
  [论文解读] FlowComposer: Composable Flows for Compositional Zero-Shot Learning
description: >-
  [CVPR 2026][多模态VLM][组合零样本学习] FlowComposer 首次将 Flow Matching 引入组合零样本学习(CZSL)，学习两个原始流(属性流和物体流)将视觉特征传输到对应文本嵌入空间，并通过可学习的 Composer 显式组合速度场得到组合流…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "组合零样本学习"
  - "Flow Matching"
  - "CLIP"
  - "速度场组合"
  - "泄露引导增强"
---

# FlowComposer: Composable Flows for Compositional Zero-Shot Learning

**会议**: CVPR 2026  
**arXiv**: [2603.16641](https://arxiv.org/abs/2603.16641)  
**代码**: [https://hkust-longgroup.github.io/FlowComposer/](https://hkust-longgroup.github.io/FlowComposer/)  
**领域**: 多模态VLM / 组合零样本学习  
**关键词**: 组合零样本学习, Flow Matching, CLIP, 速度场组合, 泄露引导增强

## 一句话总结

FlowComposer 首次将 Flow Matching 引入组合零样本学习(CZSL)，学习两个原始流(属性流和物体流)将视觉特征传输到对应文本嵌入空间，并通过可学习的 Composer 显式组合速度场得到组合流，同时利用泄露引导增强策略将不完美的特征解耦转化为辅助监督信号，作为即插即用模块在三个基准上持续提升 CZSL 性能。

## 研究背景与动机

1. **领域现状**：CZSL 旨在通过重组已见的属性和物体原语来识别未见的属性-物体组合。当前主流方法基于 CLIP 等视觉语言模型，通过参数高效微调(PEFT)进行提示学习。
2. **现有痛点**：现有方法存在两个根本缺陷——(1) 隐式组合构建：组合仅通过 token 级拼接实现，而非嵌入空间中的显式操作，未见组合的嵌入可能偏离图像嵌入；(2) 残留特征纠缠：视觉解耦器无法严格分离属性和物体特征，导致跨分支信息泄露。
3. **核心矛盾**：这两个缺陷使得现有方法容易过拟合已见组合，训练过程中已见准确率上升但未见准确率持续下降，呈现强烈的已见偏差。
4. **本文目标**：设计一个在嵌入空间中进行显式组合操作的框架，同时将不完美解耦转化为有用信号。
5. **切入角度**：Flow Matching 的速度场天然支持组合和分解——可以学习原始流再组合它们的速度场。
6. **核心 idea**：用两个 Flow Matching 模型分别学习属性和物体的传输流，再通过 Composer 网络组合速度场，实现嵌入空间的显式组合。

## 方法详解

### 整体框架

FlowComposer 建立在现有 CZSL 基线（如 CSP、Troika）之上。给定图像，通过基线的图像和文本编码器获取属性、物体、组合的视觉特征和文本嵌入。FlowComposer 在此共享特征空间中操作：学习属性流和物体流将视觉特征传输到文本嵌入，然后用 Composer 组合速度场，最终的组合流分数增强组合识别。

### 关键设计

1. **属性和物体原始流模型**:

    - 功能：学习将视觉特征传输到对应文本嵌入的速度场
    - 核心思路：对属性分支 $i \in \{a, o\}$，用 Rectified Flow 构建线性路径 $x^i_t = (1-t)x^i_0 + tx^i_1$，训练速度网络 $v_{\theta_i}$ 回归目标速度 $x^i_1 - x^i_0$，同时加入交叉熵损失确保预测端点能正确分类。推理时只需单步传输 $\hat{x}^i_1 = x^i_0 + v_{\theta_i}(x^i_0, 0)$
    - 设计动机：FM 的速度场提供了从视觉空间到文本空间的连续映射，天然支持组合操作

2. **Composer 组合器**:

    - 功能：学习如何将属性和物体的速度场组合成组合速度场
    - 核心思路：将组合速度近似为 $v^*_c = a^* v^*_a + b^* v^*_o$，先归一化原始速度得到单位方向 $\hat{\Delta}_a, \hat{\Delta}_o$，然后通过最小二乘法求解目标组合系数 $(a^*, b^*)$。Composer 网络学习从原始速度预测这些系数，训练用 MSE 损失监督
    - 设计动机：不同样本中属性和物体的贡献比例不同，需要自适应学习组合关系，超越简单的 token 拼接

3. **泄露引导增强 (Leakage-Guided Augmentation)**:

    - 功能：将解耦不完美产生的跨分支信息泄露转化为额外监督
    - 核心思路：除了标准的分支内监督（属性视觉→属性文本），额外训练每个原始流处理泄露特征：如从物体分支提取的视觉特征→属性文本，或从组合分支特征→各原始文本。这丰富了速度监督信号
    - 设计动机：完美解耦在实践中不可能实现，与其试图消除泄露不如利用它，将缺陷转化为优势

### 损失函数 / 训练策略

总损失 = 基线原始损失 + 属性流损失(MSE + CE) + 物体流损失(MSE + CE) + Composer 损失(MSE) + 泄露增强损失。FlowComposer 是模型无关的即插即用模块，可附加到任何 CZSL 管线。

## 实验关键数据

### 主实验

| 数据集 | 指标 (HM↑) | Troika | +FlowComposer | 提升 |
|--------|-----------|--------|--------------|------|
| MIT-States (CW) | HM | 39.2 | 40.2 | +1.0 |
| C-GQA (CW) | HM | 29.7 | 34.0 | +4.3 |
| UT-Zappos (CW) | HM | 55.4 | 58.6 | +3.2 |
| MIT-States (OW) | AUC | 12.5 | 15.9 | +3.4 |

在 CSP 基线上同样有显著提升：C-GQA HM 从 19.3 到 22.9 (+3.6)。

### 消融实验

| 配置 | HM (MIT-States) | AUC | 说明 |
|------|-----------------|-----|------|
| Troika 基线 | 39.2 | 12.5 | 无 FlowComposer |
| +原始流(无 Composer) | 39.7 | 13.8 | 仅学习传输流 |
| +Composer | 40.0 | 15.0 | 加入速度场组合 |
| +泄露增强 (完整) | 40.2 | 15.9 | 完整模型 |

### 关键发现

- FlowComposer 在所有三个数据集和两种设置(闭集/开集)上一致提升基线性能
- Composer 模块贡献最大，特别是在开集场景下（AUC 提升显著），说明显式组合对泛化至关重要
- 泄露引导增强在 C-GQA 上效果最明显（+4.3 HM），可能因为该数据集解耦更难
- 训练动态更稳定：已见/未见准确率更均衡，减少了已见偏差

## 亮点与洞察

- **FM 速度场的组合性**：首次指出 Flow Matching 的速度场天然适合 CZSL 的组合/分解本质，这是一个优雅的概念对应
- **缺陷变优势**：将解耦不完美（信息泄露）转化为额外监督，思路巧妙且通用
- **即插即用设计**：纯在表示空间操作，不修改编码器，可迁移到任何 CZSL 方法

## 局限与展望

- Flow 模型增加了额外参数和训练成本
- 单步推理是近似，多步可能更准但会降低效率
- 仅在 CLIP 特征空间验证，其他 VLM 需要进一步验证
- 未来可探索非线性路径（如 ODE 求解）以获得更精确的传输

## 相关工作与启发

- **vs CSP/Troika**: 它们仅在 token 级组合，FlowComposer 在嵌入空间显式组合
- **vs 扩散分类器**: 扩散分类器用生成模型做分类但不利用速度场的组合性
- **vs FM for generation**: 传统 FM 用于图像生成，FlowComposer 首次将其组合性用于分类

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ FM 用于 CZSL 是全新方向，速度场组合思路优雅
- 实验充分度: ⭐⭐⭐⭐ 三个数据集，两种基线，消融详细
- 写作质量: ⭐⭐⭐⭐ 动机清晰，公式推导清楚
- 价值: ⭐⭐⭐⭐ 即插即用设计实用性强，但领域相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](../../NeurIPS2025/multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[CVPR 2026\] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)

</div>

<!-- RELATED:END -->
