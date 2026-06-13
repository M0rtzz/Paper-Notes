---
title: >-
  [论文解读] Concept Lancet: Image Editing with Compositional Representation Transplant
description: >-
  [CVPR 2025][图像生成][图像编辑] 提出 Concept Lancet (CoLan)，一种零样本即插即用的图像编辑框架，通过将源图像的隐表示稀疏分解为视觉概念向量的线性组合，然后根据编辑任务（替换/添加/删除）进行定制化概念移植，解决了编辑强度校准难题。 领域现状 领域现状：基于扩散模型的图像编辑（如 P2P-…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像编辑"
  - "概念移植"
  - "稀疏分解"
  - "零样本即插即用"
  - "编辑强度校准"
---

# Concept Lancet: Image Editing with Compositional Representation Transplant

**会议**: CVPR 2025  
**arXiv**: [2504.02828](https://arxiv.org/abs/2504.02828)  
**代码**: [https://peterljq.github.io/project/colan](https://peterljq.github.io/project/colan) (项目页 + CoLan-150K 数据集)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 图像编辑, 概念移植, 稀疏分解, 零样本即插即用, 编辑强度校准

## 一句话总结
提出 Concept Lancet (CoLan)，一种零样本即插即用的图像编辑框架，通过将源图像的隐表示稀疏分解为视觉概念向量的线性组合，然后根据编辑任务（替换/添加/删除）进行定制化概念移植，解决了编辑强度校准难题。

## 研究背景与动机

### 领域现状

**领域现状**：基于扩散模型的图像编辑（如 P2P-Zero、InfEdit 等）通常通过在文本嵌入或 score 空间做向量加减来实现概念修改。例如替换"猫→狗"就把嵌入中减去"猫"加上"狗"。

**现有痛点**：简单的向量加减（VecAdd）面临严重的编辑强度校准问题：(1) 减去太多导致源图像结构崩坏；(2) 减去太少导致源概念残留；(3) 不同概念需要不同强度但无法自动确定。这使得编辑在效果和一致性之间难以平衡。

**核心矛盾**：向量加减假设概念在嵌入空间中是独立的，但实际上不同概念的嵌入高度纠缠。需要先精确"定位"源表示中属于目标概念的成分，再精确替换。

**本文目标** 在保持编辑一致性（不破坏非编辑区域）的同时实现精确的概念替换/添加/删除。

**切入角度**：将源图像表示分解为概念字典中向量的稀疏线性组合（Elastic Net 优化），然后只对目标概念的系数进行操作。用 VLM（GPT-4V）自动选择相关概念子集以降低计算开销。

**核心 idea**：通过 Elastic Net 将源表示稀疏分解为概念字典的线性组合，然后精确替换/添加/删除目标概念的系数实现可控编辑。

## 方法详解

### 整体框架
构建 CoLan-150K 概念数据集（5078 个概念，152971 个刺激图像）。推理时：(1) VLM 从源/目标 prompt 中解析相关概念并选择字典子集；(2) 用 Elastic Net 将源文本嵌入（或 score）分解为概念向量的稀疏组合；(3) 根据编辑类型操作概念系数——替换（交换系数）、添加（增加系数）、删除（置零系数）。即插即用，兼容多种反演方法和骨干网络。

### 关键设计

1. **稀疏概念分解**

    - 功能：精确定位源表示中属于各概念的成分
    - 核心思路：给定源文本嵌入 $e_s$ 和概念字典 $V = \{v_1, ..., v_n\}$，求解 $e_s \approx \sum_i \alpha_i v_i$，其中 $\alpha$ 通过 Elastic Net 优化（L1 + L2 正则化确保稀疏性和稳定性）。在 score 空间中类似操作。分解后每个概念的系数 $\alpha_i$ 精确量化了该概念在源表示中的贡献
    - 设计动机：VecAdd 假设减去整个"猫"嵌入，但"猫"嵌入中也包含了与其他概念共享的成分。稀疏分解只减去确实属于"猫"的那部分

2. **定制化概念移植**

    - 功能：根据编辑任务类型精确操作概念系数
    - 核心思路：**替换**: 找到源概念的系数 $\alpha_{src}$ 和目标概念的系数 $\alpha_{tgt}$（从目标 prompt 分解获得），交换系数。**添加**: 将目标概念的系数增加到当前分解中。**删除**: 将目标概念的系数置零。操作后重建新的嵌入/score 进行去噪
    - 设计动机：不同编辑原语需要不同的操作逻辑，统一框架内支持三种基本编辑

3. **CoLan-150K 概念字典**

    - 功能：提供丰富的视觉概念覆盖，支撑稀疏分解的精度
    - 核心思路：5078 个视觉概念（颜色、纹理、物体、风格等），每个概念约 30 个刺激图像。概念向量通过扩散过程的文本嵌入或 score 统计获得。VLM 自动选择与编辑相关的概念子集（~100 个），避免全字典优化
    - 设计动机：线性分解的精度取决于字典的覆盖度和多样性

### 损失函数 / 训练策略
完全无训练——Elastic Net 优化在推理时执行。VLM 调用增加一些推理开销。

## 实验关键数据

### 主实验

| 方法 | StruDist↓ (×10⁻³) | PSNR↑ | 说明 |
|------|-------------------|-------|------|
| VecAdd + P2P-Zero | 53.04 / 25.54 | 17.65 / 21.59 | 结构严重破坏 |
| **CoLan + P2P-Zero** | **15.91 / 6.61** | **23.08 / 26.08** | 一致性大幅提升 |
| VecAdd + InfEdit | 较高 | 较低 | — |
| **CoLan + InfEdit** | **13.97 / 6.20** | **23.42 / 28.46** | 所有方法最优 |

### 关键发现
- CoLan 在所有骨干网络和反演方法上一致提升编辑一致性（StruDist 降低 3-4×）和编辑精度
- 即插即用特性使其可以直接增强现有编辑方法，无需重新训练
- 在 score 空间操作通常优于文本嵌入空间

## 亮点与洞察
- **稀疏分解替代向量加减**的思路解决了编辑强度校准的根本问题——不是启发式调节强度，而是精确量化概念贡献
- **即插即用设计**使方法可广泛组合使用，增加了实用价值

## 局限与展望
- 需要 VLM (GPT-4V) 解析概念，增加推理成本
- Elastic Net 优化增加每次编辑的延迟
- 假设概念在隐空间中线性可组合，复杂非线性关系可能无法处理
- 概念字典的覆盖度决定分解质量

## 评分
- 新颖性: ⭐⭐⭐⭐ 稀疏分解+概念移植的框架设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多种骨干+多种反演+三种编辑类型
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 解决了图像编辑的实际痛点，即插即用增加实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)

</div>

<!-- RELATED:END -->
