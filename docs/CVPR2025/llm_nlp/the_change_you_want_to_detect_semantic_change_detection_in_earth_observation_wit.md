---
title: >-
  [论文解读] The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation
description: >-
  [CVPR 2025][LLM/NLP][语义变化检测] 本文提出HySCDG（Hybrid Semantic Change Detection Data Generation），一种混合数据生成流水线，结合真实超高分辨率（VHR）遥感影像和图像inpainting技术生成大规模语义变化检测训练数据…
tags:
  - "CVPR 2025"
  - "LLM/NLP"
  - "语义变化检测"
  - "VHR影像"
  - "混合数据生成"
  - "图像修复"
  - "地表覆盖"
---

# The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 语义变化检测, VHR影像, 混合数据生成, inpainting, 地表覆盖

## 一句话总结

本文提出HySCDG（Hybrid Semantic Change Detection Data Generation），一种混合数据生成流水线，结合真实超高分辨率（VHR）遥感影像和图像inpainting技术生成大规模语义变化检测训练数据，在简洁的架构设计下实现了强大的时间和空间泛化能力。

## 研究背景与动机

### 领域现状

**领域现状**：双时相变化检测是地球监测的核心能力，用于城市扩张监测、灾害评估、农业变化追踪等。基于VHR（Very High Resolution）影像的语义变化检测不仅要识别"哪里变了"（二值变化检测），还要识别"变成了什么"（语义类别），如"农田→建筑"。

**现有痛点**：(1) 标注数据严重匮乏——语义变化检测需要对两个时间点的影像分别做像素级语义标注，成本极高。现有数据集（如SECOND、HRSCD）规模小且地域覆盖有限。(2) 二值vs语义割裂——大量工作集中在简单的二值变化检测（变/没变），但实际应用需要语义级变化。(3) 合成数据不真实——纯合成数据（风格迁移、GAN生成等）与真实遥感影像有显著domain gap，导致在合成数据上训练的模型泛化差。(4) 架构复杂——许多方法采用复杂的多分支架构，需要大量标注数据才能训练。

**核心矛盾**：语义变化检测需要大规模多样的配对标注数据（两个时间点、像素级语义），但人工标注成本不可承受；纯合成数据又不够真实。

**本文目标** 如何用低成本生成大规模、高质量的语义变化检测训练数据，并训练出具有良好泛化能力的检测模型？

**切入角度**：利用"混合"策略——从真实遥感影像中选取未变化区域作为基础，用图像inpainting将特定区域修改为其他地表类型，同时自动生成变化标注。真实底图+局部合成变化=混合数据。

**核心 idea**：在真实VHR遥感影像上用inpainting局部合成地表变化，自动生成变化标注，构建大规模混合训练数据。

## 方法详解

### 整体框架

HySCDG流水线：(1) 收集大量真实VHR遥感影像及其地表覆盖语义图。(2) 选取感兴趣的变化类型（如"植被→建筑"），在语义图中定位对应区域。(3) 用inpainting模型（如SD-Inpaint）在选定区域生成目标类别的纹理（如将植被区域重绘为建筑纹理），保持周围环境不变。(4) 原始影像作为T1、修改后影像作为T2，语义图差异自动构成变化标注。(5) 在生成的混合数据上训练简洁的语义变化检测网络。

### 关键设计

1. **混合数据生成策略**：
    - 功能：低成本生成大规模、多样化的语义变化检测配对数据
    - 核心思路：利用现有的遥感影像语义分割数据集（单时相标注即可），通过inpainting"人为制造"地表变化。对于每张影像，随机选择一些语义区域，用扩散模型的inpainting功能将该区域重绘为另一种地表类型。通过控制inpainting的prompt（如"urban buildings"、"farmland"）和区域形状来生成多样化的变化。关键约束：修改区域边界应自然过渡，非修改区域保持一致
    - 设计动机：传统合成数据完全由模型生成，全图都有domain gap。混合数据只修改局部区域，保留了大部分真实影像信息，大幅减小domain gap

2. **语义一致性约束的Inpainting**：
    - 功能：确保inpainting结果在语义和视觉上与遥感场景一致
    - 核心思路：对inpainting模型进行条件控制：(1) 文本prompt指定目标地表类型。(2) 周围像素作为上下文条件。(3) 后处理检查——用预训练的语义分割模型验证inpainting区域是否确实被分类为目标类别，过滤不一致的样本。多次采样选择语义最一致的结果
    - 设计动机：通用inpainting模型对遥感场景理解有限，可能生成不符合遥感特征的纹理（如生成街景风格的建筑而非鸟瞰视角的建筑屋顶）

3. **简洁的变化检测架构**：
    - 功能：验证混合数据的有效性
    - 核心思路：采用简洁的Siamese编码器+特征差异+分割解码器架构，避免复杂设计。编码器基于预训练骨干（如ResNet或ViT），提取双时相特征后计算差异或拼接，通过轻量解码器输出语义变化图。架构简洁的目的是证明混合数据的价值而非靠复杂模型。同时在多个真实遥感场景预训练以提升空间泛化
    - 设计动机：复杂架构可能掩盖数据质量的影响

## 实验关键数据

### 关键发现

- 在SECOND和HRSCD测试集上，用混合数据训练的简洁模型达到或超越了用真实标注训练的复杂模型
- 混合数据相比纯合成数据，在真实测试集上的语义变化检测mIoU提升约10-15%
- 时间泛化性：在一个城市训练，在不同年份的同城市影像上表现良好
- 空间泛化性：在一个地区训练，在其他大洲的VHR影像上仍有合理的检测性能
- inpainting质量对最终检测性能影响显著，语义一致性过滤可提升约5% mIoU

## 亮点与洞察

- **数据生成的范式价值**：对标注数据匮乏的遥感领域具有广泛参考意义
- **混合策略是关键**：真实底图+局部合成的思路有效缩小了合成-真实domain gap
- **简洁证明有效**：不依赖复杂架构，用数据驱动性能

## 局限与展望

- Inpainting模型对遥感特有视角（鸟瞰）的生成质量仍有提升空间
- 目前主要验证了城市化相关的变化类型，更多变化类型（如洪水、森林砍伐）待验证
- 混合数据中变化区域的边界过渡可能不如真实变化自然
- 未来可结合实际时序卫星影像做半监督学习

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](../../ICML2026/llm_nlp/token-efficient_change_detection_in_llm_apis.md)
- [\[CVPR 2025\] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](../../ACL2025/llm_nlp/difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
