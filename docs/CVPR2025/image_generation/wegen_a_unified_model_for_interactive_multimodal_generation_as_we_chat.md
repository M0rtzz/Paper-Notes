---
title: >-
  [论文解读] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat
description: >-
  [CVPR 2025][图像生成][多模态统一模型] 提出WeGen统一框架，将多模态理解和视觉生成整合到单一模型中，通过动态实例身份一致性(DIIC)数据管线和提示自重写(PSR)机制，解决参考图像一致性保持和生成多样性两大挑战，实现类似对话式设计助手的交互体验。
tags:
  - CVPR 2025
  - 图像生成
  - 多模态统一模型
  - 交互式生成
  - 实例一致性
  - 生成多样性
  - 设计助手
---

# WeGen: A Unified Model for Interactive Multimodal Generation as We Chat

**会议**: CVPR 2025  
**arXiv**: [2503.01115](https://arxiv.org/abs/2503.01115)  
**代码**: [GitHub](https://github.com/hzphzp/WeGen)  
**领域**: Image Generation  
**关键词**: 多模态统一模型, 交互式生成, 实例一致性, 生成多样性, 设计助手

## 一句话总结

提出WeGen统一框架，将多模态理解和视觉生成整合到单一模型中，通过动态实例身份一致性(DIIC)数据管线和提示自重写(PSR)机制，解决参考图像一致性保持和生成多样性两大挑战，实现类似对话式设计助手的交互体验。

## 研究背景与动机

当前视觉生成工具存在两个根本问题：(1) 每种视觉任务需要专用模型，用户需设计复杂的工作流；(2) 不如ChatGPT直观，学习曲线陡峭。理想的设计助手应通过自然对话统一多种生成能力。

现有统一MLLM+扩散模型方法（如Emu、SEED-LLaMA）面临两个关键挑战：(a) **实例身份一致性** — 用户需要在生成中保留参考图像的关键属性（如花瓶、人脸），同时允许合理变化（如姿态、光照），但现有方法要么简单复制粘贴要么失去身份信息；(b) **生成多样性不足** — 现有方法直接将条件映射为扩散模型的连续特征，缺乏采样过程中的随机性，不同种子产生相似输出。

## 方法详解

### 整体框架

WeGen由三部分组成：CLIP编码器（将参考图像转为64个视觉特征）、大语言模型LLM（处理交替的文本和视觉输入）、SDXL解码器（将生成特征转为最终图像）。两阶段训练：先训练SDXL从CLIP特征重建图像，再冻结CLIP和SDXL微调LLM。

### 关键设计1：动态实例身份一致性(DIIC)数据管线

- **功能**: 从视频中构建训练数据，使模型学会在保持实例身份的同时允许自然变化
- **核心思路**: 四步流程——(1) InternVL2生成帧标注并提取名词短语识别实例；(2) Grounding DINO定位边界框；(3) SAM2跨帧跟踪实例分割；(4) 采样时间间隔$t_{ref}$的帧对构建训练数据。相比单图像训练中人工增强产生的不自然变化，视频中的自然变化更有利于学习身份保持与变化的平衡
- **设计动机**: 现有方法在单张图上训练（输入=输出），鼓励简单复制粘贴。视频数据天然包含同一实例在不同帧的自然变化（姿态、表情、上下文），是学习一致性感知生成的理想数据源

### 关键设计2：提示自重写(PSR)机制

- **功能**: 增强生成结果的多样性，避免不同种子产生近似输出
- **核心思路**: 在生成图像特征前，利用MLLM的语言能力将详细描述重写，引入离散文本token采样带来的随机性，使模型能探索不同语义解释
- **设计动机**: 直接映射条件到连续视觉特征是确定性过程，缺乏自然多样性。而将CLIP特征离散化会造成严重信息损失。PSR通过在语义层面引入随机性，避免了这一困境

### 关键设计3：大规模CLIP编码器的Scaling Law

- **功能**: 最小化编码-解码过程中的信息损失
- **核心思路**: 探索CLIP用作编码器时的Scaling Law，采用更大规模的CLIP模型减少视觉信息丢失
- **设计动机**: 信息损失是身份一致性降低的根本原因之一

### 损失函数

第一阶段使用标准扩散损失训练SDXL解码器。第二阶段微调LLM时，文本token使用分类损失，视觉特征使用回归损失。

## 实验关键数据

### 多任务性能

| 任务 | WeGen表现 |
|------|----------|
| 文本到图像生成 | SOTA |
| 主体驱动生成（身份保持） | SOTA |
| 条件驱动生成 | 竞争力 |
| 图像修复/风格迁移 | 统一支持 |

### 生成多样性对比

PSR机制使不同种子生成的结果多样性显著高于无PSR基线和其他统一模型，同时保持语义对齐。

### 关键发现

- 视频训练数据(DIIC)相比单图像+数据增强在身份一致性上表现更好
- 更大的CLIP编码器显著改善重建质量和身份保持
- PSR在不同种子下产生真正多样化的输出，而非略微变化

## 亮点与洞察

1. **交互式设计助手范式**: 用户通过多轮对话逐步完善生成结果，比单次指令更自然
2. **视频作为一致性学习的天然数据源**: 将视频序列用于训练一致性保持是简洁而有效的思路
3. **统一框架覆盖广泛任务**: 单一模型支持文生图、主体驱动、条件生成、修复、风格迁移等

## 局限与展望

- 统一模型在单一任务上可能不如专用模型
- DIIC数据管线依赖多个基础模型，处理复杂
- 多轮交互中的上下文管理可能成为瓶颈

## 相关工作与启发

- 用视频数据学习对象一致性的思路可推广到视频编辑、虚拟试穿等任务
- PSR的语义层面引入随机性可能启发其他需要多样化输出的生成任务

## 评分

⭐⭐⭐⭐ — 统一多模态生成和理解的方向有价值，DIIC数据管线和PSR机制都有实际意义。作为设计助手的定位清晰，但系统复杂度较高。

<!-- RELATED:START -->

## 相关论文

- [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)

<!-- RELATED:END -->
