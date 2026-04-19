---
title: >-
  [论文解读] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)
description: >-
  [CVPR2026][多模态][多模态大语言模型] 提出 HouseMind，通过层次化 VQ-VAE 将建筑平面图离散化为房间级空间 token，在统一的 MLLM 框架中实现平面图理解、生成和编辑三大任务，在几何有效性和可控性上全面超越扩散模型和通用 VLM 基线。
tags:
  - CVPR2026
  - 多模态
  - 多模态大语言模型
  - VQ-VAE
  - 空间token化
  - 平面图生成
  - 平面图编辑
  - 指令调优
---

# Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)

**会议**: CVPR2026  
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)  
**代码**: [housemind.github.io](https://housemind.github.io/)  
**领域**: 多模态VLM / 建筑平面图设计  
**关键词**: 多模态大语言模型, VQ-VAE, 空间token化, 平面图生成, 平面图编辑, 指令调优

## 一句话总结

提出 HouseMind，通过层次化 VQ-VAE 将建筑平面图离散化为房间级空间 token，在统一的 MLLM 框架中实现平面图理解、生成和编辑三大任务，在几何有效性和可控性上全面超越扩散模型和通用 VLM 基线。

## 研究背景与动机

**建筑平面图设计的高认知复杂度**：平面图设计需要同时推理几何、语义和空间层次关系，模式不是顺序的而是嵌在复杂关系中，是 AI 面临的重大挑战

**现有方法缺乏全局空间一致性**：扩散模型和自回归模型在视觉保真度上有所提升，但将布局合成视为纯视觉过程，缺少房间实例级的显式推理，导致局部合理但全局空间不连贯（如邻接和流通关系不一致）

**可解释性和可控性不足**：大规模视觉-语言模型常作为黑箱生成器，空间可控性和可解释性受限

**无法统一理解-生成-编辑**：现有框架难以在单一架构中同时处理理解、生成和编辑任务，尤其在建筑布局的几何和语义复杂性下

**计算开销大、难以本地部署**：大多数 AI 系统计算需求高，难以集成到实际设计工作流

**已有 LLM 驱动设计仍是模块化的**：Tell2Design、ChatHouseDiffusion、FloorPlanLLaMA 等虽提升了可解释性，但各模块独立，缺乏统一多任务推理

## 方法详解

### 整体框架

HouseMind 由两个核心组件构成：**房间实例 Token 化**（Room-Instance Tokenization）和**多模态对齐与指令调优**（Multimodal Alignment & Instruction Tuning）。

将平面图分解为轮廓 $x_o$ 和 $N$ 个房间实例 $\{x_{r_i}\}_{i=1}^N$，分别通过两个 VQ-VAE 编码为离散 token 序列，再交织为统一序列：

$$Z = [\boldsymbol{z}_o, \ell_{r_1}, \boldsymbol{z}_{r_1}, \dots, \ell_{r_N}, \boldsymbol{z}_{r_N}]$$

其中 $\ell_{r_i}$ 是房间语义标签 token，$\boldsymbol{z}_o$ 和 $\boldsymbol{z}_{r_i}$ 分别为轮廓和房间的离散 token。

### 关键设计

1. **轮廓离散化**：CNN 编码器 $E_o$ 从二值轮廓 mask 提取特征，通过轮廓码本 $\mathcal{Z}_o$ 向量量化为离散 token，解码器重建轮廓
2. **条件式房间离散化**：房间编码器 $E_r$ 将每个房间 mask 与轮廓联合编码，通过房间码本 $\mathcal{Z}_r$ 量化。条件编码使房间表示感知全局上下文，捕捉几何和空间邻接关系
3. **三阶段多模态训练**：

    - **Stage 1 - 嵌入初始化**：将 VQ-VAE 码本中的空间码映射为 LLM 词汇表中的可训练 token embedding，建立离散空间码与文本 token 的一一对应
    - **Stage 2 - 多模态预训练**：在文本描述 + 轮廓 token + 房间 token 的大规模配对数据上，用自回归语言建模目标训练，实现语言与几何的双向对齐
    - **Stage 3 - 指令调优 (SFT)**：在理解、生成、编辑三种任务的指令数据上做监督微调，赋予模型任务感知和空间推理能力
4. **统一任务建模**：理解（从 $Z$ 推断房间功能和拓扑）、生成（给定文本 $s$ 和轮廓 $\boldsymbol{z}_o$ 自回归生成布局）、编辑（给定原始布局 $Z^{src}$ 和指令 $s$ 生成修改后的 $Z^{tgt}$），均为统一的序列建模问题

### 骨干与效率

基于 Qwen3-0.6B 作为语言模型骨干，参数量小，支持单张 RTX 3090 实时推理和本地部署。

## 实验

### 数据集与基准

基于 RPLAN 数据集构建首个统一评估平面图理解/生成/编辑的 benchmark。共 80,738 个样本：76,122 训练 + 2,308 验证 + 2,308 测试。每个平面图有 JSON 表示 + 简单/详细两种文本描述（由 Qwen3-30B-A3B 生成）。

### 理解任务结果

| 方法 | RMR | LocAcc | AreaDiff↓ | AdjAcc | RelAcc | 时间(s) |
|------|-----|--------|-----------|--------|--------|---------|
| LLaVA-v1.6-Mistral-7B | 0.616 | 0.225 | 3.649 | 0.134 | 0.056 | ~6 |
| Qwen3-VL-8B | 0.698 | 0.347 | 5.837 | 0.382 | 0.128 | ~8 |
| InternVL3.5-8B | 0.847 | 0.546 | 12.234 | 0.469 | 0.157 | ~13 |
| MiniCPM-V 4.5 | 0.904 | 0.492 | 13.765 | 0.597 | 0.208 | ~14 |
| **HouseMind-U** | **0.998** | **0.969** | **0.549** | **0.990** | **0.808** | **~3** |

HouseMind 在房间定位精度和邻接准确率上提升超过 40 个绝对百分点，面积误差从数平方米降至 0.6 m² 以下。

### 生成任务结果

| 方法 | Micro IoU | Macro IoU | FID↓ | GED↓ | Node F1 | Edge Ovl. | 时间(s) |
|------|-----------|-----------|------|------|---------|-----------|---------|
| Tell2Design | 0.390 | 0.307 | 30.5 | 6.94 | 0.808 | 0.197 | ~15 |
| ChatHouseDiffusion | 0.589 | 0.521 | 11.3 | 2.36 | 0.985 | 0.710 | ~30 |
| FloorPlanLLaMA | 0.607 | 0.511 | 49.3 | 2.68 | 0.922 | 0.574 | ~1 |
| **HouseMind-G** | **0.709** | **0.653** | **1.91** | **1.01** | **0.994** | **0.880** | **~2** |

IoU 比 ChatHouseDiffusion 提升 10%+，FID 从 11.3 降至 1.9。

### 编辑任务结果

| 方法 | ΔIoU | ΔMSE↓ | Node F1 | Edge Ovl. |
|------|------|-------|---------|-----------|
| FLUX.1-Kontext-dev | 0.053 | 0.0162 | 0.765 | 0.222 |
| Qwen-Image-Edit | 0.088 | 0.0074 | 0.915 | 0.426 |
| **HouseMind-E** | **0.608** | **0.0019** | **0.998** | **0.934** |

### 消融实验

| 配置 | Train Loss↓ | Eval Loss↓ |
|------|-------------|------------|
| w/o Stage 1&2 | 0.0729 | 0.0836 |
| w/o Stage 1 | 0.0659 | 0.0840 |
| w/o Stage 2 | 0.0712 | 0.0831 |
| **Full** | **0.0644** | **0.0830** |

### 关键发现

- 去掉 Stage 1（嵌入初始化）导致优化不稳定，空间 token 无法在稳定嵌入空间中着陆
- 去掉 Stage 2（多模态预训练）使模型缺乏高层文本-布局对应关系
- 统一变体 HouseMind-O 在所有任务上接近甚至等同于各自独立训练的专用模型
- 与 GPT-5 和 Gemini 2.5 Pro 的定性对比中，HouseMind 在空间一致性和可控性上表现更优

## 亮点

- **房间级离散 token 化**是核心创新：将连续几何布局桥接到离散序列建模，使 LLM 能直接在 token 空间进行可解释的空间推理
- **统一三任务**：单一模型同时处理理解、生成和编辑，无需模块化组合
- **极轻量且可部署**：基于 0.6B 参数的 Qwen3，单卡 RTX 3090 即可推理，每样本仅需 2-3 秒
- **条件式房间编码**：房间编码以轮廓为条件，自然捕捉全局上下文和邻接关系
- **首个统一 benchmark**：构建了覆盖三任务的标准化评估协议

## 局限性

- 编辑功能仅支持简单的增删操作，不支持复杂拓扑变换（如整体重组）
- 未建模门、窗、家具等功能性组件，限制了在详细室内设计中的应用
- 生成结果未与人类设计偏好和审美约束对齐，与专业设计标准有差距
- 数据集基于 RPLAN（中国住宅为主），在其他建筑类型和文化风格上的泛化能力有待验证

## 相关工作

- **GAN-based**：Graph-constrained GAN 等提升真实感但过拟合局部几何
- **Graph/GNN-based**：Graph2Plan 等建模房间连通性但离散图限制几何保真度
- **Diffusion-based**：GSDiff、FloorPlan Diffusion 等稳定但计算昂贵且局限于单任务
- **LLM-driven**：Tell2Design 建立文本-平面图基准，ChatHouseDiffusion/FloorPlanLLaMA 引入语言控制但仍为模块化架构
- **本文定位**：首个统一多任务多模态框架，将几何、语义、拓扑表示联合学习

## 评分

- 新颖性: ⭐⭐⭐⭐ — 房间级 VQ-VAE token 化 + LLM 统一三任务的思路新颖，将空间设计转化为 token 序列建模
- 实验充分度: ⭐⭐⭐⭐ — 三任务全面评估，多个基线对比（含 GPT-5/Gemini），消融验证训练策略
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义形式化，图表丰富
- 价值: ⭐⭐⭐⭐ — 在建筑设计 AI 领域树立了统一范式，轻量可部署有实际应用潜力

<!-- RELATED:START -->

## 相关论文

- [HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](housemind_tokenization_mllm_floor_plan.md)
- [Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)
- [Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)

<!-- RELATED:END -->
