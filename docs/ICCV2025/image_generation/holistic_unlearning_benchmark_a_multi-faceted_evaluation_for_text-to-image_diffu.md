---
title: >-
  [论文解读] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning
description: >-
  [ICCV 2025][图像生成][概念遗忘] HUB 提出了首个全面评估文生图扩散模型概念遗忘（concept unlearning）方法的基准框架，覆盖 33 个目标概念和 6 大评估维度（忠实度、对齐性、精确性、多语言鲁棒性、对抗鲁棒性、效率），每个概念使用 16,000 条 prompt，发现没有任何单一方法能在所有维度上占优。
tags:
  - ICCV 2025
  - 图像生成
  - 概念遗忘
  - 评估基准
  - 文生图安全
  - 多维度评估
  - 扩散模型
---

# Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning

**会议**: ICCV 2025  
**arXiv**: [2410.05664](https://arxiv.org/abs/2410.05664)  
**代码**: [GitHub](https://github.com/ml-postech/HUB)  
**领域**: 扩散模型/概念遗忘  
**关键词**: 概念遗忘, 评估基准, 文生图安全, 多维度评估, 扩散模型

## 一句话总结
HUB 提出了首个全面评估文生图扩散模型概念遗忘（concept unlearning）方法的基准框架，覆盖 33 个目标概念和 6 大评估维度（忠实度、对齐性、精确性、多语言鲁棒性、对抗鲁棒性、效率），每个概念使用 16,000 条 prompt，发现没有任何单一方法能在所有维度上占优。

## 研究背景与动机

文生图扩散模型在训练中使用的大规模网络数据可能包含暴力、有害内容和受版权保护的知识产权。概念遗忘（concept unlearning）作为一种解决方案，旨在从预训练模型中移除特定目标概念。

**现有评估的不足**：

**评估维度单一**：大多数工作仅关注目标概念是否被移除 + 图像质量是否保持，忽略了副作用

**prompt 数量不足**：之前最多仅使用数十到数百条 prompt，无法充分测试遗忘效果

**概念覆盖有限**：不同方法使用不同概念，缺乏统一比较基础

**缺少重要维度**：如精确性（是否过度遗忘相关概念）、多语言鲁棒性（非英文 prompt 是否仍能触发）、实际效率等

核心矛盾：没有一个统一、全面的评估框架来公平比较不同遗忘方法，导致研究者无法获得关于方法真正能力的完整画面。

核心 idea：**构建覆盖 33 个概念、6 个维度、每概念 16,000 条 prompt 的 Holistic 评估基准**。

## 方法详解

### 整体框架

HUB 的评估框架包含：
- **概念分类**：4 大类（Celebrity ×10, Style ×10, IP ×10, NSFW ×3）= 33 个概念
- **Prompt 生成**：LLM 驱动的两步流程（属性提取 → prompt 组合生成），每概念约 10,000 条
- **概念检测**：专用分类器（NSFW 用 Q16、Celebrity 用 GIPHY）+ VLM 通用检测框架
- **6 维度 × 15 任务评估**

### 关键设计

1. **六维度评估体系**:

    - **忠实度（Faithfulness）**：目标概念移除比例 + 通用图像质量（FID）+ 目标图像质量（Aesthetic Score）
    - **对齐性（Alignment）**：通用对齐（PickScore, ImageReward） + 选择性对齐（QG/A 框架检测非目标实体是否保留）
    - **精确性（Pinpoint-ness）**：检测是否过度遗忘语义相近概念（用 CLIP score 最高的 100 个 WordNet 词汇测试）
    - **多语言鲁棒性**：将 prompt 翻译为 5 种语言（西/法/德/意/葡），测试非英文 prompt 是否仍生成目标概念
    - **对抗鲁棒性**：Ring-a-Bell + UDA + UoC 三种攻击方法
    - **效率**：训练时间 + GPU 显存 + 存储需求
    - 设计动机：覆盖遗忘方法在部署中可能面临的所有关键场景

2. **VLM-based 概念检测框架**:

    - 功能：对无专用分类器的概念（IP、Style），使用 VLM 进行通用概念检测
    - 核心思路：两步检测——(1) 用原模型生成 3 张参考图像做 in-context learning；(2) 对遗忘模型生成的测试图像做 chain-of-thought 推理判断概念是否存在
    - 设计动机：避免为每个概念训练专用检测器，实现概念无关的通用检测

3. **LLM-driven Prompt 生成**:

    - 功能：为每个概念生成 10,000+ 多样化 prompt
    - 核心思路：先提取关键属性（如 NSFW-violent 的属性包括"War"、"Murder"等），再随机组合 1-3 个属性让 LLM 生成多样 prompt
    - 设计动机：简单 prompt（"a photo of Mickey Mouse"）无法充分测试遗忘效果，需要多样化、现实化的 prompt

### 损失函数 / 训练策略

HUB 本身是评估框架，不涉及训练。被评估的 7 种遗忘方法包括：
- SLD（负提示引导）、AC（概念消融微调）、ESD（反向引导微调）
- UCE（交叉注意力闭式更新）、SA（选择性遗忘/持续学习）
- RECELER（适配器+掩码）、MACE（掩码引导遗忘）

## 实验关键数据

### 主实验（Overall 平均）

| 方法 | 目标比例↓ | 通用FID↓ | 通用对齐↑ | 精确性↑ | 多语言鲁棒↓ | 攻击鲁棒↓ | 训练时间(min) |
|--------|------|------|----------|------|------|------|------|
| Original | 0.649 | 13.20 | 0.172 | 0.592 | 0.489 | 0.491 | 0.0 |
| SLD | 0.228 | 16.57 | 0.077 | 0.502 | 0.101 | 0.196 | 0.0 |
| AC | 0.301 | 14.20 | 0.127 | 0.528 | 0.199 | 0.280 | 37.3 |
| ESD | **0.143** | 14.53 | -0.082 | 0.260 | **0.071** | **0.148** | 106.0 |
| UCE | 0.250 | 13.82 | **0.193** | **0.535** | 0.114 | 0.252 | **0.1** |
| SA | 0.173 | 32.57 | -0.322 | 0.131 | 0.079 | 0.166 | 29980.0 |
| RECELER | **0.086** | 15.19 | 0.006 | 0.316 | 0.030 | 0.107 | 100.0 |
| MACE | 0.148 | 15.30 | -0.345 | 0.306 | 0.111 | 0.125 | 140.3 |

### 消融：按类别差异

| 类别 | 最佳遗忘方法(目标比例) | 最佳质量保持(FID) | 最佳精确性 |
|------|---------|------|------|
| Celebrity | UCE (0.001) | MACE (12.98) | AC (0.429) |
| Style | RECELER (0.038) | MACE (13.09) | UCE (0.696) |
| IP | UCE (0.034) | AC (13.23) | AC (0.552) |
| NSFW | RECELER (0.272) | UCE (13.95) | UCE (0.571) |

### 关键发现

1. **没有方法在所有维度上占优**：
    - RECELER 在遗忘效果最强（目标比例最低 0.086），但通用质量下降明显
    - UCE 速度最快（0.1 分钟）且精确性高，但遗忘不够彻底
    - SA 训练成本极高（29,980 分钟 ≈ 20 天），且严重损害图像质量（FID 32.57）
    - ESD 对抗鲁棒性最佳，但通用对齐变为负值

2. **精确性问题突出**：所有方法的精确性都显著低于 Original（0.592），说明不同程度地过度遗忘了相关概念

3. **多语言是严重漏洞**：即使英文 prompt 被成功遗忘，非英文 prompt 仍可触发目标概念

4. **Style 类概念最难遗忘**：即使最好的方法在 Style 上的目标比例仍为 0.038（RECELER），而 Celebrity 可降至 0.001

## 亮点与洞察

1. **评估框架设计全面且严谨**：6 维度 15 任务覆盖了遗忘方法在实际部署中可能面临的所有关键挑战
2. **选择性对齐（QG/A 框架）**的概念值得借鉴——检测遗忘是否"外溢"到非目标实体
3. **VLM-based 概念检测**实现了概念无关的通用检测，具有方法论贡献
4. **16,000 条/概念的规模**远超之前工作，更能暴露方法弱点
5. **"没有银弹"的结论**对社区有重要指导意义

## 局限与展望

- 基于 SD v1.5 评估，更新的模型（SDXL、SD3.0）可能表现不同
- VLM 概念检测准确率 ~83%，可能引入检测噪声
- 未包含 Object 类概念（认为太通用不实际），但某些场景下确实需要遗忘特定物体
- 效率评估仅考虑训练阶段，未考虑遗忘后推理效率变化
- 33 个概念虽多但仍有限，未覆盖所有可能的遗忘场景

## 相关工作与启发

- 与 UnlearnCanvas（仅 Style）和 CPDM（仅 IP/Celebrity/Style）相比，HUB 的全面性是质的飞跃
- 多语言鲁棒性维度揭示了当前遗忘方法的根本性缺陷——只在嵌入空间局部"打补丁"
- 精确性维度指出了遗忘研究的重要未来方向：如何实现"手术刀式"精确遗忘

## 评分
- 新颖性: ⭐⭐⭐⭐ 作为 benchmark 论文，6 维度评估框架设计是主要创新，无新方法提出
- 实验充分度: ⭐⭐⭐⭐⭐ 33 概念 × 7 方法 × 6 维度 × 15 任务，实验规模极大
- 写作质量: ⭐⭐⭐⭐⭐ 框架描述清晰，表格组织合理，结论明确
- 价值: ⭐⭐⭐⭐⭐ 为概念遗忘领域建立了首个全面标准，对推动该领域进展有重要价值

<!-- RELATED:START -->

## 相关论文

- [Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [MUNBa: Machine Unlearning via Nash Bargaining](munba_machine_unlearning_via_nash_bargaining.md)
- [Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)

<!-- RELATED:END -->
