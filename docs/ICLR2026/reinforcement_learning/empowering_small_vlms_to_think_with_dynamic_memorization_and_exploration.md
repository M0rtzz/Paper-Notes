---
title: >-
  [论文解读] Empowering Small VLMs to Think with Dynamic Memorization and Exploration
description: >-
  [ICLR 2026][小规模视觉语言模型] 提出 DyME（Dynamic Memorize-Explore），通过逐步动态切换 SFT 记忆模式与 GRPO 探索模式，首次赋予小规模视觉语言模型（<1B 参数）在特定任务上的思维推理能力。
tags:
  - ICLR 2026
  - 小规模视觉语言模型
  - 思维能力
  - 动态切换
  - SFT与RLVR融合
  - 视觉监督
---

# Empowering Small VLMs to Think with Dynamic Memorization and Exploration

**会议**: ICLR 2026  
**arXiv**: [2506.23061](https://arxiv.org/abs/2506.23061)  
**代码**: [有](https://github.com/HKUST-LongGroup/DyME)  
**领域**: 强化学习  
**关键词**: 小规模视觉语言模型, 思维能力, 动态切换, SFT与RLVR融合, 视觉监督

## 一句话总结
提出 DyME（Dynamic Memorize-Explore），通过逐步动态切换 SFT 记忆模式与 GRPO 探索模式，首次赋予小规模视觉语言模型（<1B 参数）在特定任务上的思维推理能力。

## 研究背景与动机
- 大模型（如 Qwen2.5-VL-32B）可通过 SFT 或 RLVR 获得推理能力，但**小模型（SVLM，<1B）两种范式都失败**：
    - SFT 失败：CoT 数据冗长且含大量与视觉无关内容，SVLM 容量不足以吸收，导致"伪思维轨迹"
    - RLVR 失败：SVLM 指令遵循能力差，频繁生成无法验证的输出，引发"优势坍塌"（advantage collapse）
- 两阶段训练（SFT→RL）的静态平衡窗口极窄，SVLM 几乎不可能成功
- 实际需求：SVLM 适合边缘设备部署，赋予其思维能力有强烈实践意义

## 方法详解

### 整体框架
DyME 由两个核心机制组成：
1. **动态切换机制（Pure DyME）**：每步根据模型生成质量自适应选择 SFT 或 GRPO
2. **视觉监督机制（Visual Supervision）**：通过视觉检查器和精炼器注入图像锚定的动态指导

### 关键设计

**动态切换规则**：

给定输入 x，策略模型采样 K 个回复，每个回复经规则验证答案正确性：
- 若**至少一个正确**→ GRPO 模式（探索）
- 若**全部错误**（含格式错误）→ SFT 模式（记忆）

这个二值切换看似简单但极其有效：全部错误时优势被噪声主导，GRPO 不稳定；至少一个正确时 GRPO 可安全利用相对优势驱动探索。

**SFT 与 GRPO 的梯度兼容性**：
作者揭示两者在梯度层面形式等价：
- SFT 梯度 = 外部数据分布下的对数概率梯度
- GRPO 梯度 = 内部采样分布下加权对数概率梯度
- SFT 是 GRPO 的特殊情况（真实样本 + 单位优势值）
- 这使得统一损失函数在理论上成立

**GRPO 模式增强**：
- 引入思维轨迹的辅助奖励 r_t（通过 token-level F1 与参考 CoT 比较）
- **去除** KL 惩罚和 clipping（动态 SFT 切换已隐式提供稳定性），简化梯度形式

**视觉检查器（Visual Checker）**：
- 从图像中提取视觉事实 I_c（对象、属性、状态等细粒度视觉组件）
- 沿两个维度评估回复：(1) 是否包含足够正确的视觉元素 (2) 是否符合风格示例
- 评分作为 GRPO 的思维奖励信号

**视觉精炼器（Visual Refiner）**：
- 高评分探索轨迹存入动态示例池
- LLM 从池中采样，结合结构模板与 I_c，为 SFT 模式生成视觉锚定的参考响应
- 实现"探索成功→提升记忆目标"的正反馈循环
- 视觉事实提取、检查、精炼均通过 Qwen2.5-14B 的结构化 prompt 实现

### 损失函数 / 训练策略
完整 DyME 目标在每步根据指示函数动态选择 GRPO 损失或 SFT 损失：
- 不需要额外超参数调优（无阈值、无退火系数、无预算）
- 视觉事实通过领域工具自动提取（如 BiomedGPT 用于医学、DePlot 用于图表）
- 训练数据仅需几千条样本

## 实验关键数据

### 主实验（跨三个领域）

| 模型 | 方法 | Medical | Chart | Geometry | Avg |
|------|------|---------|-------|----------|-----|
| SmolVLM (0.5B) | 基线 | 72.1 | 63.2 | 14.6 | 49.9 |
| SmolVLM | + SFT | 60.1 | 57.7 | 14.5 | 44.1 (↓) |
| SmolVLM | + GRPO | 61.1 | 53.8 | 17.1 | 44.0 (↓) |
| SmolVLM | + Two-stage | 59.4 | 60.1 | 16.7 | 45.4 (↓) |
| SmolVLM | **+ DyME** | **78.1** | **69.7** | **18.9** | **55.6 (+5.7)** |
| LLaVA-OV-S (0.5B) | 基线 | 74.9 | 61.4 | 15.9 | 50.7 |
| LLaVA-OV-S | **+ DyME** | **78.3** | **67.5** | **20.4** | **55.4 (+4.7)** |
| InternVL2-S (0.5B) | 基线 | 78.3 | 71.9 | 18.7 | 56.3 |
| InternVL2-S | **+ DyME** | **80.0** | **74.5** | **19.8** | **58.1 (+1.8)** |

DyME 训练的 SVLM 性能超过 7B 参数的 MoVA（54.2）。

| 切换策略对比（Medium 数据） | Acc | 额外代价 |
|--------------------------|-----|---------|
| Reward Thresholding (t=0.5) | 52.4 | 无 |
| SFT Annealing (Cosine) | 64.0 | +25% |
| SFT Budget (Hard Mining) | 59.6 | 依赖预算 |
| **Binary Switch (DyME)** | **64.9** | 基线 |

### 消融实验

| DyME 变体 | Medical | Chart | Geometry | Avg |
|-----------|---------|-------|----------|-----|
| DyME (完整) | 78.3 | 67.5 | 20.4 | 55.4 |
| 去除记忆模式 | 63.2 | 53.4 | 15.0 | 43.9 (↓20.6%) |
| 去除探索模式 | 75.5 | 61.3 | 14.5 | 50.4 (↓9.0%) |
| 去除视觉精炼器 | 75.6 | 62.3 | 16.8 | 51.6 (↓6.9%) |
| 去除视觉检查器 | 76.9 | 64.3 | 17.1 | 52.8 (↓4.7%) |

### 关键发现
1. 记忆模式是基石（去除后↓20.6%），探索模式是增益引擎（去除后↓9.0%）
2. 开源模型（Qwen2.5-14B）+Full DyME 可达到 GPT-4o 数据 + Pure DyME 的同等效果
3. DyME 跨模态通用：纯文本 Qwen2.5-0.5B 在 GSM8K 上 +5.8%；7B 模型上也可 +2.3%
4. 训练效率：Pure DyME 与标准 GRPO 速度相当（约 14s/step），Full DyME 约 1.6 倍开销
5. 外部辅助模型从 14B 换成 7B 性能损失微乎其微（67.5% vs 66.8%）

## 亮点与洞察
- **参数无关的切换规则**：不需要任何阈值/退火系数等超参数，二值切换本身就是最优策略
- SFT 和 GRPO 的**梯度兼容性证明**为统一损失函数提供了理论基础
- 视觉精炼器实现了"探索成功→提升记忆目标"的正反馈循环，数据质量在训练中自我改善
- 证明了"能力弱的模型需要更智能的训练范式"这一直觉
- 使用低质量（Undesigned）CoT 数据 + DyME 也能获得显著提升，降低了数据门槛

## 局限与展望
- 视觉监督依赖视觉事实 I_c 的可提取性，对涉及抽象语义（如 meme 讽刺）或非结构化感知的场景可能失效
- 仅在 ≤7B 模型上验证，更大规模模型是否需要 DyME 待研究
- 当前依赖外部 LLM（Qwen2.5-14B）做视觉精炼，完全闭环的自主改善更理想
- 三个领域的训练样本仅几千条，更大规模数据下的行为待观察

## 相关工作与启发
- 直接动机来自 DeepSeek-R1：纯 RL 可激励推理，但需要强底模能力
- 与 R1-V、LMM-R1 等视觉 RLVR 工作互补：它们聚焦大模型，DyME 聚焦小模型
- 两阶段训练（SFT→RL）的失败分析为理解 SVLM 能力边界提供了实证
- Multimodal-CoT 是早期尝试但数据规模受限；G-LLaVA、LLaVA-CoT 依赖大规模 CoT 数据

## 评分
- 新颖性: 5/5 （首次解决 SVLM 思维能力问题，动态切换设计优雅）
- 实验充分度: 5/5 （算法验证+系统验证双轨实验，三领域三模型，消融详尽）
- 写作质量: 5/5 （动机阐述清晰，SFT/GRPO 梯度统一分析精彩）
- 价值: 5/5 （实用性极强，对边缘端 SVLM 部署有直接推动）

<!-- RELATED:START -->

## 相关论文

- [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](../../AAAI2026/reinforcement_learning/start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)

<!-- RELATED:END -->
