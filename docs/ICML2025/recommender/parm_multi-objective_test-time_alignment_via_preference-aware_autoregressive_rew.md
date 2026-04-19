---
title: >-
  [论文解读] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model
description: >-
  [ICML2025][多目标对齐] 提出PARM——统一的Preference-Aware自回归奖励模型，通过PBLoRA双线性低秩适配以偏好向量为条件，实现单个ARM替代K个独立ARM的多目标测试时对齐，还支持弱到强引导（小PARM引导大LLM）。
tags:
  - ICML2025
  - 多目标对齐
  - 测试时对齐
  - 自回归奖励模型
  - 偏好向量
  - PBLoRA
---

# PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model

**会议**: ICML2025  
**arXiv**: [2505.06274](https://arxiv.org/abs/2505.06274)  
**代码**: [GitHub - PARM](https://github.com/Baijiong-Lin/PARM)  
**领域**: recommender  
**关键词**: 多目标对齐, 测试时对齐, 自回归奖励模型, 偏好向量, PBLoRA

## 一句话总结
提出PARM——统一的Preference-Aware自回归奖励模型，通过PBLoRA双线性低秩适配以偏好向量为条件，实现单个ARM替代K个独立ARM的多目标测试时对齐，还支持弱到强引导（小PARM引导大LLM）。

## 研究背景与动机

### 多目标对齐的需求
用户希望LLM响应同时满足多维偏好（如有帮助+无害+幽默），需要在推理时根据偏好向量动态调整。

### GenARM的局限
先驱工作GenARM为每个偏好维度独立训练一个ARM：
1. K个ARM增加推理成本
2. 独立训练的ARM之间不感知彼此→引导生成与偏好不对齐

### PARM的改进
单个统一ARM，以偏好向量为条件，既减少推理成本又改善对齐。

## 方法详解

### Preference-Aware Bilinear Low-Rank Adaptation (PBLoRA)
- 用双线性形式将偏好向量注入LoRA适配层
- W = W_base + alpha * B * diag(Wv * p) * A
- p是偏好向量，Wv是偏好投影矩阵
- 实现精细的偏好权衡控制

### 训练
单个ARM在所有偏好维度的数据上联合训练，偏好向量作为输入条件。

### 推理
给定用户偏好向量p，PARM输出对应的token级奖励，引导冻结LLM的生成。

### 弱到强引导
小型PARM(如7B)可引导大型冻结LLM(如65B)，无需昂贵的大模型训练。

## 实验关键数据

### 推理成本对比

| 方法 | 偏好维度k | 推理ARM数 | 参数量 |
|------|---------|----------|-------|
| GenARM | k=3 | 3 | 3x |
| **PARM** | k=3 | **1** | **1x** |

### 对齐质量（HH-RLHF）

| 方法 | 偏好对齐度 | 有帮助性 | 无害性 |
|------|----------|---------|--------|
| GenARM(3个ARM) | 中 | 高 | 高 |
| **PARM(1个ARM)** | **高** | **高** | **更高** |

### 弱到强引导

| PARM大小 | 被引导LLM | 效果 |
|---------|----------|------|
| 7B | 13B | 有效 |
| 7B | 65B | **有效** |

### 关键发现
1. 单个PARM的对齐质量优于K个独立GenARM
2. PBLoRA的双线性形式有效编码偏好权衡
3. 弱到强引导可行——显著降低多目标对齐的门槛
4. 推理成本降低K倍
5. 对齐质量随PARM规模增大而提升

## 亮点与洞察

1. "统一ARM替代K个独立ARM"大幅降低实用成本。
2. PBLoRA的偏好条件化设计精炼——用双线性形式而非简单拼接。
3. 弱到强引导使多目标对齐对资源受限用户可行。
4. 保持LLM冻结——不改变原模型。
5. 与现有对齐方法正交——可与RLHF/DPO等结合。

## 局限与展望

1. PBLoRA的低秩维度是新超参。
2. 偏好维度增多时PBLoRA的表达力是否足够。
3. 弱到强引导在极端偏好设置下的稳定性。
4. 仅验证文本生成，多模态场景待扩展。
5. 训练数据中偏好维度的正交性假设可能不成立。

## 相关工作与启发

- 与GenARM的直接改进关系。
- 与Rewarded Soups/MOD的区别：它们需要训练基座模型。
- 启发：条件化LoRA的设计可推广到其他多目标调控场景。

## 评分
- 新颖性: 4.5/5 — PBLoRA+统一ARM
- 实验充分度: 4.5/5 — 多维度+弱到强
- 写作质量: 4.5/5
- 价值: 5.0/5 — 显著降低多目标对齐成本

## 补充设计细节

### PBLoRA的双线性形式
传统LoRA: W = W_base + B*A。PBLoRA: W = W_base + B * diag(Wp*p) * A。偏好向量p通过对角矩阵控制A各行的缩放，实现精细控制。

### 弱到强引导的工作机制
小型PARM在每步提供token级奖励信号，引导大型LLM的采样分布。大模型保持冻结，只需小模型的推理开销。

<!-- RELATED:START -->

## 相关论文

- [Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [MMPB: It's Time for Multi-Modal Personalization](../../NeurIPS2025/recommender/mmpb_its_time_for_multi-modal_personalization.md)
- [Interpretable Reward Model via Sparse Autoencoder](../../AAAI2026/recommender/interpretable_reward_model_via_sparse_autoencoder.md)
- [SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](../../ACL2025/recommender/gram_generative_recommendation.md)

<!-- RELATED:END -->
