---
title: >-
  [论文解读] 联邦学习中水印的鲁棒性与放射性可能相互矛盾
description: >-
  [ICLR 2026][视频理解][联邦学习] 首次研究联邦学习中 LLM 水印的数据溯源问题，发现水印在 FL 中具有放射性（可检测），但恶意服务器可通过强鲁棒聚合算法过滤水印更新，揭示了放射性、鲁棒性和模型效用之间的根本性三元矛盾。
tags:
  - ICLR 2026
  - 视频理解
  - 联邦学习
  - LLM水印
  - 数据溯源
  - 鲁棒聚合
  - 放射性检测
---

# 联邦学习中水印的鲁棒性与放射性可能相互矛盾

**会议**: ICLR 2026  
**arXiv**: [2510.17033](https://arxiv.org/abs/2510.17033)  
**领域**: 视频理解  
**关键词**: 联邦学习, LLM水印, 数据溯源, 鲁棒聚合, 放射性检测

## 一句话总结

首次研究联邦学习中 LLM 水印的数据溯源问题，发现水印在 FL 中具有放射性（可检测），但恶意服务器可通过强鲁棒聚合算法过滤水印更新，揭示了放射性、鲁棒性和模型效用之间的根本性三元矛盾。

## 研究背景与动机

随着 LLM 生成的合成数据在联邦学习（FL）中的广泛使用，数据溯源（data provenance）变得至关重要：

**合成数据泛滥**：FL 客户端越来越多地使用 LLM 生成的合成数据集进行本地训练

**溯源需求**：若合成数据被恶意利用进行微调，无溯源机制则无法追责

**法规要求**：EU AI 法案等法规明确要求 AI 系统的透明性和可追溯性

**核心问题**：LLM 水印在 FL 环境中是否仍然有效？恶意服务器能否在保持模型效用的同时移除水印信号？

### 关键发现

通过 t-SNE 可视化发现：水印客户端的模型更新在高维空间中呈现为**异常值**，与干净客户端的更新分布明显分离（图1），这为恶意服务器提供了攻击手柄。

## 方法详解

### 整体框架

研究设置两种 FL 场景：

- **VanillaFL**：良性服务器，简单平均所有客户端更新
- **ActiveFL**：恶意服务器，使用鲁棒聚合器过滤水印更新

### 关键定义

**放射性（Radioactivity）**：数据集 $D^w$ 对统计检验 $T$ 是 $\alpha$-放射性的，如果检验 $T$ 能以 p-value 低于 $\alpha$ 拒绝零假设 $H_0$（模型未在 $D^w$ 上训练）。

**FL 鲁棒性（Robustness）**：若存在对抗者 $\mathcal{A}$ 使得：
1. $\mathcal{A}(U_\Delta, \theta^{t_\mathcal{A}}) \approx_\mathcal{E} \mathcal{T}(C_\Delta, \theta^{t_\mathcal{A}})$（保持效用）
2. $\text{Detect}^{\mathcal{M}_{\theta^{t_\mathcal{A}+1}},\mathcal{A}}_s(D^w_i) \rightarrow \text{False}$（水印不可检测）

则数据集 $D^w_i$ 对 $\mathcal{A}$ 不鲁棒。

### 攻击机制

恶意服务器使用**拜占庭鲁棒聚合器**（如 RandEigen）替代简单平均。聚合器保证偏差上界：

$$\text{bias} = \|\text{Fil}(U_\Delta) - \mu_C\|_2 \leq \beta \cdot \|\Sigma_C\|_2^{1/2}$$

其中强鲁棒聚合器保证 $\beta = O(1)$，与向量维度 $d$ 无关，适用于 LLM 的高维参数空间。

### 评估指标

- **逃逸率（ER）**：水印客户端在聚合后未被过滤的比例
- **过度过滤率（OFR）**：被过滤客户端中非水印客户端的比例

## 实验关键数据

### 放射性检测结果（VanillaFL，ε=6.6%）

| 数据集 | 水印 | 模型 | FT前 p-value | FT后 p-value |
|--------|------|------|-------------|-------------|
| C4 | KGW+ | 160M | 0.397 | $1.27\times10^{-3}$ |
| C4 | KGW+ | 410M | 0.877 | $2.41\times10^{-8}$ |
| Alpaca | KGW+ | 410M | 0.302 | $4.96\times10^{-24}$ |
| C4 | KTH+ | 全部 | ~0.5 | ~0.5 |

### 鲁棒性检测结果（ActiveFL vs VanillaFL，ε=6.6%）

| 数据集 | 模型 | VanillaFL p-value | ActiveFL p-value |
|--------|------|-------------------|------------------|
| C4 | 160M | $1.27\times10^{-3}$ | 0.550 |
| C4 | 410M | $2.41\times10^{-8}$ | 0.613 |
| Alpaca | 160M | $1.59\times10^{-11}$ | 0.231 |
| Alpaca | 410M | $4.96\times10^{-24}$ | 0.282 |

### 关键发现

1. **KGW+ 水印在 FL 中具有强放射性**：即使仅 6.6% 数据带水印，p-value 可低至 $10^{-24}$
2. **KTH+ 水印在 FL 中不具放射性**：因其检测器无法跨 prompt 累积统计信号
3. **RandEigen 聚合器有效移除所有水印**：所有放射性水印在 ActiveFL 下 p-value 均恢复至 ~0.5
4. **更大 δ 提高放射性但降低鲁棒性**：ER 从 60.2%（δ=0）降至 0.7%（δ=5）
5. **三元矛盾**：增大 ε 同时提高放射性和鲁棒性，但降低模型效用

## 亮点与洞察

1. **首个联邦数据溯源研究**：将水印检测从集中式扩展到分布式 FL 场景
2. **攻击者视角的洞察**：水印引入的分布偏移使更新成为异常值，恰好被设计用于防御拜占庭攻击的聚合器过滤
3. **三元权衡的揭示**：放射性（可检测性）、鲁棒性（抗攻击性）和效用（模型质量）无法同时满足
4. **实用的威胁模型**：服务器仅需更换聚合函数即可移除水印，无需知道水印方案细节

## 局限性

1. 仅评估了 Pythia 系列（70M-410M），未在更大模型上验证
2. 仅考虑两种水印方案（KGW+、KTH+），覆盖面有限
3. 假设恶意服务器不知道水印密钥和方案，实际中信息可能泄露
4. 未提出有效的防御方案来应对鲁棒聚合攻击
5. 水印客户端比例 ε 设置较小（最大 30%），更极端场景未探索

## 评分 ⭐⭐⭐⭐

问题定义新颖，实验设计严谨，揭示了 FL 中水印的根本性矛盾。虽然未提出解决方案，但为后续研究指明了方向。与视频理解领域关联较弱，更偏向安全与联邦学习交叉方向。

<!-- RELATED:START -->

## 相关论文

- [VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)
- [Video-KTR: 通过关键 Token 归因增强视频推理](video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)
- [AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [FuncBenchGen: 面向可靠基准测试的无污染可控评估框架](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)

<!-- RELATED:END -->
