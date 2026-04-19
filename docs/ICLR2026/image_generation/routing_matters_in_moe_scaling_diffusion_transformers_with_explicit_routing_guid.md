---
title: >-
  [论文解读] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance
description: >-
  [ICLR 2026][图像生成][Mixture-of-Experts] 提出 ProMoE，一种针对扩散 Transformer 的 MoE 框架，通过两步路由器（条件路由 + 原型路由）和路由对比损失提供显式语义引导，促进专家特化，在 ImageNet 上显著超越现有 MoE 和稠密模型。
tags:
  - ICLR 2026
  - 图像生成
  - Mixture-of-Experts
  - DiT
  - 显式路由引导
  - 原型路由
  - 路由对比损失
---

# Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance

**会议**: ICLR 2026  
**arXiv**: [2510.24711](https://arxiv.org/abs/2510.24711)  
**代码**: [https://github.com/ali-vilab/ProMoE](https://github.com/ali-vilab/ProMoE)  
**领域**: 扩散模型 / 混合专家  
**关键词**: Mixture-of-Experts, DiT, 显式路由引导, 原型路由, 路由对比损失

## 一句话总结

提出 ProMoE，一种针对扩散 Transformer 的 MoE 框架，通过两步路由器（条件路由 + 原型路由）和路由对比损失提供显式语义引导，促进专家特化，在 ImageNet 上显著超越现有 MoE 和稠密模型。

## 研究背景与动机

MoE 在 LLM 中取得了巨大成功，但在 DiT 中的表现令人失望：
- DiT-MoE（令牌选择路由）性能甚至不如稠密模型
- EC-DiT（专家选择路由）仅获得微弱提升
- DiffMoE（全局令牌分布路由）改进也很有限

**根本原因分析**：语言令牌和视觉令牌存在本质差异：

**高空间冗余**：视觉令牌连续、空间耦合、高度冗余（类间/类内距离比仅 0.748 vs LLM 的 19.283），导致专家学习同质特征

**功能异质性**：CFG 引入了条件/无条件两种功能不同的输入类型，朴素 MoE 未区分处理

## 方法详解

### 整体框架

ProMoE 包含两步路由器 + 路由对比学习，目标是促进：
- **内部专家一致性**：同一专家持续处理相似模式
- **跨专家多样性**：不同专家特化于不同任务

### 关键设计一：条件路由

根据令牌的功能角色进行硬路由分区：
- **无条件令牌**（空标签/文本下的图像块）→ $N_u$ 个无条件专家
- **条件令牌**（特定条件下的图像块）→ 路由专家（第二步决定）

前向过程：

$$\text{MoE}(\mathbf{x}) = \underbrace{\sum_{i=1}^{N_s} E_i^S(\mathbf{x})}_{\text{Shared}} + \begin{cases}\sum_{j=1}^{N_E}\mathbf{G}_j E_j(\mathbf{x}) & \mathbf{x} \in \mathbf{X}_c \\ \sum_{k=1}^{N_u}E_k^U(\mathbf{x}) & \mathbf{x} \in \mathbf{X}_u\end{cases}$$

### 关键设计二：原型路由

引入可学习原型 $\mathbf{P} \in \mathbb{R}^{N_E \times D}$，每个原型对应一个专家。使用余弦相似度分配令牌：

$$\mathbf{Z}_{i,j} = \alpha \frac{\mathbf{x}_i \mathbf{p}_j^\top}{\|\mathbf{x}_i\| \|\mathbf{p}_j\|}$$

激活函数选择恒等函数 $\mathcal{A}(\mathbf{Z}) = \mathbf{Z}$（优于 softmax 和 sigmoid）。

### 关键设计三：路由对比损失

显式增强原型路由的语义引导，拉近每个原型与其正集质心，推开负集质心：

$$\mathcal{L}_{\text{RCL}} = -\frac{1}{N_a}\sum_{i=1}^{N_a}\log\frac{\exp(\text{sim}(\mathbf{p}_i, \mathbf{m}_i)/\tau)}{\sum_{j=1}^{N_a}\exp(\text{sim}(\mathbf{p}_i, \mathbf{m}_j)/\tau)}$$

其中 $\mathbf{m}_i$ 为分配给专家 $E_i$ 的令牌质心。RCL 的推开操作还兼具负载均衡效果。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{RCL}} \mathcal{L}_{\text{RCL}}$$

## 实验

### 与稠密模型对比（Rectified Flow，500K 步）

| 模型 | 激活参数 | 总参数 | FID↓ (cfg=1.0) | FID↓ (cfg=1.5) |
|------|---------|-------|---------------|---------------|
| Dense-DiT-B | 130M | 130M | 30.61 | 9.02 |
| ProMoE-B | 130M | 300M | 24.44 | 6.39 |
| Dense-DiT-L | 458M | 458M | 15.44 | 3.56 |
| ProMoE-L | 458M | 1.063B | 11.61 | 2.79 |
| Dense-DiT-XL | 675M | 675M | 13.38 | 3.23 |
| ProMoE-XL | 675M | 1.568B | 9.44 | 2.59 |

ProMoE-L 使用更少的激活参数（458M）即超越 Dense-DiT-XL（675M）。

### 语义引导验证

| 方法 | FID↓ (cfg=1.5) | IS↑ |
|------|---------------|-----|
| Dense-DiT-B | 9.02 | 131.13 |
| DiT-MoE-B | 8.94 | 131.66 |
| DiffMoE-B | 8.22 | 137.46 |
| 分类路由引导 | **5.91** | **165.45** |
| K-Means 路由引导 | 6.24 | 159.77 |

显式/隐式语义引导均带来显著提升，验证了视觉 MoE 需要语义引导。

### 与 MoE 基线对比

ProMoE 在所有规模和训练范式（DDPM/RF）上均超越 DiT-MoE、EC-DiT、DiffMoE。

### 关键发现

- 视觉 MoE 的核心瓶颈是专家同质化（缺乏引导时专家子空间高度相似）
- 条件路由有效消除了功能异质性对路由的干扰
- RCL 无需人工标签，比分类损失更灵活，比 K-Means 更鲁棒
- RCL 的推开操作自然替代了传统的负载均衡损失

## 亮点

- 深入分析了 MoE 在视觉与语言中效果差异的根因
- 两步路由 + 对比损失的设计简洁有效，可推广到其他视觉 MoE
- 参数效率突出：更少激活参数超越更大稠密模型
- 在 DDPM 和 Rectified Flow 两种范式上均验证有效

## 局限性

- 仅在类条件 ImageNet 上评估，未验证文生图等更复杂场景
- 条件路由要求 CFG 推理，不适用于不使用 CFG 的场景
- 聚类/对比学习的计算开销未详细分析
- 总参数量约为稠密模型的 2.3 倍

## 相关工作

- **DiT MoE**：DiT-MoE、EC-DiT、DiffMoE 等在视觉 MoE 上的尝试
- **LLM MoE**：DeepSeek-MoE、Mixtral 等语言领域的成功应用
- **扩散模型**：DiT、SiT 等 Transformer 架构的扩散模型

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 分析深入，两步路由 + RCL 组合新颖
- 技术性：⭐⭐⭐⭐ — 实验设计严谨，消融充分
- 实验：⭐⭐⭐⭐ — 多尺度多范式验证
- 影响力：⭐⭐⭐⭐⭐ — 为视觉 MoE 指明了方向

<!-- RELATED:START -->

## 相关论文

- [CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](../../CVPR2026/image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [Mixture of States: Routing Token-Level Dynamics for Multimodal Generation](../../CVPR2026/image_generation/mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)
- [Scaling Diffusion Transformers Efficiently via μP](../../NeurIPS2025/image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)
- [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)

<!-- RELATED:END -->
