---
title: >-
  [论文解读] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)
description: >-
  [ICLR 2026][图像生成][Object-Centric Learning] 提出 CODA 框架，通过引入 register slots 吸收残余注意力、微调交叉注意力投影以及对比对齐损失，解决基于扩散模型的物体中心学习中的 slot 纠缠和弱对齐问题，在合成和真实数据集上显著提升物体发现和组合式生成质量。
tags:
  - ICLR 2026
  - 图像生成
  - Object-Centric Learning
  - 注意力机制
  - Register Slots
  - 对比学习
  - 组合式生成
---

# Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)

**会议**: ICLR 2026  
**arXiv**: [2601.01224](https://arxiv.org/abs/2601.01224)  
**代码**: [GitHub](https://github.com/sony/coda)  
**领域**: 物体中心学习 / 扩散模型  
**关键词**: Object-Centric Learning, Slot Attention, Register Slots, contrastive learning, 组合式生成

## 一句话总结

提出 CODA 框架，通过引入 register slots 吸收残余注意力、微调交叉注意力投影以及对比对齐损失，解决基于扩散模型的物体中心学习中的 slot 纠缠和弱对齐问题，在合成和真实数据集上显著提升物体发现和组合式生成质量。

## 研究背景与动机

物体中心学习（OCL）旨在将复杂场景分解为结构化的、可组合的物体表示，支撑视觉推理、因果推断、世界模型和组合式生成等下游任务。Slot Attention（SA）是一种完全无监督的方法，但在真实场景中表现有限。近期将 SA 与预训练扩散模型（如 Stable Diffusion）结合的方法（Stable-LSD, SlotAdapt）虽取得进展，但仍面临两大核心问题：

**Slot 纠缠（Slot Entanglement）**：一个 slot 编码了多个物体的特征，导致单 slot 生成时图像失真或语义不一致。本质原因是 softmax 归一化迫使注意力权重在所有 slot 上求和为 1，当 U-Net 中的某些 query 不强匹配任何语义 slot 时，注意力分散到多个 slot 上。

**弱对齐（Weak Alignment）**：slot 未能与不同的图像区域一致对应，导致过分割（一个物体被拆成多个 slot）或欠分割（多个物体合并到一个 slot）。

这两个问题严重影响了物体中心表示的准确性和组合式场景生成的实用性。

## 方法详解

### 整体框架

CODA 基于 DINOv2 提取图像特征，通过 Slot Attention 产生 slot 表示，再用预训练的 Stable Diffusion v1.5 作为 slot 解码器重建输入图像。在此基础上引入三个改进模块。

### 关键设计

1. **Register Slots（注册槽）**：将纯 padding token 通过 SD 的冻结文本编码器（CLIP ViT-L/14）得到固定长度的嵌入序列 $\bar{\mathbf{r}}$，作为与输入无关的 register slots。这些 register slots 在 cross-attention 中充当"注意力吸收器"，吸收本不应分配给语义 slot 的残余注意力质量。由于 softmax 归一化约束，当 U-Net query 不匹配任何语义 slot 时，注意力会自然流向 register slots 而非干扰语义 slot，从而减轻 slot 纠缠。SD v1.5 使用 77 个 padding token，因此产生 77 个 register slots。实验表明固定 register slots 优于可训练的版本。

2. **交叉注意力微调（Cross-Attention Finetuning）**：SD 预训练于图文对，直接作为 slot 解码器会带来文本条件偏置（text-conditioning bias），模型倾向于语言驱动的语义而非 slot 级表示。CODA 仅微调 cross-attention 层中的 key/value/output 投影矩阵 $\boldsymbol{\theta}$，不引入额外层或适配器，概念简洁且计算高效。去噪目标为：

$$\mathcal{L}_{\mathrm{dm}}(\phi, \boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{z}, \mathbf{s}), \epsilon, \gamma} \left[\|\epsilon - \epsilon_{\boldsymbol{\theta}}(\mathbf{z}_\gamma, \gamma, \mathbf{s}, \bar{\mathbf{r}})\|_2^2\right]$$

3. **对比对齐目标（Contrastive Alignment）**：仅靠去噪损失无法显式保证 slot 捕捉图像中实际存在的概念。CODA 引入对比损失，使用负样本 $\tilde{\mathbf{s}}$（通过跨图像随机替换一半 slot 构造困难负样本），让模型给匹配 slot 高似然、给不匹配 slot 低似然：

$$\mathcal{L}_{\mathrm{cl}}(\phi) = -\mathbb{E}_{(\mathbf{z}, \tilde{\mathbf{s}}), \epsilon, \gamma} \left[\|\epsilon - \epsilon_{\bar{\boldsymbol{\theta}}}(\mathbf{z}_\gamma, \gamma, \tilde{\mathbf{s}}, \bar{\mathbf{r}})\|_2^2\right]$$

其中 $\bar{\boldsymbol{\theta}}$ 表示停止梯度的参数。关键设计是对比损失只更新 SA 模块而冻结扩散解码器，防止解码器走捷径。

### 损失函数 / 训练策略

总损失为去噪损失与对比损失的加权和：

$$\mathcal{L}(\phi, \boldsymbol{\theta}) = \mathcal{L}_{\mathrm{dm}}(\phi, \boldsymbol{\theta}) + \lambda_{\mathrm{cl}} \mathcal{L}_{\mathrm{cl}}(\phi)$$

理论上，该目标等价于最大化 slot 与图像之间互信息的可操作代理（Theorem 1），其中去噪差 $\Delta$ 作为互信息的实用近似。困难负样本的构造通过共享初始化来保证语义有效性。

## 实验关键数据

### 主实验：物体发现

| 数据集 | 指标 | CODA | SlotAdapt (之前SOTA) | 提升 |
|--------|------|------|----------|------|
| MOVi-C | FG-ARI | 59.19 | 51.98 (LSD) | +7.21 |
| MOVi-E | FG-ARI | 59.04 | 56.45 | +2.59 |
| VOC | FG-ARI | 32.23 | 29.6 | +2.63 |
| VOC | mBOi | 55.38 | 51.5 | +3.88 |
| VOC | mIoUc | 56.30 | 49.3 (SlotDiff) | +7.00 |
| COCO | FG-ARI | 47.54 | 41.4 | +6.14 |

### 消融实验

| 配置 | FG-ARI | mBOi | mBOc | mIoUi | mIoUc |
|------|--------|------|------|-------|-------|
| Baseline (Frozen SD) | 12.27 | 47.21 | 54.20 | 48.72 | 55.71 |
| + Register Slots | 19.21 | 55.76 | 64.02 | 49.93 | 57.14 |
| + CA Finetuning | 15.44 | 47.03 | 52.63 | 49.75 | 55.63 |
| + Contrastive | 11.96 | 47.16 | 54.17 | 49.40 | 56.56 |
| Reg + CA | 19.62 | 56.27 | 65.05 | 50.40 | 58.02 |
| Reg + CA + CO (不停止梯度) | 10.54 | 30.64 | 35.86 | 37.74 | 43.61 |
| **Reg + CA + CO (CODA)** | **32.23** | **55.38** | **61.32** | **50.77** | **56.30** |

### 关键发现

- Register Slots 是提升最显著的单一组件（mBO 提升约 10 个点），有效缓解 slot 纠缠
- 对比损失中必须对扩散解码器停止梯度，否则训练不稳定且性能严重下降
- 在组合式生成中，CODA 的 FID 从 SlotAdapt 的 40.57 降至 31.03
- 属性预测中类别分类准确率从 43.92% 提升到 78.06%（MOVi-E）

## 亮点与洞察

- Register Slots 的设计受到 LLM 中"注意力吸收"现象的启发，概念简洁且几乎零计算开销
- 从互信息最大化的理论视角统一了去噪损失和对比损失
- 仅微调 cross-attention 的 KVO 投影即可消除文本条件偏置，无需额外架构改动
- 支持精细的组合式编辑（删除物体、交换物体），实用性强

## 局限与展望

- 3D 包围盒预测表现不佳，DINOv2 特征缺乏精细的几何细节
- 目前仅在 SD v1.5 上验证，扩展到更大模型（SDXL、SD3）的效果未知
- 在极密集遮挡场景中的分割质量仍有提升空间
- Register Slots 数量（77）由 SD 文本编码器决定，换模型需重新设计

## 相关工作与启发

- DINOSAUR、SPOT 等方法从自监督特征出发改进 OCL，而 CODA 从扩散模型解码器侧改进
- ViT 中的 register token 思想可迁移到更多需要注意力竞争的场景
- 对比对齐目标的设计思路可推广到其他 slot-based 生成任务（如视频物体中心学习）

## 评分

- 新颖性: ⭐⭐⭐⭐ Register Slots 和对比对齐的组合设计新颖，但各单一技术并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实数据集，物体发现+属性预测+组合生成+消融，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实验展示完善，图表信息量大
- 价值: ⭐⭐⭐⭐ 对 OCL 社区有实质贡献，方法简洁高效且易于在现有框架上复现

<!-- RELATED:START -->

## 相关论文

- [GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](../../CVPR2025/image_generation/glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](../../CVPR2025/image_generation/ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](../../ICCV2025/image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)

<!-- RELATED:END -->
