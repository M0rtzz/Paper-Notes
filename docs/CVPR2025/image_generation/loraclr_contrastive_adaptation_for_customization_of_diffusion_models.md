---
title: >-
  [论文解读] LoRACLR: Contrastive Adaptation for Customization of Diffusion Models
description: >-
  [CVPR 2025][图像生成][LoRA合并] LoRACLR 提出一种基于对比学习目标的 LoRA 模型合并方法，通过学习一个 delta 权重将多个独立训练的单概念 LoRA 模型融合为一个统一模型，无需重训练或访问原始训练数据，即可实现高保真的多概念图像生成，合并 12 个概念仅需 5 分钟。
tags:
  - CVPR 2025
  - 图像生成
  - LoRA合并
  - 对比学习
  - 多概念生成
  - 模型融合
  - 扩散模型定制
---

# LoRACLR: Contrastive Adaptation for Customization of Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.09622](https://arxiv.org/abs/2412.09622)  
**代码**: https://loraclr.github.io  
**领域**: 扩散模型 / 个性化图像生成  
**关键词**: LoRA合并, 对比学习, 多概念生成, 模型融合, 扩散模型定制

## 一句话总结

LoRACLR 提出一种基于对比学习目标的 LoRA 模型合并方法，通过学习一个 delta 权重将多个独立训练的单概念 LoRA 模型融合为一个统一模型，无需重训练或访问原始训练数据，即可实现高保真的多概念图像生成，合并 12 个概念仅需 5 分钟。

## 研究背景与动机

**领域现状**：文本到图像扩散模型（如 Stable Diffusion）结合 LoRA 微调实现了高效的个性化图像生成。社区已有大量针对单一概念（角色、物体、风格）训练好的 LoRA 模型。

**现有痛点**：当需要在一张图中同时生成多个个性化概念时，现有方法面临严重挑战。加权求和（FedAvg）导致特征干扰；Custom Diffusion 需要同时在多概念上训练；Mix-of-Show 需要特殊的 ED-LoRA 格式，与社区标准 LoRA 不兼容；ZipLoRA 只能合并风格+内容 LoRA；OMG 依赖分割模型的精度；Orthogonal Adaptation 需要在原始数据上重新微调每个 LoRA。

**核心矛盾**：独立训练的 LoRA 模型在各自的权重空间中编码了不同概念，直接合并会导致概念交叉（如把 A 的头发特征混到 B 上）或概念丢失。但用户期望能直接使用已有的社区 LoRA 模型进行组合，不想重训练。

**本文目标** 如何在不重训练、不访问原始数据的条件下，将多个独立的 LoRA 模型合并为一个能准确生成所有概念的统一模型？

**切入角度**：用对比学习来对齐合并后模型的权重空间——同一概念的输出应与原始 LoRA 的输出一致（正对），不同概念的输出应互相远离（负对），从而自然地保持各概念的独立性。

**核心 idea**：通过学习一个可加的 delta 权重，用对比损失使合并模型对每个概念的特征输出趋近原始 LoRA、远离其他 LoRA，实现无需重训练的多概念 LoRA 合并。

## 方法详解

### 整体框架

LoRACLR 的输入是 $N$ 个独立训练的单概念 LoRA 模型 $\{V_i\}_{i=1}^N$。第一阶段，用每个 LoRA 模型分别生成概念专属的输入-输出特征对 $(X_i, Y_i)$，建立正负样本对。第二阶段，初始化一个全零的 delta 权重 $\Delta W$，通过最小化对比损失 + L2 正则化来学习 $\Delta W$，使得合并后的模型 $W + \Delta W$ 能正确再现所有概念。优化完成后，将所有 LoRA 权重加上 $\Delta W$ 得到最终的统一模型。整个过程在 A100 上约 5 分钟（12 个概念）。

### 关键设计

1. **对比合并目标（Contrastive Merging Objective）**:

    - 功能：在权重空间中对齐多个 LoRA 模型，保持概念独立性
    - 核心思路：对每个概念 $i$，正样本距离 $d_{p,i} = \|Y_i - \hat{Y}_i\|_2$（原始 LoRA 输出 vs 合并模型输出），负样本距离 $d_{n,i} = \min_{j \neq i} \|Y_i - \hat{Y}_j\|_2$（概念 $i$ 的原始输出 vs 其他概念的合并模型输出）。对比损失 $\mathcal{L}_{contrastive} = \frac{1}{N} \sum_{i=1}^{N} (d_{p,i}^2 + \max(0, m - d_{n,i})^2)$，其中 $m$ 是 margin 参数。正对拉近确保身份保留，负对推远防止概念交叉。
    - 设计动机：简单的加权平均合并（FedAvg）无法处理概念间的干扰，因为不同 LoRA 可能在相似的权重方向上编码了不同概念。对比学习天然适合处理"保持各自特性、防止混淆"的需求。

2. **Delta-Based 合并策略**:

    - 功能：通过增量更新合并 LoRA，保护各模型的原始权重完整性
    - 核心思路：不直接修改任何 LoRA 模型的权重，而是学习一个附加的 $\Delta W$（初始化为零）。对 $\Delta W$ 施加 L2 正则 $\mathcal{L}_{delta} = \lambda_{delta} \|\Delta W\|_2$，鼓励稀疏和最小调整。总损失 $\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \mathcal{L}_{delta}$。通过梯度下降优化 $\Delta W$。
    - 设计动机：直接修改 LoRA 权重可能破坏原始概念表示。Delta-based 策略将合并过程限制在最小必要调整范围内，保持了向后兼容性——原始 LoRA 仍可单独使用。

3. **兼容社区 LoRA 的即插即用设计**:

    - 功能：支持直接使用社区平台（如 civitai）上的预训练 LoRA 模型
    - 核心思路：LoRACLR 的整个流程不需要访问原始训练数据，也不需要特殊的 LoRA 变体（如 ED-LoRA）。只需要现有 LoRA 模型的权重文件即可生成特征对并执行合并优化。支持标准 LoRA 和 ED-LoRA 格式。合并后的模型可以用任意 prompt 生成，无需额外推理开销。
    - 设计动机：现有方法（如 Mix-of-Show、Orthogonal Adaptation）要么需要特殊 LoRA 格式，要么需要在原始数据上重训练，严重限制了实用性。LoRACLR 的设计面向实际应用场景。

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \mathcal{L}_{delta}$。使用 Stable Diffusion + ChilloutMix checkpoint。学习率 1e-4，margin $m = 0.5$，$\lambda_{delta} = 0.001$。LoRA 应用于交叉注意力层的 $W_{in}$ 和 $W_{out}$。在 NVIDIA A100 上运行，合并 12 个概念约 5 分钟，生成图像约 10 秒。

## 实验关键数据

### 主实验

12 个概念身份的合并结果：

| 方法 | Text Align ↑ | Image Align ↑ (Δ) | Identity Align ↑ (Δ) |
|------|-------------|-------------------|---------------------|
| P+ | .643 | .683 (—) | .515 (—) |
| Custom Diffusion | .673 | .623 (-.025) | .408 (-.096) |
| DB-LoRA (FedAvg) | .682 | .531 (-.213) | .098 (-.585) |
| MoS (Grad Fusion) | .631 | .729 (-.016) | .717 (-.011) |
| Orthogonal Adapt. | .644 | .741 (-.007) | .745 (+.005) |
| **LoRACLR** | .665 | **.776 (+.010)** | **.828 (+.029)** |

用户研究（50 人，Identity Alignment 1-5分）：

| 方法 | 平均得分 |
|------|---------|
| **LoRACLR** | **3.42** |
| Orthogonal Adapt. | 2.41 |
| Mix-of-Show | 2.21 |
| Prompt+ | 2.01 |

### 消融实验

| 配置 | 关键观察 |
|------|---------|
| margin = 0.25-0.5 | 最优范围，身份保留和视觉一致性最好 |
| margin > 0.5 | 性能下降，过度分离导致概念退化 |
| λ_delta = 0.001 | 最优，平衡合并效果和稀疏性 |
| λ_delta > 0.01 | 限制过强，合并不充分 |
| 概念数 2→12 | 所有指标保持稳定，可扩展性好 |

### 关键发现

- LoRACLR 是唯一在合并后 Image 和 Identity Alignment 都提升（而非下降）的方法：Image +.010，Identity +.029。其他方法合并后普遍退化
- DB-LoRA (FedAvg) 合并后 Identity Alignment 从 .683 暴跌至 .098（下降 .585），说明简单平均在概念多时完全失效
- 随着概念数从 2 增加到 12，LoRACLR 的各项指标保持稳定，而其他方法在 6+ 概念时明显退化（如无法保留 Messi 的身份）
- 合并时间仅 5 分钟 vs Orthogonal Adaptation 需 120 分钟，效率优势显著
- 支持风格 LoRA 的集成，可以在保持概念身份的同时改变艺术风格

## 亮点与洞察

- **对比学习用于权重空间对齐**：将对比学习从特征空间迁移到模型权重的合并过程中，正负样本对的定义优雅自然——同概念吸引、跨概念排斥。这个思路可以迁移到其他模型合并场景（如多任务 LoRA 合并、联邦学习中的模型聚合）。
- **Post-training 一次性合并**：LoRACLR 是一个 post-training 方法，合并后的模型可以用任意 prompt 生成任意组合的图像，无需逐次优化。这使得它在实际生产环境中极具价值——用户可以自由组合社区 LoRA 而无需重新训练。
- **概念数增加时的鲁棒性**：从 2 到 12 个概念指标几乎不降，这在多概念方法中非常罕见。说明对比损失有效地防止了"概念拥挤"导致的相互干扰。

## 局限与展望

- 方法性能受限于底层 LoRA 模型的质量——如果原始单概念 LoRA 效果不好，合并也不会改善
- 当前仅在 Stable Diffusion 1.5 上验证，在 SDXL 或 Flux 等更新模型上的效果未知
- 论文未讨论概念之间存在细粒度相似性时的表现（如两个外观相近的人）
- 可能存在被恶意使用生成 deepfake 的风险
- 对比损失中的 margin 参数在不同概念组合间可能需要自适应调整

## 相关工作与启发

- **vs Mix-of-Show**: MoS 需要特殊的 ED-LoRA 格式和原始训练数据，限制了与社区模型的兼容性。LoRACLR 支持任意标准 LoRA，合并后 Identity Alignment 更高（.828 vs .717）
- **vs Orthogonal Adaptation**: OA 需要在原始数据上重新微调每个 LoRA（~120分钟），且在高概念数时出现特征转移。LoRACLR 仅需 5 分钟且无需原始数据
- **vs ZipLoRA**: ZipLoRA 只支持 style+content 的二元合并，不支持多个内容 LoRA 的合并。LoRACLR 无此限制
- **对比学习 + 模型合并的新范式**: 本文提出了一种新的模型合并范式——与其手工设计合并规则（加权平均、梯度融合），不如用学习方法（对比损失）自动找到最优合并点

## 评分

- 新颖性: ⭐⭐⭐⭐ 对比学习用于 LoRA 合并有新意，delta-based 策略简洁有效
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+用户研究+消融，但缺少在更新模型上的验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图例丰富
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，直接解决社区用户的真实需求，5分钟合并12个概念

<!-- RELATED:START -->

## 相关论文

- [Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [DreamRelation: Bridging Customization and Relation Generation](dreamrelation_bridging_customization_and_relation_generation.md)
- [Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)
- [IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)

<!-- RELATED:END -->
