---
title: >-
  [论文解读] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization
description: >-
  [CVPR 2026][多模态][后训练量化] 本文提出 Quant Experts (QE)，一种基于混合专家（MoE）的 token 感知自适应量化误差补偿框架，通过将重要通道分为 token 无关和 token 依赖两组，分别用共享专家和路由专家进行全局和局部量化误差重建，在 2B-72B 规模的 VLM 上实现了显著的量化精度恢复。
tags:
  - CVPR 2026
  - 多模态
  - 后训练量化
  - 混合专家
  - 视觉语言模型
  - 通道重要性
  - 低秩重建
---

# Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization

**会议**: CVPR 2026  
**arXiv**: [2602.24059](https://arxiv.org/abs/2602.24059)  
**代码**: 无  
**领域**: 多模态VLM / 模型压缩  
**关键词**: 后训练量化、混合专家、视觉语言模型、通道重要性、低秩重建

## 一句话总结

本文提出 Quant Experts (QE)，一种基于混合专家（MoE）的 token 感知自适应量化误差补偿框架，通过将重要通道分为 token 无关和 token 依赖两组，分别用共享专家和路由专家进行全局和局部量化误差重建，在 2B-72B 规模的 VLM 上实现了显著的量化精度恢复。

## 研究背景与动机

1. **领域现状**：后训练量化（PTQ）是压缩大规模视觉语言模型的主流手段，通过将权重和激活映射到低比特表示来降低计算和内存开销。现有方法如 SmoothQuant、AWQ、GPTQ 等主要依赖静态地识别敏感/异常通道并全局补偿量化误差。

2. **现有痛点**：这些方法假设通道重要性是固定的，用校准数据估计一组固定的通道缩放系数或低秩重建矩阵，对所有输入 token 统一处理。然而在多模态场景下，不同模态、甚至同一模态内不同 token 的通道重要性分布差异很大。

3. **核心矛盾**：作者通过实验观察到两个关键现象：（1）重要通道的位置在不同模态和 token 之间显著变化——即使在同一模态内，不同 token 的语义和上下文差异也会导致激活分布变化，使重要通道动态迁移；（2）重要通道的出现频率呈现高度不均匀分布——只有一小部分通道在大多数 token 中稳定出现（token 无关），而大部分重要通道仅对特定 token 激活（token 依赖）。

4. **本文目标**：如何在量化过程中同时补偿全局稳定出现的通道误差和动态变化的局部通道误差。

5. **切入角度**：将重要通道按出现频率分为两组，用不同类型的专家分别处理——共享专家处理全局误差，路由专家处理 token 依赖的局部误差。

6. **核心 idea**：将 MoE 思想引入量化误差补偿，通过共享专家+路由专家的组合实现 token 级自适应的量化误差重建。

## 方法详解

### 整体框架

给定校准数据，QE 首先估计每层重要通道的出现频率分布，将其二分为 token 无关通道（高频）和 token 依赖通道（低频）。然后使用两类专家来分别处理：共享专家（SE）用一个低秩适配器重建 token 无关通道的全局量化误差；路由专家（REs）通过对 token 依赖通道进行共现聚类，为每个聚类簇训练一个路由低秩适配器。推理时，共享专家固定补偿全局误差，路由器根据输入 token 动态选择最优路由专家进行局部误差补偿。

### 关键设计

1. **通道依赖性划分（Channel Dependence Partitioning）**

    - 功能：将重要通道分为全局稳定组和局部动态组
    - 核心思路：对校准数据中的每个 token $x_t$，通过 $\mathcal{C}_t = \text{Top-}k(|x_t| \odot \mathbf{w})$ 识别 top-k 重要通道（其中 $\mathbf{w}$ 为权重矩阵行均值），统计每个通道的出现频率 $f_c$，按频率降序排列，前 $k$ 个为 token 无关通道 $\mathcal{C}_s$，接下来的 $N_r \cdot k$ 个为 token 依赖通道 $\mathcal{C}_r$
    - 设计动机：直接利用统计频率来区分全局和局部重要通道，避免了对所有 token 做逐一处理的计算开销

2. **共享专家（Shared Expert）**

    - 功能：补偿由 token 无关重要通道引起的全局量化误差
    - 核心思路：将 token 无关通道从直接量化中豁免，使用白化 SVD 分解出低秩适配器 $(\mathbf{L}_{SA}, \mathbf{L}_{SB})$ 来重建这些通道的权重。同时采用通道缩放来压制激活中的异常值，降低激活量化误差。共享专家对所有 token 统一应用
    - 设计动机：token 无关通道在大多数 token 中都是重要的，使用统一的全局补偿即可有效恢复这部分误差，这与 ASER 等先前工作的思路一致

3. **路由专家（Routed Experts）+ 共现聚类**

    - 功能：自适应补偿 token 依赖通道引起的局部量化误差
    - 核心思路：首先构建 token 依赖通道的共现矩阵 $\mathcal{O}$，通过归一化逐点互信息（NPMI）转化为相似性矩阵，再用谱聚类将通道分为 $N_r$ 个簇。每个簇对应一个路由专家，使用加权 SVD 构建低秩适配器。推理时，轻量级路由器 $\mathbf{R}$ 根据输入 token 的绝对值估计各专家的残余误差，选择误差最小的专家激活
    - 设计动机：共现频繁的通道往往同时出现在同类 token 中，聚类后每个专家只需专注于一簇相关通道的误差补偿，比逐 token 定制高效得多。路由器使用残余误差作为选择依据，确保选出最适合当前 token 的专家

### 损失函数 / 训练策略

QE 的共享专家和路由专家均通过 SVD 直接构造（无需梯度训练），但提供了一个可选的轻量级微调策略：仅微调路由专家和路由器参数，其他参数冻结，逐层优化而非端到端训练，进一步提升量化精度。

## 实验关键数据

### 主实验

在 Qwen2VL-2B 上的 W4A6 量化结果（11个多模态基准平均准确率）：

| 方法 | 平均准确率 | vs RTN | vs LQER | vs MBQ |
|------|-----------|--------|---------|--------|
| FP16 (全精度) | 62.97 | - | - | - |
| RTN | 53.62 | - | - | - |
| LQER | 55.92 | +2.30 | - | - |
| MBQ | 54.73 | +1.11 | - | - |
| **QE** | **58.74** | **+5.12** | **+2.82** | **+4.01** |

在 Qwen2VL-72B 上的 W4A6 量化结果：

| 方法 | MMMU | OCRBench | ScienceQA | TextVQA | VizWiz |
|------|------|----------|-----------|---------|--------|
| FP16 | 61.44 | 78.70 | 91.22 | 82.26 | 76.27 |
| MBQ | 52.67 | 69.70 | 86.32 | 76.08 | 67.99 |
| **QE** | **58.11** | **76.60** | **90.33** | **79.27** | **73.91** |

### 消融实验

Qwen2VL-2B W4A6 设置下的组件消融：

| 配置 | MMMU | ScienceQA |
|------|------|-----------|
| 仅路由专家 (REs) | 34.56 | 68.72 |
| 仅共享专家 (SE) | 35.22 | 69.61 |
| SE + 随机路由 | 35.89 | 70.00 |
| SE + 随机聚类 | 35.33 | 69.71 |
| **QE (SE+REs)** | **36.89** | **70.85** |

### 关键发现

- 共享专家和路由专家都不可或缺：去掉任一组件性能都明显下降，二者互补
- 共现聚类比随机聚类高约 1%，表明基于 NPMI 的谱聚类确实捕捉了通道间的有意义关联
- 路由器选择比随机路由高约 0.8%，验证了自适应选择的必要性
- 路由专家数量从 2 增加到 16 时性能持续提升（平均准确率从 67.08 到 68.06），但收益递减
- 模型规模越大，QE 的改进越显著：72B 模型上 W4A6 提升 5.09%，几乎恢复到全精度水平

## 亮点与洞察

- **将 MoE 引入量化误差补偿是很新颖的思路**：之前的工作要么全局补偿要么按模态区分，QE 首次在 token 粒度实现自适应补偿，这个框架可以灵活扩展
- **共现聚类的设计很巧妙**：利用 NPMI + 谱聚类发现 token 依赖通道的共现模式，将无限的 token 空间映射到有限的专家数量上，是对"每个 token 定制一个补偿策略"这一不可行方案的优雅近似
- **开销极低**：低秩适配器的额外 FLOPs 和参数量相对于原始线性层可以忽略不计，且仅在校准阶段构造，不需要端到端训练

## 局限与展望

- 路由器设计比较简单（仅用绝对均值估计误差），可以考虑更复杂的路由策略如门控网络
- 仅在 Qwen2VL 和 InternVL2 系列上验证，缺少对更多架构（如 LLaVA、Phi-Vision）的评估
- token 无关和 token 依赖的二分策略较为硬性，可以探索连续频率谱上的软分组
- 对极端低比特（如 W2）场景的效果未知

## 相关工作与启发

- **vs LQER/ASER**: 它们用单一全局低秩适配器补偿全部量化误差，QE 将其拆分为全局+局部两组，精度更高
- **vs MBQ**: MBQ 基于模态感知（图像/文本分别处理），QE 进一步细化到 token 级别，在 MBQ 的基础上提升 4%+
- **vs SmoothQuant/AWQ**: 这些方法用固定缩放系数平衡激活和权重分布，QE 提供了更灵活的自适应补偿机制

## 评分

- 新颖性: ⭐⭐⭐⭐ MoE 用于量化误差补偿的想法新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 2B-72B 多模型多设置，消融全面
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，观察→方法逻辑链完整
- 价值: ⭐⭐⭐⭐ 实用性强，适用于大规模 VLM 部署

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_moe_dynamic_expert_skipping.md)

<!-- RELATED:END -->
