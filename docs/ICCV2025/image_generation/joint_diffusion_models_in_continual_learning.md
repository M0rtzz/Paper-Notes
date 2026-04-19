---
title: >-
  [论文解读] Joint Diffusion Models in Continual Learning
description: >-
  [图像生成] > 提出 JDCL，将分类器与扩散生成模型统一为一个联合参数化的网络，结合知识蒸馏和两阶段训练策略，在生成重放式持续学习中大幅缓解灾难性遗忘，超越现有生成重放方法。
tags:
  - 图像生成
---

# Joint Diffusion Models in Continual Learning

| 信息 | 内容 |
|------|------|
| 会议 | ICCV 2025 |
| arXiv | [2411.08224](https://arxiv.org/abs/2411.08224) |
| 代码 | [GitHub](https://github.com/pskiers/Joint-Diffusion-in-Latent-Space) |
| 领域 | 持续学习 · 扩散模型 · 生成重放 |
| 关键词 | continual learning, generative replay, joint diffusion, knowledge distillation, catastrophic forgetting |

## 一句话总结

> 提出 JDCL，将分类器与扩散生成模型统一为一个联合参数化的网络，结合知识蒸馏和两阶段训练策略，在生成重放式持续学习中大幅缓解灾难性遗忘，超越现有生成重放方法。

## 研究背景与动机

### 灾难性遗忘与生成重放

神经网络在面对新任务数据时，往往会急剧丧失对旧任务的性能（灾难性遗忘）。生成重放(Generative Replay)方法使用生成模型合成过往数据来缓解这一问题。然而传统方法存在根本性缺陷：

**生成模型与分类器的解耦问题**：标准流程中生成模型和分类器是独立的。分类器高度依赖生成样本的质量，两者之间存在知识传递的瓶颈。

**生成样本质量退化**：即使 SOTA 扩散模型也无法精确建模数据分布，反复用生成数据训练分类器会导致分类器性能持续下降。

**可塑性-稳定性权衡**：现有方法要么过于稳定（无法学新任务），要么过于可塑（快速遗忘旧任务）。

### 核心观察

作者做了一个关键实验（Fig. 1）：先在 CIFAR10 上训练分类器和联合扩散模型至收敛，然后仅用扩散模型生成的数据继续训练。结果发现：
- 独立分类器的性能**急剧下降**
- 联合扩散模型的下降**显著更小**
- 加入知识蒸馏后退化**进一步减少**

这表明联合建模+知识蒸馏是生成重放持续学习中保持知识的关键。

## 方法详解

### 整体框架

JDCL 包含三个核心组件：**联合扩散建模**、**两阶段局部-全局训练**和**知识蒸馏**。

### 1. 联合扩散模型

将 UNet 去噪网络与分类器统一在同一参数化中。UNet 编码器 $e_\nu$ 从不同层提取特征集 $\mathcal{Z}_t = \{z_t^1, z_t^2, \ldots, z_t^n\}$，通过平均池化聚合为向量 $z_t = f(\mathcal{Z}_t)$，再送入分类器 $g_\omega$ 预测类别。

联合概率建模为：

$$p_{\nu,\psi,\omega}(x_{0:T}, y) = p_{\nu,\omega}(y|x_0) \cdot p_{\nu,\psi}(x_{0:T})$$

联合训练损失：

$$L_{JD}(\nu,\psi,\omega) = \alpha \cdot L_{\text{class}}(\nu,\omega) - \sum_{t=2}^{T} L_{t,\text{diff}}(\nu,\psi) - L_0 - L_T$$

其中扩散损失为简化的 DDPM 目标 $L_{t,\text{diff}} = \mathbb{E}[\|\epsilon - \hat{\epsilon}\|^2]$，分类损失为标准交叉熵。

### 2. 两阶段局部-全局训练

为平衡可塑性和稳定性，采用两阶段方案：

- **局部阶段**：复制当前全局模型，仅在新任务数据 $D_\tau$ 上训练（保证完美可塑性）
- **全局阶段**：用全局模型生成旧任务数据 $S_{1\ldots\tau-1}$，用局部模型生成当前任务数据 $S_\tau$，合并后微调全局模型

### 3. 知识蒸馏

同时对扩散和分类部分施加 KD：

**扩散蒸馏损失**：

$$L_{t,\text{diff}}^{KD}(\nu,\psi) = \mathbb{E}[\|\epsilon_f - \hat{\epsilon}\|^2]$$

**分类蒸馏损失**：

$$L_{\text{class}}^{KD}(\nu,\omega) = -\mathbb{E}\left[\sum_{k} \log \frac{\exp(\varphi_k)}{\sum_c \exp(\varphi_c)} \varphi_k^f\right]$$

最终持续学习目标：

$$L_{CL} = \mathbb{E}_{S_{1\ldots t-1}}[L_{JDKD}(\cdot; p^f)] + \mathbb{E}_{S_t}[L_{JDKD}(\cdot; p_n)] + \beta \cdot \mathbb{E}_{S_{1\ldots t}}[L_{JD}]$$

### 4. 半监督扩展

利用联合建模的灵活性，将未标注数据仅用于生成部分训练，并结合伪标签和一致性正则化：使用弱增广生成伪标签，强增广计算半监督损失。

## 实验

### 主实验：全监督持续学习

| 方法 | CIFAR-10 (T=5) | CIFAR-100 (T=5) | CIFAR-100 (T=10) | ImageNet100 (T=5) |
|------|:-:|:-:|:-:|:-:|
| Continual Joint (上界) | 86.41 | 73.07 | 64.15 | 50.59 |
| GUIDE (前SOTA) | 64.47 | 41.66 | 26.13 | 39.07 |
| DGR diffusion | 59.00 | 28.25 | 15.90 | 23.92 |
| GFR | 26.70 | 34.80 | 21.90 | 32.95 |
| **JDCL (本文)** | **83.69** | **47.95** | **29.04** | **54.53** |

关键发现：JDCL 在 CIFAR-10 上超越前 SOTA 19+点（约30%提升），在 ImageNet100 上超越15+点（约40%提升），接近无限缓冲区上界。

### 半监督持续学习

| 方法 | CIFAR-10 0.8% | CIFAR-10 5% | CIFAR-100 0.8% | CIFAR-100 5% |
|------|:-:|:-:|:-:|:-:|
| NNCSL (5120) | 73.7 | 79.3 | 27.5 | 46.0 |
| **JDCL (无缓冲区)** | **78.93** | **79.96** | 22.19 | 26.39 |

JDCL 在 CIFAR-10 上**无需记忆缓冲区**即超越使用 5120 样本缓冲区的 NNCSL。

### 消融实验

| 联合建模 | 知识蒸馏 | 两阶段训练 | 准确率 |
|:-:|:-:|:-:|:-:|
| ✓ | ✓ | ✓ | 83.7 |
| ✗ | ✓ | ✓ | 68.4 |
| ✓ | ✗ | ✓ | 48.2 |

联合建模和知识蒸馏缺一不可，尤其知识蒸馏对性能贡献极大（移除后降35.5点）。

## 亮点与洞察

1. **核心创新**：将生成和判别模型统一参数化，从根本上消除了知识传递瓶颈
2. **自监督重放**：联合模型可以用自己的生成部分进行重放，生成部分和判别部分共享编码器，天然避免分布偏移
3. **灵活性**：局部训练阶段与全局训练阶段完全解耦，使得半监督扩展非常自然
4. **计算效率**：相比训练独立的生成模型+分类器，联合训练减少了总计算开销

## 局限性

- 在 CIFAR-100 半监督设置下表现不够理想，可能因为标注信号太弱导致联合训练中分类目标与生成目标不平衡
- 实验仅在小规模数据集（CIFAR、ImageNet100）上验证，缺少大规模场景评估
- 模型基于 UNet 架构，未探讨与 Transformer 架构的结合

## 相关工作

- **生成重放方法**：DGR、RTF、DDGR、GUIDE 等使用 GAN/VAE/扩散模型进行重放
- **正则化方法**：EWC、LwF 等限制关键参数更新
- **H-space 利用**：利用 UNet 中间特征进行分割、分类等下游任务
- **半监督持续学习**：ORDisCo、CCIC、NNCSL 等

## 评分

| 维度 | 分数 |
|------|:----:|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)

<!-- RELATED:END -->
