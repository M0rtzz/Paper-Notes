---
title: >-
  [论文解读] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models
description: >-
  [ECCV 2024][图像生成] 提出 M2D2M，基于离散扩散模型生成多段连续人体运动序列，通过动态转移概率和两阶段采样策略（TPS）实现动作间平滑过渡，且无需额外的多运动训练数据。 现有文本到运动方法主要关注单动作序列生成，但实际应用（叙事、游戏、模拟训练）需要生成包含一系列连续动作的多运动序列…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2407.14502](https://arxiv.org/abs/2407.14502)  
**领域**: 图像生成

## 一句话总结

提出 M2D2M，基于离散扩散模型生成多段连续人体运动序列，通过动态转移概率和两阶段采样策略（TPS）实现动作间平滑过渡，且无需额外的多运动训练数据。

## 研究背景与动机

现有文本到运动方法主要关注单动作序列生成，但实际应用（叙事、游戏、模拟训练）需要生成包含一系列连续动作的多运动序列。现有多运动方法（如 PriorMDM 的 Handshake 算法、TEACH 的 SLERP 插值）先独立生成各动作再后处理连接，常导致：
- 动作边界处过渡突兀
- 个别动作的保真度下降
- 需要额外的过渡长度超参数

本文基于离散扩散模型提出统一的生成方案，**用单运动训练的模型直接生成多运动序列**，无需额外训练或后处理。

## 方法详解

### 整体框架

M2D2M 由三个模块组成：
1. **Motion VQ-VAE**：将运动序列编码为离散 token
2. **去噪 Transformer**：在离散扩散框架下学习条件去噪
3. **两阶段采样（TPS）**：联合采样建立粗略轮廓 → 独立采样精调各段运动

### 关键设计

**动态转移概率**：改进 VQ-Diffusion 中的均匀转移概率，根据 codebook token 间的距离动态调整：

$$\beta(t, d) = (1 - \gamma_t - \alpha_t) \cdot \text{softmax}_d\left(\eta \cdot \frac{t}{T} \cdot \frac{d}{K}\right)$$

在扩散早期（t 大）优先探索远距离 token 以促进多样性，后期逐渐收敛为均匀分布以精确收敛。这种探索-利用的策略对多运动边界处的模式混合至关重要。

**两阶段采样（TPS）**：
- **联合采样阶段**（步骤 T → Ts+1）：将所有动作的 mask token 合并，用去噪 Transformer 联合去噪，通过自注意力机制让不同动作的 token 相互影响，确保过渡平滑
- **独立采样阶段**（步骤 Ts → 1）：各动作独立去噪，与对应文本描述对齐，保持个体保真度

关键优势：无需多运动训练数据，用单运动训练的模型即可生成。

**新评估指标 Jerk**：衡量多运动序列在动作边界处的平滑度：

$$Jerk = \sum_p \ln \frac{1}{v_{p,\text{peak}}^2} \int_{t_1}^{t_2} \left\| \frac{d}{dt} \mathbf{a}_p(t) \right\|_2^2 dt$$

首次将 Jerk 引入多运动生成评估。

### 损失函数

离散扩散标准目标：变分下界 + 去噪交叉熵损失：

$$\mathcal{L} = \mathcal{L}_{\text{vlb}} + \lambda \mathbb{E}_{z_t \sim q(z_t|z_0)} [-\log p_\theta(z_0 | z_t, y)]$$

采用 CLIP 文本编码器、相对位置编码和 classifier-free guidance（10% 无条件丢弃率）。

## 实验关键数据

### 主实验

HumanML3D 多运动生成（N=4 个动作）：

| 方法 | R-Top3 ↑ | FID ↓ | MMdist ↓ | Jerk → |
|------|:-:|:-:|:-:|:-:|
| GT (Single) | 0.791 | 0.002 | 2.707 | 1.192 |
| GT (Concat) | — | — | — | 1.371 |
| PriorMDM | 0.586 | 0.832 | 5.901 | 0.476 |
| T2M-GPT | 0.719 | 0.342 | 3.512 | 1.321 |
| **M2D2M** | **0.733** | **0.253** | **3.165** | **1.238** |

M2D2M 在所有个体运动指标上显著领先，且 Jerk 值接近真实单运动（1.238 vs 1.192），远优于简单拼接（1.371）。PriorMDM 的 Jerk 仅 0.476，说明过度平滑导致运动缺乏真实感。

HumanML3D 单运动生成对比（与 13 种方法比较，部分结果）：

| 方法 | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | MModality ↑ |
|------|:-:|:-:|:-:|:-:|
| MotionGPT | 0.778 | 0.232 | 3.096 | 2.008 |
| ReMoDiffuse | **0.795** | 0.103 | **2.974** | 1.795 |
| **M2D2M** | 0.788 | **0.057** | 3.040 | 2.473 |

### 消融实验

KIT-ML 多运动生成（N=4）：

| 方法 | R-Top3 ↑ | FID ↓ | Jerk → |
|------|:-:|:-:|:-:|
| PriorMDM | 0.292 | 3.311 | 0.594 |
| T2M-GPT | 0.667 | 0.907 | 1.388 |
| **M2D2M** | **0.711** | **0.817** | **1.351** |

TPS 与动态转移概率的联合效果（消融实验表明两者协同工作效果最佳，单独使用效果有限）。

### 关键发现

- TPS 是单阶段多运动生成算法，不需要已完成的独立运动或过渡长度超参数
- 动态转移概率在扩散早期促进模式混合对多运动边界过渡至关重要
- PriorMDM 的 Handshake 算法过度平滑边界（Jerk 过低），丢失了运动的细节特征
- 相对位置编码允许模型外推到训练时未见的长序列

## 亮点与洞察

- **零额外训练成本的多运动生成**：用单运动训练的模型直接生成多运动，解决了多运动标注数据稀缺的问题
- **Jerk 指标的引入**：填补了多运动过渡平滑度评估的空白
- **动态转移概率**的探索-利用策略：扩散早期鼓励远距离 token 混合，为多运动边界处不同动作模式的融合提供了理论支持
- TPS 的联合→独立两阶段设计简洁优雅

## 局限性

- 联合采样阶段的步数 Ts 为超参数，需要手动调节
- 基于 VQ-VAE 的离散表示可能引入量化误差
- 未与 FineMoGen 等同期工作做充分对比
- 多运动过渡区域的 Diversity 评估依赖随机组合测试集，可能引入偏差

## 评分

⭐⭐⭐⭐ 离散扩散视角下的多运动生成新方案，动态转移概率和 TPS 设计新颖，Jerk 指标有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
