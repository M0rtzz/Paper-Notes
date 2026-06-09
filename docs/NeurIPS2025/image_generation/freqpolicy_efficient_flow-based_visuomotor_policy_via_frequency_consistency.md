---
title: >-
  [论文解读] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency
description: >-
  [NEURIPS2025][图像生成][flow matching] 首次在 flow-based 视觉运动策略中引入频域一致性约束，利用 DCT 变换将动作块的速度场投影到频域并施加自适应频率分量损失，实现了高质量一步动作生成（93.5 Hz），在仿真和真实机器人任务中均优于现有一步生成方法。
tags:
  - "NEURIPS2025"
  - "图像生成"
  - "flow matching"
  - "visuomotor policy"
  - "one-step generation"
  - "frequency consistency"
  - "robotic manipulation"
---

# FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency

**会议**: NEURIPS2025  
**arXiv**: [2506.08822](https://arxiv.org/abs/2506.08822)  
**代码**: 未公开  
**领域**: 图像生成  
**关键词**: flow matching, visuomotor policy, one-step generation, frequency consistency, robotic manipulation

## 一句话总结

首次在 flow-based 视觉运动策略中引入频域一致性约束，利用 DCT 变换将动作块的速度场投影到频域并施加自适应频率分量损失，实现了高质量一步动作生成（93.5 Hz），在仿真和真实机器人任务中均优于现有一步生成方法。

## 背景与动机

- 基于生成模型（diffusion / flow matching）的视觉运动策略在机器人操作中取得了显著进展，但其**多步迭代采样**带来高推理延迟，限制了实时部署。
- 现有加速方法（Consistency Policy、OneDP、SDM 等）大多直接借鉴图像生成领域的加速技术。然而，图像生成产生的是**独立样本**，而机器人操作需要生成具有**时序连贯性**的动作轨迹——这一关键区别被忽视了。
- 时序信号处理和语音处理领域的大量研究表明，频域特征在建模非平稳和振荡模式方面比时域特征更有效。机器人高频采样的动作块本质上是时序信号，频域表示能更精细地区分平滑轨迹中的细微变化。
- 因此，作者提出从**频域视角**对 flow matching 施加一致性约束，首次利用动作块的时序特性来加速一步动作生成。

## 核心问题

如何在 flow matching 的框架下，利用动作块的时序结构信息，在**不需要预训练教师模型**的前提下实现高质量的一步动作生成？

## 方法详解

### 1. 基本 Flow Matching 目标

给定初始噪声 $a_0 \sim \mathcal{N}(0, I)$ 和专家动作 $a_1$，学习速度场 $v_\theta(t, a_t)$ 来映射噪声到动作：

$$\mathcal{L}_{\text{fm}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \|v_\theta(t, a_t) - (a_1 - a_0)\|_2$$

其中 $a_t = (1-t) \cdot a_0 + t \cdot a_1$ 为线性插值。这个基本目标虽然能训练出可用的策略模型，但仍需多步采样才能生成高质量动作。

### 2. 频率一致性约束目标（核心贡献）

**关键思路**：将多维动作块视为**时序信号**（而非静态向量），在频域中对不同时间步的速度向量施加一致性约束，促进直线流和一步生成。

具体步骤：

**(a)** 在两个随机时间步 $s, r \in [0,1]$ 构造插值状态 $a_r, a_s$

**(b)** 使用 type-II 离散余弦变换（DCT）将速度向量投影到频域：

$$F(v_t)_k = \sum_{n=0}^{H-1} v_t(n) \cdot \cos\left[\frac{\pi}{N}\left(n+\frac{1}{2}\right)k\right]$$

**(c)** 频率一致性损失包含两部分：
- **速度一致性**：直接约束不同时间步的速度在频域中一致
- **轨迹一致性**：从不同起点出发，经过各自速度场传播后应收敛到同一目标点

$$\mathcal{L}_{\text{freq}} = \mathbb{E}\left[\text{Sim}(v_\theta(s, a_s), v_\theta(r, a_r))\right] + \mathbb{E}\left[\text{Sim}(a_s + (u-s)v_\theta(s, a_s), a_r + (u-r)v_\theta(r, a_r))\right]$$

### 3. 自适应频率分量损失（第二个贡献）

**动机**：机器人操作序列在低动态阶段（如移动到目标位置）和高动态阶段（如技能切换、接触交互）之间交替，不同阶段的频率分量分布差异很大。

受 Focal Loss 启发，设计自适应加权方案，对差异更大的频率分量施加更大权重：

$$w_k = \frac{\exp(\|F(v_r)_k - F(v_s)_k\|_2)}{\sum_{j=0}^{H-1} \exp(\|F(v_r)_j - F(v_s)_j\|_2)}$$

最终的相似度函数为加权求和：

$$\text{Sim}(v_r, v_s) = \sum_{k=0}^{H-1} w_k \cdot \|F(v_r)_k - F(v_s)_k\|_2$$

### 4. 总训练目标

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fm}} + \mathcal{L}_{\text{freq}}$$

两个损失分别负责 (1) 学习准确的噪声到动作映射，(2) 强制时序一致的频域流，共同实现可靠的一步动作生成。

### 5. 模型架构

- 使用标准 1D CNN-based U-Net 作为主干（与先前工作保持公平对比）
- 2D 输入使用 ResNet-18 编码，3D 输入使用轻量 MLP 编码点云
- 可作为 VLA 模型的策略头（已集成到 OpenVLA）

## 实验关键数据

### Robomimic（2D 输入，5个任务）

| 方法 | NFE | Transport | Tool Hang |
|------|-----|-----------|-----------|
| DDPM | 15 | 80% | 52% |
| Consistency Policy | 1 | 78% | 70% |
| Consistency-FM | 1 | 84% | 80% |
| IMLE Policy | 1 | 90% | 81% |
| **FreqPolicy** | **1** | **90%** | **85%** |

- 在所有一步方法中取得最优，Transport 比 Consistency-FM 高 6%，Tool Hang 高 5%

### MetaWorld + Adroit（3D 输入，53个任务）

| 方法 | NFE | 平均成功率 |
|------|-----|-----------|
| DP3（10步） | 10 | 76.1% |
| SDM（需教师） | 1 | 74.8% |
| FlowPolicy | 1 | 77.2% |
| **FreqPolicy** | **1** | **78.5%** |

- 无需预训练教师模型，MetaWorld 超过 SDM 3.7 个百分点

### VLA 集成（LIBERO，40个任务）

| 方法 | NFE | 平均成功率 | 推理速度 |
|------|-----|-----------|---------|
| OpenVLA-DP | 50 | 68.1% | 0.32 Hz |
| OpenVLA-FM | 10 | 93.7% | 1.26 Hz |
| OpenVLA-FM | 1 | 93.5% | 5.92 Hz |
| **OpenVLA-FreqPolicy** | **1** | **94.8%** | **6.05 Hz** |

- 1步 FreqPolicy 超越 10步 FlowMatching（94.8% vs 93.7%），速度快 5 倍

### 真实机器人（3个长horizon任务）
- FreqPolicy 在所有任务上取得最高或并列最高成功率，推理频率 **93.5 Hz**
- Task 1 水果分拣：70% 成功率 @ 93.5 Hz（vs Diffusion Policy 55% @ 19.8 Hz）

### 消融实验

| 配置 | Transport | Tool Hang |
|------|-----------|-----------|
| Vanilla FM (1步) | 84% | 76% |
| + 空间一致性约束 | 90% | 78% |
| + 频率一致性（全频段） | 92% | 82% |
| + **自适应频率损失** | **92%** | **88%** |

- 从 vanilla 到完整 FreqPolicy，Tool Hang 提升 12 个百分点

## 亮点

1. **视角新颖**：首次将动作块视为时序信号并在频域施加一致性约束，跳出了图像生成加速技术直接套用到机器人领域的思维定式
2. **不依赖教师模型**：与 SDM / OneDP 等需要预训练 diffusion 教师的方法不同，FreqPolicy 训练流程更简洁
3. **自适应加权设计合理**：受 Focal Loss 启发的频率分量自适应权重，能根据操作阶段动态调整关注重点
4. **实验覆盖广泛**：93 个仿真任务 + 3 个真实任务，涵盖 2D/3D 输入，还验证了作为 VLA 头的泛化能力
5. **推理速度极快**：93.5 Hz 的真实机器人推理频率，远超多步方法

## 局限与展望

1. **仅验证了 flow matching**：核心思想应可扩展到 diffusion-based 策略，但论文未做验证
2. **计算开销翻倍**：频率一致性损失需要对两个随机样本做前向传播，训练成本约为基本 FM 的两倍
3. **DCT 变换的选择缺乏 ablation**：为什么选 type-II DCT 而非 FFT 或小波变换？论文未充分讨论
4. **真实机器人实验规模有限**：仅 3 个任务、单一操作场景，泛化性验证不够充分
5. **对动作块长度 H 的敏感度未讨论**：频域表示的分辨率直接取决于 H，不同 chunking size 对性能的影响未分析

## 与相关工作的对比

| 方法 | 策略类型 | 需要教师 | NFE | 一致性域 |
|------|---------|---------|-----|---------|
| Consistency Policy | Diffusion | ✅ | 1-3 | 去噪轨迹 |
| SDM / OneDP | Diffusion | ✅ | 1 | 分布匹配 |
| FlowPolicy | Flow Matching | ❌ | 1 | 空间域速度 |
| **FreqPolicy** | **Flow Matching** | **❌** | **1** | **频域速度** |

- FreqPolicy 与 FlowPolicy 最相近，关键区别在于从空间域约束转向频域约束，利用了动作块的时序结构信息
- 相比需要教师的 SDM / OneDP，FreqPolicy 训练流程更简洁，性能更好

## 启发与关联

- 频域一致性的思路可能迁移到其他时序生成任务（运动规划、轨迹预测）
- 自适应频率加权的设计可以借鉴到时间序列预测中不同频段的加权学习
- 作为 VLA 策略头的集成方式（与 OpenVLA 结合）展示了与大模型协同的可能性，值得关注后续是否有与 π₀ 等更强 VLA 的集成

## 评分
- 新颖性: ⭐⭐⭐⭐ （频域一致性约束视角新颖，但核心想法相对简单）
- 实验充分度: ⭐⭐⭐⭐ （93个仿真 + 3个真实任务 + VLA集成，消融完整）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机阐述合理）
- 价值: ⭐⭐⭐⭐ （对机器人实时部署有实际意义，推理速度提升显著）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](../../AAAI2026/image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](../../ICLR2026/image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](../../ICLR2026/image_generation/cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[NeurIPS 2025\] Riemannian Consistency Model](riemannian_consistency_model.md)

</div>

<!-- RELATED:END -->
