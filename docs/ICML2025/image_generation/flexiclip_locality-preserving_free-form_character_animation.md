---
title: >-
  [论文解读] FlexiClip: Locality-Preserving Free-Form Character Animation
description: >-
  [图像生成] FlexiClip 提出了一种基于时域Jacobian校正、概率流ODE连续时间建模和GFlowNet流匹配损失的剪贴画动画框架，在保持视觉一致性的同时显著提升了动画的时间平滑性和几何完整性。
tags:
  - 图像生成
---

# FlexiClip: Locality-Preserving Free-Form Character Animation

| 信息 | 内容 |
|------|------|
| 会议 | ICML 2025 |
| arXiv | [2501.08676](https://arxiv.org/abs/2501.08676) |
| 代码 | [项目主页](https://creative-gen.github.io/flexiclip.github.io/) |
| 领域 | 图像生成/动画 |
| 关键词 | clipart动画, 时间一致性, 贝塞尔曲线, 概率流ODE, Flow Matching, GFlowNet, Video SDS |

## 一句话总结

FlexiClip 提出了一种基于时域Jacobian校正、概率流ODE连续时间建模和GFlowNet流匹配损失的剪贴画动画框架，在保持视觉一致性的同时显著提升了动画的时间平滑性和几何完整性。

## 研究背景与动机

将静态剪贴画（clipart）转化为流畅动画是计算机图形学中的经典难题。现有方法面临两大核心挑战：

**时间不一致性**：AniClipart 等方法通过三次贝塞尔曲线建模关键点轨迹，并用 ARAP（As Rigid As Possible）变形保持几何一致性，但在帧间过渡时容易出现突变运动和几何扭曲

**域差距问题**：Text-to-Video (T2V) 和 Image-to-Video (I2V) 模型在处理剪贴画时效果不佳，因为自然视频与剪贴画在统计特性上存在显著差异

具体来说，AniClipart 独立预测每帧的运动动力学，缺乏跨帧噪声累积的纠正机制，导致快速姿态转换时出现不自然的运动伪影。Gal23 虽然类似地学习神经位移场，但同样无法解决时间噪声问题。

## 方法详解

### 整体框架

FlexiClip 的核心思路是将动画生成分解为**空间姿态建模**和**时间平滑校正**两个阶段：

1. 使用 UniPose 检测关键点并构建骨架
2. 通过三次贝塞尔曲线定义空间运动轨迹
3. 引入时域Jacobian和概率流ODE处理时间噪声
4. 利用GFlowNet流匹配损失减少时间噪声
5. 通过 Video SDS 损失从预训练视频扩散模型中蒸馏知识

### 空间姿态建模（Spatial Posing）

给定初始网格 $\mathcal{M}_0 = (\mathbf{V}_0, \mathbf{F}_0)$，其中 $\mathbf{V}_0 \in \mathbb{R}^{V \times 2}$ 为顶点位置，$\mathbf{F}_0$ 为三角面。关键点通过指示矩阵 $\mathbf{K}_c$ 定义，其目标位置为 $\mathbf{T}_c = \mathbf{V}_c + \mathbf{D}_c$。

网格变形通过Jacobian场描述，求解优化问题：

$$\mathbf{V}^* = \arg\min_{\mathbf{V}} \|\mathbf{L}\mathbf{V} - \nabla^T \mathcal{A} \mathbf{J}\|^2 + \lambda \|\mathbf{K}_c \mathbf{V} - \mathbf{T}_c\|^2$$

其中 $\mathbf{L}$ 为余切Laplacian算子，$\mathcal{A}$ 为质量矩阵。关键点沿三次贝塞尔曲线演化：

$$p_t(i) = \sum_{j=0}^{3} B_j(u_t) c_j(i)$$

其中 $u_t \in [0,1]$ 为归一化时间，$B_j$ 为Bernstein基函数。

### 时域平滑（Temporal Smoothing）

这是本文最核心的贡献。将空间Jacobian $\mathbf{J}_t^P$ 和时域Jacobian $\mathbf{J}_t^R$（校正项）分解：

$$\mathbf{J}_t = \mathbf{J}_t^P + \mathbf{J}_t^R$$

时域Jacobian通过ODE建模其连续时间演化：

$$\frac{d\mathbf{J}_t^R}{dt} = f_R(\mathbf{J}_0^P, C_W^P, C_{W-1}^R, t; \theta_R)$$

其中 $C_W^P$ 是当前窗口空间Jacobian的注意力编码特征，$C_{W-1}^R$ 是过去窗口时域Jacobian的注意力编码特征。通过积分求解：

$$\mathbf{J}_t^R = \mathbf{J}_0^R + \int_0^t f_R(\mathbf{J}_0^P, C_W^P, C_{W-1}^R, \tau; \theta_R) d\tau$$

初始条件 $\mathbf{J}_0^R = \mathbf{0}$，确保第一帧无校正。这种设计将pfODE中的噪声项 $C(t)$ 映射为 $C_W^P$，将缩放项 $A(t)$ 映射为 $C_{W-1}^R$。

### 损失函数

**Video SDS 损失**：从预训练视频扩散模型蒸馏知识

$$\nabla_\theta \mathcal{L}_{\text{SDS}}(\phi, \mathbf{X}) = \mathbb{E}_{t', \epsilon}\left[w(t')(\epsilon_\phi(\mathbf{z}_{t'}; \mathbf{y}, t') - \epsilon) \frac{\partial \mathbf{X}}{\partial \theta}\right]$$

**流匹配损失**（受GFlowNet详细平衡条件启发）：

$$L_{flow} = \mathbb{E}_{t',t} \|\nabla_\mathbf{X} \log p_{t'}(\mathbf{X}, \mathbf{J}_t) - \nabla_\mathbf{X} \log p_{t'}(\mathbf{X}, \mathbf{J}_t^P)\|^2 + \mathbb{E}_t \|\mathbf{J}_t - \mathbf{J}_t^P\|^2$$

第二项为校正最小化项，鼓励时域Jacobian尽可能小。总损失为：$L_{SDS} + \lambda \cdot L_{flow}$，实验中 $\lambda = 15$。

### 网络结构

- 空间姿态：4层MLP + LeakyReLU，最后一层线性
- 时域Jacobian（pfODE）：3层MLP
- 注意力网络：2个网络，32维key/value，2个注意力头

## 实验结果

### 主实验

| 方法 | CLIP Score ↑ | X-CLIP Score ↑ |
|------|-------------|----------------|
| DynamiCrafter | 0.8031 | 0.1732 |
| Gal23 | 0.8395 | 0.1865 |
| VideoCrafter2 | 0.8410 | 0.1988 |
| AniClipart | 0.9401 | 0.2075 |
| **FlexiClip** | **0.9563** | **0.2102** |

| 方法 | MV ↑ | TC ↓ | GD ↓ | DS ↓ | AE (×10³) ↑ |
|------|------|------|------|------|-------------|
| AniClipart | 20.87 | 8.51 | 50.98 | 18.49 | 75.23 |
| **FlexiClip** | **25.33** | **8.14** | 52.34 | **13.76** | **113.44** |

FlexiClip 在视觉保真度（CLIP 0.9563 vs 0.9401）和文本-视频对齐（X-CLIP 0.2102 vs 0.2075）上均超越 AniClipart。动画指标方面运动活力提升21%，变形平滑度改善26%，动画能量提升51%。

### 消融实验

| 变体 | MV ↑ | TC ↓ | GD ↓ | DS ↓ | AE ↑ |
|------|------|------|------|------|------|
| w/o 时域Jacobian | 23.00 | 8.80 | 51.50 | 14.00 | 105.00 |
| w/o 流匹配损失 | 24.50 | 8.40 | 53.00 | 14.20 | 95.00 |
| **完整模型** | **25.33** | **8.14** | 52.34 | **13.76** | **113.44** |

- 去除时域Jacobian：运动活力下降、时间一致性变差、出现刚性变形
- 去除流匹配损失：几何扭曲增大、动画能量降低、肢体运动不稳定

### 用户研究

55 个剪贴画、30 名参与者、6 种方法对比。FlexiClip 在所有维度领先：身份保持 94.9%、文本对齐 94.5%、平滑度 93.8%，远超 AniClipart（分别为 83.6%、80.7%、76.4%）。

## 亮点

1. **Jacobian分解策略**：将总Jacobian分解为空间和时域校正项，实现精细化的运动控制
2. **pfODE连续时间建模**：相较离散时间方法，更好地处理帧间噪声累积
3. **GFlowNet启发的流匹配损失**：巧妙利用详细平衡条件，使前向过程消除后向过程引入的时间噪声
4. **多功能支持**：支持旋转、多文本条件、多物体交互和分层动画
5. **端到端可微**：整个管线可微分，贝塞尔参数和时域参数可联合优化

## 局限性

1. **GD 指标略高**：由于未使用 ARAP 变形，几何偏差比 AniClipart 略高（52.34 vs 50.98）
2. **计算开销**：V100 上生成24帧动画需约40分钟，占用26GB显存
3. **依赖预训练模型**：Video SDS 损失依赖 ModelScope T2V 模型的质量
4. **超参数敏感**：$\lambda$ 的选择对运动质量和收敛速度有显著影响（过低收敛慢，过高运动不自然）
5. **仅限2D**：当前框架限于2D剪贴画，未扩展到3D动画

## 评分

⭐⭐⭐⭐ (4/5)

创新性强，将pfODE和GFlowNet引入剪贴画动画是有趣的跨领域融合。实验全面，包含定量评估、消融和用户研究。但计算成本较高，GD指标略逊，且限于2D场景，实际应用场景相对有限。

<!-- RELATED:START -->

## 相关论文

- [MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)
- [FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](../../ICCV2025/image_generation/emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)

<!-- RELATED:END -->
