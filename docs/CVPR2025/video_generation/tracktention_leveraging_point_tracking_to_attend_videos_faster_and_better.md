---
title: >-
  [论文解读] Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better
description: >-
  [CVPR 2025][点追踪] Tracktention 提出了一种基于点追踪的新型注意力层，通过将预提取的点轨迹信息注入 Vision Transformer，实现运动感知的时序特征聚合，能将纯图像模型升级为 SOTA 视频模型，在视频深度预测和视频着色任务上显著提升时序一致性。
tags:
  - CVPR 2025
  - 点追踪
  - 时序一致性
  - 视频深度估计
  - 注意力机制
  - 即插即用
---

# Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better

**会议**: CVPR 2025  
**arXiv**: [2503.19904](https://arxiv.org/abs/2503.19904)  
**代码**: https://zlai0.github.io/TrackTention  
**领域**: 视频理解  
**关键词**: 点追踪, 时序一致性, 视频深度估计, 注意力机制, 即插即用

## 一句话总结

Tracktention 提出了一种基于点追踪的新型注意力层，通过将预提取的点轨迹信息注入 Vision Transformer，实现运动感知的时序特征聚合，能将纯图像模型升级为 SOTA 视频模型，在视频深度预测和视频着色任务上显著提升时序一致性。

## 研究背景与动机

**领域现状**：视频分析任务（如视频分割、深度估计、着色）相比图像任务额外要求输出的时序一致性。现有的时序建模方法主要包括 3D 卷积和时空注意力两类。与此同时，点追踪领域近年取得了重大进展，PIPs、TAPIR、CoTracker 等追踪器能在长视频中可靠地追踪大量点。

**现有痛点**：3D 卷积假设局部时空相关性，无法处理大位移运动；时空注意力为控制计算开销通常需要降低空间分辨率或限制时序范围，难以精确表示运动。光流方法在遮挡和大位移场景下表现不佳。这些方法本质上都在"隐式"地建立时空对应关系。

**核心矛盾**：精确的运动建模需要细粒度的时空对应关系，但现有方法要么计算代价过高（全时空注意力），要么建模能力不足（3D卷积、分离注意力）。

**本文目标**：设计一种高效的、显式利用运动信息的时序建模组件，能即插即用地升级图像模型为视频模型。

**切入角度**：现代点追踪器已经是很强的"运动专家"，能提供精确的跨帧对应关系。与其让网络隐式学习这些对应关系，不如直接利用追踪器的输出作为桥梁。

**核心 idea**：用预提取的点轨迹作为"中间人"——先从图像特征中沿轨迹位置采样信息，沿时间轴传播，再将信息写回图像特征，实现显式的运动感知时序对齐。

## 方法详解

### 整体框架

Tracktention 层由三个子模块组成：（1）Attentional Sampling 从视频特征图中沿点轨迹位置采样信息到 track tokens；（2）Track Transformer 沿时间维度更新 track tokens，实现时序信息传播；（3）Attentional Splatting 将更新后的 track tokens 写回到视频特征图。整个过程以残差连接方式嵌入现有网络：$F' = F + \text{Tracktention}(F)$。

### 关键设计

1. **Attentional Sampling（注意力采样）**:

    - 功能：从视频特征图中沿点轨迹位置池化信息到 track tokens
    - 核心思路：将 track tokens 的位置嵌入作为 query，特征图 token 作为 key/value，进行交叉注意力。关键是引入高斯位置偏置 $B_{tij} = -\frac{\|P_{ti} - \text{pos}(j)\|^2}{2\sigma^2}$（$\sigma=1/2$），鼓励注意力集中在轨迹点附近。同时使用 RoPE 编码捕获相对空间关系，QK-normalization 稳定训练
    - 设计动机：比简单的双线性采样更灵活，让模型可以学习更好的采样策略；高斯偏置确保注意力在空间上受轨迹位置约束

2. **Track Transformer（轨迹变换器）**:

    - 功能：沿时间维度传播和平滑 track tokens 的信息
    - 核心思路：将 track tokens 从 $T \times M \times D_f$ reshape 为 $M \times T \times D_f$，每条轨迹独立地在时间维度上做自注意力。使用 2 层 transformer encoder，加正弦位置编码
    - 设计动机：不在轨迹之间交换信息，因为空间信息交换已由 ViT 本身完成。实验验证跨轨迹注意力效果反而略差且更慢

3. **Attentional Splatting（注意力泼溅）**:

    - 功能：将更新后的 track tokens 信息写回到视频特征图
    - 核心思路：与 Attentional Sampling 对称设计——特征图网格坐标作为 query，track tokens 作为 key/value，使用转置的偏置矩阵 $B_t$，最终通过输出投影 $W_{\text{out}}$ 生成结果。$W_{\text{out}}$ 初始化为零，保证训练初始阶段网络输出不变
    - 设计动机：对称的采样-泼溅设计保证了信息处理的一致性；零初始化使得插入 Tracktention 层不会破坏预训练权重

### 损失函数 / 训练策略

使用 CoTracker3 作为点追踪预处理器，在时空体积中均匀随机采样 576 个点并双向追踪。Tracktention 层以残差方式插入 ViT 的每个（或部分）transformer block 之后，输出投影初始化为零以保留预训练效果。在下游视频任务的训练集上微调即可。

## 实验关键数据

### 主实验（视频深度预测）

| 方法 | 类型 | 参数量 | Sintel AbsRel↓ | Scannet AbsRel↓ | KITTI AbsRel↓ | Bonn AbsRel↓ | Avg AbsRel↓ |
|------|------|--------|----------|----------|----------|----------|----------|
| DepthCrafter | 视频 | 1521M | 0.343 | 0.125 | 0.110 | 0.075 | 0.163 |
| DepthAnything | 图像 | 343M | 0.325 | 0.130 | 0.142 | 0.078 | 0.169 |
| **Tracktention** | **视频** | **140M** | **0.295** | **0.087** | **0.104** | **0.066** | **0.138** |

### 消融实验

关键消融对比（从论文中提取的设计选择）：

| 配置 | 效果说明 |
|------|---------|
| 跨轨迹注意力 | 略差且更慢，空间信息交换已由 ViT 完成 |
| 无高斯位置偏置 | 注意力分散，退化为全局注意力 |
| 双线性采样替代 Attentional Sampling | 性能下降，固定采样不如可学习采样 |
| 随机初始化 vs 网格初始化轨迹 | 随机更好，网格初始化在运动下覆盖不均 |

### 关键发现

- **参数量最小但性能最优**：仅 140M 参数（基于 DepthAnything-Base 97M），超越 1521M 的 DepthCrafter
- 将纯图像模型（DepthAnything）升级为视频模型后，AbsRel 从 0.169 降至 0.138，提升 18.3%
- 在所有数据集上均超越专门设计的视频深度模型和大参数量的图像模型
- Tracktention 还成功应用于视频着色任务，超越原生视频设计的模型

## 亮点与洞察

- **点追踪作为视频理解的通用基础设施**——不局限于特定任务，而是作为通用组件注入任何图像模型，这种"专家模块组合"的思路非常优雅
- **零初始化的残差插入策略**使得 Tracktention 可以无损地嵌入预训练模型，然后通过微调逐步激活时序能力。这个 trick 在 ControlNet 等工作中也用到，但在视频时序建模中是新颖应用
- **以更少参数实现更好效果**的原因在于：显式利用追踪器输出避免了网络需要自行学习对应关系的开销

## 局限与展望

- 依赖外部点追踪器的质量——如果追踪器在某些场景（如高度重复纹理、严重遮挡）失败，Tracktention 的效果会受限
- 预提取点轨迹增加了前处理时间，在实时应用中可能是瓶颈
- 目前仅在深度预测和着色上验证，可以进一步扩展到视频分割、视频生成等更多任务
- 未来可以探索端到端训练追踪器和 Tracktention 层

## 相关工作与启发

- **vs DepthCrafter**: DepthCrafter 基于视频扩散模型生成视频深度，参数量 1521M；Tracktention 仅 140M 参数但性能更优，因为显式运动建模比隐式学习更高效
- **vs 3D 卷积 / 时空注意力**: 这些方法要么感受野有限（3D conv），要么计算量大且精度不足（分离注意力）；Tracktention 在追踪点分辨率（远高于特征图分辨率）上显式对齐
- **vs 光流方法**: 光流只建模相邻帧关系，且在遮挡/大位移时失败；点追踪可处理长程依赖和遮挡

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将点追踪提升为视频理解的通用基础组件是非常创新的视角
- 实验充分度: ⭐⭐⭐⭐ 在深度估计和着色两个任务上验证，对比充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法描述精准，图示直观
- 价值: ⭐⭐⭐⭐⭐ 即插即用设计使其有极高的实用价值和广泛的应用前景

<!-- RELATED:START -->

## 相关论文

- [Exploring Temporally-Aware Features for Point Tracking](exploring_temporally-aware_features_for_point_tracking.md)
- [Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](../../NeurIPS2025/video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)
- [TransPixeler: Advancing Text-to-Video Generation with Transparency](transpixeler_advancing_text-to-video_generation_with_transparency.md)

<!-- RELATED:END -->
