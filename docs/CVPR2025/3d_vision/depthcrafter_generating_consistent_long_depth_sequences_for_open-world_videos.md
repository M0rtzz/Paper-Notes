---
title: >-
  [论文解读] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos
description: >-
  [CVPR 2025][3D视觉][视频深度估计] 利用预训练的视频扩散模型 (SVD) 进行视频深度估计，通过三阶段训练策略实现可变长度（最长110帧）的时间一致深度序列生成，并设计分段推理策略支持极长视频，在零样本设置下全面超越现有方法。 单目深度估计领域的基础模型（Depth Anything V2、Marigold…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "视频深度估计"
  - "扩散模型"
  - "时间一致性"
  - "开放世界"
  - "长序列深度"
---

# DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos

**会议**: CVPR 2025  
**arXiv**: [2409.02095](https://arxiv.org/abs/2409.02095)  
**代码**: [https://depthcrafter.github.io](https://depthcrafter.github.io)  
**领域**: 3D视觉  
**关键词**: 视频深度估计, 扩散模型, 时间一致性, 开放世界, 长序列深度

## 一句话总结

利用预训练的视频扩散模型 (SVD) 进行视频深度估计，通过三阶段训练策略实现可变长度（最长110帧）的时间一致深度序列生成，并设计分段推理策略支持极长视频，在零样本设置下全面超越现有方法。

## 研究背景与动机

单目深度估计领域的基础模型（Depth Anything V2、Marigold 等）在生成单帧深度时表现出色，但直接逐帧应用于视频会产生严重的时间不一致（闪烁）问题。现有视频深度方法面临多重挑战：

- **测试时优化方法**（如 Robust CVD）需要相机位姿或光流，难以适用于开放世界视频
- **前馈预测方法**（如 NVDS）受限于有限的训练数据，无法处理多样化的开放场景
- **时间上下文不足**：现有视频扩散模型仅支持固定的少量帧数（如 SVD 仅25帧），不足以准确安排整个视频的深度分布
- 开放世界视频在内容、运动、相机移动和长度上差异巨大

核心动机：利用视频扩散模型的强大生成能力来学习时间一致的深度分布，通过精心设计的训练策略同时获取内容多样性和深度精度。

## 方法详解

### 整体框架

DepthCrafter 是一个条件扩散模型，建模 $p(\mathbf{d}|\mathbf{v})$，从输入视频 $\mathbf{v}$ 生成深度序列 $\mathbf{d}$。基于预训练的 Stable Video Diffusion (SVD)，在潜在空间中操作，使用 VAE 进行空间编码/解码，通过三阶段训练策略渐进式适配。

### 关键设计

1. **视频条件机制适配**:
    - 功能：将 SVD 的单图条件生成适配为视频到深度的逐帧条件生成
    - 核心思路：原始 SVD 仅将第一帧的潜在表示拼接到输入，本文改为将所有视频帧的潜在表示逐帧拼接到噪声深度潜在表示；高层语义信息通过 CLIP 嵌入逐帧注入交叉注意力。深度序列采用仿射不变表示（归一化到 $[0,1]$），关键区别是用**全序列共享的 scale 和 shift** 而非逐帧归一化
    - 设计动机：逐帧条件提供完整的视频信息，共享归一化确保时间一致性

2. **三阶段训练策略**:
    - 功能：渐进式训练使模型同时获得内容多样性、长时间上下文和精细深度细节
    - 核心思路：
        - **阶段一**（80K迭代）：在大规模真实数据集（~200K视频）上训练全模型，序列长度随机采样 $[1,25]$ 帧，学习视频到深度的基本任务和可变长度生成
        - **阶段二**（40K迭代）：仅微调时间层，仍在真实数据上训练，序列长度扩展到 $[1,110]$ 帧，学习长时间上下文
        - **阶段三**（10K迭代）：仅微调空间层，在小型合成数据集（~3K视频，DynamicReplica + MatrixCity）上训练，固定45帧，学习精细深度细节
    - 设计动机：直接训练长序列内存不够（40GB GPU 仅支持25帧）；仅微调时间层大幅降低内存；时间层对序列长度敏感而空间层已在阶段一适配；合成数据有更精确的深度标注

3. **极长视频推理策略**:
    - 功能：支持超过110帧的任意长度视频深度估计
    - 核心思路：将视频分为重叠段，逐段估计深度。关键技巧是**噪声初始化锚定**——对重叠帧的潜在表示不用纯高斯噪声，而是对前一段的去噪结果加噪声来初始化，锚定深度分布的 scale 和 shift。然后用**榫卯式潜在插值**拼接相邻段——对重叠帧用线性递减的权重 $w_i$ 插值两段的潜在表示
    - 设计动机：独立推理各段会导致段间深度分布不一致；噪声初始化锚定 scale/shift，线性插值确保平滑过渡

### 损失函数 / 训练策略

- 使用 EDM 框架的去噪分数匹配损失：$\mathbb{E}[\lambda_{\sigma_t}\|D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) - \mathbf{x}_0\|_2^2]$
- VAE 编码器/解码器直接复用 SVD 的预训练权重，已验证对深度序列的重建误差可忽略
- 分辨率 $320 \times 640$ 训练，推理时可用任意分辨率（如 $576 \times 1024$）
- 8×A100 GPU，总训练约5天
- 推理时默认5步去噪

## 实验关键数据

### 主实验（零样本视频深度估计）

| 数据集 | 指标 | DepthCrafter | Depth-Anything-V2 | 提升 |
|--------|------|------|----------|------|
| Sintel (~50帧) | AbsRel↓ | **0.270** | 0.367 | 26.4% |
| Sintel (~50帧) | $\delta_1$↑ | **0.697** | 0.554 | 25.8% |
| ScanNet (90帧) | AbsRel↓ | **0.123** | 0.135 | 8.9% |
| KITTI (110帧) | AbsRel↓ | **0.104** | 0.140 | 25.7% |
| KITTI (110帧) | $\delta_1$↑ | **0.896** | 0.804 | 11.4% |
| Bonn (110帧) | AbsRel↓ | **0.071** | 0.078 | 9.0% |

### 消融实验（Sintel 数据集）

| 训练阶段 | AbsRel↓ | $\delta_1$↑ | 说明 |
|------|---------|------|------|
| Stage 1 only | 0.322 | 0.626 | 基础适配 |
| Stage 1+2 | 0.316 | 0.675 | +长时间上下文 |
| Stage 1+2+3 | **0.270** | **0.697** | +精细深度细节 |

### 关键发现

- 在所有四个视频数据集上取得 SOTA，在 Sintel 和 KITTI 等大运动场景上优势最为显著
- 每帧推理465.84ms（$1024 \times 576$），比 Marigold (1070ms) 快一倍多，但慢于 Depth-Anything-V2 (180ms)
- 时间一致性显著优于所有基线——时间剖面图无锯齿伪影
- 单帧深度估计（NYU-v2）也有竞争力，$\delta_1=0.948$
- 推理策略的噪声初始化+潜在插值缺一不可：仅用初始化可解决静态区域闪烁，但动态区域仍有问题
- 40帧段长仅需12GB显存，可适配大多数现代GPU

## 亮点与洞察

- **视频扩散模型的强大先验**: SVD 的时空注意力机制天然适合建模时间一致的深度分布，迁移效果远超预期
- **三阶段训练的精妙设计**: 数据集质量-数量的 trade-off 通过分阶段网络层选择性微调优雅解决
- **全序列共享 scale/shift**: 这是确保视频深度时间一致性的关键设计，比逐帧归一化更具挑战但更实用
- **推理策略的榫卯式插值**: 命名来源于中国传统木工技术，实际是latent space中简单但有效的线性混合
- **开放世界泛化能力**: 在 DAVIS、Sora 生成视频、卡通、游戏等极度多样化场景上均表现出色

## 局限与展望

- 推理速度（465ms/帧）仍较慢，主要瓶颈在扩散模型的迭代去噪
- 需要约24GB GPU 内存处理 $1024 \times 576$、110帧视频
- 仅预测相对深度（仿射不变），无法直接获得度量深度
- 合成训练数据仅~3K，如果扩大精细合成数据的规模可能进一步提升
- 真实数据集的深度GT通过立体匹配获得，存在一定噪声

## 相关工作与启发

- 与 ChronoDepth 相比：ChronoDepth 仅支持10帧，DepthCrafter 支持110帧且可推理任意长度，这是决定性优势
- 测试时优化方法（Robust CVD 等）需要相机位姿，限制了实际应用；DepthCrafter 不需要任何额外信息
- 启发：视频扩散模型是视频几何估计的强大基座，三阶段式渐进训练是利用异质数据集的有效范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将视频扩散模型成功用于长序列深度估计，三阶段训练和推理策略设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集零样本评测，覆盖室内外/动静态/真实合成，消融和下游应用展示充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构组织出色，时间剖面图可视化直观，动机-方案-验证链条完整
- 价值: ⭐⭐⭐⭐⭐ 开创了视频深度估计的新范式，对后续深度基础模型和视频理解有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [\[CVPR 2025\] Open-World Amodal Appearance Completion](open-world_amodal_appearance_completion.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] On Denoising Walking Videos for Gait Recognition](on_denoising_walking_videos_for_gait_recognition.md)

</div>

<!-- RELATED:END -->
