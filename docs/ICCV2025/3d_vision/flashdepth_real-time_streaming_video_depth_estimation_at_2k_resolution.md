---
title: >-
  [论文解读] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution
description: >-
  [ICCV 2025][3D视觉][视频深度估计] 提出FlashDepth，在Depth Anything v2基础上添加Mamba循环模块实现帧间尺度一致性，并设计Small-Large混合架构在2K分辨率下达到24 FPS的实时流式视频深度估计，边界清晰度远超现有方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 视频深度估计
  - 实时流式处理
  - 2K分辨率
  - Mamba循环网络
  - 混合模型
---

# FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

**会议**: ICCV 2025  
**arXiv**: [2504.07093](https://arxiv.org/abs/2504.07093)  
**代码**: [GitHub](https://github.com/Eyeline-Research/FlashDepth)  
**领域**: 3D视觉 / 视频深度估计  
**关键词**: 视频深度估计, 实时流式处理, 2K分辨率, Mamba循环网络, 混合模型

## 一句话总结

提出FlashDepth，在Depth Anything v2基础上添加Mamba循环模块实现帧间尺度一致性，并设计Small-Large混合架构在2K分辨率下达到24 FPS的实时流式视频深度估计，边界清晰度远超现有方法。

## 研究背景与动机

- **视频深度估计的三大核心需求**：
  1. 跨帧一致性和准确性
  2. 高分辨率和清晰边界
  3. 实时流式处理
- **现有方法无法同时满足**：
    - 基于扩散的视频深度模型（DepthCrafter）：精度高但慢（2.1 FPS），需处理110帧批次，无法实时
    - 单帧模型（DAv2）：精度高但帧间闪烁
    - 早期在线方法（LSTM/全局点云）：无法扩展到高分辨率，精度低
    - Video Depth Anything：需要处理32帧批次，分辨率受限
    - CUT3R：支持流式但深度图模糊
- **关键观察**：
    - 深度图精度（低频）对分辨率不敏感——边界仅占约1%的像素
    - 但边界清晰度（高频）高度依赖分辨率
    - 帧间一致性本质上是尺度对齐问题，轻量级循环网络即可解决

## 方法详解

### 整体架构

基于Depth Anything v2（DAv2），包含三个关键设计：
1. **Mamba循环模块**：在DPT decoder后、最终卷积头前添加，实现帧间尺度对齐
2. **Small-Large混合模型**：高分辨率轻量流 + 低分辨率精准流 + 交叉注意力融合
3. **两阶段训练策略**：先学一致性（低分辨率），再学高分辨率

### 关键设计1：Mamba循环模块

**设计考量**：
- 使用Mamba（而非LSTM/GRU），因其选择性保留长序列相关信息的能力
- 对DPT decoder输出特征先下采样再处理，降低计算开销
- 仅占模型总参数的约1%

**具体实现**：
$$f, H_t = \text{Mamba}(\text{Flatten}(\text{Down}(F)), H_{t-1})$$
$$F_{\text{align}} = F + \text{Up}(\text{UnFlatten}(f))$$

其中F为DPT decoder的特征输出，H_t为Mamba的隐状态。通过残差连接，零初始化Mamba输出层确保第一次迭代保持原始DAv2的有效深度图。

**Mamba模块结构**：4个block，每个包含LayerNorm + Mamba层 + MLP。使用vanilla Mamba而非双向扫描。

### 关键设计2：Small-Large混合模型

**动机**：DAv2-Large在2K分辨率下仅6 FPS，DAv2-Small虽然实时但精度大幅下降。

**解决方案**：
- **FlashDepth-S**（主流）：基于DAv2-S微调，处理2K全分辨率（2044×1148）
- **FlashDepth-L**（辅助流）：基于DAv2-L微调，处理低分辨率（924×518）

通过交叉注意力融合两流中间特征：
$$F_{\text{fused}_i} = \text{CrossAttn}(Q=F_{S_i}, KV=F_{L_i})$$

交叉注意力模块包含2个transformer block，零初始化确保训练初始时不改变原始能力。仅融合第一个DPT层的输出特征，避免低分辨率伪影渗入高分辨率流。

**推理优化**：两个流通过CUDA Graph并行运行，综合实现24 FPS。

### 两阶段训练

**第一阶段**（低分辨率一致性训练）：
- 数据集：MVS-Synth, Spring, TartanAir, PointOdyssey, Dynamic Replica，共约50万图像-深度对
- 分辨率：518×518随机裁剪
- 损失：简单L1损失（因数据为合成度量深度）
- 分别训练FlashDepth-L和FlashDepth-S的temporal模块
- 通过增大帧间步长增强长视频泛化能力

**第二阶段**（高分辨率混合训练）：
- 数据集：MVS-Synth + Spring，约16,000个2K深度标注
- 冻结FlashDepth-L，仅微调FlashDepth-S和交叉注意力模块
- 学习率：Mamba/CrossAttn 1e-4，其余 1e-6
- 训练时长：每阶段各半天（8×A100）

## 实验关键数据

### 主实验：边界清晰度与速度

| 方法 | Unreal4K F1↑ | UrbanSyn F1↑ | FPS↑ | 分辨率 | 流式 |
|------|-------------|-------------|------|--------|------|
| DAv2 | 0.058 | 0.118 | 30 | 924×518 | 逐帧 |
| DepthCrafter | 0.021 | 0.044 | 2.1 | 1024×576 | ✗(110帧) |
| VidDepthAny | 0.049 | 0.097 | 24 | 924×518 | ✗(32帧) |
| CUT3R | 0.007 | 0.019 | 14 | 512×288 | ✓ |
| FlashDepth-L | 0.048 | 0.136 | 30 | 924×518 | ✓ |
| FlashDepth-L(high) | 0.143 | 0.271 | 6.0 | 2044×1148 | ✓ |
| **FlashDepth(Full)** | **0.109** | **0.185** | **24** | **2044×1148** | **✓** |

关键发现：FlashDepth在2K分辨率24 FPS下的边界F1分数远超所有基线方法，验证了高分辨率处理对边界清晰度的关键作用。

### 深度精度对比

| 方法 | ETH3D δ1↑ | Sintel δ1↑ | Waymo δ1↑ | Unreal4K δ1↑ | UrbanSyn δ1↑ | 平均δ1排名 |
|------|----------|----------|----------|-------------|-------------|----------|
| DAv2 | 0.633 | 0.561 | 0.897 | 0.379 | 0.622 | 5.4 |
| DepthCrafter | 0.745 | 0.697 | 0.790 | 0.399 | 0.556 | 4.6 |
| VidDepthAny | 0.864 | 0.660 | 0.944 | 0.606 | 0.892 | 1.6 |
| CUT3R | 0.836 | 0.509 | 0.902 | 0.851 | 0.878 | 3.6 |
| FlashDepth-L | 0.875 | 0.642 | 0.924 | 0.566 | 0.882 | 2.2 |
| **FlashDepth(Full)** | **0.848** | **0.642** | **0.916** | **0.545** | **0.862** | **3.4** |

FlashDepth精度仅次于VidDepthAny（处理32帧批次），且超过处理110帧的DepthCrafter。

### 消融实验：超分辨率方法对比

| 方法 | Unreal4K δ1↑ | Unreal4K F1↑ | UrbanSyn δ1↑ | UrbanSyn F1↑ | FPS |
|------|-------------|-------------|-------------|-------------|-----|
| FlashDepth-L（基线） | 0.566 | 0.048 | 0.882 | 0.136 | 30 |
| + UNet | 0.490 | 0.034 | 0.811 | 0.107 | 24 |
| + DAGF | 0.377 | 0.055 | 0.703 | 0.133 | 7.7 |
| **FlashDepth(Full)** | **0.545** | **0.109** | **0.862** | **0.185** | **24** |

关键发现：直接使用depths超分辨率网络（UNet/DAGF）会损害精度或泛化能力，而混合模型方案同时保持精度和边界清晰度。

### 消融：模型大小 vs 混合架构

| 方法 | ETH3D δ1↑ | Waymo δ1↑ | UrbanSyn δ1↑ |
|------|----------|----------|-------------|
| FlashDepth-S (~21M参数) | 0.786 | 0.898 | 0.784 |
| FlashDepth-L (~300M参数) | 0.875 | 0.924 | 0.882 |
| FlashDepth(Full) | 0.848 | 0.916 | 0.862 |

直接缩小模型（S vs L）导致3-10%精度下降，混合模型仅损失1-3%但速度与S相同。

## 亮点与洞察

1. **极简但有效的时序一致性方案**：Mamba模块仅占1%参数，无需光流监督或时序损失，即可实现长达1000帧的一致性
2. **深度-分辨率解耦洞察**：精度（低频）对分辨率不敏感但边界（高频）强依赖，据此设计混合架构
3. **规避数据稀缺**：基于预训练DAv2微调，仅需少量合成视频数据即可训练，不需要大规模高分辨率视频深度数据
4. **工程完成度高**：开源代码和权重，实际可在A100上2K 24FPS运行，具有直接应用价值

## 局限性

- 仍存在残余闪烁（residual flickering），特别在剧烈运动场景中
- 混合架构需要同时加载两个模型（DAv2-S + DAv2-L），显存开销较大
- 第二阶段训练仅用~16,000张高分辨率深度标注，可能限制泛化
- 无法产生度量深度（metric depth），输出为相对深度

## 相关工作与启发

- **与Video Depth Anything的对比**：同样基于DAv2微调但采用不同时序模块；VidDepthAny需32帧批处理，FlashDepth支持真正流式
- **与DepthCrafter的对比**：视频扩散模型的一致性更优但计算代价过大
- **启发**：对于视频深度估计，在高质量单帧模型上添加轻量时序模块是比训练视频扩散模型更务实的路径

## 评分 ⭐⭐⭐⭐

实用性极强的工作。在"精度-分辨率-速度"三角中找到了优秀的平衡点。混合架构的设计动机清晰、实验验证充分。Mamba时序模块的简洁有效令人印象深刻。Netflix Eyeline Studios的工业背景为其工程质量提供了保障。

<!-- RELATED:START -->

## 相关论文

- [QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](../../CVPR2025/3d_vision/quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](../../CVPR2025/3d_vision/video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

<!-- RELATED:END -->
