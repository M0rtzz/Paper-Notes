---
title: >-
  [论文解读] Towards Open-World Generation of Stereo Images and Unsupervised Matching
description: >-
  [ICCV 2025][自动驾驶][立体图像生成] 提出 GenStereo，一种基于扩散模型的立体图像生成框架，通过视差感知坐标嵌入、跨视图注意力和自适应融合机制，同时实现高视觉质量和高几何精度的立体图像生成，并推动无监督立体匹配达到新 SOTA。
tags:
  - ICCV 2025
  - 自动驾驶
  - 立体图像生成
  - 扩散模型
  - 无监督立体匹配
  - 视差感知
  - 自适应融合
---

# Towards Open-World Generation of Stereo Images and Unsupervised Matching

**会议**: ICCV 2025  
**arXiv**: [2503.12720](https://arxiv.org/abs/2503.12720)  
**代码**: [项目页面](https://qjizhi.github.io/genstereo)  
**领域**: 自动驾驶  
**关键词**: 立体图像生成, 扩散模型, 无监督立体匹配, 视差感知, 自适应融合

## 一句话总结

提出 GenStereo，一种基于扩散模型的立体图像生成框架，通过视差感知坐标嵌入、跨视图注意力和自适应融合机制，同时实现高视觉质量和高几何精度的立体图像生成，并推动无监督立体匹配达到新 SOTA。

## 研究背景与动机

立体图像是 XR 设备、自动驾驶和机器人等领域的基础数据形式，但获取高质量立体图像面临多重挑战：

**真实数据获取困难**：双目相机需要精确标定，真实数据集要么只有稀疏视差标注（如 KITTI），要么仅限于特定场景（如室内）。合成数据集虽有精确视差但存在域差距。

**现有生成方法的二难困境**：
   - **基于 warping 的方法**（如 MfS）：几何精度高但遮挡区域用随机背景填充，语义不一致。
   - **基于扩散的方法**（如 StereoDiffusion）：语义连贯但在 latent space 做视差偏移，缺乏像素级精度。
   - SD-Inpainting 填充的遮挡区域语义不恰当，与周围纹理不连续。

**无监督立体匹配的瓶颈**：之前的无监督方法要么受限于简单 warping + 随机填充（MfS），要么限于小规模静态场景（NeRF-Stereo），泛化能力有限。

作者的核心洞察：需要一个统一框架同时解决视觉质量和几何精度——在 warped 图像可靠的区域使用几何精确的 warp 结果，在遮挡区域利用扩散模型生成语义一致的内容，并通过学习的融合权重实现无缝过渡。

## 方法详解

### 整体框架

GenStereo 采用双流 U-Net 架构（从 Stable Diffusion 预训练权重 fine-tune）：
- **参考 U-Net**：处理左视图 $(I_l, C_l)$，提取参考特征
- **去噪 U-Net**：以 $(I_{warp}, C_r)$ 为条件，合成右视图 $\hat{I}_r$
- 最后通过自适应融合模块将生成图像与 warp 图像加权混合

### 关键设计

1. **视差感知坐标嵌入 (Disparity-Aware Coordinate Embedding)**：

    - **核心思路**：构建标准化 2D 坐标图 $X \in \mathbb{R}^{h \times w \times 2}$，通过 Fourier 位置编码 $\phi$ 变换为坐标嵌入。左视图嵌入 $C_l = \phi(X)$ 保持不变，右视图嵌入 $C_r = \text{warp}(C_l, D_l)$ 根据视差 warp。
    - **设计动机**：传统 inpainting 方法在 warp 区域和填充区域之间存在明显边界。坐标嵌入提供了隐式几何引导，使模型理解每个像素的空间对应关系。相比 GenWarp 使用相机矩阵，使用视差图进行 warp 可实现更精确的像素级控制。
    - 同时将 warp 后的图像 $I_{warp}$ 作为额外条件输入去噪 U-Net。

2. **跨视图特征增强 (Cross-View Feature Enhancement)**：

    - **核心思路**：在注意力机制中拼接左视图特征和右视图特征：$q = F_r, \; k = [F_l, F_r], \; v = [F_l, F_r]$。
    - **设计动机**：右视图生成需要参考左视图的语义信息。双流注意力使模型自适应平衡参考视图的语义一致性和 warp 视图的几何精度。文本条件替换为左图的 CLIP 图像嵌入。

3. **像素空间对齐与自适应融合**：

    - **双空间监督**：标准 latent space 损失 $L_{latent}$ 外增加像素空间损失 $L_{pixel} = \| \mathcal{D}(z_{pred}) - \mathcal{D}(z_{target}) \|_2^2$，最终 $L = L_{latent} + \alpha L_{pixel}$（$\alpha = 1$）。
    - **自适应融合模块**：轻量卷积网络预测空间变化的融合权重 $W = \sigma(f_\theta(\text{concat}(I_{gen}, I_{warp}, M)))$，最终右视图 $\hat{I}_r = M \odot W \odot I_{warp} + (1 - M \odot W) \odot I_{gen}$。
    - **设计动机**：LDM 在 latent space 操作可能损失像素级精度。像素空间监督直接约束输出质量。自适应融合在高置信区域（$M \approx 1$）倾向使用 warp 内容，在遮挡区域使用生成内容，确保平滑过渡。

### 损失函数 / 训练策略

- **混合数据集训练**：11 个合成立体数据集，共 684K 图像对，覆盖室内外场景
- 不使用真实数据集（即使微小标定误差也会影响性能）
- 重采样策略：小数据集重复采样至最大数据集的 10%
- 随机方形裁剪 + resize 到 512×512（SD v1.5）或 768×768（SD v2.1）
- 训练 3 个 epoch fine-tune 预训练 SD UNet
- **随机视差丢弃**：10% 训练样本随机丢弃部分视差，模拟 KITTI 等稀疏 GT 场景

## 实验关键数据

### 主实验

**立体图像生成质量（Table 2，Middlebury 2014 + KITTI 2015）：**

| 方法 | SD版本 | Middlebury PSNR↑ | Middlebury SSIM↑ | KITTI PSNR↑ | KITTI SSIM↑ |
|------|--------|-----------------|-----------------|-------------|-------------|
| StereoDiffusion | 1.5 | 15.456 | 0.468 | 15.679 | 0.481 |
| SD-Inpainting | 1.5 | 15.740 | 0.412 | 9.792 | 0.230 |
| **GenStereo+Pseudo** | **2.1** | **25.142** | **0.911** | **23.488** | **0.849** |

GenStereo 在 PSNR 上提升约 10 dB，SSIM 提升近一倍。

**无监督立体匹配（Table 3，KITTI 2012/2015）：**

| 方法 | KITTI 2012 D1-all↓ | KITTI 2012 EPE↓ | KITTI 2015 D1-all↓ | KITTI 2015 EPE↓ |
|------|-------------------|----------------|-------------------|----------------|
| SD-Inpainting | 3.907 | 0.894 | 4.490 | 1.059 |
| StereoDiffusion | 15.213 | 2.220 | 5.651 | 1.154 |
| **GenStereo** | **3.802** | **0.815** | **3.933** | **0.991** |

### 消融实验

**关键组件消融（从论文描述推断的组件效果）：**

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无坐标嵌入 | 生成图像几何精度下降 | 缺乏像素级空间对应引导 |
| 无像素空间损失 | PSNR 下降 | latent space 操作损失像素精度 |
| 无自适应融合 | warp/生成边界明显 | 无法在高/低置信区域间平滑过渡 |
| 无随机视差丢弃 | 稀疏 GT 场景性能差 | 模型未见过稀疏输入 |
| 使用真实数据训练 | 性能反而下降 | 微小标定误差影响学习 |

### 关键发现

- 使用伪视差（MDE 预测）生成的立体图像质量竟然与 GT 视差相当甚至略好（在 KITTI 上 PSNR 23.488 vs 19.836），因为 MDE 模型提供了更密集的视差
- 仅用合成数据训练就能在真实场景（Middlebury、KITTI）上获得强泛化能力
- SD v2.1 比 v1.5 效果更好（PSNR 25.142 vs 23.835），得益于更高分辨率训练
- 生成的立体图像可直接用于训练无监督立体匹配网络，显著缩小了与有监督方法的差距

## 亮点与洞察

- **统一视觉质量和几何精度的框架设计**是核心贡献：不是简单的 inpainting，而是将 warp 和生成有机结合
- **只用合成数据训练**这一决策反直觉但效果好——真实数据的标定误差反而有害
- **伪视差优于 GT 视差**的发现意义重大：说明 MDE 模型的密集深度比 LiDAR 稀疏深度更适合生成任务
- 为无监督立体匹配提供了一条新路径：用单目图像 + 深度估计生成训练数据，避免了昂贵的双目标定

## 局限与展望

- 推理速度受限于扩散模型的多步采样（未报告推理时间）
- 分辨率限于 512×512 或 768×768，不适合高分辨率应用
- 依赖 MDE 模型质量——视差估计的误差会传播到生成结果
- 未探索动态场景中运动物体的时序一致性
- 自适应融合模块仅用 3×3 卷积，感受野有限

## 相关工作与启发

- 架构灵感来自 GenWarp 和 Animate-Anyone 的双流注意力设计，但创新地加入视差条件
- 与 Mono2Stereo 相比，GenStereo 在遮挡区域的生成质量显著更好
- 启发：单目深度估计 + 扩散模型的组合可推广到多视图生成、视频深度估计等任务
- 混合数据集训练策略（重采样平衡 + 仅用合成数据）值得在其他跨域任务中借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐ 视差感知坐标嵌入和双空间监督的组合设计新颖，但双流 U-Net 框架借鉴较多
- **实验充分度**: ⭐⭐⭐⭐ 多基准测试，消融覆盖关键组件，但缺少推理效率分析
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，图表说明性强
- **价值**: ⭐⭐⭐⭐⭐ 同时推进了立体图像生成和无监督立体匹配两个方向，实用价值高

<!-- RELATED:START -->

## 相关论文

- [AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](../../NeurIPS2025/autonomous_driving/towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](../../CVPR2025/autonomous_driving/helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)

<!-- RELATED:END -->
