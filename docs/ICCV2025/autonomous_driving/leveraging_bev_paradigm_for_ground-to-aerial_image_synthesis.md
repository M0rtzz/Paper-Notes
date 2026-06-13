---
title: >-
  [论文解读] SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis
description: >-
  [ICCV 2025][自动驾驶][跨视角图像合成] 提出SkyDiffusion，结合Curved-BEV变换和BEV引导扩散模型，实现从地面街景图像到航拍/卫星图像的高质量跨视角合成，并引入Ground2Aerial-3多场景数据集。 跨视角地面到航拍图像合成旨在从街景图像生成对应位置的鸟瞰航拍图像…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "跨视角图像合成"
  - "鸟瞰图变换"
  - "扩散模型"
  - "地面到航拍生成"
  - "街景到卫星图"
---

# SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis

**会议**: ICCV 2025  
**arXiv**: [2408.01812](https://arxiv.org/abs/2408.01812)  
**代码**: [https://opendatalab.github.io/skydiffusion/](https://opendatalab.github.io/skydiffusion/)  
**领域**: 自动驾驶  
**关键词**: 跨视角图像合成, 鸟瞰图变换, 扩散模型, 地面到航拍生成, 街景到卫星图

## 一句话总结

提出SkyDiffusion，结合Curved-BEV变换和BEV引导扩散模型，实现从地面街景图像到航拍/卫星图像的高质量跨视角合成，并引入Ground2Aerial-3多场景数据集。

## 研究背景与动机

跨视角地面到航拍图像合成旨在从街景图像生成对应位置的鸟瞰航拍图像，在土地覆盖分类、城市规划、灾害应急响应等领域有重要应用。街景图像可通过Google Street View等平台便捷获取，具有高时间覆盖频率和灵活性，因此利用街景合成航拍图可以在卫星图不可用时提供替代方案。

**然而该任务面临两大核心挑战**：

**视角域差距巨大**：街景图像主要捕捉地面细节和建筑立面，而卫星图展现的是屋顶和宏观布局信息。即使使用SOTA扩散模型，直接从街景生成的航拍图虽然看起来逼真，但与实际地面内容严重不一致——例如道路方向、建筑位置等关键结构特征不匹配。

**密集遮挡问题**：在城市场景中，高建筑和树木严重限制了街景的观测范围，单张街景图远不足以覆盖对应卫星图的全部区域。需要融合多张不同位置的街景图来填补盲区。

现有方法的不足：早期GAN方法（X-Seq、SelectionGAN）生成质量差且模糊；基于扩散模型的方法（AerialDiffusion、Instr-p2p）虽生成效果逼真但**缺乏内容一致性**；依赖语义地图的方法在实际场景中不可行；并行工作GPG2A使用现有BEV方法+文本引导+语义分割，需要额外数据和多阶段处理。

**SkyDiffusion的切入角度**：BEV范式在自动驾驶中已被广泛用于统一多视角感知。将BEV变换应用于跨视角合成，可以自然地桥接街景和航拍之间的视角差距——先将街景变换为BEV空间（完成域对齐），再以BEV图作为条件引导扩散模型生成航拍图。

## 方法详解

### 整体框架

SkyDiffusion包含两个核心模块：(1) Curved-BEV变换，将街景全景图映射到鸟瞰视角，支持One-to-One和Multi-to-One两种映射模式；(2) BEV引导扩散模型，以BEV图作为条件输入，控制扩散模型生成内容一致的卫星图像。

### 关键设计

#### 1. Curved-BEV变换

- **功能**：将全景街景图像转换为鸟瞰（BEV）视角的图像，同时保留上方（如远处道路、建筑顶部）的信息。
- **核心思路**：改进传统BEV的"地面平面假设"（$z=0$），提出**上弯曲面假设**——BEV平面的高度随距中心距离增加而快速升高：
  $z = d_{norm}^4 \times \lambda = \left(\frac{\sqrt{x^2+y^2}}{d_{max}}\right)^4 \times \lambda$
  
  然后将弯曲面上的3D点 $P(x,y,z)$ 转换为球坐标 $P(\theta,\varphi)$，再利用等矩形投影关系映射到全景图的像素坐标 $P(u,v)$：
  $u = [\text{arctan2}(y,x) + \pi] \frac{w}{2\pi}, \quad v = [\frac{\pi}{2} + \text{arctan2}(z-H, \sqrt{x^2+y^2})] \frac{h}{\pi}$
- **设计动机**：传统地面平面BEV无法映射街景图像上半部分的内容（如远处道路、建筑高处），导致信息丢失。弯曲面设计使远处区域的映射"抬头"看向上方，从而捕获更多跨视角关联信息。且由于映射关系是固定的，计算开销极低。

#### 2. Multi-to-One BEV映射

- **功能**：将多张不同位置拍摄的街景BEV映射结果统一到同一卫星坐标系中。
- **核心思路**：基于相机位置关系，将各街景的BEV映射结果 $\text{BEV}_{cam_k}$ 偏移到卫星坐标系。对于重叠区域，选择距离拍摄点最近的BEV映射结果：
  $k = \arg\min_i \sqrt{(x - x_{cam_i})^2 + (y - y_{cam_i})^2}$
- **设计动机**：城市密集场景中单张街景的BEV感知范围非常有限。通过融合多张附近街景的BEV信息，有效扩展了感知范围，解决了建筑遮挡导致的大面积盲区问题。

#### 3. BEV控制扩散模型

- **功能**：以BEV变换后的图像为条件，引导预训练扩散模型生成内容一致的卫星图像。
- **核心思路**：使用条件编码器将BEV图 $I_{bev}$ 编码为特征 $c_{bev} = \mathcal{E}(I_{bev})$（包含空间注意力以增强特征、抑制变形区域），然后通过零卷积层和复制的SD编码器/中间块（类似ControlNet架构）将BEV特征注入扩散模型。训练损失为标准去噪目标：
  $\mathcal{L} = \mathbb{E}_{sat_0, t, c_{bev}, \epsilon \sim \mathcal{N}(0,1)} \left[\|\epsilon - \epsilon_\theta(sat_t, t, c_{bev})\|_2^2\right]$
- **设计动机**：BEV图已经与卫星视角大致对齐，但存在信息不完整和变形区域。扩散模型可以"填充"BEV图中缺失的内容（如建筑屋顶纹理），同时通过空间注意力自动抑制变形严重的区域的影响。

### 损失函数 / 训练策略

- 基于Stable Diffusion v1.5预训练权重，解锁扩散解码器。
- Classifier-free guidance scale设为9.0。
- DDIM采样50步；8块A100 GPU，batch size 128，训练100 epochs。
- 文本提示不是优化目标——该任务本质上是图像条件生成，文本无法充分描述街景场景的复杂性。

## 实验关键数据

### 主实验——与SOTA方法对比

| 方法 | CVUSA FID↓ | CVUSA SSIM↑ | CVACT FID↓ | VIGOR FID↓ | VIGOR SSIM↑ |
|------|-----------|-------------|-----------|-----------|-------------|
| X-Seq (GAN) | 161.16 | 0.084 | 190.12 | — | — |
| SelectionGAN | 116.57 | 0.129 | 100.21 | 149.53 | 0.127 |
| CUT | 72.83 | 0.121 | 62.22 | 69.42 | 0.169 |
| ControlNet | 32.45 | 0.149 | 62.21 | 53.27 | 0.170 |
| GPG2A | 58.80 | 0.135 | 63.50 | 70.19 | 0.159 |
| **SkyDiffusion** | **29.18** | **0.168** | **36.48** | **45.29** | **0.186** |

SkyDiffusion在所有数据集上全面领先：CVUSA上FID降低25.72%，SSIM提升7.68%；城市数据集VIGOR上FID降低14.98%，SSIM提升9.41%。

### 消融实验——Curved-BEV模块

| 配置 | CVACT FID↓ | CVACT SSIM↑ | VIGOR FID↓ | VIGOR SSIM↑ |
|------|-----------|-------------|-----------|-------------|
| 直接街景→扩散 (Baseline) | 62.21 | 0.115 | 53.27 | 0.170 |
| 标准BEV + 扩散 | 42.84 | 0.117 | 48.63 | 0.175 |
| Curved-BEV + 扩散 | 36.48 | 0.118 | 45.29 | 0.186 |
| Curved-BEV Multi-to-One | — | — | **31.90** | **0.205** |

逐步加入每个组件都带来一致性提升，Multi-to-One在VIGOR城市数据集上将FID从45.29进一步降低到31.90（+29.6%），SSIM从0.186提升到0.205。

### 关键发现

- **GAN方法生成大量伪影和模糊**，不适合跨视角合成。
- **现有扩散方法（AerialDiffusion、Instr-p2p）生成图像逼真但内容不一致**——缺乏BEV域对齐导致道路方向、建筑位置等结构信息丢失。
- **G2A-3数据集**上三个应用场景（灾害应急、低空无人机、历史卫星图）均significantly优于ControlNet基线，FID平均降低25.81%，SSIM平均提升12.88%。
- 灾害场景生成中，关键受灾区域在合成卫星图中清晰可见，可辅助灾后评估。

## 亮点与洞察

- **BEV范式用于生成任务**是一个巧妙的跨界迁移。BEV在自动驾驶感知中已经成熟，但将其用于图像生成领域的跨视角域对齐是新思路。
- **Curved-BEV的弯曲面假设**解决了传统地面平面BEV丢失上半视场信息的固有缺陷，且计算代价几乎为零（固定映射关系）。
- **G2A-3数据集**提供了三个有实际应用价值的新场景，特别是灾害应急和历史图像补全，弥补了现有跨视角数据集仅面向检索任务的局限。

## 局限与展望

- 当BEV中缺失区域过大时（如建筑屋顶信息），扩散模型只能"幻想"填充，无法保证真实性。
- Multi-to-One映射需要已知多张街景的精确地理位置关系，在实际应用中获取成本较高。
- 仅在512×512分辨率下实验，更高分辨率的合成结果有待验证。
- BEV变换中的超参数 $\lambda$ 对不同场景可能需要不同设置，缺乏自适应机制。

## 相关工作与启发

- 与Pix2Pix、ControlNet等通用图像翻译框架相关，但SkyDiffusion针对跨视角的巨大域差进行了专门设计。
- 与自动驾驶BEV感知（BEVFormer、LSS等）有方法论上的联系，将BEV从感知工具扩展为生成工具。
- 启发：能否反向利用——从航拍图合成街景？或将此技术用于自动驾驶模拟器的场景生成？

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification](../../NeurIPS2025/autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)
- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](../../ICML2025/autonomous_driving/geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[ICCV 2025\] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)

</div>

<!-- RELATED:END -->
