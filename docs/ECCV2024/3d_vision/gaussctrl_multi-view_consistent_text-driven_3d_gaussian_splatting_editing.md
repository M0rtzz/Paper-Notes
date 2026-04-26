---
title: >-
  [论文解读] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing
description: >-
  [ECCV 2024][3D视觉][3D编辑] 提出GaussCtrl，利用深度条件化的ControlNet编辑和注意力对齐模块实现多视角一致的文本驱动3DGS场景编辑，支持一次编辑所有视角并仅需一次3D模型更新。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D编辑
  - 3D Gaussian Splatting
  - 扩散模型
  - 多视角一致性
  - ControlNet
---

# GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

**会议**: ECCV 2024  
**arXiv**: [2403.08733](https://arxiv.org/abs/2403.08733)  
**代码**: https://gaussctrl.active.vision/ (有)  
**领域**: 3D视觉  
**关键词**: 3D编辑, 3D Gaussian Splatting, 扩散模型, 多视角一致性, ControlNet

## 一句话总结

提出GaussCtrl，利用深度条件化的ControlNet编辑和注意力对齐模块实现多视角一致的文本驱动3DGS场景编辑，支持一次编辑所有视角并仅需一次3D模型更新。

## 研究背景与动机

**领域现状**: NeRF和3DGS已实现高质量3D重建和新视角渲染，但对这些3D表示的编辑能力仍被低估。Instruct-NeRF2NeRF (IN2N)开创了"通过2D编辑实现3D编辑"的范式——用2D扩散模型逐帧编辑渲染图像，再迭代更新3D模型。

**现有痛点**: IN2N及其后续方法的核心问题是**多视角不一致性**——2D扩散模型独立处理每张图像，不保证几何一致性和外观一致性，导致3D编辑出现模糊、伪影、"背面出现正面"等问题。且迭代优化方式导致收敛缓慢。

**核心矛盾**: 2D扩散模型天生缺乏3D几何意识，但3D编辑又依赖2D编辑结果——如何在2D编辑阶段就注入3D一致性约束？

**本文目标**: 在所有视角图像的编辑过程中显式强制多视角一致性，使之能一次性编辑所有图像、仅更新一次3D模型。

**切入角度**: 利用3DGS本身能提供的一致性信息（深度图天然几何一致）和cross-view attention机制（统一文本编辑→外观一致）双管齐下。

**核心 idea**: 深度图条件化编辑保证几何一致性 + 注意力latent code对齐保证外观一致性 = 多视角一致的3D编辑。

## 方法详解

### 整体框架

GaussCtrl的流程分为四步：
1. 从已重建的3DGS渲染所有训练视角图像及对应深度图
2. 用DDIM Inversion将每张图像反转为latent noise
3. 用ControlNet（深度条件）配合注意力对齐模块，基于编辑后的文本prompt去噪生成编辑后的图像
4. 用编辑后的图像重新训练3DGS得到编辑后的3D模型

### 关键设计

1. **深度条件化图像编辑**: 利用3DGS渲染的深度图$\mathcal{D}$作为ControlNet的条件，保证编辑后的图像保持原始几何结构。核心流程：

    - DDIM Inversion: 将原始图像$\mathcal{I}$通过ControlNet的VAE编码为$z^0$，再迭代反转为噪声$z^T$：
   
   $$z^{t+1} = \sqrt{\alpha_{t+1}} \frac{z^t - \sqrt{1-\alpha_t} \cdot \epsilon^t}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t+1}} \epsilon^t$$
   
   - 编辑去噪: 替换为编辑prompt $\hat{p}_e$，通过classifier-free guidance去噪：
   
   $$\epsilon^t = \epsilon_\emptyset^t + \omega \cdot (\epsilon_p^t - \epsilon_\emptyset^t)$$
   
   设计动机：深度图来自同一3DGS，天然跨视角几何一致，以此条件化编辑可避免几何不协调。DDIM Inversion保证初始latent codes继承原始图像的一致颜色和几何。

2. **基于注意力的Latent Code对齐模块**: 深度条件保证几何一致，但各视角仍独立编辑，外观可能不一致（颜色偏差、困难视角异常）。模块通过混合self-attention和cross-view attention统一外观：

   $$\text{AttnAlign}_e = \lambda \cdot \text{Attn}_{e,e} + (1-\lambda) \cdot \frac{1}{N_r} \sum_{i=1}^{N_r} \text{Attn}_{e,i}$$

   其中attention操作定义为：
   
   $$\text{Attn}_{i,j} = \text{Softmax}\left(\frac{W_q(z_i) W_k(z_j)^\top}{\sqrt{c}}\right) W_v(z_j)$$

   - Self-attention $\text{Attn}_{e,e}$: 保持每张编辑图像的独特性
   - Cross-view attention $\text{Attn}_{e,i}$: 将外观对齐到$N_r$个参考视角
   - $\lambda = 0.6$，$N_r = 4$个随机采样的参考视角
   
   设计动机：已有研究表明扩散模型中self-attention的key-value对决定了生成图像的外观，通过注入参考视角的K/V来统一所有视角的外观。

3. **可选的语义分割掩码**: 使用Language-based SAM（Lang SAM）生成掩码，在编辑局部物体时过滤背景区域，提升编辑质量。

### 损失函数 / 训练策略

- 编辑后的图像直接用于重新训练3DGS（使用NeRFStudio的splatfacto模型）
- 基于Stable Diffusion v1.5及其对应的ControlNet
- 所有图像统一预处理为$512 \times 512$分辨率
- 编辑一个场景约需9分钟（NVIDIA RTX A5000, 24GB显存）
- 对齐模块同时替换U-Net和ControlNet block中的所有self-attention

## 实验关键数据

### 主实验（CLIP方向相似度 + 编辑时间）

| 场景 | IN2N CLIPdir | IN2N(GS) CLIPdir | ViCA-NeRF CLIPdir | **Ours CLIPdir** | Ours时间 |
|------|:---:|:---:|:---:|:---:|:---:|
| Bear Statue | 0.1019 | 0.1165 | 0.1104 | **0.1388** | ~9min |
| Dinosaur | 0.1466 | 0.1490 | 0.0723 | **0.1584** | ~9min |
| Garden | **0.3027** | 0.1663 | 0.2903 | 0.2891 | ~9min |
| Stone Horse | 0.1654 | 0.1947 | 0.1926 | **0.2268** | ~9min |
| Fangzhou | 0.1598 | 0.2032 | 0.1809 | 0.1887 | ~9min |
| Face | 0.1332 | 0.1357 | 0.1119 | **0.1503** | ~9min |

编辑时间对比：IN2N ~1.5h, IN2N(GS) ~13.5min, ViCA-NeRF ~38.5min, **Ours ~9min**

### 消融实验

| 配置 | 效果描述 |
|------|---------|
| (b) Instruct Pix2Pix一次编辑 | 困难视角（背面#4,6,7,8）完全失败，正面也有伪影 |
| (c) ControlNet + 随机噪声 | 几何一致但风格偏离原图，困难视角强行生成正面 |
| (d) ControlNet + 反转latent（无对齐） | 风格明显改善，但困难视角仍有伪影和"背面正脸"问题 |
| **(e) + AttnAlign（完整方法）** | 伪影大幅缓解，外观统一，困难视角语义正确 |

### 关键发现

- 4/6个场景上CLIPdir最优，且编辑速度最快（~9min vs IN2N的~1.5h，加速10倍）
- CLIPdir指标不能完全反映编辑质量——它衡量全局文本-图像相似性而忽视局部细节（论文展示了CLIPdir高但视觉质量差的反例）
- 360度场景比前向场景更能体现方法优势，因视角变化更极端
- DDIM Inversion + 深度条件是一致性的基础，AttnAlign进一步消除外观不一致和语义异常

## 亮点与洞察

- **"一次编辑所有视角"的范式转变**: 之前方法（IN2N）需要迭代编辑单帧+更新3D模型的循环，本文实现了batch编辑+单次3D更新，速度提升10倍
- **巧妙利用3DGS的深度信息**: 3DGS天然提供多视角一致的深度图，将其作为ControlNet条件是一个优雅的设计
- **对attention机制的深刻理解**: self-attention的K/V决定外观这一洞察被有效利用，cross-view attention的引入本质上建立了跨视角的外观通信
- **对CLIPdir指标局限性的分析**: 指出该指标的不足并提供了反例，对领域评估方法有重要启示

## 局限与展望

- **无法改变几何结构**: 深度条件保持了原始几何，因此无法完成需要大幅几何变化的编辑（如将熊变成长颈鹿）——但论文指出IN2N等方法同样无法完成
- **依赖ControlNet的能力**: 如果ControlNet不认识某些概念（如"Hulk"），编辑将失败
- **参考视角选择策略**: 当前随机采样4个参考视角，可能不是最优策略，基于覆盖度的选择可能更好
- **评估指标不完善**: CLIPdir不足以评估3D编辑质量，需要更好的指标

## 相关工作与启发

- IN2N开创了"2D编辑→3D编辑"范式，但迭代方式天然导致不一致性
- ViCA-NeRF尝试用参考视角投影混合缓解不一致，但导致模糊
- Prompt-to-Prompt和MasaCtrl等工作揭示了attention中K/V对外观的控制作用
- 本文思路可推广到视频编辑、4D场景编辑等需要时序/空间一致性的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 深度条件+注意力对齐的组合虽非全新组件，但在3D编辑中的应用设计精巧
- 实验充分度: ⭐⭐⭐⭐ 360度和前向场景、多种编辑类型、详细消融、一致性可视化（10视角对比）
- 写作质量: ⭐⭐⭐⭐⭐ 图示丰富，问题阐述清晰，消融设计层层递进，对指标局限的讨论坦诚
- 价值: ⭐⭐⭐⭐ 10倍加速+更好质量，实用价值高，定义了3D编辑的新baseline

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] 3DEgo: 3D Editing on the Go!](3dego_3d_editing_on_the_go.md)
- [\[ECCV 2024\] DATENeRF: Depth-Aware Text-based Editing of NeRFs](datenerf_depth-aware_text-based_editing_of_nerfs.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)

<!-- RELATED:END -->
