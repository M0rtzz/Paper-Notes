---
title: >-
  [论文解读] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars
description: >-
  [CVPR 2026][3D视觉][渐进式3D表示] 提出 ProgressiveAvatars，一种基于模板网格自适应隐式细分构建层级3DGS的渐进式头像表示，支持在不同带宽和算力约束下渐进传输和渲染——仅传输5%数据（2.6MB）即可获得可用头像，后续增量加载平滑提升质量至与 SOTA 方法可比。
tags:
  - CVPR 2026
  - 3D视觉
  - 渐进式3D表示
  - 可动画化头部Avatar
  - 3D高斯溅射
  - 流式传输
  - 自适应细分
---

# ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars

**会议**: CVPR 2026  
**arXiv**: [2603.16447](https://arxiv.org/abs/2603.16447)  
**代码**: [GitHub](https://ustc3dv/ProgressiveAvatars)  
**领域**: 3D视觉  
**关键词**: 渐进式3D表示, 可动画化头部Avatar, 3D高斯溅射, 流式传输, 自适应细分

## 一句话总结
提出 ProgressiveAvatars，一种基于模板网格自适应隐式细分构建层级3DGS的渐进式头像表示，支持在不同带宽和算力约束下渐进传输和渲染——仅传输5%数据（2.6MB）即可获得可用头像，后续增量加载平滑提升质量至与 SOTA 方法可比。

## 研究背景与动机
**领域现状**：高保真实时头部 Avatar 是沉浸式交互的关键技术。3DGS 因其高效渲染已成为主流显式表示。GaussianAvatars、FlashAvatar、MeGA 等方法实现了高质量可动画化头像。

**现有痛点**：
   - 在社交 VR 等多用户动态场景中，以传统静态资产方式传输高保真 Avatar 会导致严重启动延迟和带宽尖峰，用户必须等待完整下载才能看到任何渲染
   - 现有 3DGS Avatar 缺少增量加载机制，无法在传输过程中平滑累积细节
   - 已有 LOD 方法（LoDAvatar、ArchitectHead）依赖离散 LOD 切换范式，需要存储多个独立模型副本，存在严重存储冗余和资源切换延迟
   - 均匀细分（LoDAvatar）在平滑区域过度细化而在高频区域细化不足，浪费资源

**核心矛盾**：如何在一个统一资产中实现渐进式传输和渲染，支持任意传输比例下的即时可动画化渲染，同时不引入离散资产切换和存储冗余

**切入角度**：在 FLAME 模板网格上构建面局部坐标系的层级3DGS，通过自适应隐式细分按需增长细节，按重要性排序实现连续流式传输

**核心 idea**：通过模板网格面上的自适应隐式细分构建层级森林，利用每个面的重要性评分实现增量加载的渐进传输和渲染

## 方法详解

### 整体框架
输入：头部视频 → FLAME 网格跟踪。训练：在 FLAME 面上绑定3D高斯，用屏幕空间梯度驱动隐式细分构建多层级结构。推理：预计算重要性评分，按重要性递减顺序发送高斯，接收端增量添加并渲染。

### 关键设计

1. **隐式细分（Implicit Subdivision）**

    - 功能：在 FLAME 模板网格的每个三角面上递归产生子面，形成每面的层级树结构
    - 核心思路：对父面 $f = (i,j,k)$，通过重心插值创建新顶点 $\mathbf{p} = \beta_1 \mathbf{v}_i + \beta_2 \mathbf{v}_j + \beta_3 \mathbf{v}_k$，重心坐标初始化为 $(1/3, 1/3, 1/3)$ 并在训练中在单纯形约束下优化。不同表情和姿态下细分点的位置通过同一重心映射重新计算
    - "隐式"含义：不显式创建子面的新顶点和拓扑，而是通过可学习的重心坐标在父面内部隐式定义新点，允许其在三角面内自由移动以适应不同面部区域的最优位置
    - 设计动机：相比显式均匀细分，隐式细分通过可学习重心可以为不同大小和结构的面部区域适配不同的有效尺度和形状

2. **面局部高斯绑定（Face-Local Gaussian Binding）**

    - 功能：将3D高斯绑定到层级中每个面的局部坐标系
    - 核心思路：每个高斯的旋转 $\mathbf{R} = \Delta\mathbf{R}\mathbf{r}$、缩放 $\mathbf{S} = \Delta\mathbf{S} s$、中心 $\boldsymbol{\mu} = s\mathbf{r}\Delta\boldsymbol{\mu} + \mathbf{t}$，其中 $\mathbf{r}$ 是与面法向对齐的旋转，$\mathbf{t}$ 是面重心，$s$ 是三边均长，$\Delta\mathbf{R}, \Delta\mathbf{S}, \Delta\boldsymbol{\mu}$ 是可训练残差
    - 设计动机：面局部参数化保证高斯随面的变形（表情/头部运动）共同移动，在不同层级间保持一致外观

3. **自适应生长（Adaptive Growing）**

    - 功能：在训练中按需扩展层级，将细节集中在最需要的区域
    - 核心思路：仅在当前最细层级 $\ell_{\max}$ 累积屏幕空间梯度 $g_i$，每隔 $k$ 次迭代选择满足 $g_i > \varepsilon$ 的叶面进行细分，为新子面绑定高斯。循环直到达到最大深度 $L$
    - 设计动机：均匀细分无差别扩展所有区域，浪费计算和存储。自适应策略将有限资源集中在五官、毛发等高频区域，平滑区域（额头、脸颊）保持较少高斯即可

4. **重要性排序与渐进传输**

    - 功能：确定每层内高斯的传输优先级
    - 核心思路：每个面的重要性评分定义为其绑定高斯在所有像素上的渲染贡献总和：$W_i = \sum_{j \in \mathcal{G}_i} \sum_p \alpha_{j,p} T_{j,p}$。按评分降序传输，高贡献高斯优先到达
    - 设计动机：先传输高重要性高斯能最小化部分渲染与完整渲染之间的颜色漂移。实验证明（Fig. 3）重要性优先传输显著优于随机传输

### 损失函数 / 训练策略
- 多层级联合监督：$\mathcal{L}_{\text{rgb}} = \sum_{\ell \in \mathcal{S}} w_\ell [(1-\lambda_s)\mathcal{L}_1 + \lambda_s \mathcal{L}_{\text{ssim}}]$
- 粗到细优化：初始化深度上限为1，每50k迭代提升上限并触发自适应细分
- 正则化：$\mathcal{L}_{\text{scale}}$（缩放约束）+ $\mathcal{L}_{\text{pos}}$（位置约束），防止高斯偏离绑定面
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{scale}}\mathcal{L}_{\text{scale}} + \lambda_{\text{pos}}\mathcal{L}_{\text{pos}}$
- Adam 优化器，60k 迭代，每2k次自适应扩展

## 实验关键数据

### 主实验（NeRSemble 数据集，不同传输预算）

| 传输比例 | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ | #高斯 | 传输数据 | FPS |
|---------|----------|----------|-----------|-------|---------|-----|
| 5% (Base) | 27.89 | 0.851 | 0.186 | 10,144 | 2.60MB | 291 |
| 25% | 29.14 | 0.892 | 0.080 | 37,302 | 9.56MB | 278 |
| 50% | 30.03 | 0.904 | 0.073 | 84,132 | 21.56MB | 258 |
| 100% | 31.47 | 0.929 | 0.068 | 169,438 | 43.42MB | 260 |
| GaussianAvatars | 31.10 | 0.937 | 0.064 | 163,829 | 41.90MB | 271 |

### 与 SOTA 对比

| 方法 | NVS PSNR↑ | NVS LPIPS↓ | NES PSNR↑ | NES LPIPS↓ |
|------|----------|-----------|----------|-----------|
| PointAvatar | 25.8 | 0.097 | 23.4 | 0.102 |
| GaussianAvatars | 31.1 | 0.064 | 25.8 | 0.076 |
| Ours (5%) | 27.9 | 0.186 | 25.1 | 0.176 |
| **Ours (100%)** | **31.5** | 0.068 | **25.9** | 0.080 |

### 关键发现
- 仅5%数据（2.6MB）即可获得可用头像（PSNR 27.89），GaussianAvatars 必须等待几乎全部数据才能渲染
- 100%传输时 PSNR 31.47 超过 GaussianAvatars 的 31.10，NES（新表情合成）也略优
- 帧率始终保持在 258-291 FPS（4090, 550×802），高斯数量增加未导致明显帧率下降
- 自适应细分优于均匀细分：同等高斯数量下重建质量更高（Fig. 6），高频区域（胡须）获得更深的细分
- 多层级监督对渐进传输至关重要：仅监督最细层时，低层级无法学到完整头像（Tab. 3 中去掉多层级监督后35%预算下 PSNR 从 29.87 降至 20.06）

## 亮点与洞察
- **从离散 LOD 到连续渐进流**：核心范式转变。传统 LOD 需要多个独立模型且有切换延迟，ProgressiveAvatars 的单一连续资产支持任意传输比例下的即时渲染。对 Social VR 等延迟敏感场景有直接价值
- **自适应隐式细分比均匀细分高效得多**：高频区域（眼睛、嘴巴、胡须）获得更深细分，平滑区域保持浅层细分，实现更好的质量-成本权衡
- **重要性排序保证了渐进质量的单调提升**：先传输高贡献高斯，确保每一步增量都能最大化渲染质量改善
- **面局部绑定+重心映射**保证了在任意传输阶段Avatar都可动画化，这是技术路线选择的关键

## 局限与展望
- 100%传输时的 LPIPS（0.068）略逊于 GaussianAvatars（0.064），感知质量上仍有小差距
- 最大层级深度 $D=4$ 限制了最终能达到的精细度
- 仅在 NeRSemble 数据集上验证，泛化到更多角色/更复杂场景未知
- 重要性评分在训练时预计算且固定，无法根据运行时视角动态调整
- 多人场景中多个Avatar的渐进传输优先级策略未讨论

## 相关工作与启发
- **vs GaussianAvatars**：GPU们共享相似的面绑定高斯设计，但 GaussianAvatars 没有层级结构和渐进传输能力。ProgressiveAvatars 以可比质量实现了从2.6MB即可渲染的渐进体验
- **vs LoDAvatar**：LoDAvatar 也在模板网格上做 LOD，但用均匀细分+手工 mask 选择性密实化。ProgressiveAvatars 的自适应生长更优雅且质量更好
- **vs GA+LightGaussian（离散压缩）**：GA+LG 需要 227.2MB 存10个 LOD 级别，而 ProgressiveAvatars 仅需 43.4MB 的 single asset 支持连续任意比例渲染

## 评分
- 新颖性: ⭐⭐⭐⭐ 从离散LOD到连续渐进流的范式转换有创新性，自适应隐式细分设计精巧
- 实验充分度: ⭐⭐⭐⭐ 渐进传输模拟、消融充分，但仅在一个数据集上验证
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 对 VR/远程呈现等延迟敏感的3D头像传输场景有重要实用价值

<!-- RELATED:START -->

## 相关论文

- [Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)
- [PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](../../ECCV2024/3d_vision/headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)

<!-- RELATED:END -->
