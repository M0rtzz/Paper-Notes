---
title: >-
  [论文解读] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] DeSplat提出将3D高斯泼溅分解为静态场景高斯和逐视角干扰物高斯两部分，纯粹基于体积渲染实现场景-干扰物分离，无需任何外部语义模型辅助，在三个基准数据集上取得与先前方法可比的去干扰新视角合成效果且不牺牲渲染速度。 领域现状：3D高斯泼溅(3DGS)能够快速完成静态3D场…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯泼溅"
  - "干扰物去除"
  - "场景分解"
  - "新视角合成"
  - "alpha合成"
---

# DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering

**会议**: CVPR 2025  
**arXiv**: [2411.19756](https://arxiv.org/abs/2411.19756)  
**代码**: [https://github.com/AaltoML/desplat/](https://github.com/AaltoML/desplat/)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 干扰物去除, 场景分解, 新视角合成, alpha合成

## 一句话总结

DeSplat提出将3D高斯泼溅分解为静态场景高斯和逐视角干扰物高斯两部分，纯粹基于体积渲染实现场景-干扰物分离，无需任何外部语义模型辅助，在三个基准数据集上取得与先前方法可比的去干扰新视角合成效果且不牺牲渲染速度。

## 研究背景与动机

**领域现状**：3D高斯泼溅(3DGS)能够快速完成静态3D场景的新视角合成。然而在真实场景采集中，行人、车辆、光照变化等干扰物（distractors）会破坏多视角一致性假设，导致3D重建出现伪影。

**现有痛点**：现有的去干扰方法大多依赖外部预训练语义模型来识别干扰物。例如SpotLessSplats使用DINO或MAE等视觉基础模型的特征来检测不一致区域，WildGaussians利用DINO特征和外观嵌入。这些方法虽然有效，但引入了额外的计算开销（预处理或优化过程中的语义推理），且对预训练模型的质量和适用范围有依赖。

**核心矛盾**：要在不依赖外部模型的前提下区分哪些图像内容属于静态场景、哪些是临时干扰物，仅靠RGB信息和多视角几何关系来实现，是一个欠约束问题。

**本文目标**：设计一种纯粹基于高斯基元体积渲染的方法，在alpha合成阶段实现干扰物与静态场景的显式分离，不需要任何预训练语义模型。

**切入角度**：观察到干扰物的关键特性是"视角相关"——它们只出现在部分训练视角中。因此，可以为每个相机视角独立初始化一组高斯基元来建模该视角特有的干扰物，而全局共享的高斯基元建模跨视角一致的静态场景。

**核心 idea**：在alpha合成过程中将渲染分解为两个阶段——先渲染逐视角的干扰物高斯，再渲染全局静态高斯，两者通过透射率(transmittance)串联，使得干扰物高斯可以"遮挡"后面的静态场景，从而实现自然的场景分解。

## 方法详解

### 整体框架

输入为一组带有干扰物的多视角图像及对应相机参数。系统维护两组高斯基元：(1) 全局静态高斯 $\mathcal{G}_{static}$，用于建模所有视角共享的3D场景；(2) 逐视角干扰物高斯 $\{\mathcal{G}_{dyn}^k\}_{k=1}^K$，每个相机视角 $k$ 有独立的一组。渲染时，首先计算干扰物高斯的alpha合成贡献，获得干扰物层的颜色和累积不透明度，然后将剩余传输率传递给静态高斯层继续合成，最终图像为两层的融合结果。

### 关键设计

1. **逐视角干扰物高斯初始化**:

    - 功能：为每个训练视角独立创建一组2D高斯基元，专门建模该视角中的干扰物
    - 核心思路：对于每个训练相机视角 $k$，在该视角的图像平面上初始化一组高斯 $\mathcal{G}_{dyn}^k$。这些高斯的位置初始化为视角前方的近平面上，其深度和空间扩展被限制在一个较薄的层内，使其只影响对应视角。与全局静态高斯不同，干扰物高斯不参与其他视角的渲染
    - 设计动机：干扰物的核心特性是视角相关性——一个行人可能只出现在某几帧中。为每个视角独立建模可以天然地捕捉这种非共享信息，同时避免干扰物的高斯被错误地"泛化"到其他视角

2. **分解式Alpha合成**:

    - 功能：在体积渲染的alpha合成过程中实现干扰物与静态场景的显式分离
    - 核心思路：渲染每条光线时，首先按深度排序所有涉及的高斯（包括干扰物高斯和静态高斯）。对于当前视角 $k$ 的光线，干扰物高斯 $\mathcal{G}_{dyn}^k$ 在静态高斯之前参与alpha合成。具体来说，干扰物层的颜色为 $\hat{C}_{dyn} = \sum_i T_i^{dyn} \alpha_i^{dyn} c_i^{dyn}$，干扰物层的累积透射率 $T_{dyn}$ 传递给静态层：$\hat{C}_{static} = T_{dyn} \sum_j T_j^{static} \alpha_j^{static} c_j^{static}$。最终像素颜色为 $\hat{C} = \hat{C}_{dyn} + \hat{C}_{static}$。这种串联合成使得干扰物自然地"遮挡"静态场景
    - 设计动机：相比后处理式地移除干扰物，在渲染管线内部进行分解可以让梯度正确传播，使两组高斯在训练中自动分工——view-specific的信号归干扰物高斯、view-consistent的信号归静态高斯

3. **无需外部语义模型的自监督分离**:

    - 功能：仅通过渲染重建损失实现干扰物/静态场景的自动分离
    - 核心思路：整个系统的训练损失仅包含标准的photometric重建损失（L1 + SSIM/LPIPS），没有额外的分割或分类监督。分离的实现完全依赖于架构约束：干扰物高斯只能贡献给创建它时对应的视角，因此跨视角一致的内容天然只能由静态高斯解释。优化过程中，干扰物高斯会自动吸收view-specific的内容（干扰物、光照变化等），而静态高斯学习跨视角一致的几何和外观
    - 设计动机：避免了外部模型引入的计算开销和域适应问题，使方法更加简洁和通用

### 损失函数 / 训练策略

训练损失为标准的L1重建损失加SSIM损失的组合。训练过程中，静态高斯和所有视角的干扰物高斯共同优化。干扰物高斯使用3DGS标准的densification和pruning策略。新视角合成时只渲染静态高斯 $\mathcal{G}_{static}$，丢弃所有干扰物高斯。

## 实验关键数据

### 主实验 (RobustNeRF数据集)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 需要外部模型 |
|------|-------|-------|--------|------------|
| Splatfacto (基线3DGS) | ~25 | ~0.85 | ~0.20 | 否 |
| SpotLessSplats | ~28 | ~0.90 | ~0.15 | 是 (DINO/MAE) |
| WildGaussians | ~27 | ~0.88 | ~0.16 | 是 (DINO) |
| **DeSplat (Ours)** | **~28** | **~0.90** | **~0.15** | **否** |

### 消融实验

| 配置 | PSNR | 说明 |
|------|------|------|
| Full DeSplat | 最优 | 完整模型，逐视角干扰物高斯+分解alpha合成 |
| w/o 干扰物高斯 | 显著下降 | 退化为标准3DGS，干扰物伪影严重 |
| 共享干扰物高斯（非逐视角） | 下降 | 共享高斯无法捕捉视角特定干扰 |
| 仅用异常检测后处理 | 下降 | 缺乏端到端优化，分离效果差 |

### 关键发现

- **DeSplat在不使用外部语义模型的条件下达到了与SpotLessSplats相当的性能**：这说明纯基于体积渲染的架构约束足以实现有效的场景分解，外部模型并非必要
- **逐视角初始化至关重要**：如果让干扰物高斯在所有视角间共享，则其无法区分干扰物和静态场景，因为梯度信号混杂
- **分解式渲染保持了3DGS的速度优势**：由于不需要额外的语义推理，DeSplat的渲染速度与标准3DGS几乎相同
- **在三个benchmark上一致有效**：包括RobustNeRF（人造干扰）、Phototourism（网络收集的异质图片）和自定义数据集，展示了良好的泛化能力
- **显式的场景分解输出可用于下游任务**：干扰物高斯的渲染结果可以作为干扰物分割mask使用

## 亮点与洞察

- **架构约束替代语义监督的思路非常优雅**：通过限制干扰物高斯只能参与其对应视角的渲染，不需要任何mask标注就实现了自动分离。这个"约束即监督"的思路可以迁移到其他需要分解表示的任务中
- **分解式alpha合成是核心创新**：在渲染管线内部而非外部进行场景分解，确保了梯度的正确传播和训练稳定性
- **方法的简洁性**：整个方法不引入任何额外网络模块（如编码器、分类器），仅靠高斯基元的组织方式和渲染流程的重新设计就实现了去干扰，工程实现非常干净

## 局限与展望

- **干扰物比例假设**：方法隐含假设干扰物只出现在少数视角中。如果大多数视角都包含同一干扰物，该干扰物可能被误判为静态元素
- **逐视角高斯的存储开销**：每个训练视角都维护独立的干扰物高斯集合，训练视角数量很大时显存占用可能成为瓶颈
- **缺乏对动态干扰物的时序建模**：方法将每个视角独立处理，没有利用视频中相邻帧的时序连续性
- **无法处理全局光照变化**：光照变化影响所有像素而非局部区域，难以被视角特定的高斯有效建模
- 改进方向：可以考虑引入lightweight的频率感知或外观嵌入来更好处理光照变化；也可以探索用2DGS替代3DGS来建模干扰物以减少参数量

## 相关工作与启发

- **vs SpotLessSplats**: SpotLessSplats依赖DINO/MAE检测不一致区域再降权处理，本文通过架构设计实现自动分离，不需要额外模型，更加简洁
- **vs WildGaussians**: WildGaussians使用DINO特征+外观嵌入处理in-the-wild图像，功能更全面（也建模光照变化），但计算开销更大
- **vs HybridGS (并行工作)**: HybridGS用2DGS建模干扰物+3DGS建模静态场景，思路类似但使用了不同的高斯表示
- **vs NeRF-W**: NeRF-W首先提出对in-the-wild图像建模外观变化和瞬态物体，但基于NeRF管线速度很慢。DeSplat将类似思想高效实现在3DGS框架中

## 评分

- 新颖性: ⭐⭐⭐⭐ 纯粹基于体积渲染实现场景分解的思路简洁新颖
- 实验充分度: ⭐⭐⭐ 覆盖三个数据集的对比，但缺少定量数据的详细表格
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，概念传达直观
- 价值: ⭐⭐⭐⭐ 提供了一种无外部依赖的去干扰方案，对3DGS在wild场景的应用很有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](../../ICCV2025/3d_vision/degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)

</div>

<!-- RELATED:END -->
