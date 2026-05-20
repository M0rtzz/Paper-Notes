---
title: >-
  [论文解读] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][文本到3D生成] 提出DreamScene360，利用全景图像作为中间表示，结合GPT-4V自精炼机制和全景3D高斯溅射技术，实现从文本到沉浸式360°3D场景的快速生成。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "文本到3D生成"
  - "全景图像"
  - "3D高斯溅射"
  - "场景生成"
  - "扩散模型"
---

# DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2404.06903](https://arxiv.org/abs/2404.06903)  
**代码**: [https://github.com/dreamscene360/dreamscene360](https://github.com/dreamscene360/dreamscene360)  
**领域**: 3D视觉  
**关键词**: 文本到3D生成, 全景图像, 3D高斯溅射, 场景生成, 扩散模型

## 一句话总结

提出DreamScene360，利用全景图像作为中间表示，结合GPT-4V自精炼机制和全景3D高斯溅射技术，实现从文本到沉浸式360°3D场景的快速生成。

## 研究背景与动机

**领域现状**：文本到3D场景生成主要有两条技术路线：(a) 基于Score Distillation Sampling (SDS)的方法（如DreamFusion），通过从2D扩散模型蒸馏先验来优化NeRF/3DGS表示；(b) 基于显式表示的渐进式方法（如LucidDreamer、Text2Room），通过逐步扩展3D表示来覆盖更广的视野。

**现有痛点**：
   - SDS方法渲染质量低，受限于2D模型的多视角不一致性，且难以扩展到场景级别的3D结构
   - 渐进式方法在填充大面积缺失区域时效果差，尤其对于360°场景会产生明显的扭曲和不连贯结构
   - 文本到图像的提示工程问题在3D生成中更加突出，需要大量试错

**核心矛盾**：现有方法缺乏全局一致性的2D场景表示，导致无法在360°范围内保持语义和几何一致性

**本文目标** 从任意文本提示生成全局一致的沉浸式360°3D场景

**切入角度**：以全景图像作为中间表示，既能保证全局一致性，又能利用GPT-4V实现自动提示优化

**核心 idea**：全景图像提供了完整360°场景的全局一致2D表示，结合单目深度初始化和语义/几何正则化，可以高效地提升为3D高斯表示

## 方法详解

### 整体框架

DreamScene360包含三个阶段：(1) 使用扩散模型生成360°全景图，并通过GPT-4V自精炼迭代优化；(2) 利用单目深度估计和可学习几何场对全景图进行2D-to-3D的初始化；(3) 通过语义和几何正则化优化全景3D高斯，填充单视角输入的不可见区域。

### 关键设计

1. **文本到360°全景生成 + 自精炼机制**:

    - **功能**：从文本生成高质量、全局一致的360°全景图像
    - **核心思路**：基于MultiDiffusion的滑窗扩散过程，使用StitchDiffusion确保左右边界连续性。生成分辨率为 $H \times 2H$ 的全景图，去噪过程中每个patch的更新通过加权平均融合：
    $\Phi(I_{t-1}) = \sum_{i=1}^{n} \frac{P_i^{-1}(W_i)}{\sum_{j=1}^{n} P_j^{-1}(W_j)} \otimes P_i^{-1}(\Phi(P_i(I_t)))$
    - 在每个去噪时间步，不仅在原始分辨率上扩散，还将最左和最右区域拼接后一起扩散，确保边界一致性
    - **自精炼**：集成GPT-4V进行多轮自我改进，从用户简单提示出发，GPT-4V对生成图像在对象数量、属性、关系、外观等方面评分(0-10)，并给出改进建议来修改提示词，最终选出最高分的全景图
    - **设计动机**：全景图为后续3D生成提供全局一致的2D表示，GPT-4V消除了手动提示工程的需要

2. **全景几何场初始化**:

    - **功能**：将2D全景图提升为具有一致性的3D点云，作为高斯初始化
    - **核心思路**：将全景图投影到N=20个重叠的透视切线图像，使用DPT单目深度估计器获取每个视角的深度图 $D_i^{\text{Mono}}$。由于单目深度存在仿射歧义，引入可学习的全局几何场(MLP)和每视角的尺度/偏移参数进行全局对齐：
    $\min_{\alpha,\beta,\Theta} \left\{ \|\alpha \cdot D^{\text{Mono}} + \beta - \text{MLPs}(v;\Theta)\|_2^2 + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(\beta) + \lambda_\alpha \|\gamma(\alpha) - 1\|^2 \right\}$
    - 其中 $\alpha_i$ 为每视角尺度参数，$\beta_i$ 为逐像素偏移参数，$\Theta$ 为MLP参数，$\gamma(\cdot)$ 为softplus函数
    - TV损失确保偏移参数的空间平滑性
    - **设计动机**：户外场景没有结构化布局先验，需要通过可变形对齐实现跨视角的尺度一致性

3. **虚拟相机合成视差 + 语义/几何正则化**:

    - **功能**：解决单视角全景缺乏视差信息的问题，填充不可见区域
    - **核心思路**：通过在全景视点坐标上引入渐进式扰动来合成虚拟相机：
    $(x', y', z') = (x, y, z) + \delta(d_x, d_y, d_z)$
      扰动范围为 $[-0.05, +0.05] \times \gamma$，$\gamma \in \{1, 2, 4\}$ 表示3阶段渐进扰动
    - **语义正则化**：使用DINOv2的[CLS]特征，约束训练视角与虚拟视角的语义一致性：
    $\mathcal{L}_{\text{sem}} = 1 - \text{Cos}([\text{CLS}](I_i), [\text{CLS}](I_i'))$
    - **几何正则化**：使用DPT估计渲染图像深度，通过Pearson相关正则化渲染深度的相对关系：
    $\mathcal{L}_{\text{geo}}(I_i, D_i) = 1 - \frac{\text{Cov}(D_i, \text{DPT}(I_i))}{\sqrt{\text{Var}(D_i) \cdot \text{Var}(\text{DPT}(I_i))}}$
    - **设计动机**：单视角全景没有视差信息(无法通过双目视差感知深度)，需要通过虚拟视角结合2D模型先验来弥补

### 损失函数 / 训练策略

总体损失函数：
$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_1 \cdot \mathcal{L}_{\text{sem}} + \lambda_2 \cdot \mathcal{L}_{\text{geo}}$$

- $\mathcal{L}_{\text{RGB}}$：光度损失，包含L1和D-SSIM项
- $\lambda_1 = \lambda_2 = 0.05$
- 输入全景分辨率 $1024 \times 2048$
- 禁用3DGS的自适应密度控制(densification)，因为已有高质量点云初始化
- FoV设为80°进行几何场优化

## 实验关键数据

### 主实验

| 指标 | DreamScene360 | LucidDreamer | 说明 |
|------|--------------|--------------|------|
| CLIP Distance ↓ | **0.8732** | 0.8900 | 文本-图像对齐 |
| Q-Align ↑ | **3.1094** | 3.0566 | SOTA感知质量评估 |
| NIQE ↓ | **4.9165** | 6.2305 | 无参考图像质量 |
| BRISQUE ↓ | **38.3911** | 51.9764 | 无参考图像质量 |
| 运行时间 | 7min 20sec | 6min 15sec | 略慢但可接受 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅光度损失 | 虚拟视角出现伪影 | 缺少遮挡区域的约束 |
| + 几何正则化 | 伪影减少 | 深度一致性提升 |
| + 语义正则化 | 伪影减少 | 高层语义补充 |
| 完整模型 | **最优视觉质量** | 两种正则化互补 |
| 随机初始化 | 模糊结果 | 缺乏几何先验 |
| 单目深度+对齐 | **清晰一致** | 良好的初始化关键 |

### 关键发现
- LucidDreamer的渐进式修复容易在复杂场景中产生重复内容（如一个卧室被复制成多个卧室）
- GPT-4V自精炼能显著提升全景图的视觉质量和细节丰富度
- 禁用3DGS的densification反而有助于提升质量和加速收敛

## 亮点与洞察
- **全景图作为中间表示**是极好的设计选择——它自然解决了360°场景的全局一致性问题，同时使得GPT-4V的质量评估成为可能（之前的方法没有全局2D表示，无法做此类评估）
- 整个pipeline实现了"一键式"3D场景生成，约7分钟完成，用户体验显著优于需要反复调试的SDS方法
- 将单目深度的仿射歧义问题转化为可学习的几何场优化，是实用且优雅的解决方案

## 局限与展望
- 生成分辨率受限于预训练全景扩散模型的默认分辨率（$512 \times 1024$）
- 未来可探索更高分辨率生成和4D动态场景扩展
- 场景的几何精度仍受限于单目深度估计的质量

## 相关工作与启发
- **vs LucidDreamer**: LucidDreamer通过渐进式修复扩展视角，但无法保证360°全局一致性；DreamScene360通过全景中间表示解决了这个根本问题
- **vs DreamFusion**: DreamFusion使用SDS蒸馏，生成速度慢且质量低；DreamScene360避免了耗时的score distillation过程
- **vs Text2Room**: Text2Room使用网格表示，渐进式构建室内场景；DreamScene360支持室内外无约束场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 全景图作为中间表示+GPT-4V自精炼的设计简洁有效
- 实验充分度: ⭐⭐⭐ 仅与LucidDreamer对比，缺乏更多baseline和定量消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，图示优质
- 价值: ⭐⭐⭐⭐ 为360°场景生成提供了实用的端到端方案，有明确的工业应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] GVGEN: Text-to-3D Generation with Volumetric Representation](gvgen_text-to-3d_generation_with_volumetric_representation.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
