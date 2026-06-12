---
title: >-
  [论文解读] Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling
description: >-
  [CVPR 2025][医学图像][虚拟试穿] 本文提出通过人体图像反向提取合成服装对来增强虚拟试穿训练数据，并设计了基于错误感知噪声调度的Schrödinger Bridge精炼模型（EARSB），对已有试穿模型的生成结果进行局部纠错…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "虚拟试穿"
  - "合成数据增强"
  - "Schrödinger Bridge"
  - "错误感知精炼"
  - "扩散模型"
---

# Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling

**会议**: CVPR 2025  
**arXiv**: [2501.04666](https://arxiv.org/abs/2501.04666)  
**代码**: 有（论文中提到 code available）  
**领域**: 医学图像  
**关键词**: 虚拟试穿, 合成数据增强, Schrödinger Bridge, 错误感知精炼, 扩散模型

## 一句话总结
本文提出通过人体图像反向提取合成服装对来增强虚拟试穿训练数据，并设计了基于错误感知噪声调度的Schrödinger Bridge精炼模型（EARSB），对已有试穿模型的生成结果进行局部纠错，在VITON-HD和DressCode上取得了SOTA效果且用户更偏好本文结果（59%）。

## 研究背景与动机

1. **领域现状**：虚拟试穿旨在生成将目标服装穿在目标人身上的真实感图像，近年来从GAN方法转向扩散模型方法，取得了显著进展。

2. **现有痛点**：当前方法面临两个核心挑战——(a) 配对训练数据（人体图像+对应产品视图服装）数量有限，版权保护限制了大规模数据获取；(b) 生成的服装纹理常出现伪影，如文字扭曲、纹理褪色等局部错误。

3. **核心矛盾**：需要更多样化的训练数据来覆盖人体姿态、肤色和服装属性的组合空间，但高质量配对数据获取成本极高；同时，基础模型的局部生成错误难以通过端到端训练消除。

4. **本文目标**：(1) 如何低成本获取更多训练配对数据？(2) 如何针对性地修复基础试穿模型的局部生成伪影？

5. **切入角度**：作者观察到人体→服装的反向任务（从穿着照中提取服装正面图）比正向任务更简单，可以用来制造合成配对数据；同时受经典boosting思想启发，构建级联精炼模型专门修复前序模型的错误。

6. **核心 idea**：用反向人→服装模型生成合成数据增强训练 + 用弱监督错误分类器引导的空间自适应Schrödinger Bridge精炼生成结果。

## 方法详解

整个方法由两个相互独立但可互补的部分组成：合成数据增强和EARSB精炼模型。

### 整体框架
输入为掩码人体图像、服装图像和姿态表示，先用基础试穿模型生成初始结果，然后通过弱监督错误分类器定位伪影区域，最后用空间自适应噪声调度的Schrödinger Bridge对错误区域进行定向精炼。合成数据增强在训练阶段独立使用，为任何试穿模型提供额外训练对。

### 关键设计

1. **Human-to-Garment 合成数据生成**:

    - 功能：从单张穿着图像生成（人体, 合成服装）配对数据
    - 核心思路：训练一个服装提取模型，先用分割模型从人体图像中提取clothing region，再用基于flow机制的UNet生成其正面标准视图。发出合成数据后按三条标准过滤——背景干净、正面视角、LPIPS重建误差小。最终从DeepFashion2和UPT构建了12,730个上半身和8,939个全身合成配对。训练时采用实/合成flag条件化策略，优于预训练+微调的两阶段方案。
    - 设计动机：产品视图服装图像受版权保护难以大规模获取，而单人图像容易获得。利用人→服装的对称性，将数据获取成本从需要版权图推到只需单张人像。

2. **弱监督错误分类器 (WSC)**:

    - 功能：定位基础试穿模型生成结果中的局部伪影区域，输出错误置信度热力图 $M$
    - 核心思路：双编码器架构，分别编码初始生成图 $x_1$ 和服装 $C$，通过交叉注意力预测sigmoid激活的错误图。训练使用图像级和patch级联合损失——图像级损失 $\mathcal{L}_{img}$ 通过max-pooling让合成图的最大错误分数高、真实图分数低；patch级损失 $\mathcal{L}_{pat}$ 利用少量手工标注的bounding box在标注区域内最大化、区域外最小化错误分数。
    - 设计动机：完整标注所有伪影区域代价极高，弱监督只需几小时的patch级标注即可训练有效的错误定位器，且可针对特定基础模型的错误模式定制。

3. **Error-Aware Refinement Schrödinger Bridge (EARSB)**:

    - 功能：利用错误图引导的空间自适应噪声调度，定向精炼初始生成图像的伪影区域
    - 核心思路：基于I2SB框架构建从初始图 $x_1$ 到真实图 $x_0$ 的Schrödinger Bridge。关键创新是将噪声 $\epsilon$ 替换为 $\epsilon^r = M \cdot \epsilon$，即用错误图 $M$ 空间缩放噪声——正确区域几乎不加噪（直接复制像素），错误区域加大噪声（允许更多修改自由度）。采样过程也相应地在每步将噪声乘以 $M$。去噪网络使用cloth-flow-learning UNet实现更精确的服装变形。进一步引入WSC分类器指导（类似classifier guidance）和expert denoiser（在 $t \in [0,0.5]$ 和 $t \in [0.5,1]$ 分别微调）。
    - 设计动机：朴素I2SB需隐式学习"什么该修什么该留"，而用错误图显式控制噪声分布可将精炼聚焦到伪影区域，正确区域直接保留，既提高精炼效果又节省采样步数。

### 损失函数 / 训练策略
EARSB使用MSE损失: $\mathcal{L}_{EARSB} = \mathbb{E}_{t \sim U(0,1)} \|\epsilon_\theta^r(M,P,x_t,C;t) - \epsilon^r\|^2$，其中 $\epsilon^r = M \cdot \epsilon$。WSC使用图像级+patch级联合损失。训练分三步：先用基础模型生成初始图→用WSC生成错误图→用自适应噪声训练EARSB。训练后将EARSB拆分为两个expert denoiser分别负责不同时间范围。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | EARSB(SD)+H2G | StableVITON | IDM-VTON | 提升 |
|--------|------|------|--------------|-------------|----------|------|
| VITON-HD | unpaired | FID↓ | **8.04** | 8.20 | 8.59 | 2.0% |
| VITON-HD | paired | SSIM↑ | **0.925** | 0.917 | 0.902 | 0.9% |
| VITON-HD | paired | LPIPS↓ | **0.053** | 0.057 | 0.061 | 7.0% |
| DressCode | unpaired | FID↓ | **10.41** | - | 11.09 | 6.1% |
| DressCode | paired | SSIM↑ | **0.968** | - | 0.956 | 1.3% |

### 消融实验

| 配置 | 用户偏好率 | 说明 |
|------|-----------|------|
| EARSB vs GP-VTON | 59.5% | 精炼GAN基础模型显著提升 |
| EARSB vs StableVITON | 58.5% | 精炼扩散基础模型也有效 |
| +合成数据(H2G-UH/FH) | FID进一步↓ | 合成数据一致性提升各指标 |
| 不同采样步数 | 5步仍稳定 | EARSB在少步数下性能降低小 |

### 关键发现
- EARSB可以级联在不同基础模型之上（GAN或扩散），均能带来提升，体现了精炼框架的通用性
- 合成数据增强对不同基础模型和数据集均有一致性提升
- 空间自适应噪声调度使EARSB在低采样步数下（5步、10步）仍保持稳定性能，优于其他需要25+步的扩散方法
- 用户研究中，本文方法在纹理一致性和图像保真度两方面均被大多数用户偏好

## 亮点与洞察
- **空间自适应Schrödinger Bridge**：将错误图融入噪声调度是非常优雅的设计，让扩散过程"按需精炼"。这个思路可迁移到任何需要局部编辑的图像任务。
- **弱监督错误检测器**：只需几小时标注就能训练出有效的伪影定位器，且可针对特定模型定制，实用性强。
- **对称任务利用**：将人→服装的反向任务作为数据增强手段，巧妙地绕过了版权数据获取问题。

## 局限与展望
- 合成服装质量受限于human-to-garment模型，复杂纹理/多层服装的提取效果可能不佳
- WSC需要针对每个基础模型分别标注和训练，跨模型迁移性有待验证
- EARSB增加了额外的推理时间（需要先跑基础模型再精炼），实时应用受限
- 仅评估了上装试穿，下装/全身/多服装场景未涉及

## 相关工作与启发
- **vs StableVITON/IDM-VTON**: 这些方法直接端到端生成，本文采用"基础+精炼"的级联策略，类似boosting思想，可以持续叠加提升
- **vs CAT-DM**: CAT-DM也用GAN初始图+小噪声初始化扩散，但噪声调度是全局统一的；本文的空间自适应噪声调度更精细
- **vs I2SB**: 本文在I2SB基础上引入空间变化的噪声，从"全局精炼"进化为"局部精炼"

## 评分
- 新颖性: ⭐⭐⭐⭐ 空间自适应SB和合成数据增强各有新意，但整体是组合创新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+用户研究+消融+采样效率分析，较全面
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，流程描述通顺
- 价值: ⭐⭐⭐⭐ 精炼框架通用性强，对虚拟试穿应用有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)
- [\[CVPR 2026\] RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](../../CVPR2026/medical_imaging/rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation](enhancing_sam_with_efficient_prompting_and_preference_optimization_for_semi-supe.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)

</div>

<!-- RELATED:END -->
