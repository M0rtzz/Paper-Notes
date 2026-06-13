---
title: >-
  [论文解读] Collaborative Control for Geometry-Conditioned PBR Image Generation
description: >-
  [ECCV 2024][图像生成][PBR材质生成] 提出 Collaborative Control 范式，通过冻结预训练RGB扩散模型并训练一个并行PBR模型，利用双向跨网络通信层联合建模RGB与PBR图像分布，在有限数据下实现高质量的几何条件PBR材质图像生成。 领域现状：扩散模型在RGB图像生成领域取得了巨大成功…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "PBR材质生成"
  - "多模态扩散"
  - "跨网络控制"
  - "几何条件生成"
  - "物理渲染"
---

# Collaborative Control for Geometry-Conditioned PBR Image Generation

**会议**: ECCV 2024  
**arXiv**: [2402.05919](https://arxiv.org/abs/2402.05919)  
**代码**: [https://unity-research.github.io/holo-gen](https://unity-research.github.io/holo-gen) (项目页面)  
**领域**: 扩散模型 / 图像生成  
**关键词**: PBR材质生成, 多模态扩散, 跨网络控制, 几何条件生成, 物理渲染

## 一句话总结

提出 Collaborative Control 范式，通过冻结预训练RGB扩散模型并训练一个并行PBR模型，利用双向跨网络通信层联合建模RGB与PBR图像分布，在有限数据下实现高质量的几何条件PBR材质图像生成。

## 研究背景与动机

**领域现状**：扩散模型在RGB图像生成领域取得了巨大成功，Text-to-3D和Text-to-Texture方法也成功将其扩展到3D内容生成。然而，下游3D工作流（如游戏引擎）需要的是PBR（Physically-Based Rendering）材质，而非简单的RGB图像。

**现有痛点**：

**逆渲染的固有缺陷**：当前方法先生成RGB图像再通过逆渲染提取PBR属性，但扩散模型生成的RGB图像光照不符合物理规律（模型偏好理想化和艺术化外观），导致逆渲染结果存在严重歧义

**数据稀缺**：最大的PBR数据集Objaverse仅有~80万对象，比LAION-5B（50亿）小几个数量级，直接训练生成模型会导致泛化能力不足

**高维度困境**：PBR图像包含Albedo（3通道）、Metallic（1通道）、Roughness（1通道）和Bump Map（3通道），共8通道，无法良好压缩进现有RGB VAE的低维潜空间

**微调导致灾难性遗忘**：在有限PBR数据上微调预训练RGB模型会丧失泛化能力

**核心矛盾**：如何在数据极度稀缺的条件下，利用预训练RGB模型的丰富先验知识，直接建模PBR图像的联合分布？

**切入角度**：保持预训练RGB模型完全冻结，训练一个并行的PBR模型，通过双向跨网络通信机制紧密关联两个模型，使PBR模型既能从RGB模型获取语义信息，又能引导RGB模型生成与PBR对齐的渲染图像。

**核心idea**：将联合逆过程分解为两个耦合过程——RGB模型生成渲染图像并提供丰富的内部表示，PBR模型利用这些表示生成对应的PBR材质。

## 方法详解

### 整体框架

系统包含两个并行运行的扩散模型：
- **左分支**：预训练的冻结RGB扩散模型 $\mathcal{D}_{rgb}$，生成渲染后的RGB图像
- **右分支**：新训练的PBR扩散模型 $\mathcal{D}_{pbr}$，生成PBR材质图像

两个模型在每个self-attention层后通过跨网络通信层连接，实现双向信息交换。PBR模型的输入还拼接了屏幕空间几何法线作为条件。

### 关键设计

1. **Collaborative Control 双向通信机制**：

    - **功能**：在两个模型的每个self-attention模块后插入连接层，实现双向信息流
    - **核心思路**：将两个模型的隐状态拼接，通过一个简单的逐像素线性层处理，然后将结果残差地分配回两个模型：
    $h_{rgb}' = h_{rgb} + \text{Linear}([h_{rgb}; h_{pbr}])$
    $h_{pbr}' = h_{pbr} + \text{Linear}([h_{rgb}; h_{pbr}])$
    - **设计动机**：PBR分支需要从RGB模型中提取相关信息，同时引导RGB输出趋向渲染图像域 $\text{Im}(f)$。单向通信（如ControlNet）无法让RGB模型对齐到条件分布，顺时针通信（如AnimateAnyone）无法让PBR模型在编码器阶段获取 $z'_{rgb,t-1}$。实验证明双向通信是不可或缺的。

2. **PBR专用VAE**：

    - **功能**：训练一个专门用于PBR图像压缩的VAE，潜空间维度设为14通道
    - **核心思路**：采用StableDiffusion v1.5的VAE架构，但将潜空间通道数从4扩展到14，以平衡PBR图像（8通道）的质量与压缩比
    - **设计动机**：PBR图像的分布与RGB差异巨大，直接使用RGB VAE编码PBR通道三元组会导致严重的分布不匹配，实验证实CMMD指标从6.30急剧恶化到84.66

3. **几何切线空间Bump Map表示**：

    - **功能**：将bump map定义在仅依赖于几何体的切线空间中，而非传统的UV切线空间
    - **核心思路**：对于点 $\bm{p}$ 和几何法线 $\bm{n}$，构造局部切线向量 $\bm{t} = \bm{n} \times ([-p_y, p_x, 0]^T \times \bm{n})$
    - **设计动机**：UV切线空间依赖于任意的UV展开，导致世界空间中相似的表面凹凸在UV空间中表现迥异，解耦纹理与UV映射有助于模型学习

4. **禁用PBR分支的文本交叉注意力**：

    - **功能**：在PBR模型中关闭prompt交叉注意力，所有文本引导仅通过冻结的RGB模型流入
    - **设计动机**：在有限数据上PBR模型的文本注意力层容易过拟合，数据越少效果越差。强制文本注意力通过冻结的RGB模型可防止过拟合

### 损失函数 / 训练策略

- 联合优化RGB和PBR去噪的训练损失，仅更新PBR模型权重和跨网络通信层权重
- RGB模型使用固定环境贴图和固定相机设置渲染，简化对齐问题
- 训练数据：Objaverse过滤后约30万对象，每个从16个视角渲染
- 训练配置：512×512分辨率，200K步，batch=12，lr=3e-5，单张A100约2天

## 实验关键数据

### 主实验（通信范式对比）

| 通信方式 | CMMD(PBR)↓ | CMMD(Relit)↓ | FID(PBR)↓ | CLIPScore(Albedo)↑ | CLIPScore(Relit)↑ |
|---------|-----------|-------------|----------|-------------------|------------------|
| One-way | 16.44 | 13.38 | 20.90 | 23.08 | 23.40 |
| Clockwise | 6.78 | 2.76 | 12.21 | 26.45 | 24.53 |
| **Bi-directional** | **6.30** | **1.79** | **11.65** | **26.76** | **25.41** |

### 消融实验

| 配置 | CMMD(PBR)↓ | FID(PBR)↓ | CLIPScore↑ | 说明 |
|-----|-----------|----------|-----------|------|
| PBR VAE | 6.30 | 11.65 | 26.76 | 专用VAE（基线） |
| RGB VAE三元组 | 84.66 | 25.81 | 25.27 | 分布不匹配严重 |
| Fine-tuning(含RGB) | 13.40 | 14.42 | 25.04 | OOD性能下降 |
| Fine-tuning(不含RGB) | 5.25 | 11.41 | 25.66 | OOD严重过拟合 |
| Pixel-wise MLP | 5.43 | 11.43 | 27.15 | 略优但更复杂 |
| Global Attention | 7.60 | 13.61 | 24.50 | 缺乏像素对应 |

**数据效率实验**（无PBR prompt attention）：

| 训练数据比例 | CMMD(PBR)↓ | FID(PBR)↓ | 说明 |
|------------|-----------|----------|------|
| 1% (~6万图) | 6.25 | 11.87 | 仅数千对象仍可工作 |
| 5% | 5.77 | 11.49 | 接近完整数据性能 |
| 98% | 6.30 | 11.65 | 完整训练集 |

### 关键发现

- 双向通信对PBR生成至关重要，单向通信的one-way方案甚至无法对齐对象位置
- 简单的逐像素线性层即可作为通信层，MLP和注意力机制并无显著优势
- 禁用PBR分支的prompt注意力对OOD泛化至关重要，尤其在小数据集上
- 方法数据效率极高，仅用1%数据也能生成合理的PBR材质
- 与IPAdapter兼容，因为RGB模型完全冻结

## 亮点与洞察

- **冻结+并行**的设计范式非常优雅：既利用了预训练模型的先验，又不破坏其权重，还保持了与第三方控制技术的兼容性
- 用贝叶斯规则分解联合逆过程的理论动机清晰
- 几何切线空间bump map的设计考虑了实际应用中UV映射的任意性

## 局限与展望

- 最常见的失败案例是roughness、metallic或bump map生成为常数图（缺乏细节）
- 训练数据仅来自Objaverse，限制了对真实场景的泛化
- 固定环境贴图和相机的简化可能限制了某些应用场景
- 仅在StableDiffusion 1.5/2.1上验证，未扩展到更大模型

## 相关工作与启发

- **ControlNet/ControlNet-XS**：控制范式的单向和半双向变体，本文证明完全双向对PBR任务不可或缺
- **AnimateAnyone**：仅单向通信（控制模型→生成），不适用于需要双向信息流的场景
- **Wonder3D/UniDream**：跨域self-attention方案，但随模态数增加扩展性差
- 启发：对于需要在有限数据上训练新模态生成的任务，冻结基础模型+并行分支+双向通信是一个通用的有效范式

## 评分
- 新颖性: ⭐⭐⭐⭐ Collaborative Control范式新颖，解决了PBR生成的实际痛点
- 实验充分度: ⭐⭐⭐⭐⭐ 消融非常全面，涵盖通信方式/类型/VAE/数据量/分辨率/训练预算等
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，动机和实验组织良好
- 价值: ⭐⭐⭐⭐ 对3D内容生成管线有直接实用价值，范式可推广到其他多模态生成任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation](anycontrol_create_your_artwork_with_versatile_control_on_text-to-image_generatio.md)
- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)

</div>

<!-- RELATED:END -->
