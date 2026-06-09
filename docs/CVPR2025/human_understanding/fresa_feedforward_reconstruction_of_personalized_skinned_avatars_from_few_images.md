---
title: >-
  [论文解读] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images
description: >-
  [CVPR 2025][人体理解][Avatar重建] 提出 FRESA，通过学习一个通用着装人体先验模型，从少量图像前馈式（18秒）联合推理个性化 canonical 形状、蒙皮权重和姿态依赖变形，实现零样本泛化到手机照片的高质量可动画化 3D 人体 Avatar 重建。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "Avatar重建"
  - "前馈推理"
  - "个性化蒙皮"
  - "线性混合蒙皮"
  - "可动画化"
---

# FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images

**会议**: CVPR 2025  
**arXiv**: [2503.19207](https://arxiv.org/abs/2503.19207)  
**代码**: [https://github.com/rongakowang/FRESA](https://github.com/rongakowang/FRESA)  
**领域**: 人体理解  
**关键词**: Avatar重建, 前馈推理, 个性化蒙皮, 线性混合蒙皮, 可动画化

## 一句话总结

提出 FRESA，通过学习一个通用着装人体先验模型，从少量图像前馈式（18秒）联合推理个性化 canonical 形状、蒙皮权重和姿态依赖变形，实现零样本泛化到手机照片的高质量可动画化 3D 人体 Avatar 重建。

## 研究背景与动机

**领域现状**：3D 着装人体重建已取得显著进展（PIFu、ICON 等），但大多只重建单帧静态形状。想要获得可动画化的 Avatar，需要在 canonical 空间重建几何并配合蒙皮权重通过 LBS 驱动动画。

**现有痛点**：当前可动画 Avatar 重建方法存在两类主要问题。(1) ARCH++等前馈方法虽然快速，但使用模板身体的最近邻蒙皮权重来绑定 Avatar，这在极端姿态和体型下会产生变形伪影（如腋下三角形过度拉伸）。(2) 一些方法尝试联合优化个性化蒙皮权重，但缺乏跨体型/服装类型的统一先验，只能逐人优化，需要数小时的测试时间。

**核心矛盾**：个性化蒙皮权重对动画质量至关重要（不同体型、不同服装需要不同的蒙皮策略），但学习这样的权重需要大量多样化数据来建立通用先验。同时，canonical 形状和蒙皮权重之间存在耦合歧义——错误的 canonical 形状配合错误的蒙皮权重可能意外地产生正确的 posed 形状。

**本文目标** 如何在不做逐人优化的情况下，从少量图像前馈式地联合推理个性化的 canonical 几何、蒙皮权重和姿态依赖变形？

**切入角度**：作者收集了超过 1100 个穿着不同类型衣物的受试者的大规模 dome 捕获数据集，每人多达 100 个姿态，学习跨体型和服装类型的通用先验。通过显式的 3D canonicalization 产生像素对齐的初始条件，使特征提取更容易；通过多帧聚合消除 canonicalization 伪影并融合人物本征信息。

**核心 idea**：用千人规模数据学通用先验，通过 3D canonicalization + 多帧聚合 + 多阶段训练，实现前馈式联合推理个性化蒙皮 Avatar。

## 方法详解

### 整体框架

输入：N 帧着装人体图像（前后视角），估计的 3D 姿态。输出：canonical 空间的 Avatar 网格 $M$、蒙皮权重矩阵 $W$、以及任意目标姿态下的姿态依赖位移 $\Delta V$。Pipeline 分三步：(1) 3D Canonicalization 将 posed 图像 unpose 到 canonical 空间产生像素对齐的初始条件；(2) 多帧编码器聚合 + 解码器联合预测几何/蒙皮/变形；(3) 多阶段训练解耦 canonical 监督和 posed 监督。

### 关键设计

1. **3D Canonicalization（规范化）**:

    - 功能：消除输入图像中的姿态差异，产生统一空间下的像素对齐初始条件
    - 核心思路：先用基础模型估计法线图和分割图，通过法线积分 lift 成 3D 前后表面网格。然后通过 LBS 逆变换 unpose 到 canonical 空间：$[v;1] = (\sum_{j=1}^J w_j T_j)^{-1}[\hat{u};1]$。此时用模板最近邻蒙皮权重做确定性 unpose（虽然会有伪影，但这些伪影模式一致，可被后续网络学会修正）。最后用固定正交相机渲染 canonical 法线和分割图作为网络输入
    - 设计动机：直接从 posed 图像采样特征会因姿态差异导致特征不对齐，产生过度平滑的重建。Canonicalization 后同一身体部位总是出现在特征图的同一位置，极大降低了特征学习难度

2. **多帧特征聚合**:

    - 功能：跨帧融合消除 canonicalization 伪影，提取人物本征特征
    - 核心思路：每帧的 canonical 法线+分割图通过浅层 CNN 提取高分辨率特征 $H_i^v$ 和 DeepLabV3 提取低分辨率全局特征 $L_i^v$。多帧特征通过简单平均聚合为单一双平面特征 $B = (B^f \oplus B^b)$，其中 $B^v = \frac{1}{N}\sum_{i=1}^N f_b(H_i^v \oplus L_i^v)$
    - 设计动机：不同姿态下的 unposing 伪影不同，但人物本征信息（体型、衣物类型）跨帧一致。平均操作天然保留共性、过滤帧特异性伪影。实验显示 5 帧即可收敛到足够好的结果

3. **联合解码：几何 + 蒙皮 + 姿态变形**:

    - 功能：从聚合特征同时预测三个相互耦合的输出
    - 核心思路：
        - 几何解码器：在 canonical 四面体网格上，每个顶点投影采样双平面特征，通过 MLP 预测 SDF 值和位移，用 Marching Tetrahedra 提取网格
        - 蒙皮权重解码器：独立 MLP 对每个 canonical 顶点预测 $J$ 个关节的蒙皮权重（Softmax 归一化保证有效性），以模板最近邻权重为正则化目标
        - 姿态变形模块：给定目标姿态渲染 position map 作为条件，结合 canonical 网格的渲染法线，通过 CNN + MLP 预测逐顶点位移 $\Delta v_t$。最终动画：$[\hat{v}_t;1] = \text{LBS}(v + \Delta v_t, w, \hat{T})$
    - 设计动机：联合优化三个输出比分开优化更有效（蒙皮权重影响几何质量，几何形状影响蒙皮合理性）。但为解决耦合歧义，需要多阶段训练

### 损失函数 / 训练策略

**多阶段训练**解决 canonical 形状和蒙皮权重的耦合歧义：

- **Canonical 阶段**：只训练编码器和几何解码器，用 pseudo GT canonical 网格监督（通过优化得到的高质量 unpose 结果）: $\mathcal{L}_c = \|\mathcal{N} - \mathcal{N}_i^\star\|_1 + \|\mathcal{D} - \mathcal{D}_i^\star\|_1$
- **Posed 阶段**：联合训练所有模块，用 posed 空间 GT 扫描监督: $\mathcal{L} = \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s + \lambda_e \mathcal{L}_e$。其中 $\mathcal{L}_p$ 包含法线 L1 + 深度 L1 + 感知损失；$\mathcal{L}_s$ 正则化蒙皮权重偏离模板不要太远；$\mathcal{L}_e$ 惩罚过度拉伸的三角形边

## 实验关键数据

### 主实验

| 方法 | Normal↓ | P2S(cm)↓ | CD(cm)↓ | 推理时间 |
|------|---------|----------|---------|---------|
| ARCH++ (前馈) | 0.338 | 4.52 | 5.07 | 26s |
| PuzzleAvatar (扩散) | 0.104 | 1.47 | 1.63 | 3h |
| Vid2Avatar (优化) | 0.072 | 0.98 | 1.12 | 8h |
| **FRESA (LBS Only)** | 0.030 | 0.43 | 0.49 | 18s |
| **FRESA (Full)** | **0.026** | **0.37** | **0.43** | **18s** |

在 RenderPeople 数据集上零样本泛化同样大幅领先（CD: 0.34 vs 1.91），且可直接泛化到手机照片。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无 Canonicalization | 几何过度平滑 | 直接从 posed 采样特征对不齐 |
| 单帧 vs 5帧聚合 | 多帧更准 | 伪影被平均消除、裙子和头发更合理 |
| 模板蒙皮 vs 个性化蒙皮 | 个性化减少腋下伪影 | 多帧训练的蒙皮更鲁棒 |
| 无姿态变形 | 缺少动态褶皱 | 变形模块纠正 LBS 伪影 + 生成合理褶皱 |

### 关键发现

- 前馈推理仅需 18 秒，比优化方法快 600-1600 倍，质量反而更好，归功于大规模数据学到的通用先验
- 个性化蒙皮权重对极端姿态的动画质量提升显著，尤其在腋下、肘部弯曲等区域
- 姿态依赖变形模块带来三项收益：纠正 LBS 伪影、生成合理的衣物动态（如抬手时袖子下垂）、整体细节精细化

## 亮点与洞察

- **"先 unpose 再学修正"的策略非常实用**：虽然 unpose 伪影不完美，但它提供了像素对齐的初始条件，让网络只需学习"修正残差"而非"从头理解姿态"。这大幅降低了学习难度
- **多帧平均聚合的简洁有效令人印象深刻**：不需要复杂的注意力机制或对齐操作，简单平均就能利用帧间一致性过滤伪影。这说明当初始条件足够好时，简单方法就能奏效
- **大规模数据驱动的通用先验是核心优势**：1100+ 人的 dome 数据是这篇工作的"护城河"，这种先验使得前馈推理在质量和速度上同时超越优化方法

## 局限与展望

- 几何精度受四面体网格分辨率限制，微小配饰（如耳环、项链）可能丢失
- 只建模姿态驱动的变形，忽略了身体-衣物交互动力学和非常宽松衣物/长发的复杂运动
- 依赖前后双视角输入，单视角场景需要额外的视角补全策略
- 训练数据集不公开（Meta Reality Labs 内部 dome 数据），限制了可复现性
- Canonical 伪 GT 的生成需要每帧 20 分钟的优化过程，限制了训练数据的规模扩展

## 相关工作与启发

- **vs ARCH++**：ARCH++ 也是前馈方法但使用固定蒙皮权重和手工空间编码，导致动画质量差、几何粗糙。FRESA 通过个性化蒙皮和 canonicalization 实现了量级级别的提升
- **vs PuzzleAvatar**：PuzzleAvatar 用扩散模型 SDS 损失生成 Avatar，质量尚可但需 3 小时且可能丢失面部身份。FRESA 保留了个性化细节且速度快 600 倍
- **vs Vid2Avatar**：基于多视角视频优化，质量好但极度耗时。FRESA 在通用先验加持下以前馈方式超越了优化方法的质量

## 评分

- 新颖性: ⭐⭐⭐⭐ 联合推理几何/蒙皮/变形的前馈框架有较强新颖性，多阶段训练解耦也有巧思
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集评估 + 在手机照片上的零样本泛化 + 详尽消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式完整，图表质量高
- 价值: ⭐⭐⭐⭐⭐ 在速度和质量上同时取得突破，对虚拟人产业有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 3D Face Reconstruction From Radar Images](3d_face_reconstruction_from_radar_images.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](../../ICCV2025/human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2025\] ShowMak3r++: Compositional Entertainment Video Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)

</div>

<!-- RELATED:END -->
