---
title: >-
  [论文解读] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians
description: >-
  [ECCV 2024][3D视觉] 提出WaSt-3D，利用3D高斯溅射表示将风格迁移重新定义为两个高斯分布之间的最优传输问题，通过Sinkhorn散度匹配内容场景和风格场景的3D分布，首次实现了3D场景到场景的几何风格迁移。
tags:
  - "ECCV 2024"
  - "3D视觉"
---

# WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians

**会议**: ECCV 2024  
**arXiv**: [2409.17917](https://arxiv.org/abs/2409.17917)  
**代码**: [项目页](https://compvis.github.io/wast3d/)  
**领域**: 3D视觉

## 一句话总结

提出WaSt-3D，利用3D高斯溅射表示将风格迁移重新定义为两个高斯分布之间的最优传输问题，通过Sinkhorn散度匹配内容场景和风格场景的3D分布，首次实现了3D场景到场景的几何风格迁移。

## 研究背景与动机

### 领域现状

**领域现状**：现有3D风格迁移主要修改纹理/颜色，几乎无法迁移几何风格

### 现有痛点

**现有痛点**：画作中的"集合"技法（如Arcimboldo用蔬菜拼人像、Picasso用木块拼乐器）体现了从局部风格元素组装整体内容的艺术思想

### 核心矛盾

**核心矛盾**：基于NeRF的方法（ARF、StyleRF、SNeRF）在RGB特征空间优化，无法改变底层几何

### 解决思路

**解决思路**：核心洞察**：将风格迁移从"特征空间生成"转变为"两个3D粒子分布的显式匹配"

## 方法详解

### 整体框架

1. 将内容和风格场景分别训练为正则化高斯溅射
2. 将内容场景聚类为N个子区域
3. 为每个内容簇找到最佳匹配的风格区域（约束优化）
4. 最小化每对内容-风格簇之间的Sinkhorn散度

### 关键设计

**各向同性高斯正则化**：
- 标准3DGS训练中高斯可能被拉伸成针状，分割时引起伪影
- 添加各向异性正则化：最小化最大/最小scale的比值
- 添加均匀尺寸正则化：约束所有高斯趋近相同大小

**Wasserstein-2距离与Sinkhorn散度**：
- 使用熵正则化的W-2距离使最优传输可解且平滑
- Sinkhorn散度 = W_2(p_s,p_c) - W_2(p_s,p_s) - W_2(p_c,p_c)，消除"去偏"保证距离为零当两分布相同
- γ控制传输计划平滑度：大γ产生全局平均效果，小γ实现精确一对一匹配

**场景分割（解决大规模OT不可解）**：
- K-Means聚类内容场景为N=400个簇
- 对每个内容簇通过平移+旋转+缩放拟合到风格场景中找最佳匹配
- 用k近邻选取对应的风格高斯子集
- 问题分解为N个小规模OT子问题

**优化目标**：
$$\mathcal{L}_{opt} = \sum_{i=1}^{N} \mathcal{SD}_{2,\gamma}^2(C_i, D_i(C_i))$$

在坐标和亮度/颜色通道上同时计算Sinkhorn散度。优化亮度有助于保持原始场景的光影体积感。

### 损失函数

两阶段优化：第一阶段优化 {t_i, R_i, S_i} 找匹配（Eq.8）；第二阶段优化风格簇的颜色和坐标最小化Sinkhorn散度（Eq.10）。可选添加ARAP弹性变形损失防止风格簇过度拉伸。

## 实验关键数据

### 主实验

| 方法 | CLIP高层相似度↑ | 人类偏好↑ | 时间↓ | VRAM↓ |
|------|----------------|----------|-------|-------|
| ARF | 74.79% | 12.5% | 11 min | 9GB |
| StyleRF | 74.94% | 1.5% | 18 min | 6GB |
| SNeRF | 76.99% | 10.5% | 30 min | 8GB |
| **WaSt-3D** | **84.40%** | **75.5%** | **8 min** | 16GB |

### 消融实验

Sinkhorn散度优化的不同参数组合效果：

| 优化参数 | 效果描述 |
|----------|----------|
| 仅坐标 | 形状匹配但缺乏光影层次 |
| 坐标+亮度 | 保持原始场景光影，推荐配置 |
| 坐标+亮度+颜色 | 最完整但可能过度约束 |

法线可视化对比：WaSt-3D完整保留风格场景的3D几何纹理，其他方法因在RGB空间优化引入噪声。

### 关键发现

- 人类偏好75.5%远超其他方法（<12.5%），验证了几何风格迁移的视觉吸引力
- CLIP高层细节相似度84.40%显著领先（vs SNeRF 76.99%），说明风格的3D细节被忠实保留
- 优化时间8分钟最短，因为扩散/NeRF方法需要额外训练而WaSt-3D仅做OT优化
- 减少内容簇数量（400→200）会降低内容保真度
- 用ARAP弹性损失替代Sinkhorn散度效果较差，说明分布匹配比几何约束更合适

## 亮点与洞察

- 范式转换：将风格迁移从"在隐空间优化"转变为"3D分布的显式匹配"
- 用最优传输理论（Wasserstein距离）处理3D风格迁移是全新的技术路线
- 内容场景由风格元素"组装"而成的思想与艺术中的集合技法直接对应
- 正则化高斯溅射是3D风格迁移的良好基底表示——显式、可操作、高效

## 局限与展望

- 风格场景必须也是3D高斯溅射表示，无法直接使用2D风格图
- 需要16GB VRAM，大场景内存开销较大
- 聚类数量N需要手动调整
- 部分极端几何变形可能产生不自然效果

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 3D-to-3D风格迁移 + 最优传输的独特组合
- 有效性：⭐⭐⭐⭐ — 用户研究大幅领先
- 实用性：⭐⭐⭐ — 需要3D风格场景作为输入
- 推荐度：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
