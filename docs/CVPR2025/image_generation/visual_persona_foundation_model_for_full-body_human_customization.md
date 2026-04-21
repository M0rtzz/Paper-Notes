---
title: >-
  [论文解读] Visual Persona: Foundation Model for Full-Body Human Customization
description: >-
  [CVPR 2025][图像生成][全身人体定制] 提出 Visual Persona，首个面向全身人体定制的基础模型，通过大规模配对数据集策展（580K图像/100K身份）和身体部位分区 Transformer 解码器架构，实现高保真的全身外观保持与文本引导的多样化生成。
tags:
  - CVPR 2025
  - 图像生成
  - 全身人体定制
  - 文本到图像生成
  - 扩散模型
  - 身份保持
  - 数据策展
---

# Visual Persona: Foundation Model for Full-Body Human Customization

**会议**: CVPR 2025  
**arXiv**: [2503.15406](https://arxiv.org/abs/2503.15406)  
**代码**: [项目主页](https://cvlab-kaist.github.io/Visual-Persona)  
**领域**: 图像生成  
**关键词**: 全身人体定制, 文本到图像生成, 扩散模型, 身份保持, 数据策展

## 一句话总结

提出 Visual Persona，首个面向全身人体定制的基础模型，通过大规模配对数据集策展（580K图像/100K身份）和身体部位分区 Transformer 解码器架构，实现高保真的全身外观保持与文本引导的多样化生成。

## 研究背景与动机

- 文本到图像定制化生成近年取得显著进展，但大多数方法仅关注面部身份保持
- 全身人体定制（包括面部、服装、配饰的完整保持）是一个被严重忽视的领域
- 实现全身定制需要大规模配对人体数据（同一人多张图片且全身身份一致），这是极难获取的
- 现有方法如 StoryMaker 只能使用单张图片训练（无配对数据），导致身份保持与文本对齐之间的权衡
- 使用 CLIP 或人脸识别模型的全局编码方式无法捕捉全身的精细局部特征
- 将输入特征压缩到少量 token 嵌入（通常 $l_H=16$）会丢失关键的身体部位细节
- 需要一种新的数据策展方法和模型架构来同时实现身份保持和文本对齐

## 方法详解

### 整体框架

Visual Persona 包含两大创新：(1) 数据策展管线——利用视觉语言模型（VLM）从大量未配对人体图像中筛选出全身身份一致的配对数据，构建 Visual Persona-500K 数据集；(2) 模型架构——基于身体部位分解的 Transformer 编码器-解码器架构，适配预训练 T2I 扩散模型（冻结参数），通过稠密身份嵌入条件化扩散过程。仅训练身体分区 Transformer 解码器和身份跨注意力模块。

### 关键设计

**1. Visual Persona-500K 数据集策展**
- **功能**: 构建大规模全身身份一致的配对人体数据集
- **核心思路**: 三阶段策展——先用人脸识别模型余弦相似度筛选面部身份一致的图像子集；再用 LLAVA 评估全身着装一致性（提示："Are they wearing exactly the same clothes?"）；最后用 Phi-3 生成不含身份信息的描述性文本（聚焦表情、姿态、动作、环境）
- **设计动机**: 配对数据是同时实现身份保持和文本对齐的关键（单图训练易过拟合位置/姿态等身份无关属性）。VLM 是评估复杂全身外观一致性的简单有效工具，最终收集 580K 图像、100K 身份

**2. 身体部位分解 + DINOv2 编码器**
- **功能**: 将输入人体图像分解为独立身体区域（全身、面部、上身、腿部、鞋子共 $N=5$ 个），提取细粒度局部外观特征
- **核心思路**: 使用 off-the-shelf 身体解析方法分割身体区域，裁剪、零填充、缩放为独立图像，用预训练 DINOv2 编码器提取局部 token 特征 $F \in \mathbb{R}^{N \times h \times w \times d_F}$（使用全部局部 token 而非 CLS）
- **设计动机**: DINOv2 的自监督训练使其比 CLIP 更擅长捕获细粒度结构和纹理信息。身体分解让扩散模型能独立关注各部位，避免全局编码导致的局部细节混合

**3. 身体分区 Transformer 解码器**
- **功能**: 将各身体部位的编码特征投影为对应的稠密身份嵌入，条件化扩散模型
- **核心思路**: 每层包含跨注意力（将可学习隐嵌入 $H^{i,j}$ 与对应身体部位特征 $F^i$ 关联）+自注意力（学习嵌入内部关系）+ MLP。经 $M$ 层迭代后将所有部位嵌入沿 token 长度拼接：$C_H^* = \text{Concat}([C_H^1, ..., C_H^N]) \in \mathbb{R}^{(N \times l_H) \times d_H}$。通过解耦跨注意力注入扩散模型
- **设计动机**: 使用稠密嵌入（$l_H = 16 \times 16$）保留远多于传统方法（$l_H=16$）的细节。分区设计防止不同身体部位特征混合，消融实验证明身体分解使 D-T 从 6.13 提升至 6.67

### 损失函数

标准扩散模型噪声预测损失：

$$L = \mathbb{E}_{z_{Y,t}, \epsilon, t, C_T, C_H^*}\left[\|\epsilon - \epsilon_\theta(z_{Y,t}, t, C_T, C_H^*)\|_2^2\right]$$

仅反向传播更新身体分区 Transformer 解码器和身份跨注意力模块的参数。

## 实验关键数据

### 主实验：定量比较（GPT-based DreamBench++ 评估）

| 方法 | D-I↑ (SSHQ) | D-T↑ | D-H↑ | D-I↑ (PPR10K) | D-T↑ | D-H↑ |
|------|------------|------|------|--------------|------|------|
| IP-Adapter-FaceID | 1.78 | 7.50 | 2.76 | 1.86 | 7.49 | 2.81 |
| InstantID | 1.52 | 6.94 | 2.37 | 1.70 | 7.12 | 2.63 |
| PhotoMaker | 1.70 | 7.72 | 2.64 | 2.03 | 7.64 | 3.03 |
| StoryMaker | 6.74 | 7.08 | 6.71 | 6.80 | 6.77 | 6.63 |
| **Visual Persona** | **7.10** | **7.15** | **6.99** | **7.30** | 6.67 | **6.85** |

*Visual Persona 在身份保持(D-I)上大幅领先，谐波平均(D-H)最高*

### 消融实验

| 配置 | D-I↑ | D-T↑ | D-H↑ |
|------|------|------|------|
| (I) MLP only | 6.66 | 7.11 | 6.74 |
| (II) + Self-Attention | 6.54 | 7.01 | 6.63 |
| (III) + Cross-Attention | 7.47 | 6.13 | 6.40 |
| **(IV) + Body Part Decomp.** | **7.30** | **6.67** | **6.85** |

| Token 长度 $l_H$ | D-I↑ | D-T↑ | D-H↑ |
|-------------------|------|------|------|
| $4 \times 4$ | 5.51 | 6.90 | 5.81 |
| $8 \times 8$ | 6.56 | 6.50 | 6.52 |
| $16 \times 16$ | **7.30** | **6.67** | **6.85** |

### 关键发现

- 身体部位分解对保持身份保持的同时改善文本对齐至关重要（D-T: 6.13→6.67）
- 稠密身份嵌入（$16 \times 16$）比稀疏嵌入（$4 \times 4$）提升 D-I 达 32.5%
- 跨图像训练（配对数据）比重建训练（单图）更好地支持大幅度几何变形

## 亮点与洞察

1. **数据驱动的突破**: 通过 VLM 自动化全身一致性评估，解决了配对数据获取的核心瓶颈
2. **稠密嵌入范式**: 打破了将身份压缩为少量 token 的惯例，证明更多 token 更好地保留全身细节
3. **零样本多场景应用**: 无需额外训练即可支持多人定制、虚拟试穿、人物风格化和故事连贯生成

## 局限与展望

- 对 VLM 着装一致性判断的准确性有依赖，某些细微差异可能被忽略
- 身体部位解析模型的质量直接影响最终效果
- 当前固定 $N=5$ 个身体区域，更灵活的动态分区可能进一步提升效果
- 数据集中的人种和年龄分布可能存在偏差

## 相关工作与启发

- 与 IP-Adapter、InstantID 等面部定制方法相比，Visual Persona 将定制范围从面部扩展到全身
- DINOv2 作为编码器的选择突显了自监督特征在细粒度保持任务中的优势
- 配对数据+跨图像训练的模式可推广到其他身份一致性生成任务

## 评分

⭐⭐⭐⭐ — 系统性地解决了全身人体定制的数据和模型两大瓶颈，方法设计合理，消融分析充分。数据策展管线具有很强的实用价值，多种下游应用展示了方法的通用性。评估指标依赖GPT-based打分，可能存在一定局限性。

<!-- RELATED:START -->

## 相关论文

- [DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects](../../ECCV2024/image_generation/himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)
- [InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)

<!-- RELATED:END -->
