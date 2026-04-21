---
title: >-
  [论文解读] ArtiFade: Learning to Generate High-quality Subject from Blemished Images
description: >-
  [CVPR 2025][图像生成][主题驱动生成] 本文提出ArtiFade，首个解决"瑕疵主题驱动生成"问题的方法，通过构建瑕疵-无瑕疵配对数据集、部分微调扩散模型的cross-attention权重并优化artifact-free embedding，使得现有主题驱动方法（Textual Inversion、DreamBooth）能从带水印/贴纸/对抗噪声等瑕疵的图像中生成高质量无伪影的主题图像。
tags:
  - CVPR 2025
  - 图像生成
  - 主题驱动生成
  - 瑕疵图像
  - 水印去除
  - Textual Inversion
  - 扩散模型微调
---

# ArtiFade: Learning to Generate High-quality Subject from Blemished Images

**会议**: CVPR 2025  
**arXiv**: [2409.03745](https://arxiv.org/abs/2409.03745)  
**代码**: 无  
**领域**: 扩散模型 / 主题驱动图像生成  
**关键词**: 主题驱动生成, 瑕疵图像, 水印去除, Textual Inversion, 扩散模型微调

## 一句话总结

本文提出ArtiFade，首个解决"瑕疵主题驱动生成"问题的方法，通过构建瑕疵-无瑕疵配对数据集、部分微调扩散模型的cross-attention权重并优化artifact-free embedding，使得现有主题驱动方法（Textual Inversion、DreamBooth）能从带水印/贴纸/对抗噪声等瑕疵的图像中生成高质量无伪影的主题图像。

## 研究背景与动机

主题驱动文本到图像生成（如Textual Inversion、DreamBooth）旨在从少量主题图像中学习主题特征，然后结合文本提示生成包含该主题的多样化图像。这些方法取得了显著进展，但**都依赖高质量的无瑕疵输入图像**。

在实际应用中，获取完美无瑕的主题图像往往昂贵甚至不可能。例如，从网络抓取的主题图像可能带有各种**可见瑕疵**（水印、贴纸、涂鸦）或**不可见瑕疵**（对抗噪声，如Anti-DreamBooth生成的保护性扰动）。

**核心问题**：现有方法无法区分主题特征和瑕疵干扰。Textual Inversion学到的embedding同时编码了主题信息和瑕疵信息，导致生成的图像包含失真的背景、扭曲的主题和瑕疵伪影。DreamBooth也会过拟合到瑕疵上。

**本文的切入角度**：不是直接去除输入图像的瑕疵（传统图像处理思路），而是在扩散模型层面学习将"瑕疵embedding"映射到"无瑕疵图像"的生成能力。通过在配对的瑕疵/无瑕疵数据上微调扩散模型的文本条件处理模块，让模型学会区分瑕疵模式与主题特征。

## 方法详解

### 整体框架

ArtiFade的pipeline分为两个阶段：
1. **Artifact Rectification Training**（离线，一次性）：构建包含多种瑕疵类型的配对数据集，微调扩散模型使其能从瑕疵embedding生成无瑕图像
2. **推理阶段**：对新的瑕疵测试图像做Textual Inversion得到瑕疵embedding，直接用ArtiFade模型生成无瑕主题图像

### 关键设计

1. **配对数据集构建**:

    - 收集N=20个主题的无瑕疵图像集合（涵盖宠物、植物、容器、玩具、穿戴物等）
    - 定义L种瑕疵增强变换（如10种水印：不同字体、方向、颜色、大小、文本）
    - 对每个主题的每张图像应用每种瑕疵，构成N×L=200个瑕疵子集
    - 对每个瑕疵子集用Textual Inversion训练5000步得到瑕疵textual embedding
    - 最终形成（瑕疵embedding，无瑕疵原图）的配对训练数据

2. **部分微调策略（Partial Fine-tuning）**:

    - 核心洞察：瑕疵信息编码在textual embedding中，通过cross-attention层影响生成
    - 只微调扩散模型cross-attention层中处理文本条件的**key权重W^k和value权重W^v**
    - 不微调query权重W^q（处理图像特征的参数）——消融实验证明微调W^q反而降低效果
    - 冻结扩散模型其他所有参数
    - 这种策略确保：优化文本条件相关参数来"纠正"瑕疵embedding，同时保留模型原有生成能力

3. **Artifact-free Embedding ⟨Φ⟩**:

    - 在文本空间额外优化一个可学习的embedding
    - 推理时构造prompt："a ⟨Φ⟩ photo of [V_test^β']"
    - 作用：增强prompt fidelity，帮助模型更好地保留文本信息
    - 消融实验表明：单独用⟨Φ⟩不够（会过拟合），但与部分微调结合能提升文本忠实度

### 损失函数 / 训练策略

训练损失为标准LDM重建损失的变体：
```
L_ArtiFade = E[||ε - ε_{W^k, W^v, ⟨Φ⟩}(z_t, t, y_i^β_k)||²]
```

关键点：
- 输入为**无瑕疵图像**的潜在表示z
- 条件为**瑕疵textual embedding**构造的文本条件y_i^β_k
- 优化目标是让模型在瑕疵条件下重建无瑕疵图像，即学会"自动纠正"

训练细节：
- 基础模型：Stable Diffusion v1-5
- 学习率：⟨Φ⟩用5e-3，W^k和W^v用3e-5
- 训练16k步（TI-based模型），2块RTX 3090 GPU
- 每次迭代随机采样一个无瑕疵图像和一种瑕疵类型

## 实验关键数据

### 主实验

分布内（In-Distribution）水印测试：

| 方法 | I^DINO↑ | R^DINO↑ | I^CLIP↑ | R^CLIP↑ | T^CLIP↑ |
|------|---------|---------|---------|---------|---------|
| TI (无瑕疵输入) | 0.488 | 1.349 | 0.730 | 1.070 | 0.283 |
| TI (瑕疵输入) | 0.217 | 0.852 | 0.576 | 0.909 | 0.263 |
| ArtiFade (TI) | **0.337** | **1.300** | **0.649** | **1.020** | **0.282** |

分布外（Out-of-Distribution）水印测试：

| 方法 | I^DINO↑ | R^DINO↑ | I^CLIP↑ | R^CLIP↑ | T^CLIP↑ |
|------|---------|---------|---------|---------|---------|
| TI (瑕疵输入) | 0.229 | 0.858 | 0.575 | 0.929 | 0.262 |
| ArtiFade (TI) | **0.356** | **1.237** | **0.654** | **1.079** | **0.282** |

DreamBooth集成结果（ID测试）：

| 方法 | I^DINO↑ | R^DINO↑ | T^CLIP↑ |
|------|---------|---------|---------|
| DB (瑕疵输入) | 0.503 | 0.874 | 0.272 |
| ArtiFade (DB) | **0.589** | **1.308** | **0.284** |

### 消融实验

| 配置 | W^kv | W^q | ⟨Φ⟩ | I^DINO↑ | R^DINO↑ | T^CLIP↑ | 说明 |
|------|------|-----|------|---------|---------|---------|------|
| Var_A | - | - | ✓ | 0.154 | 1.412 | 0.265 | 仅embedding，过拟合严重 |
| Var_B | - | ✓ | ✓ | 0.283 | 1.230 | 0.277 | 微调W^q，效果差 |
| Var_C | ✓ | - | - | 0.342 | 1.292 | 0.280 | 无embedding，主题保真度高但去瑕疵弱 |
| 完整 | ✓ | - | ✓ | **0.337** | **1.300** | **0.282** | 最佳平衡 |

### 关键发现

- R^DINO/R^CLIP >1 表示生成图像更接近无瑕疵原图而非瑕疵图像——ArtiFade在ID和OOD场景均达到此标准
- 瑕疵输入使TI的I^DINO下降55%（0.488→0.217），ArtiFade恢复到0.337（+55%提升）
- OOD泛化能力出色：在训练时未见过的水印类型上仍有显著提升
- DreamBooth+ArtiFade组合效果最好，I^DINO甚至超过无瑕疵输入的DreamBooth
- 微调W^q（图像特征参数）反而有害——因为瑕疵信息在文本embedding中，应该修改文本处理路径

## 亮点与洞察

- **问题定义有价值**："瑕疵主题驱动生成"是一个实际但被忽视的问题，首次被明确提出
- **方法设计逻辑清晰**：瑕疵→embedding→cross-attention→生成，微调文本条件路径的key/value权重是合理且最小化的干预
- **OOD泛化能力**：仅在10种水印上训练，却能泛化到贴纸、玻璃效果、对抗噪声等完全不同的瑕疵类型，说明模型学到了"区分瑕疵vs主题"的通用能力
- **评估体系完善**：设计了包含ID/OOD测试集和R^DINO/R^CLIP相对比率指标的专用benchmark
- **框架通用性**：同一框架可适配Textual Inversion和DreamBooth，且ArtiFade的微调是一次性的

## 局限与展望

- 主题重建保真度与无瑕疵输入相比仍有差距（I^DINO 0.337 vs 0.488）——部分主题信息确实在瑕疵embedding中丢失了
- 瑕疵embedding的训练（每个子集5k步Textual Inversion）在数据集构建阶段计算成本较高
- 仅在Stable Diffusion v1-5上验证，未扩展到SDXL或更新的基础模型
- 对极端遮挡（如主体大部分被遮挡）的处理能力未讨论
- 评估仅使用CLIP和DINO相似度，缺少人类评估和FID等生成质量指标
- 每个新的瑕疵测试场景仍需运行Textual Inversion来获取embedding，不是端到端的解决方案

## 相关工作与启发

- **Textual Inversion / DreamBooth**：本文是这些方法在"鲁棒性"维度上的扩展——从"干净输入"扩展到"瑕疵输入"
- **水印/阴影去除**：传统方法直接在像素级去除，本文在生成模型的嵌入空间级别处理，更通用
- **Anti-DreamBooth**：这是一种隐私保护技术（加对抗噪声防止身份被复制），ArtiFade反过来突破了这种保护——引发隐私攻防的思考
- **Custom Diffusion / Break-A-Scene**：其他主题微调方法也可能受益于类似的瑕疵纠正训练
- 对数据清洗领域的启发：与其费力清洗训练数据，不如让模型学会在噪声数据上工作

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次提出并系统解决瑕疵主题驱动生成问题，设计合理但方法本身不算复杂
- 实验充分度: ⭐⭐⭐⭐ ID/OOD分析、DreamBooth扩展、不可见瑕疵、多种应用场景、消融研究全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数学符号规范，实验组织系统，但部分符号较繁琐
- 价值: ⭐⭐⭐⭐ 解决了实际痛点，OOD泛化性使其真正可用，但与最新扩散模型的结合有待探索

<!-- RELATED:START -->

## 相关论文

- [EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)
- [OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)

<!-- RELATED:END -->
