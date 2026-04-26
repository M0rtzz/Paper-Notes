---
title: >-
  [论文解读] One Wave To Explain Them All: A Unifying Perspective On Feature Attribution
description: >-
  [ICML2025][语音][小波变换] 提出 Wavelet Attribution Method (WAM)，将特征归因从像素域迁移到小波域，利用小波系数的空间-尺度局部性为音频、图像、体数据提供统一且更具结构信息的模型解释。
tags:
  - ICML2025
  - 语音
  - 小波变换
  - 特征归因
  - 可解释AI
  - 多模态
  - SmoothGrad
  - Integrated Gradients
---

# One Wave To Explain Them All: A Unifying Perspective On Feature Attribution

**会议**: ICML2025  
**arXiv**: [2410.01482](https://arxiv.org/abs/2410.01482)  
**代码**: [项目主页](https://gabrielkasmi.github.io/wam/)  
**领域**: 可解释性 (Explainability / Feature Attribution)  
**关键词**: 小波变换, 特征归因, 可解释AI, 多模态, SmoothGrad, Integrated Gradients

## 一句话总结

提出 Wavelet Attribution Method (WAM)，将特征归因从像素域迁移到小波域，利用小波系数的空间-尺度局部性为音频、图像、体数据提供统一且更具结构信息的模型解释。

## 研究背景与动机

- **特征归因的核心问题**：现有方法（Saliency、SmoothGrad、Integrated Gradients、GradCAM 等）在像素域计算梯度，生成热力图指示"模型看了哪里"，但像素域不包含信号的尺度/频率结构信息。
- **像素域的缺陷**：
    - 像素之间仅存在空间平移关系，无法揭示模型依赖的纹理、边缘、形状等多尺度特征。
    - 对于 1D 音频和 3D 体数据，通常先投射到 2D 再做归因，进一步丢失结构信息。
- **小波的优势**：小波变换同时保留空间位置和频率/尺度信息，分解出的系数天然对应边缘、纹理、瞬态等可解释的低级特征，且适用于任意平方可积信号（1D/2D/3D）。
- **研究空白**：仅有极少数工作（CartoonX、ShearletX、WCAM）尝试在小波/Shearlet 域做归因，且均局限于图像、需要扰动采样，缺乏跨模态统一框架。

## 方法详解

### 核心思路

WAM 的关键操作：将输入 $\boldsymbol{x}$ 做离散小波变换得到系数 $\boldsymbol{z} = \mathcal{W}(\boldsymbol{x})$，然后计算分类器输出对小波系数的梯度，而非对原始像素的梯度。

### 小波域显著性图

对分类器 $\boldsymbol{f}_c$ 在小波域的显著性定义为：

$$\boldsymbol{\gamma}_{\text{Sa}}(\boldsymbol{z}) = \left| \frac{\partial \boldsymbol{f}_c(\boldsymbol{x})}{\partial \boldsymbol{z}} \right| = \left| \frac{\partial \boldsymbol{f}_c(\boldsymbol{x})}{\partial \boldsymbol{x}} \cdot \frac{\partial \mathcal{W}^{-1}(\boldsymbol{z})}{\partial \boldsymbol{z}} \right|$$

其中 $\mathcal{W}^{-1}$ 为逆小波变换的 Jacobian。实际计算时，先对输入做小波变换，再通过自动微分对 $\boldsymbol{z}$ 求梯度，只需一次反向传播。

### WAM$_{\text{SG}}$（SmoothGrad 变体）

在小波域加噪平滑，对 $n$ 个噪声样本取平均：

$$\boldsymbol{\gamma}_{\text{SG}}(\boldsymbol{z}) = \frac{1}{n} \sum_{i=1}^{n} \nabla_{\tilde{\boldsymbol{z}}} \boldsymbol{f}(\mathcal{W}^{-1}(\tilde{\boldsymbol{z}})), \quad \tilde{\boldsymbol{z}} = \mathcal{W}(\boldsymbol{x} + \boldsymbol{\delta}), \; \boldsymbol{\delta} \sim \mathcal{N}(0, I\sigma^2)$$

### WAM$_{\text{IG}}$（Integrated Gradients 变体）

在小波域做路径积分，从基线 $\boldsymbol{z}_0$ 到当前 $\boldsymbol{z}$：

$$\boldsymbol{\gamma}_{\text{IG}} = (\boldsymbol{z} - \boldsymbol{z}_0) \cdot \int_0^1 \frac{\partial \boldsymbol{f}_c(\mathcal{W}^{-1}(\boldsymbol{z}_0 + \alpha(\boldsymbol{z} - \boldsymbol{z}_0)))}{\partial \boldsymbol{z}} \, d\alpha$$

WAM$_{\text{IG}}$ 继承了 Integrated Gradients 的灵敏度（Sensitivity）和实现不变性（Implementation Invariance）公理。

### 多尺度解读

- 小波系数按尺度（scale level $j$）和方向（水平/垂直/对角）分层，归因结果自然分解为"在哪里"（空间定位）和"看到了什么"（尺度/纹理/边缘）。
- 对每个尺度的归因求和，可量化模型对不同频段的依赖程度。

### 模态无关性

- 小波变换对 1D（音频波形）、2D（图像）、3D（体数据/医学影像）均有定义，因此 WAM 无需任何模态特定适配即可跨模态应用。

## 实验关键数据

### 评估指标

- **Faithfulness** = Insertion − Deletion（越高越好）
- **Insertion**（↑）：按归因分数从高到低逐步插入特征，观察预测概率上升速度。
- **Deletion**（↓）：按归因分数从高到低逐步删除特征，观察预测概率下降速度。

### 主实验结果

| 模态 | 模型 | 数据集 | 方法 | Ins ↑ | Del ↓ | Faith ↑ |
|------|------|--------|------|-------|-------|---------|
| 音频 | ResNet | ESC-50 | Integrated Gradients | 0.267 | 0.047 | 0.264 |
| 音频 | ResNet | ESC-50 | SmoothGrad | 0.251 | 0.067 | 0.184 |
| 音频 | ResNet | ESC-50 | GradCAM | 0.274 | 0.201 | 0.072 |
| 音频 | ResNet | ESC-50 | **WAM$_{\text{IG}}$** | **0.436** | 0.260 | 0.176 |
| 音频 | ResNet | ESC-50 | **WAM$_{\text{SG}}$** | **0.449** | 0.252 | **0.197** |
| 图像 | EfficientNet | ImageNet | GradCAM | 0.364 | 0.303 | 0.061 |
| 图像 | EfficientNet | ImageNet | **WAM$_{\text{IG}}$** | **0.447** | **0.049** | **0.370** |
| 体数据 | 3D Former | AdrenalMNIST3D | Saliency | 0.751 | 0.742 | 0.009 |
| 体数据 | 3D Former | AdrenalMNIST3D | **WAM$_{\text{IG}}$** | 0.719 | **0.621** | **0.098** |

**关键发现**：

- 图像模态上 WAM$_{\text{IG}}$ Faithfulness 达 0.370，远超 GradCAM 的 0.061 和 SmoothGrad 的 0.010。
- 体数据上 WAM 是首个达到正 Faithfulness 的方法（其他方法均为负或接近零）。
- 音频上 WAM 在 Insertion 分数上大幅领先（0.449 vs 0.274），但 Deletion 略高于最优（trade-off）。

## 亮点与洞察

1. **统一框架**：首次用同一套方法（小波域梯度）统一解释音频、图像、体数据三种模态，无需模态特定设计。
2. **多尺度分解的新视角**：
    - 对图像，可将每个空间位置的归因拆解为细纹理（蓝/高频）、中尺度轮廓（红/中频）、粗边缘（黄/低频），这是像素域归因无法实现的。
    - 对 3D 体数据，首次展示了多尺度归因分解——粗尺度捕获器官/病灶轮廓，细尺度捕获纹理细节。
3. **鲁棒性评估的桥梁**：通过对各尺度归因求和，可直接量化模型对低/高频特征的依赖度。实验表明对抗训练模型更依赖粗尺度（低频）特征，与鲁棒性文献一致，且无需复杂的频率扰动实验。
4. **音频去噪应用**：在 0 dB 白噪声实验中，WAM 通过选取重要小波系数重建音频，能有效滤除噪声、保留目标声音成分，且完全是 post-hoc 方式，无需训练 NMF 等模型。
5. **理论保障**：WAM$_{\text{IG}}$ 保持了灵敏度和实现不变性公理。

## 局限与展望

- **不支持点云**：小波变换要求结构化网格，无法直接处理非结构化的点云数据；未来可探索图小波变换。
- **音频可听解释质量有限**：贪心提取重要系数的方式不够平滑，生成的可听解释可能存在伪影。
- **不适用于文本**：小波变换形式上无法应用于离散文本数据，限制了在 NLP 和多模态 VLM 中的应用。
- **小波基的选择**：不同母小波（Haar、Daubechies 等）会影响归因结果的解释，论文未充分讨论如何自动选择最优小波基。
- **计算开销**：WAM$_{\text{SG}}$ 需要多次采样，WAM$_{\text{IG}}$ 需要路径积分，在高分辨率输入上的计算成本未详细报告。

## 相关工作与启发

- **CartoonX / ShearletX**（Kolek et al., 2022/2023）：在小波/Shearlet 域做 Meaningful Perturbation，但属于扰动方法，计算昂贵。
- **WCAM**（Kasmi et al., 2023）：在小波域做 Sobol 归因，WAM 是其梯度方法推广。
- **散射变换**（Bruna & Mallat, 2013）：小波系数构建的固定特征提取器，天然可解释，是 WAM 理论基础的一部分。
- **频率偏置文献**（Zhang et al., 2022; Wang et al., 2020）：模型鲁棒性与频率依赖的关系，WAM 提供了更简洁的量化手段。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 小波域归因的统一框架思路新颖，跨模态泛化有价值
- 实验充分度: ⭐⭐⭐⭐ — 三种模态、多个数据集、多种基线和指标，附录补充实验丰富
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰、可视化优秀，结构完整
- 价值: ⭐⭐⭐⭐ — 为可解释 AI 提供了新的域选择视角，实用性强（鲁棒性评估、音频去噪）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](../../NeurIPS2025/audio_speech/from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](../../NeurIPS2025/audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)

<!-- RELATED:END -->
