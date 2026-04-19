---
title: >-
  [论文解读] LEGION: Learning to Ground and Explain for Synthetic Image Detection
description: >-
  [ICCV 2025][图像分割][合成图像检测] 提出 LEGION 框架和 SynthScars 数据集，利用多模态大语言模型（MLLM）实现合成图像的伪影检测、像素级分割和文本解释三位一体，并创新性地将检测器从"防御者"扩展为"控制者"，引导生成模型产出更高质量的图像。
tags:
  - ICCV 2025
  - 图像分割
  - 合成图像检测
  - 伪影定位
  - MLLM
  - 可解释性
  - 图像修复
---

# LEGION: Learning to Ground and Explain for Synthetic Image Detection

**会议**: ICCV 2025  
**arXiv**: [2503.15264](https://arxiv.org/abs/2503.15264)  
**代码**: [opendatalab.github.io/LEGION](https://opendatalab.github.io/LEGION)  
**领域**: 图像分割  
**关键词**: 合成图像检测, 伪影定位, MLLM, 可解释性, 图像修复

## 一句话总结

提出 LEGION 框架和 SynthScars 数据集，利用多模态大语言模型（MLLM）实现合成图像的伪影检测、像素级分割和文本解释三位一体，并创新性地将检测器从"防御者"扩展为"控制者"，引导生成模型产出更高质量的图像。

## 研究背景与动机

生成技术（GAN → Diffusion → 自回归模型）快速发展，合成图像日益逼真，带来隐私侵犯、版权争议和虚假信息传播等风险。现有合成图像检测方法存在三方面不足：

**数据集过时**：OpenForensics 等数据集主要包含早期 GAN 生成的低质量/动漫风格图像，模型难以泛化到 Stable Diffusion 3.5、FLUX 等现代生成器；RichHF-18K 仅用点标注，空间精度低；SID-Set 仅适用于篡改图像

**方法局限**：传统方法（PAL4VST）依赖底层结构线索，难以处理需要全局推理的伪影（如违反物理光影规律）；现有 MLLM 方法主要关注篡改图像，对完全合成图像的研究不足

**检测与生成脱节**：现有检测方法仅作为"防御者"，未探索利用检测反馈来**提升生成质量**的可能性

本文的核心动机是将检测范式从 **Defender（防御者）** 升级为 **Controller（控制者）**，不仅检测伪影，还引导生成模型消除伪影。

## 方法详解

### 整体框架

LEGION 包含四个核心组件：(i) 全局图像编码器（ViT-H/14 CLIP）, (ii) LLM（Vicuna-based）, (iii) Grounding 图像编码器（SAM encoder）, (iv) 像素解码器（SAM decoder 变体），支持三个任务：伪影检测（二分类）、伪影定位（像素级分割）、解释生成（自然语言）。

### 关键设计

1. **伪造检测（Deepfake Detection）**: 提取 CLIP 全局编码器的 CLS token，通过两层 MLP 进行真/假二分类：
$$y_d = \text{MLP}(\text{CLS}(I_x))$$
简洁有效，利用预训练 CLIP 的强大特征表达能力。

2. **解释生成（Explanation Generation）**: 将 CLIP 全局编码器的 256 个图像 token（不含 CLS）通过 V-L 投影层映射到 LLM 的输入空间，结合伪造分析 prompt 生成文本解释：
$$y_e = \mathcal{L}(x_p, \mathcal{P}_{vl}(I'_x))$$
采用 prompt 模板："The \<image\> provides an overview of the image." + 伪造分析专用 prompt。

3. **伪影定位（Artifact Localization）**: LLM 输出在每个伪影位置描述后附加 `<SEG>` token，通过 L-P 投影层将其嵌入转换到解码器特征空间，SAM 解码器生成二值掩码：
$$M = \mathcal{D}(\mathcal{E}_l(x_i), \mathcal{P}_{lp}(v_{seg}))$$
实现了**语言引导的像素级伪影分割**。

4. **图像修复管线（Image Refinement Pipeline）**: 

    - **重生成（Regeneration）**：LEGION 检测伪影 → 解释记入 memory bank → 文本修改器修订 prompt → T2I 模型重新生成
    - **修复（Inpainting）**：LEGION 输出区域级三元组 $(L_i, M_i, E_i)$（位置、掩码、解释）→ 逐区域修复，保留非伪影区域

### 损失函数 / 训练策略

采用**两阶段独立训练**：

**Stage 1**（定位 + 解释）:
$$\mathcal{L}_{s1} = \lambda_{bce}\mathcal{L}_{BCE}(M, \hat{M}) + \lambda_{dice}\mathcal{L}_{Dice}(M, \hat{M}) + \lambda_{ce}\mathcal{L}_{CE}(y_e, \hat{y}_e)$$
其中 $\lambda_{ce}=1.0, \lambda_{dice}=0.2, \lambda_{bce}=0.4$。

**Stage 2**（检测）: 典型交叉熵损失用于分类：
$$\mathcal{L}_{s2} = \mathcal{L}_{CE}(y_d, \hat{y}_d)$$

基于 GLaMM 预训练权重，使用 LoRA（$\alpha=8$）在 8×A100 GPU 上微调。

## 实验关键数据

### 主实验（伪影定位）

| 方法 | 类型 | SynthScars F1 | SynthScars mIoU | LOKI F1 | RichHF-18K F1 |
|------|------|--------------|----------------|---------|--------------|
| PAL4VST | 传统专家 | 50.46~52.55 | 11.58~21.61 | 49.88 | 14.78 |
| LISA-v1-7B* | MLLM | 31.10~37.56 | 9.29~23.70 | 35.90 | 21.94 |
| InternVL2-8B | MLLM | 41.08~42.03 | 3.91~13.36 | 39.90 | 9.58 |
| **LEGION** | **本文** | **48.66~60.82** | **16.71~39.44** | **50.07** | **17.41** |

提升幅度：在 SynthScars 上比最强传统专家 PAL4VST 的 mIoU 高 **+3.31%**，F1 高 **+7.75%**。

### 解释质量

| 方法 | 参数量 | SynthScars ROUGE-L | SynthScars CSS | LOKI ROUGE-L |
|------|--------|-------------------|----------------|-------------|
| Qwen2-VL | 72B | 25.84 | 58.15 | 11.80 |
| LLaVA-v1.6 | 7B | 29.61 | 61.75 | 16.07 |
| **LEGION** | - | **最优** | **最优** | **最优** |

### 消融实验

| SynthScars 数据集统计 | 数值 |
|---------------------|------|
| 全合成图像数 | 12,236 |
| 图像内容类型 | 4种（Object/Animal/Human/Scene） |
| 伪影类别 | 3种（physics/distortion/structure） |
| 标注完整性 | 100%（像素级掩码 + 文本解释 + 伪影类型） |

### 关键发现

- LEGION 在三个评测集上的大多数指标 SOTA，尤其在**对象类别**上比 PAL4VST 的 F1 高出 10.65 分
- 通用 MLLM（Ferret, Griffon, Qwen2-VL）在伪影定位上存在两种极端：要么完全失败，要么将大部分图像标为伪影
- 作为 Controller 的图像修复管线在 HPS（人类偏好分数）上显著优于基线
- LEGION 在各种扰动（压缩、噪声、模糊）下展现强鲁棒性

## 亮点与洞察

- **Defender → Controller 的范式转换**：首次系统性地将伪影检测反馈用于引导更高质量的图像生成，开辟新研究方向
- **SynthScars 数据集填补空白**：首个同时提供像素级掩码、文本解释和伪影类型标签的全合成图像基准
- **多任务统一框架**：检测、定位、解释在一个 MLLM 中统一，比分离式方案更高效
- **`<SEG>` token 的语言引导分割**：LLM 自然语言描述与像素级分割无缝衔接

## 局限与展望

- 依赖预训练的 SAM 和 CLIP，对于极新生成器的图像可能需要更新基础模型
- 两阶段训练增加复杂度，可探索端到端联合训练
- 图像修复管线需要多轮迭代，效率待提升
- RichHF-18K 上的 F1 低于 LISA-v1-7B，跨域泛化仍有提升空间

## 相关工作与启发

- 与 LISA（CVPR 2024）的语言引导分割方法类似，但专门针对伪影检测任务
- 修复管线思路类似 Idea2Img 的迭代优化，但增加了空间定位信息
- SynthScars 的标注流程使用 Qwen2-VL-72B 进行质量过滤，值得参考

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — Defender→Controller 范式 + 高质量数据集
- 实验充分度：⭐⭐⭐⭐ — 4个基准 + 19个方法对比 + 鲁棒性分析
- 实用性：⭐⭐⭐⭐ — 可直接用于内容审核和生成质量提升
- 总体：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation](learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)
- [HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](../../ECCV2024/segmentation/frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)
- [Learning Camouflaged Object Detection from Noisy Pseudo Label](../../ECCV2024/segmentation/learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](../../NeurIPS2025/segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)

<!-- RELATED:END -->
