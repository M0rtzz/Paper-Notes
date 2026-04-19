---
title: >-
  [论文解读] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation
description: >-
  [AAAI 2026][图像分割][伪装图像生成] 提出CT-CIG，首个文本引导的可控伪装图像生成方法。利用VLM设计伪装揭示对话机制（CRDM）生成高质量文本提示，结合轻量控制网络和频率交互精炼模块（FIRM），在Stable Diffusion框架上生成逻辑合理、纹理真实的伪装图像，开创了Text-guided CIG新范式。
tags:
  - AAAI 2026
  - 图像分割
  - 伪装图像生成
  - 扩散模型
  - 文本引导
  - 频率交互
  - Vision Language Model
---

# Text-guided Controllable Diffusion for Realistic Camouflage Images Generation

**会议**: AAAI 2026  
**arXiv**: [2511.20218](https://arxiv.org/abs/2511.20218)  
**代码**: [github.com/NikoNairre/CT-CIG](https://github.com/NikoNairre/CT-CIG)  
**领域**: 分割  
**关键词**: 伪装图像生成, 扩散模型, 文本引导, 频率交互, Vision Language Model

## 一句话总结

提出CT-CIG，首个文本引导的可控伪装图像生成方法。利用VLM设计伪装揭示对话机制（CRDM）生成高质量文本提示，结合轻量控制网络和频率交互精炼模块（FIRM），在Stable Diffusion框架上生成逻辑合理、纹理真实的伪装图像，开创了Text-guided CIG新范式。

## 研究背景与动机

伪装（camouflage）是生物的本能生存策略——通过与环境融合使自身视觉上难以辨识。伪装图像生成（CIG）对于扩充伪装检测（COD）训练数据具有重要意义，但自然伪装图像采集困难。

**现有CIG方法的两大范式及其不足**：

**背景适配（Background fitting）**：改变物体的颜色和纹理以融入任意背景（DCI、LCG-Net、PTDiffusion）。问题在于：**破坏了物体外观**，且忽略了前景物体与背景环境之间的逻辑关系（如老虎脸出现在山中），产生的是视觉艺术而非自然伪装。

**前景引导（Foreground guiding）**：利用生成模型基于前景物体特征向外绘制背景（LAKE-RED、FACIG）。问题在于：**缺乏背景语义理解**，导致严重的纹理伪影，背景看起来不真实。

**核心洞察**：自然伪装不仅需要视觉一致性（颜色、纹理相似），还需要**逻辑合理性**——伪装物体与环境之间存在语义上合理的对应关系。这种逻辑关系无法从像素域直接学习，但可以通过语义域的文本提示显式地引入。

**关键挑战**：COD数据集缺少配对的文本描述，需要一种方法自动生成高质量、伪装感知的文本提示。

## 方法详解

### 整体框架

CT-CIG基于Stable Diffusion (SDXL)，接收三种输入：
- RGB伪装图像 $x \in \mathbb{R}^{3 \times h \times w}$
- 二值掩码 $c_f \in \mathbb{R}^{1 \times h \times w}$
- 通过CRDM+VLM生成的文本提示 $c$

流程：VAE编码图像到潜空间→添加高斯噪声→轻量控制器编码掩码→FIRM频率精炼→Cross Normalization归一化→UNet扩散去噪→VAE解码。仅训练控制器、FIRM和UNet交叉注意力的线性投影层（约4%参数）。

### 关键设计

#### 1. **伪装揭示对话机制（CRDM）**

核心思路：利用VLM的视觉感知和上下文理解能力，通过精心设计的多轮对话生成伪装感知的文本描述。

**预处理**：对所有图像使用随机半透明彩色轮廓线标注物体边界——这些边界恰好是伪装的关键所在。半透明效果既帮助VLM定位伪装物体，又保留了边界像素细节。

**对话设计**（4轮问答）：
- **问题1**：获取伪装物体的描述
- **问题2**：获取周围环境的描述及其与物体的关系
- **问题3**：将上述描述重组为详细提示 $T_{detail}$
- **问题4**：审查所有内容并总结为一句话 $T_{simple}$

**非伪装图像处理**：对于显著性物体或一般图像，问题2改为让VLM**想象**一个能成功伪装该物体的理想场景，然后生成相应提示。

$T_{detail}$ 用于训练（包含更丰富信息，防止灾难性遗忘），$T_{simple}$ 用于推理（增加生成多样性）。

**VLM选择**：经CLIPScore评估，Qwen2.5-VL-7B在文本-图像对齐上表现最优（0.3242），超过BLIP2、LLaVA、Gemma3。

#### 2. **频率交互精炼模块（FIRM）**

核心问题：二值掩码仅提供粗略的位置和几何线索，缺少空间层次和物体内部外观信息。控制器编码的 $x_{cf}$ 信息不足，可能产生纹理伪影和不自然幻觉。

设计思路：利用傅里叶变换从图像潜表征 $z_t$ 中学习高频纹理信息来增强控制特征。

流程：
1. 对 $x_{cf}$ 和 $z_t$ 进行FFT得到频率域表征
2. 用注意力生成器（2层卷积）从 $|z_t|$ 的频谱生成注意力图 $A$（先fftshift使频谱连续化以适配卷积）
3. 交互增强控制频谱：$\hat{x}_{facf} = \hat{x}_{cf} \otimes A$
4. 通过可学习门控自适应叠加精炼增益：$\hat{x}_{frcf} = \hat{x}_{cf} + gate \times (\hat{x}_{facf} - \hat{x}_{cf})$
5. IFFT变回特征域

设计动机：根据傅里叶谱理论，低频贡献整体结构信息，高频贡献纹理和精细模式。FIRM使控制特征获得来自图像的细节纹理表征，确保生成复杂伪装纹理的鲁棒性。

#### 3. **交叉归一化（Cross Normalization）**

FIRM精炼后的控制特征与噪声潜表征之间存在分布差异，可能导致颜色不稳定。CN通过对控制特征进行标准化并用潜表征的统计量进行仿射变换：

$$x'_{frcf} = \mu_z + \frac{x_{frcf} - \mu_{cf}}{\sqrt{\sigma^2_{cf} + \varepsilon}} \times \sigma_z$$

使最终控制信号与噪声潜表征分布一致，替代ControlNet中的"零卷积"层。

### 损失函数 / 训练策略

总损失结合条件扩散损失和LPIPS感知损失：

$$\mathcal{L} = \mathcal{L}_{SD} + \lambda_{Lpips} \cdot \mathcal{L}_{Lpips}$$

其中：
- $\mathcal{L}_{SD}$：标准的条件扩散噪声预测MSE损失
- $\mathcal{L}_{Lpips}$：LPIPS感知损失，最小化生成结果与输入图像的VGG特征差异
- $\lambda_{Lpips} = 1\text{e-3}$
- 控制器和FIRM学习率1e-4，UNet学习率5e-6
- 训练80 epochs，4×RTX A5000约8小时
- 控制缩放因子1.2

## 实验关键数据

### 主实验

| 范式 | 方法 | 伪装FID↓ | 显著FID↓ | 一般FID↓ | 总体FID↓ | 总体KID↓ | CLIP↑ |
|------|------|----------|----------|----------|----------|----------|-------|
| 背景适配 | LCGNet | 129.80 | 136.24 | 132.64 | 129.88 | 0.0550 | — |
| 前景引导 | LAKERED | 39.55 | 88.70 | 102.67 | 64.27 | 0.0355 | — |
| 文本引导 | ControlNet | 39.67 | 81.72 | 102.94 | 59.52 | 0.0227 | 0.2950 |
| 文本引导 | SOO | **30.92** | 89.46 | 117.31 | 59.75 | 0.0187 | 0.3043 |
| **文本引导** | **CT-CIG** | 30.59 | **81.60** | **104.46** | **52.88** | **0.0169** | **0.3243** |

CT-CIG在总体FID上以52.88大幅领先，CLIPScore最高表明最佳语义对齐。

### 消融实验

**FIRM和CN的作用**：

| 配置 | FID↓ | KID↓ | 说明 |
|------|------|------|------|
| w/o FIRM & CN | 32.37 | 0.0079 | 基线，纹理伪影明显 |
| w/o CN | 33.99 | 0.0114 | 缺少分布对齐 |
| w/o FIRM | 31.66 | 0.0080 | 缺少高频纹理细节 |
| **CT-CIG完整** | **30.59** | **0.0085** | 最佳 |

**文本提示配置的影响**：

| 配置 | CLIP↑ | FID↓ | KID↓ | 说明 |
|------|-------|------|------|------|
| 简单文本训练 | 0.3183 | 54.92 | 0.0387 | 灾难性遗忘，结果模糊 |
| 无物体轮廓 | 0.3247 | 39.24 | 0.0112 | CLIP高但不匹配形状引导 |
| 提及轮廓 | 0.3218 | 39.79 | 0.0138 | 产生线条画伪影 |
| **静默轮廓（ours）** | 0.3242 | **30.59** | **0.0085** | 最佳平衡 |

**VLM选择**（CLIPScore）：

| VLM | CLIP simple | CLIP detail |
|-----|-------------|-------------|
| BLIP2-2.7B | 0.2461 | 0.2859 |
| LLaVA-13B | 0.2986 | 0.2969 |
| Gemma3-4B | 0.3127 | 0.3136 |
| **Qwen2.5-VL-7B** | **0.3183** | **0.3242** |

### 关键发现

1. 文本引导范式全面优于背景适配和前景引导，语义理解是实现自然伪装的关键
2. 半透明轮廓标注在帮助VLM感知伪装的同时，必须在文本中保持"静默"——提及轮廓会误导生成
3. 详细提示训练+简单提示推理的策略，在防止遗忘的同时保持生成多样性
4. FIRM的高频纹理增强和CN的分布对齐在视觉质量上互补，二者都不可或缺

## 亮点与洞察

1. **开创Text-guided CIG新范式**：首次将伪装生成从纯视觉任务提升为视觉-语言联合任务，引入逻辑合理性约束
2. **CRDM设计精巧**：4轮对话逐步引导VLM从感知→理解→描述→总结，对伪装/非伪装图像设计不同对话策略
3. **半透明轮廓的巧思**：既辅助VLM定位伪装物体，又保留边界像素信息，平衡了定位辅助和信息保真
4. **频率域增强控制信号**：在扩散模型中引入傅里叶变换来增强信息贫乏的二值掩码控制，是少见但有效的设计
5. **参数效率高**：仅微调约4%参数即可适配伪装场景

## 局限与展望

1. 依赖COD数据集的掩码质量，掩码标注噪声会影响训练效果
2. VLM生成的文本提示质量有上限，复杂场景的描述可能不够准确
3. 生成图像的分辨率受限于SDXL（512×512），难以满足高分辨率需求
4. 训练数据量有限（LAKE-RED仅4040张训练图），可能限制泛化性
5. 未评估生成图像对下游COD任务的实际数据增强效果

## 相关工作与启发

- **LAKE-RED**（ECCV 2024）：前景引导CIG的代表，通过VQVAE和知识检索生成颜色一致的背景
- **ControlNeXt**：CT-CIG的控制网络基础，用轻量网络替代ControlNet的并行块
- **Qwen2.5-VL**：作为VLM骨干，其视觉-语言对齐能力直接决定文本提示质量
- **Camobj-LLaVA**：首个专注伪装场景理解的VLM，与CRDM的目标互补
- CRDM的多轮对话策略可推广到其他需要特定领域描述的图像生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ （文本引导伪装生成是新范式，但各组件相对标准）
- 实验充分度: ⭐⭐⭐⭐ （与11种方法比较，消融详尽，但缺少下游任务评估）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富，范式分类明确）
- 价值: ⭐⭐⭐⭐ （开创新范式，对COD数据增强有应用价值）

<!-- RELATED:START -->

## 相关论文

- [UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models](../../ECCV2024/segmentation/udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)
- [VSC: Visual Search Compositional Text-to-Image Diffusion Model](../../ICCV2025/segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](../../NeurIPS2025/segmentation/seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

<!-- RELATED:END -->
