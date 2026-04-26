---
title: >-
  [论文解读] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework
description: >-
  [ICML2025][图像生成][Concept Erasing] 提出 Co-Erasing，首次将图像监督引入概念擦除流程，通过文本-图像协同的负引导和文本引导的图像概念精炼模块，在保持良性生成质量（usability）的同时显著提升不良概念的擦除效果（efficacy）。
tags:
  - ICML2025
  - 图像生成
  - Concept Erasing
  - 扩散模型
  - Text-Image Collaboration
  - Negative Guidance
  - 注意力机制
---

# One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework

**会议**: ICML2025  
**arXiv**: [2505.11131](https://arxiv.org/abs/2505.11131)  
**代码**: [Ferry-Li/Co-Erasing](https://github.com/Ferry-Li/Co-Erasing)  
**领域**: 扩散模型安全 / 概念擦除  
**关键词**: Concept Erasing, Diffusion Model Safety, Text-Image Collaboration, Negative Guidance, Cross-Attention

## 一句话总结

提出 Co-Erasing，首次将图像监督引入概念擦除流程，通过文本-图像协同的负引导和文本引导的图像概念精炼模块，在保持良性生成质量（usability）的同时显著提升不良概念的擦除效果（efficacy）。

## 研究背景与动机

文本到图像扩散模型（如 Stable Diffusion）可能生成 NSFW 等有害内容。**概念擦除（Concept Erasing）** 旨在通过微调模型权重，使其无法生成指定的不良概念。现有方法（ESD、AdvUnlearn、SalUn 等）完全依赖文本 prompt 作为擦除指导，面临两个核心困境：

**困境一：文本-图像模态鸿沟（Modality Gap）**
- 语义上完全无关的词（如 "rhodesian"、"birth"）仍可诱导模型生成裸露内容
- 文本空间中的擦除无法覆盖图像空间中所有触发路径

**困境二：概念纠缠与文本描述有限性**
- 用 ESD 框架测试，从 1 个词扩展到 10 个词描述 nudity，擦除效果很快饱和
- 更多文本描述反而误伤无关概念，导致 FID 恶化

作者通过 CLIP 相似度实验验证：干净模型自生成的 NSFW 图像与残余不良生成的语义相似度远高于文本 "nudity"，说明**图像比文本更能代表模型对不良概念的内在知识**。

## 方法详解

### 总体框架

Co-Erasing 在 ESD 的负引导框架上引入图像分支，核心修改有三处：

1. **解耦交叉注意力**：文本和图像分别通过独立 K/V 投影与 U-Net 交互
2. **文本引导的图像概念精炼**：用注意力机制从图像中提取与目标概念相关的视觉特征
3. **自生成图像作为视觉模板**：用干净模型按 "a photo of c" 生成不良图像作为擦除引导

### 解耦交叉注意力（Decoupled Cross-Attention）

文本嵌入 $c_{\text{text}} \in \mathbb{R}^{b \times 77 \times 768}$ 和图像嵌入 $c_{\text{image}} \in \mathbb{R}^{b \times 4 \times 768}$ 分别与共享 Query 做交叉注意力：

$$\mathbf{Z}_t^{\text{text}} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}, \quad \mathbf{Z}_t^{\text{image}} = \text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{K}')^\top}{\sqrt{d}}\right)\mathbf{V}'$$

其中文本和图像各有独立的 $\mathbf{W}_k, \mathbf{W}_v$ / $\mathbf{W}'_k, \mathbf{W}'_v$，最终 $\mathbf{Z}_t^{\text{att}} = \mathbf{Z}_t^{\text{text}} + \mathbf{Z}_t^{\text{image}}$。

**关键设计**：图像分支仅在训练时使用，推理时模型结构不变——通过全参微调将概念知识从 U-Net 中移除。

### 文本引导的图像概念精炼（Text-Guided Image Concept Refinement）

图像包含多个概念（如一张含教堂的图也有树木），直接使用会误伤无关概念。精炼模块通过交叉注意力提取与目标文本最相关的视觉特征：

$$c_{\text{image}} = \text{Softmax}\left(\frac{\mathbf{Q}^r (\mathbf{K}^r)^\top}{\sqrt{d^r}}\right)\mathbf{V}^r$$

其中 $\mathbf{Q}^r = \mathcal{E}_{\text{image}}(\mathbf{X})$ 为图像编码，$\mathbf{K}^r = \mathbf{V}^r = \mathcal{E}_{\text{text}}(\mathbf{Y})$ 为目标概念的文本编码。效果类似用文本作为注意力 mask 来聚焦图像中概念相关的区域（如从教堂图中聚焦十字架和哥特窗而忽略树木）。

### 负引导擦除目标

基于 ESD 的负引导公式，结合 Tweedie 公式：

$$\epsilon_{\theta^*}(\mathbf{Z}_t, c, t) \leftarrow \epsilon_\theta(\mathbf{Z}_t, t) - \eta \left[\epsilon_\theta(\mathbf{Z}_t, c, t) - \epsilon_\theta(\mathbf{Z}_t, t)\right]$$

其中 $c = [c_{\text{text}}, c_{\text{image}}]$，$\eta$ 为引导尺度。训练时对 U-Net 做全参微调，减小模型在给定文本 + 图像条件下生成目标概念的概率。

### 自生成图像策略

使用模型自身生成的图像而非真实数据集，理由：
- t-SNE 可视化表明，自生成图像的分布与模型内部概念知识更一致
- 真实 NSFW 数据集与模型学到的分布存在统计差异
- 每次迭代从预生成的 $n$ 张图像中随机采样一张

## 训练与实验设置

- **基础模型**：Stable Diffusion v1.4
- **图像编码器**：预训练 CLIP 图像编码器（冻结），仅训练时使用
- **擦除任务**：nudity（裸露）、style（Van Gogh 风格）、object（parachute/church/tench）
- **图像来源**：干净 SD 按 "a photo of c" 生成 $n$ 张图像（具体 $n$ 值见附录）
- **竞争方法**：ESD、FMN、AC、UCE、SPM、SH、ED、SalUn、AdvUnlearn（共 9 种）
- **评价指标**：
    - 擦除效果：pre-ASR、ASR、P4D、CCE（越低越好）
    - 生成质量：FID（越低越好）、CLIP Score（越高越好）
    - 鲁棒性：Ring-A-Bell（RAB）red-team 攻击

## 主要结果

### 擦除效果 vs. 生成质量

Co-Erasing 在 efficacy-usability 权衡上全面优于现有方法：

| 维度 | Co-Erasing 优势 |
|------|--------------|
| 擦除裸露 | ASR 从 ESD 的 76.05% 降至 16.96%，同时 FID 仅微升至 18.77（ESD 为 17.29） |
| 风格擦除 | Van Gogh 擦除效果与 SalUn 相当，但 FID/CLIP 显著更好 |
| 物体擦除 | parachute/church/tench 擦除成功率高，且无显著质量退化 |

### 消融实验（nudity 擦除）

| 配置 | pre-ASR↓ | ASR↓ | FID↓ | CLIP↑ |
|------|---------|------|------|-------|
| text only | 20.42 | 76.05 | 17.29 | 0.302 |
| text + image (no refine) | 4.30 | 32.60 | 22.98 | 0.301 |
| image only | 5.08 | 41.52 | 25.82 | 0.298 |
| text + image + refine | **0.85** | **16.96** | **18.77** | **0.302** |

精炼模块使 FID 从 22.98 大幅降低至 18.77（接近 text-only），同时 ASR 进一步降至 16.96%。

### 跨框架迁移（RAB 攻击）

Co-Erasing 可作为即插即用模块嵌入其他擦除框架：
- ESD + Co-Erasing：RAB_K16 从 0.47 降至 0.18
- MACE + Co-Erasing：RAB 三项均降至 0.00，FID 几乎不变
- SLD + Co-Erasing：I2P 数据集上 nudity 检测从 125 降至 22

### 多概念擦除

与 MACE 结合擦除 5 类物体（dog/cat/bird/fish/horse），$H_c$ 综合指标在大部分类别上持平或提升，证明方法的可扩展性。

## 局限与展望

1. **自生成图像质量依赖**：擦除效果取决于干净模型能否准确生成目标概念的代表性图像；对于模型本身表征不佳的概念，图像引导的提升可能有限
2. **训练开销增加**：引入图像编码器分支和精炼模块增加了训练时的计算量，尽管推理不受影响
3. **仅在 SD v1.4（UNet 架构）上验证**：未扩展到 DiT 架构（如 FLUX、SD3）或更大规模模型
4. **多概念同时擦除的可扩展性**：虽与 MACE 兼容，但每个概念需独立生成图像模板
5. **擦除仍非完美**：ASR 仍有 16.96%（nudity），存在漏网之鱼

## 可复现性要点

- 代码已开源：[github.com/Ferry-Li/Co-Erasing](https://github.com/Ferry-Li/Co-Erasing)
- 基于 Stable Diffusion v1.4，图像分支使用预训练 CLIP 图像编码器
- 自生成图像使用 "a photo of c" 模板生成
- 评估依赖 NudeNet（nudity 检测）、UDA（对抗 prompt 生成）、CLIP Score、FID 等标准工具
- 训练细节（学习率、迭代次数、$\eta$ 值等）在论文附录 A.4 中提供

## 个人点评

**优点**：
- 从模态鸿沟角度分析纯文本擦除方法的不足，动机论证清晰有说服力（实验设计巧妙）
- 文本引导精炼模块设计优雅，消融实验证明它是 usability 保持的关键（FID 从 22.98 降至 18.77）
- 框架通用性好，可即插即用到 ESD/SLD/MACE 等多种基线

**不足**：
- 对 "modality gap" 的定义和分析偏直觉性，缺乏理论支撑
- 自生成图像的数量 $n$、采样策略等超参敏感性分析不够充分
- 在 DiT 架构和 SDXL 等主流新模型上的验证缺失，限制了实际应用价值

**启发**：
- "用模型自身生成的不良样本来教模型遗忘" 这一思路简洁有效，类似 self-play 的理念
- 精炼模块的 text-query-image-as-key 的设计可迁移到其他需要概念级图像特征提取的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次引入图像模态辅助概念擦除)
- 实验充分度: ⭐⭐⭐⭐ (消融完整，多任务多框架验证)
- 写作质量: ⭐⭐⭐⭐ (动机分析清晰，结构工整)
- 价值: ⭐⭐⭐⭐ (AI 安全方向实用性强，但模型覆盖面待扩展)

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](../../CVPR2025/image_generation/from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/mca_ctrl_attention_control_customization.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](../../NeurIPS2025/image_generation/one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)

<!-- RELATED:END -->
