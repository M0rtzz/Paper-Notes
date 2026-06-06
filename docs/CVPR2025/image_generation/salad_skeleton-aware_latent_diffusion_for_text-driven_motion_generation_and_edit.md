---
title: >-
  [论文解读] SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing
description: >-
  [CVPR 2025][图像生成][文本驱动动作生成] 提出 SALAD，一种骨骼感知的潜在扩散模型，通过骨骼-时间结构化的 VAE 和去噪器显式建模关节、帧与文本的细粒度交互，并利用交叉注意力图实现零样本文本驱动动作编辑。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本驱动动作生成"
  - "骨骼感知扩散"
  - "潜在空间"
  - "注意力调制"
  - "零样本编辑"
---

# SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing

**会议**: CVPR 2025  
**arXiv**: [2503.13836](https://arxiv.org/abs/2503.13836)  
**代码**: [项目页面](https://kaist-visual-ai-group.github.io/SALAD/)  
**领域**: 图像生成/动作生成  
**关键词**: 文本驱动动作生成, 骨骼感知扩散, 潜在空间, 注意力调制, 零样本编辑

## 一句话总结

提出 SALAD，一种骨骼感知的潜在扩散模型，通过骨骼-时间结构化的 VAE 和去噪器显式建模关节、帧与文本的细粒度交互，并利用交叉注意力图实现零样本文本驱动动作编辑。

## 研究背景与动机

文本驱动动作生成在游戏、电影和交互媒体中有重要应用。尽管扩散模型在此领域取得了显著进展，现有方法存在两个关键限制：(1) 将姿态表示为单一向量，忽略了关节间的空间交互；将文本压缩为单一向量，忽略了词级别的细微变化，导致生成结果丢失细节；(2) 预训练模型缺乏可解释的中间表示，下游编辑任务需要手动掩码、优化或微调等额外工作。

在图像领域，Prompt-to-Prompt 等方法已证明交叉注意力图可建立文本与空间布局的对应关系，从而实现零样本编辑。但动作生成领域由于过度简化的表示限制了文本-动作间的丰富交互，缺乏类似能力。SALAD 旨在通过骨骼-时间结构化的潜在空间和解耦的注意力机制同时解决这两个问题。

## 方法详解

### 整体框架

SALAD 包含三个组件：(1) 骨骼-时间 VAE 构建结构化潜在空间 $\mathbf{z} \in \mathbb{R}^{N' \times J' \times D}$；(2) 骨骼感知去噪器在此空间中进行文本条件扩散生成；(3) 基于交叉注意力调制的零样本编辑方法。

### 关键设计1：骨骼-时间 VAE

**功能**：构建保持骨骼和时间结构的紧凑动作潜在空间。

**核心思路**：使用骨骼-时间卷积（STConv）解耦关节和帧维度，分别对相邻关节用图卷积、相邻帧用 1D 卷积进行信息交换：$\text{STConv}(\mathbf{h}) = \text{SkelConv}(\mathbf{h}) + \text{TempConv}(\mathbf{h})$。通过骨骼-时间池化（STPool）降维，在骨骼维度上聚合相邻关节保持拓扑同胚、在时间维度上进行 1D 池化，最终保留 7 个原子关节（root, spine, head, 双臂, 双腿）。编码器将 $N \times J \times D$ 压缩为 $N' \times J' \times D$。

**设计动机**：直接在原始空间（N 帧 × J 关节 × D 维）上运行扩散模型面临维度灾难和计算瓶颈。骨骼-时间池化保持拓扑结构的同时压缩维度，使扩散采样更高效。

### 关键设计2：骨骼感知去噪器

**功能**：在结构化潜在空间中建模关节、帧和文本三者的细粒度交互。

**核心思路**：去噪器由 $L$ 层 Transformer 组成，每层包含：(1) 时间注意力（TempAttn）建模帧间关系；(2) 骨骼注意力（SkelAttn）建模关节间关系；(3) 交叉注意力（CrossAttn）与 CLIP 编码的文本词级特征交互。每个模块后接 FiLM 层基于扩散时间步调制特征。采用 $\mathbf{v}$-prediction 参数化：$\mathbf{v}_t = \alpha_t \epsilon - \sigma_t \mathbf{x}$，在高噪声水平下比 $\epsilon$-prediction 更稳定。

**设计动机**：分离骨骼和时间注意力使模型能独立处理空间关系（哪些关节协调运动）和时间关系（帧序列的节奏），而交叉注意力提供了与每个词 token 对每个骨骼-时间单元的细粒度关联。

### 关键设计3：注意力调制零样本编辑

**功能**：利用预训练 SALAD 的交叉注意力图实现无需微调的文本驱动动作编辑。

**核心思路**：四种调制策略——(1) 词替换：交换源/目标prompt的注意力图；(2) Prompt 精化：为追加词添加新注意力图以丰富语义；(3) 注意力重加权：放大/缩减特定词的注意力值；(4) 注意力镜像：交换对称身体部位（如左右臂）的注意力值产生镜像动作。

**设计动机**：与图像扩散模型类似，SALAD 的交叉注意力图捕获了文本词与动作骨骼-时间单元的对应关系，使得直接调制这些图即可实现编辑，无需额外优化或训练。

### 损失函数

VAE 训练：$\mathcal{L}_{\text{VAE}} = \mathcal{L}_\mathbf{m} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}$（动作重建 + 关节位置 + 关节速度 + KL 散度）。去噪器训练：$\mathcal{L}_{\text{denoiser}} = \|\hat{\mathbf{v}}_t - \mathbf{v}_t\|_2^2$（速度预测 MSE），辅以 classifier-free guidance。

## 实验关键数据

### 主实验：HumanML3D 文本驱动动作生成

| 方法 | R-Precision Top-1 ↑ | FID ↓ | MM-Dist ↓ | Diversity→ |
|------|-----|-----|-----|-----|
| MLD | 0.481 | 0.473 | 3.196 | 9.724 |
| MoMask | 0.521 | 0.045 | 2.958 | - |
| MotionGPT | 0.492 | 0.232 | 3.096 | 9.528 |
| **SALAD** | **0.524** | 0.064 | **2.926** | 9.549 |
| Real motion | 0.511 | 0.002 | 2.974 | 9.503 |

### KIT-ML 数据集

| 方法 | R-Precision Top-1 ↑ | FID ↓ | MM-Dist ↓ |
|------|-----|-----|-----|
| MLD | 0.390 | 0.404 | 3.204 |
| **SALAD** | **0.424** | **0.321** | **3.054** |

### 关键发现

- SALAD 在文本-动作对齐度（R-Precision, MM-Dist）上显著超越所有先前方法，证明了骨骼-时间结构化表示+词级交叉注意力的有效性。
- FID 仅次于 MoMask（0.064 vs 0.045），但在文本对齐度上更优，说明方法更注重语义精确性。
- 注意力镜像实验验证了交叉注意力图确实编码了身体部位与文本词的对应关系。
- 零样本编辑完全无需额外优化/微调，仅通过注意力图操作即可实现多样化编辑。

## 亮点与洞察

1. **结构化潜在空间**：7 个原子关节的骨骼-时间压缩既保持了拓扑信息又大幅降低了扩散采样的计算开销。
2. **从图像到动作的范式迁移**：成功将 Prompt-to-Prompt 的注意力调制编辑思路从 2D 图像迁移到 3D 动作领域。
3. **v-prediction 的选择**：平衡了 $\epsilon$ 和 $\mathbf{x}$ 预测的优势，在高噪声水平下更稳定。

## 局限与展望

- 骨骼池化到 7 个原子关节可能丢失手指等精细动作信息。
- 零样本编辑依赖注意力图的质量，复杂语义编辑可能不够精确。
- 当前仅处理单人动作，多人交互场景未涉及。
- FID 略逊于 token-based 方法（MoMask），生成质量还有提升空间。

## 相关工作与启发

- **Prompt-to-Prompt**：图像编辑中的注意力调制方法，被成功迁移到动作领域。
- **AnimatableGaussians / Skeleton-aware Networks**：骨骼-时间卷积架构的先驱，本文将其引入扩散模型的 VAE 组件。
- **MLD**：动作潜在扩散的先驱工作，SALAD 通过结构化潜在空间显著超越其性能。

## 评分

⭐⭐⭐⭐ — 骨骼-时间结构化的设计思路清晰，从 VAE 到去噪器再到编辑形成完整的技术栈。文本-动作对齐度 SOTA，零样本编辑能力是实际亮点。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[ICCV 2025\] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](../../ICCV2025/image_generation/float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
