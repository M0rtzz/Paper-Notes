---
title: >-
  [论文解读] Masked Representation Modeling for Domain-Adaptive Segmentation
description: >-
  [CVPR 2026][图像分割][无监督域自适应] 提出在潜在空间而非输入空间做掩码建模的辅助任务MRM，通过轻量级Rebuilder模块对编码器特征做掩码-重建并用分割损失监督，在GTA→Cityscapes上为四种UDA基线平均带来+2.3 mIoU提升，推理时零额外开销。
tags:
  - CVPR 2026
  - 图像分割
  - 无监督域自适应
  - 掩码表示建模
  - 语义分割
  - 辅助任务
  - 特征重建
---

# Masked Representation Modeling for Domain-Adaptive Segmentation

**会议**: CVPR 2026  
**arXiv**: [2509.13801](https://arxiv.org/abs/2509.13801)  
**代码**: [https://github.com/Wenlve-Zhou/MRM](https://github.com/Wenlve-Zhou/MRM)  
**领域**: 语义分割 / 无监督域自适应  
**关键词**: unsupervised domain adaptation, masked representation modeling, semantic segmentation, auxiliary task, Rebuilder

## 一句话总结

提出 Masked Representation Modeling (MRM)，在编码器输出的潜在特征空间做随机掩码与重建，以像素分类损失监督重建结果，作为即插即用辅助任务在四种 UDA 基线上平均提升 +2.3/+2.8 mIoU (GTA→CS / Synthia→CS)，推理时零额外开销。

## 研究背景与动机

**领域现状**：无监督域自适应 (UDA) 语义分割旨在将有标注源域（如合成数据 GTA）的知识迁移到无标注目标域（如真实场景 Cityscapes），主流方法包括对抗训练、自训练和高效架构设计。辅助自监督任务（如对比学习）已被证明能增强特征判别性，但另一重要范式——掩码图像建模 (MIM, 如 MAE) 在 UDA 分割中几乎没有被探索。

**现有痛点**：MIM 未被采用有两个根本原因。第一，输入结构约束：MIM 需要在输入端遮掉 patch 并只将可见 patch 送入编码器，这与 DeepLab、DAFormer 等需要处理完整图像的分割架构直接冲突。第二，优化目标冲突：MIM 的目标是逐元素重建被掩码区域的像素值，这一低层次重建目标与分割所需的高层语义分类目标不一致，可能引入优化干扰。

**核心矛盾**：掩码建模能带来全局上下文理解和特征鲁棒性提升，但其输入端操作和像素重建目标阻碍了它在 UDA 分割中的应用。

**本文目标** 如何在保留掩码建模优势（全局推理、特征正则化）的同时，消除其与分割架构的兼容性问题和与分割任务的目标冲突。

**切入角度**：将掩码操作从输入空间挪到潜在特征空间——编码器正常处理完整图像，在编码器输出的特征图上执行随机掩码，然后用轻量模块重建被掩码特征，最后将重建特征送入原始分割解码器做像素分类。这样既避免了修改输入流程，又让辅助任务与主任务共享同一优化目标。

**核心 idea**：在特征空间而非输入空间做掩码建模，用分类损失而非回归损失监督重建。

## 方法详解

### 整体框架

MRM 作为辅助训练任务嵌入现有 UDA 流程，不修改原有网络结构。完整流程：(1) 目标域图像 $x^t$ 通过编码器 $E(\cdot)$ 得到特征 $f^t$；(2) Rebuilder $R(\cdot)$ 对 $f^t$ 做掩码、重建，输出重建特征 $f^r$；(3) $f^r$ 送入原始分割解码器 $D(\cdot)$ 做像素分类，用伪标签 $\tilde{y}$ 计算交叉熵损失。训练完成后，Rebuilder 被完全移除，推理与原始模型一致。总损失为：

$$\mathcal{L}_{overall} = \mathcal{L}_{sup} + \mathcal{L}_{uda} + \lambda \mathcal{L}_{mrm}$$

其中 $\mathcal{L}_{mrm}$ 是重建特征经解码器后的像素分类损失，$\lambda=1.0$。

### 关键设计

1. **潜在空间掩码 (Representation-level Masking)**：

    - 功能：在编码器输出的特征图上执行随机块掩码，而非在输入图像上掩码
    - 核心思路：编码器处理完整输入图像，输出特征 $f^t \in \mathbb{R}^{C \times H \times W}$；在特征图上生成二值掩码 $M$，用可学习的 mask token 填充被掩码位置，保留未掩码位置的原始特征
    - 设计动机：避免修改编码器的输入处理流程，使得 MRM 与任意分割架构（CNN-based 的 DeepLab 或 Transformer-based 的 DAFormer）完全兼容。与 MAE 只送可见 patch 给编码器形成根本区别

2. **任务对齐的分类重建目标 (Task-aligned Classification Objective)**：

    - 功能：用像素级分类交叉熵损失（而非像素回归 MSE）监督重建质量
    - 核心思路：将重建后的特征 $f^r$ 直接送入分割解码器 $D(\cdot)$ 做逐像素分类，用目标域伪标签计算交叉熵。这使辅助任务与主任务共享同一优化方向
    - 设计动机：消融实验表明，像素回归目标（如 MAE）反而降低性能（-0.3 mIoU），而分类目标带来 +3.8 mIoU 提升。低层次重建与高层语义分类存在根本性目标冲突，辅助任务必须与主任务对齐

3. **轻量 Rebuilder 模块**：

    - 功能：对编码器特征做掩码-重建，训练后移除，推理零开销
    - 核心思路：包含四个步骤——(a) Representation Embedding：线性层调整通道维度 + 双线性插值调整空间维度至 $16 \times 16 \times 512$；(b) Masking：40% 均匀随机掩码，用可学习 mask token 填充；(c) Transformer Blocks：仅 2 个 Transformer 块加绝对位置编码处理序列；(d) Projector：转置卷积恢复原始分辨率。最终通过残差融合 $f^r = M^s \odot f^o + (1-M^s) \odot f^t$ 仅替换被掩码区域
    - 设计动机：保持极致轻量以避免训练不稳定（实验表明超过 2 个 Transformer 块性能反而下降），同时对多尺度模型（如 DAFormer）仅从最后阶段特征通过上采样生成多尺度重建，不需要在每个阶段都实例化 Rebuilder

### 损失函数 / 训练策略

MRM 损失 $\mathcal{L}_{mrm}$ 为目标域伪标签的像素级交叉熵，权重 $\lambda=1.0$，在 0.1–2.0 范围内性能稳定。MRM 仅在目标域图像上应用——在源域应用 MRM 反而有害（会将特征偏向源域分布，削弱域对齐）。MRM 必须同时训练编码器和解码器才能获得最佳效果，冻结任一组件都会显著降低增益。Rebuilder 学习率 $2 \times 10^{-4}$。

## 实验关键数据

### 主实验

GTA→Cityscapes 基准：

| 基线方法 | w/o MRM | w/ MRM | 提升 |
|---------|---------|--------|------|
| DACS | 52.1 | 55.9 | +3.8 |
| DAFormer | 68.3 | 70.3 | +2.0 |
| HRDA | 73.8 | 75.4 | +1.6 |
| MIC | 75.9 | **77.5** | +1.6 |

Synthia→Cityscapes 基准（13类 mIoU）：

| 基线方法 | w/o MRM | w/ MRM | 提升 |
|---------|---------|--------|------|
| DACS | 48.3 | 55.8 | +7.5 |
| DAFormer | 60.9 | 62.6 | +1.7 |
| HRDA | 65.8 | 67.1 | +1.3 |
| MIC | 67.3 | **68.1** | +0.8 |

MIC+MRM 在 GTA→CS 上达到 77.5 mIoU，超越此前最优 QuadMix (76.1) +1.4 mIoU。

### 消融实验

Rebuilder 设计参数消融（DACS + DeepLabV2-ResNet101, GTA→CS）：

| 消融维度 | 设置 | mIoU |
|---------|------|------|
| 嵌入维度 | 128 / 256 / **512** / 768 | 54.7 / 55.1 / **55.9** / 53.6 |
| Transformer 块数 | 1 / **2** / 4 / 8 | 55.4 / **55.9** / 54.4 / 52.9 |
| 空间分辨率 H'=W' | 8 / **16** / 32 / 64 | 55.6 / **55.9** / 54.1 / OOM |
| 掩码率 | 最优 **40%** | **55.9** |
| 权重 λ | 0.1 / 0.5 / **1.0** / 2.0 / 10.0 | 54.7 / 55.8 / **55.9** / 55.4 / 52.1 |

重建目标对比：

| 重建目标 | mIoU | 提升 |
|---------|------|------|
| Baseline (无 MRM) | 52.1 | — |
| 像素回归 (MAE 风格) | 51.8 | -0.3 |
| 教师特征重建 | 53.5 | +1.4 |
| 学生特征重建 | 53.7 | +1.6 |
| **像素分类 (本文)** | **55.9** | **+3.8** |

跨架构泛化（GTA→CS）：

| 编码器 | 解码器 | UDA 方法 | w/o MRM | w/ MRM |
|--------|--------|---------|---------|--------|
| ResNet-50 | DeepLabV2 | DACS | 52.0 | 55.1 (+3.1) |
| ResNet-101 | DeepLabV3+ | DACS | 54.7 | 59.3 (+4.6) |
| ResNet-101 | DeepLabV2 | MIC | 64.2 | 67.1 (+2.9) |
| MiT-B2 | DAFormer Head | DAFormer | 64.2 | 66.3 (+2.1) |
| MiT-B3 | DAFormer Head | MIC | 73.6 | 75.8 (+2.2) |

### 关键发现

- 仅掩码不重建反而有害（-0.2 mIoU），说明特征空间的掩码造成不可逆语义丢失，重建过程是关键
- MRM 仅在目标域有效（+3.8），仅源域 +0.8，源+目标域 +3.1——MRM 本质是目标域自适应正则化，而非通用自监督
- 编码器和解码器必须共同训练：仅训练编码器 +2.8，仅训练解码器 +2.1，共同训练 +3.8，证实 MRM 的优势在于全链路优化
- 最优掩码率 40% 远低于 MAE 的 75%——Rebuilder 容量较小，过高掩码率使语义信息丧失不可逆

## 亮点与洞察

- **极致简洁的核心设计**：将掩码操作从输入空间挪到特征空间，将重建目标从像素回归换成分类损失——两个简单改动解决了 MIM 在 UDA 分割中的两大阻碍（架构不兼容 + 目标不对齐），且推理完全无开销。这种"在正确的空间做正确的事"的思路值得借鉴。
- **反直觉的实验发现具有指导意义**：像素重建目标对分割有害、辅助任务仅在目标域有效、对比学习只优化编码器而 MRM 同时优化编解码器——这些发现对设计 UDA 辅助任务有普遍参考价值。
- **真正的即插即用泛化性**：在 CNN (DeepLabV2/V3+) 和 Transformer (DAFormer) 架构上均有效，跨 5 种编码器-解码器-UDA 方法组合一致提升 +2.1~+4.6 mIoU，证明方法的通用性而非特定配置的调参结果。

## 局限与展望

- Rebuilder 容量有限（仅 2 个 Transformer 块），消融表明增大到 4/8 块时性能反降（训练不稳定），更强且稳定的 Rebuilder 设计是未来方向
- 掩码策略较简单（均匀随机），语义引导的自适应掩码（如对高不确定性区域加大掩码率）可能带来更大增益
- 仅验证了 UDA 设置，能否推广到 domain generalization、source-free UDA、test-time adaptation 等更广泛设置有待探索
- 依赖伪标签质量，在伪标签噪声极大的早期训练阶段 MRM 的有效性值得进一步分析

## 相关工作与启发

- **vs MAE/MIM**：MAE 在输入空间掩码并重建像素，与分割架构不兼容且目标不对齐；MRM 在特征空间掩码并用分类目标重建，完美兼容且效果显著更好（像素回归 -0.3 vs 像素分类 +3.8）
- **vs 对比学习 (SePiCo, PiPa)**：对比学习仅增强编码器特征判别性，不训练解码器；MRM 同时优化编码器和解码器，提供更全面的正则化
- **vs MIC**：MIC 在图像空间做掩码一致性（类似 CutOut），MRM 在特征空间做掩码重建，两者机制正交互补——MIC+MRM 组合达到 77.5 mIoU，超过各自单独使用
- **信息瓶颈视角**：掩码相当于结构化噪声注入，减少 $I(Z;X)$（压缩冗余信息）同时保留 $I(Z;Y)$（保持任务相关信息），提升特征的域不变性

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心设计简洁巧妙——组件不全新但组合方式解决了实际难题
- 实验充分度: ⭐⭐⭐⭐⭐ 四种基线×两个基准、十余项消融、五种架构泛化验证，非常完整
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导清晰，设计选择的逻辑链连贯，消融分析系统
- 价值: ⭐⭐⭐⭐ 即插即用、推理零开销、跨架构泛化，对 UDA 分割社区有直接实用价值
---
title: >-
  [论文解读] Masked Representation Modeling for Domain-Adaptive Segmentation
description: >-
  [CVPR 2026][图像分割][无监督域自适应] 提出在潜在空间而非输入空间做掩码建模的辅助任务MRM，通过轻量级Rebuilder模块对编码器特征做掩码-重建并用分割损失监督，在GTA→Cityscapes上为四种UDA基线平均带来+2.3 mIoU提升，推理时零额外开销。
tags:
  - CVPR 2026
  - 图像分割
  - 无监督域自适应
  - 掩码表示建模
  - 语义分割
  - 辅助任务
  - 特征重建
---

# Masked Representation Modeling for Domain-Adaptive Segmentation

**会议**: CVPR 2026  
**arXiv**: [2509.13801](https://arxiv.org/abs/2509.13801)  
**代码**: 无  
**领域**: 语义分割 / 域自适应 / 自监督学习  
**关键词**: 无监督域自适应, 掩码表示建模, 语义分割, 辅助任务, 特征重建  

## 一句话总结
提出在潜在空间而非输入空间做掩码建模的辅助任务MRM，通过轻量级Rebuilder模块对编码器特征做掩码-重建并用分割损失监督，在GTA→Cityscapes上为四种UDA基线平均带来+2.3 mIoU提升，推理时零额外开销。

## 背景与动机
无监督域自适应(UDA)语义分割需要将源域标注知识迁移到无标注目标域。对比学习等辅助自监督任务已被证明能提升特征判别性，但掩码图像建模(MIM, 如MAE)在UDA分割中几乎无人探索。核心原因有二：(1) MIM需要修改输入结构（遮掉patch只送可见部分），与DeepLab、DAFormer等分割架构不兼容；(2) MIM的像素级重建目标与分割的语义分类目标不一致，存在优化冲突。

## 核心问题
如何将掩码建模的优势（全局上下文理解、特征鲁棒性）引入UDA语义分割，同时解决架构兼容性和目标对齐两大问题？

## 方法详解

### 整体框架
MRM作为即插即用的辅助任务，嵌入现有UDA流程。输入完整图像通过编码器得到特征 $f_t$，然后在**特征空间**中随机掩码40%的区域，由轻量Rebuilder重建被掩码部分，重建后的特征送入分割解码器做像素级分类，用伪标签监督。训练结束后Rebuilder移除，推理与原始模型完全一致。总损失：$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{uda} + \lambda \mathcal{L}_{mrm}$。

### 关键设计
1. **潜在空间掩码而非输入空间掩码**：编码器处理完整输入，在编码器输出的特征图上做随机块掩码。这保证了与任何分割架构（CNN/Transformer）的兼容性——不需要修改编码器的输入处理方式。与MAE形成鲜明对比：MAE只送可见patch给编码器，而MRM送完整图像。

2. **任务对齐的重建目标**：不像MAE那样重建正像素值，MRM将重建后的特征送入分割解码器做像素级分类（cross-entropy loss + 伪标签），使辅助任务的优化目标与主任务完全一致。消融实验证实，像素级回归反而有害（-0.3 mIoU），而分类目标带来+3.8 mIoU提升。

3. **轻量Rebuilder设计**：包含特征嵌入（线性变换+空间插值到16×16×512）、掩码/填充（可学习mask token替代被掩码区域）、少量Transformer块（仅2个）、投影器（转置卷积恢复原始分辨率）。重建后通过残差融合：$f_r = M_s \odot f_o + (1-M_s) \odot f_t$，仅替换被掩码区域。训练时Rebuilder与主网络联合优化但推理时完全移除。

### 损失函数 / 训练策略
MRM损失为目标域伪标签的cross-entropy分类损失，权重$\lambda=1.0$。仅在目标域图像上应用MRM（源域MRM反而有害——会将特征偏向源域分布）。关键发现：MRM必须同时训练编码器和解码器才能获得最佳效果，冻结任一者都会降低增益。

## 实验关键数据
| 基线方法 | GTA→CS (baseline) | GTA→CS (+MRM) | 提升 | Synthia→CS (+MRM) | 提升 |
|--------|------|------|------|------|------|
| DACS | 52.1 | 55.9 | +3.8 | 55.8 | +7.5 |
| DAFormer | 68.3 | 70.3 | +2.0 | 62.6 | +1.7 |
| HRDA | 73.8 | 75.4 | +1.6 | 67.1 | +1.3 |
| MIC | 75.9 | **77.5** | +1.6 | 68.1 | +0.8 |

MIC+MRM达到77.5 mIoU，超越当时所有SOTA方法（QuadMix 76.1、GANDA 74.5）。

### 消融实验要点
- **掩码率40%最优**：低于MAE的75%，因为MRM的Rebuilder容量更小，过高掩码率使语义信息丧失不可逆
- **仅掩码无重建有害(-0.2)**：说明特征空间的掩码造成不可逆语义丢失，重建过程是关键
- **重建目标对比**：像素回归(-0.3) < 教师特征重建(+1.4/+1.6) < **像素分类(+3.8)**，辅助任务必须与主任务目标对齐
- **应用域选择**：仅目标域(+3.8) > 源+目标域(+3.1) > 仅源域(+0.8)，MRM的本质是目标域自适应正则化
- **跨架构泛化**：ResNet50/101、MiT-B2/B3、DeepLabV2/V3+均有效，增益+2.1~+4.6

## 亮点
- **极致简洁**：一个公式说清楚核心设计（潜在空间掩码+分类重建），且完全即插即用，推理零开销
- MRM通过information bottleneck视角的分析很有说服力：掩码相当于结构化噪声注入，减少$I(Z;X)$同时保留$I(Z;Y)$
- 发现"像素重建目标对分割任务有害"这一反直觉结论对社区有参考价值
- 仅在目标域应用MRM才有效，揭示了辅助任务在UDA中的正确使用方式

## 局限与展望
- Rebuilder容量有限（仅2个Transformer块），扩大容量时训练不稳定
- 仅验证了UDA设置，能否推广到domain generalization、source-free UDA等更广泛设置未知
- 掩码策略较简单（均匀随机），语义引导的掩码可能带来更大增益
- 仅适用于像素级分类任务，深度估计、全景分割等需要进一步研究

## 与相关工作的对比
- **vs MAE/MIM**：MAE在输入空间掩码并重建像素，与分割架构不兼容且目标不对齐。MRM在特征空间掩码并用分类目标重建，完美兼容且效果更好
- **vs 对比学习辅助任务（SePiCo, PiPa）**：对比学习只增强编码器特征，MRM同时训练编码器和解码器，提供更全面的正则化
- **vs MIC**：MIC在图像空间做掩码一致性（类似高比例CutOut），MRM在特征空间做掩码重建，两者正交互补——MIC+MRM达到77.5 mIoU

## 启发与关联
- "将复杂的自监督任务在潜在空间而非输入空间执行"这一思路可以推广到其他视觉任务——比如视频理解中的时序掩码建模
- 辅助任务的目标必须与主任务对齐——这对设计新的预训练或微调策略有指导意义
- 可以考虑将MRM与知识蒸馏结合，用教师模型的特征作为重建目标的增强

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心思想（特征空间掩码+分类重建）简洁但有效，虽然组件不全新，组合方式巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 四种基线×两个基准、详尽消融、多架构泛化验证、理论分析，非常完整
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，motivation和设计选择的逻辑链非常连贯
- 价值: ⭐⭐⭐⭐ 作为即插即用模块，对UDA分割社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)

<!-- RELATED:END -->
