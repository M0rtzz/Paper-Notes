---
title: >-
  [论文解读] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement
description: >-
  [CVPR 2025][语义分割][SAM鲁棒性] GleSAM将预训练潜在扩散模型（LDM）的去噪能力引入SAM的隐空间，通过单步去噪增强低质量图像的特征表示，实现对任意质量图像的鲁棒分割。 SAM/SAM2虽然在清晰图像上表现出色，但在真实场景中常见的低质量图像（噪声、模糊、压缩失真等）上性能显著下降…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "SAM鲁棒性"
  - "低质量图像分割"
  - "潜在扩散模型"
  - "特征增强"
  - "退化鲁棒"
---

# GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement

**会议**: CVPR 2025  
**arXiv**: [2503.12507](https://arxiv.org/abs/2503.12507)  
**代码**: 即将开源  
**领域**: 图像分割  
**关键词**: SAM鲁棒性, 低质量图像分割, 潜在扩散模型, 特征增强, 退化鲁棒

## 一句话总结

GleSAM将预训练潜在扩散模型（LDM）的去噪能力引入SAM的隐空间，通过单步去噪增强低质量图像的特征表示，实现对任意质量图像的鲁棒分割。

## 研究背景与动机

SAM/SAM2虽然在清晰图像上表现出色，但在真实场景中常见的低质量图像（噪声、模糊、压缩失真等）上性能显著下降。现有的RobustSAM等方法通过一致性学习来增强退化特征，但在严重退化和复合退化情况下仍表现不佳。

核心观察：SAM从严重退化图像中提取的隐空间特征包含大量噪声，破坏了原始表示。低质量与高质量特征之间的巨大差距使得一致性学习难以收敛。

关键洞察：预训练的LDM（如Stable Diffusion）在大规模数据上学到了强大的表示先验和去噪能力，可以将其引入SAM的隐空间来"修复"退化特征。

与图像复原+分割的级联方法不同，GleSAM**直接在特征空间**进行增强，更高效且避免了图像域复原的信息损失。

## 方法详解

### 整体框架

GleSAM在SAM的编码器和解码器之间插入一个生成式隐空间增强（GLE）模块。低质量图像经SAM编码器得到退化特征 $z_L$，GLE模块通过单步去噪将其增强为接近高质量特征的 $\hat{z}_H$，然后送入微调的解码器生成精确mask。训练分两步：先训练U-Net去噪重建，再微调解码器对齐增强特征。

### 关键设计

**1. 隐空间单步去噪增强（GLE）**

- **功能**：将SAM编码器输出的低质量特征 $z_L$ 通过单步扩散去噪恢复为高质量特征
- **核心思路**：将低质量特征 $z_L$ 视为高质量特征 $z_H$ 的噪声版本，在扩散时间步 $T$ 处进行单步去噪：$\hat{z}_H = \frac{z_L - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta(z_L; T)}{\sqrt{\bar{\alpha}_T}}$。使用预训练LDM的U-Net作为去噪骨干，加LoRA层微调
- **设计动机**：多步去噪虽然质量更好但计算开销大，单步去噪在效率与效果之间取得良好平衡。从 $z_L$ 出发而非随机噪声，保留了原始内容信息

**2. 特征分布对齐（FDA）**

- **功能**：弥合SAM编码器特征空间与LDM的VAE隐空间之间的分布差距
- **核心思路**：引入自适应缩放权重 $\gamma$ 调整分割特征的方差以匹配VAE隐空间分布：$\hat{z}_H = \frac{\gamma z_L - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta(\gamma z_L; T)}{\gamma \sqrt{\bar{\alpha}_T}}$，去噪后除以 $\gamma$ 恢复原始分布
- **设计动机**：直接将SAM特征输入LDM的U-Net会因分布不匹配而无法发挥去噪能力。简单的缩放操作即可有效对齐分布

**3. 通道复制与扩展（CRE）**

- **功能**：解决LDM U-Net（4通道输入/输出）与SAM隐空间（256通道）的通道维度不匹配
- **核心思路**：复制并拼接U-Net首尾层的预训练4通道权重至256通道维度，冻结这些权重，仅通过LoRA层适配分割特征
- **设计动机**：实验发现微调新的首尾层或encoder-decoder会破坏预训练泛化能力。权重复制保留了预训练知识，LoRA提供轻量适配

### 损失函数

- U-Net训练阶段：MSE重建损失 $\mathcal{L}_{\text{Rec}} = \mathcal{L}_{\text{MSE}}(\text{GLE}(z_L), z_H)$
- 解码器训练阶段：Dice Loss + Focal Loss

## 实验关键数据

### 主实验：低质量图像分割（ThinObject-5K & LVIS测试集）

| 方法 | ThinObject LQ-3 IoU | ThinObject LQ-1 IoU | LVIS LQ-3 IoU | LVIS LQ-1 IoU |
|------|-------------------|-------------------|--------------|--------------|
| SAM | 0.6285 | 0.7527 | 0.4041 | 0.5325 |
| RobustSAM | 0.7015 | 0.7922 | 0.4517 | 0.5262 |
| DiffBIR-SAM | 0.7055 | 0.7927 | 0.5316 | 0.6021 |
| **GleSAM** | **0.7594** | **0.8277** | **0.5535** | **0.6131** |

### 消融实验：各组件有效性

| 组件 | ThinObject LQ-3 IoU |
|------|-------------------|
| SAM baseline | 0.6285 |
| + GLE (无FDA) | 0.7201 |
| + GLE + FDA | 0.7452 |
| + GLE + FDA + CRE | **0.7594** |

### 关键发现

- 退化越严重（LQ-3 vs LQ-1），GleSAM相对SAM/RobustSAM的优势越明显
- 在未见过的退化类型上也表现良好（ECSSD/COCO-val unseen sets），证明泛化性
- FDA的简单缩放操作带来约2.5%的IoU提升，说明分布对齐的重要性
- 基于SAM2的GleSAM同样有效，框架通用性强
- 整体仅增加少量可学习参数（LoRA + 解码器token），训练30小时/4 GPU

## 亮点与洞察

1. **将扩散模型的去噪能力从图像空间迁移到分割特征空间**是一个新颖且有效的范式，避免了图像复原的信息损失和计算开销
2. **分布对齐**（FDA）和**通道适配**（CRE）的技术方案简洁优雅，且有广泛的迁移价值——任何需要将LDM引入非图像隐空间的场景都可参考
3. 构建的LQSeg数据集包含多类型多级别退化组合，填补了低质量分割评估的空白

## 局限与展望

- 单步去噪的增强能力有上限，对极端退化（如几乎丢失内容的退化）可能不足
- 目前仅处理合成退化，对真实世界退化（如雨雾、水下等）的效果有待验证
- U-Net增加了推理开销，虽然是单步但仍为端到端分割引入了延迟
- 未来可探索条件式增强（根据退化类型自适应调整去噪强度）

## 相关工作与启发

- **与RobustSAM的关系**：RobustSAM用蒸馏一致性学习增强鲁棒性，GleSAM用生成式去噪增强表示，思路互补
- **与VPD等工作的关系**：VPD用扩散模型作为backbone提取特征，GleSAM则在现有分割模型的隐空间中引入扩散去噪
- **启发**：预训练生成模型的先验知识不仅可用于生成，还可用于"修复"判别式模型的退化表示

## 评分

⭐⭐⭐⭐

将扩散模型引入SAM隐空间进行特征增强的思路新颖且有效。技术路线完整（GLE+FDA+CRE），实验覆盖多退化类型/级别。构建的LQSeg数据集有独立价值。稍显不足的是合成退化与真实退化的差距，以及推理效率的权衡。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Segment Any Motion in Videos](segment_any_motion_in_videos.md)
- [\[CVPR 2025\] SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2025\] Image Quality Assessment: From Human to Machine Preference](image_quality_assessment_from_human_to_machine_preference.md)
- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)

</div>

<!-- RELATED:END -->
