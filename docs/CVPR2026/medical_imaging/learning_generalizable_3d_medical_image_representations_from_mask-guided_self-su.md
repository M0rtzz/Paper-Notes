---
title: >-
  [论文解读] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision
description: >-
  [CVPR 2026][医学图像][自监督学习] 提出 MASS（MAsk-guided Self-Supervised learning），利用 SAM2 自动生成的类别无关 mask 作为伪标注，以 in-context 分割为 pretext task 进行自监督预训练，无需任何人工标注即可学到语义丰富、泛化性强的 3D 医学图像表征，在 few-shot 分割和冻结编码器分类上均取得优异表现。
tags:
  - CVPR 2026
  - 医学图像
  - 自监督学习
  - 3D医学图像
  - mask引导预训练
  - in-context分割
  - 基础模型
---

# Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision

**会议**: CVPR 2026  
**arXiv**: [2603.13660](https://arxiv.org/abs/2603.13660)  
**代码**: 有（论文中提到 Code is available）  
**领域**: 医学图像  
**关键词**: 自监督学习, 3D医学图像, mask引导预训练, in-context分割, 基础模型

## 一句话总结

提出 MASS（MAsk-guided Self-Supervised learning），利用 SAM2 自动生成的类别无关 mask 作为伪标注，以 in-context 分割为 pretext task 进行自监督预训练，无需任何人工标注即可学到语义丰富、泛化性强的 3D 医学图像表征，在 few-shot 分割和冻结编码器分类上均取得优异表现。

## 研究背景与动机

**基础模型缺位**：GPT、CLIP、DINO 等在自然图像/语言领域已通过大规模无标注数据学到通用表征，但 3D 医学图像领域尚无对应的基础模型范式。

**现有自监督方法不足**：对比学习（SimCLR、MoCo）侧重全局特征，MAE 等重建方法侧重低层纹理，均无法捕获医学影像所需的解剖语义和空间精确性。

**监督预训练的局限**：SuPreM、STU-Net 等依赖大量专家标注，受限于预定义类别体系（如 25 个器官 + 7 种肿瘤），无法扩展到临床实践中数千种解剖变体和病理。

**医学图像的独特挑战**：与自然图像不同，医学扫描中几乎所有体素都有临床意义（骨密度→骨折、软组织纹理→肿瘤、血管模式→缺血），且空间精度至关重要。

**标注成本障碍**：3D 医学图像的像素级标注需要专业知识且极其昂贵，限制了以分割为 pretext task 的预训练方法的扩展性。

**核心洞察**：语义分割是最符合临床推理方式的 pretext task（临床医生通过识别结构"是什么"和"在哪里"进行推理），而自动生成的类别无关 mask 虽无语义标签且含噪声，但足以捕获解剖和病理上有意义的区域。

## 方法详解

### 整体框架

MASS 包含两个阶段：

**阶段一：无标注 Mask 生成**

- 使用 SAM2（在自然图像上训练，无医学知识）对无标注 3D 体积进行自动分割
- 具体流程：创建 3 通道输入（CT 用不同窗宽窗位、MRI/PET 用分位数归一化），沿最优成像轴均匀采样 2D 切片，应用 SAM2 的自动 mask 生成（密集点提示），通过 SAM2 的视频预测能力将 mask 传播到整个体积
- 每个体积可生成数百到数千个 3D mask，覆盖器官、血管、肿瘤、病灶等多种结构

**阶段二：Mask 引导的自监督学习**

- 采用 in-context segmentation（ICS）框架，沿用 Iris 架构
- 模型由三部分组成：图像编码器 $E_\theta$、任务编码模块 $T_\phi$、mask 解码器 $D_\psi$
- 每次迭代：采样一张无标注 3D 图像 $x$ 及其自动生成的 mask $m$，创建两个增强视图——参考视图 $(x_s, y_s)$ 和查询视图 $(x_q, y_q)$
- 参考视图提供"在哪里"的位置信息，外观变换迫使模型学习跨不同视觉表现的语义一致性（"是什么"）

### 关键设计

1. **任务嵌入机制**：编码参考图像 $F_s = E_\theta(x_s)$，提取任务嵌入 $\mathcal{T} = T_\phi(F_s, y_s)$，捕获"要分割什么解剖结构"的信息，引导解码器预测查询 mask $\hat{y}_q = D_\psi(E_\theta(x_q), \mathcal{T})$
2. **隐式语义学习**：无需语义标签，通过不变性机制学习语义——外观增强（亮度、对比度、gamma、高斯噪声）破坏强度匹配、纹理模式等捷径，空间增强（旋转、缩放、平移）消除位置和方向线索。模型只能学到跨所有变换不变的东西——解剖结构的本质语义身份
3. **开放集 mask 多样性**：训练时使用数千个类别无关 mask，覆盖从器官级到亚解剖区域到病理的多种粒度，迫使模型学习广泛的医学概念和可组合的视觉原语（纹理模式、边界特征、空间配置、强度分布）
4. **多模态兼容**：mask 生成流程适用于 CT、MRI、PET 等多种模态，通过不同的预处理策略（CT 用窗宽窗位、MRI/PET 用分位数归一化）统一输入

### 损失函数/训练策略

- **损失函数**：$\mathcal{L}_{Seg} = \mathcal{L}_{Dice}(\hat{y}_q, y_q) + \mathcal{L}_{BCE}(\hat{y}_q, y_q)$，Dice Loss + 二值交叉熵联合优化
- **数据增强**：空间变换（旋转、缩放、平移）同时作用于图像和 mask 以保持对应关系；外观变换（亮度、对比度、gamma、高斯噪声）仅作用于图像
- **默认骨干**：3D ResUNet
- **预训练规模**：小规模（单数据集 20-200 扫描）到大规模（5K 多模态 CT/MRI/PET 体积，12 个数据集）
- **下游使用三种模式**：(1) 免训练 in-context 分割（无需参数更新）；(2) 任务特定微调；(3) 冻结编码器做分类

## 实验关键数据

### 主实验

**表1：单数据集 few-shot 分割（Dice %）**

| 方法 | BCV 1-shot | BCV 10-shot | AMOS MR 1-shot | AMOS MR 10-shot | SS H&N 1-shot | KiTS 30-shot |
|------|-----------|------------|----------------|----------------|--------------|-------------|
| Scratch | 27.3 | 75.2 | 32.8 | 75.9 | 51.8 | 35.7 |
| SimCLR | 44.9 | 78.4 | 35.9 | 78.0 | 53.6 | 41.5 |
| MASS-IC | **65.5** | 73.6 | **62.1** | 71.6 | **59.3** | 3.8 |
| MASS-FT | **68.8** | **83.7** | **65.9** | **84.7** | **66.9** | **64.3** |
| 全监督 | 83.6 | — | 85.5 | — | 78.2 | 81.7 |

**表2：大规模多模态预训练分割（Dice %，5K 体积预训练）**

| 方法 | BCV 1-shot | AMOS MR 1-shot | KiTS 30-shot | Pelvic 1-shot |
|------|-----------|----------------|-------------|--------------|
| SuPreM (监督) | 63.9 | 55.1 | 64.1 | 85.4 |
| Iris-FT (监督) | 83.4 | 83.6 | 78.3 | 86.9 |
| AnatoMix | 53.1 | 35.9 | 40.6 | 82.2 |
| Merlin | 50.1 | 37.9 | 51.1 | 79.3 |
| **MASS-FT** | **70.2** | **74.3** | **68.5** | **92.8** |

**表3：分类性能（AUC %，冻结编码器）**

| 方法 | RSNA ICH 5% | RSNA ICH 100% | Liver Trauma 30% | Kidney Trauma 30% |
|------|------------|--------------|-----------------|-------------------|
| Scratch (全训练) | 72.8 | 89.5 | 74.4 | 75.0 |
| SuPreM | 73.5 | 78.3 | 68.3 | 54.9 |
| Merlin | 57.3 | 65.5 | 60.1 | 58.0 |
| **MASS** | **75.4** | **81.5** | **86.7** | **82.9** |

### 消融实验

**Mask 质量分析**：自动 mask 与 GT 的平均 Dice 仅 15.2%（BCV）和 7.1%（SS H&N），仅 14%/13% 的 mask Dice > 40，但 MASS 仍取得 65.5% 和 59.3% 的 1-shot 性能——说明弱监督即足够。

**Mask 生成方法对比**：

| Mask 来源 | BCV 1-shot | SS H&N 1-shot |
|----------|-----------|--------------|
| TotalSegmentator | 80.7 | 13.5（类别不覆盖） |
| SAM2 | 65.5 | 59.3 |
| SLIC 超像素 | 54.3 | 43.8 |

**数据多样性 > 数量**：从单器官腹部 CT（BCV，42.7%）扩展到全身 CT + 多模态达到 73.9%。解剖和模态多样性驱动性能提升，而同域数据堆叠迅速饱和。

**架构泛化**：ResUNet 和 I3DResNet152 在相同设置下性能相当（分割 73.87 vs 72.56，分类 75.42 vs 75.98），验证方法与具体编码器设计无关。

### 关键发现

1. **解剖 vs 病理**：MASS-IC 在解剖结构（器官）上有强 few-shot 能力，但在高变异性肿瘤上 in-context 性能有限（KiTS 仅 2.7%）；微调后 MASS-FT 显著超越基线（64.3% vs 42.2%）
2. **20-40% 标注即匹配全监督**：在解剖结构数据集上，MASS-FT 仅用 10-shot（25-40% 训练数据）即可达到全监督性能
3. **冻结编码器超越全训练**：在 RSNA ICH 5% 数据上，MASS 冻结编码器（75.4%）超越从头全训练（72.8%）；Trauma 30% 数据上提升更显著（肝 86.7 vs 74.4、肾 82.9 vs 75.0）
4. **OOD 泛化**：在完全未见的数据集上（BraTS、ACDC、Pelvic），MASS 展现竞争力甚至超越监督预训练（Pelvic 92.8 vs Iris 86.9）

## 亮点与洞察

- **范式创新**：首次将"类别无关 mask 引导的 in-context 分割"确立为 3D 医学图像自监督预训练的 pretext task，绕开了标注瓶颈
- **从弱到强**：自动 mask 与 GT 平均重叠仅 7-15%，但通过在数千个"大致正确"的分割任务上训练，模型学到了超越单个 mask 边界的语义概念
- **数据效率极高**：仅用 5K 体积（远少于 OpenMind 的 114K）就超越所有自监督基线，且 BCV 单数据集（23 scans）预训练在 ICH 分类上已接近 SuPreM
- **语义从不变性中涌现**：不需要语义标签，增强引起的外观/空间变化迫使模型学到唯一不变的因素——解剖结构的本质语义身份
- **开放集优势**：不受预定义类别约束，SAM2 mask 天然覆盖多粒度多结构，在 taxonomy 不匹配的场景下（如 SS H&N）远优于 TotalSegmentator

## 局限性

1. **病理结构 in-context 能力弱**：高变异性肿瘤（如 KiTS）的 zero-shot in-context 分割效果差（2.7%），必须通过微调才能有效处理
2. **未探索弱 mask + 专家标注的协同**：刻意排除了有标注数据，未研究自动 mask 与少量专家标注结合的潜力
3. **依赖 SAM2 的边界检测能力**：mask 质量受限于 SAM2 在医学图像上的域迁移性能，对边界模糊的结构可能效果不佳
4. **缺少视觉-语言对齐**：未与放射学报告等文本模态对齐，限制了在报告生成等任务上的应用
5. **与监督预训练的差距**：在评估目标与监督标注一致时（如 BCV），监督方法（Iris 83.2）仍领先 MASS（70.2）约 10-15 点

## 相关工作

- **自监督学习**：Model Genesis（图像恢复）、MAE/SimMIM（mask 重建）、DINO（自蒸馏）、SimCLR/MoCo（对比学习）——侧重通用视觉特征，缺乏语义分割的空间精度
- **监督预训练**：SuPreM（25M 标注体素）、STU-Net（TotalSegmentator 全身 CT）——受限于标注规模和预定义类别
- **合成数据**：AnatoMix（从 TotalSegmentator mask 生成合成 CT）——分布差异限制迁移性能
- **通用分割与交互模型**：UniverSeg、Iris（in-context 学习但需标注）、SAM 医学适配版——仍需标注或推理时人工交互
- **MASS 的差异**：唯一的无标注 + 开放集 mask + 语义自涌现 + 多模态兼容的自监督方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次用类别无关自动 mask 做医学图像自监督预训练，pretext task 设计巧妙且直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖 4 模态 12+ 数据集、分割+分类两大任务线、从 20 scans 到 5K volumes 的规模实验、多维消融
- 写作质量: ⭐⭐⭐⭐⭐ — 动机-方法-实验逻辑链完整，"不变性→语义涌现"的解释优雅且有说服力
- 价值: ⭐⭐⭐⭐⭐ — 为 3D 医学图像基础模型提供了无标注可扩展的新路径，实用性极强

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](../../NeurIPS2025/medical_imaging/self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)

<!-- RELATED:END -->
