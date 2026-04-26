---
title: >-
  [论文解读] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography
description: >-
  [CVPR2026][医学图像][PET 分割] 构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 masks），并提出 SegAnyPET——首个面向功能性 PET 影像的 3D 可提示分割基础模型，在多中心、多示踪剂、多疾病场景下实现了强零样本泛化能力。
tags:
  - CVPR2026
  - 医学图像
  - PET 分割
  - 基础模型
  - 通用分割
  - 交互式分割
  - 全身 PET
  - SAM
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**会议**: CVPR2026  
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)  
**代码**: [YichiZhang98/SegAnyPET](https://github.com/YichiZhang98/SegAnyPET)  
**领域**: 医学图像分割 (Medical Image Segmentation)  
**关键词**: PET 分割, 基础模型, 通用分割, 交互式分割, 全身 PET, SAM

## 一句话总结

构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 masks），并提出 SegAnyPET——首个面向功能性 PET 影像的 3D 可提示分割基础模型，在多中心、多示踪剂、多疾病场景下实现了强零样本泛化能力。

## 研究背景与动机

**PET 的临床不可替代性**：PET 是唯一能在活体内可视化代谢过程的核医学影像模态，在肿瘤学和神经学中的早期诊断中扮演关键角色，但其图像天然缺乏解剖对比度，分割难度远超 CT/MRI。

**数据稀缺与标注成本高**：PET 影像采集和标注费用昂贵，现有公开 PET 分割数据集局限于特定肿瘤任务，缺乏覆盖全身多器官、多病灶的大规模标注。

**任务特定模型泛化差**：传统深度学习模型仅能分割训练时见过的固定类别，面对新靶器官或病灶需要额外收集标注并重新训练，无法满足全身 PET 解读中多样化、不可预测的临床需求。

**现有基础模型跨模态失效**：SAM-Med3D、SegVol、SAT 等 3D 医学分割基础模型主要在 CT/MRI 结构影像上训练，其学到的解剖先验和外观统计与功能性 PET 影像存在巨大域差距，直接迁移效果极差。

**文本提示模型在 PET 上完全失败**：依赖文本提示的模型（如 SAT）在 PET 器官分割上 DSC 接近零，说明其跨模态对齐严重过拟合到结构影像。

**手动勾画低效且不一致**：临床中 PET 体积的手动标注耗时长、观察者间差异大、可重复性差，亟需自动化+可交互的高精度分割工具。

## 方法详解

### 整体框架

SegAnyPET 遵循 SAM 的 "图像编码器 + 提示编码器 + 掩码解码器" 三组件设计，但将所有 2D Transformer 组件扩展为**全 3D 架构**，直接在体积级输入上捕获切片间空间关联：

- **Image Encoder $\mathcal{E}_I$**：输入 $X \in \mathbb{R}^{H \times W \times D}$ 的原始 PET 体积，结合 3D 绝对位置编码提取高维特征嵌入
- **Prompt Encoder $\mathcal{E}_P$**：支持稀疏提示（点击点，通过 3D 位置编码）和密集提示（粗糙掩码，通过 3D 卷积 + LayerNorm + GELU 对齐到图像潜空间）
- **Mask Decoder $\mathcal{D}$**：通过 3D Transformer 块融合图像与提示嵌入，经 3D 转置卷积逐步上采样，最终输出体积级分割掩码

### 关键设计

1. **迭代评估循环（Human-in-the-Loop）**：每轮预测后自动比较 $\hat{Y}^{(t)}$ 与 GT，从 FN 区采样正点、FP 区采样负点，累积提示逐步细化分割，公式为 $\hat{Y}^{(t+1)} = \mathcal{D}(\mathcal{E}_I(X), \mathcal{E}_P(p^{(t)}, \hat{Y}^{(t)}))$
2. **双变体设计**：通用模型 SegAnyPET（全数据训练）+ 专用模型 SegAnyPET-Lesion（在病灶数据上微调 200 epochs，提升小异质病灶敏感度）
3. **训练时交互模拟**：每次迭代随机采样 1–20 个点击点，增强模型对不同交互场景的适应性
4. **Patch-based 推理**：裁剪 $128^3$ 的 patch，若预测区域到达边界则使用 50% 重叠的滑窗策略

### 训练细节

- **硬件**：8× NVIDIA A100 80GB GPU，DDP + NCCL 后端
- **优化器**：AdamW，图像编码器 lr=8e-5，提示编码器和掩码解码器 lr=8e-6，weight decay=0.1
- **训练**：500 epochs，输入 patch $128^3$，全局 batch size=96
- **SegAnyPET-Lesion**：在病灶数据上额外微调 200 epochs
- **学习率调度**：MultiStepLR，在第 120/180 epoch 各降 10×
- **数据增强**：三轴随机翻转 + 自适应裁剪/填充

### 损失函数

Dice Loss + Cross-Entropy Loss（带 sigmoid 激活和平方预测），联合优化体积分割任务。

## 实验

### 数据集

| 集合 | 来源 | 示踪剂 | 扫描数 | Mask数 |
|------|------|--------|--------|--------|
| 开发集 (C1-C4) | AutoPET + UDPET + 两个私有队列 | FDG | 11,041 | 59,831 |
| 内部验证 | 同中心独立样本（器官/淋巴瘤/肺癌） | FDG | 886 | 9,828 |
| 外部验证 | 独立中心 PET/CT + PET/MRI + PSMA-PET | FDG/PSMA | 1,551 | 34,579 |

### 与任务特定模型对比（内部评估，5 个器官 DSC）

| 模型 | Liver | Kidney-L | Kidney-R | Heart | Spleen |
|------|-------|----------|----------|-------|--------|
| nnUNet | 0.938 | 0.903 | 0.870 | 0.912 | 0.887 |
| SegResNet | 0.936 | 0.907 | 0.859 | 0.903 | 0.894 |
| STUNet | 0.935 | 0.903 | 0.877 | 0.918 | 0.880 |
| SwinUNETR | 0.935 | 0.899 | 0.846 | 0.890 | 0.886 |
| SegAnyPET (1p) | 0.926 | 0.875 | 0.870 | 0.892 | 0.876 |
| SegAnyPET (3p) | 0.939 | 0.887 | 0.888 | 0.905 | 0.891 |
| **SegAnyPET (5p)** | **0.949** | **0.898** | **0.898** | **0.916** | **0.905** |

SegAnyPET 使用 5 个点提示即可在所有 5 个器官上超越 nnUNet 等四个全监督专用模型，仅 1 个点时已与专用模型持平。

### 与其他基础模型对比

2D 模型（SAM、MedSAM）采用逐切片处理策略，在 PET 上 DSC 极低（0.17–0.39），且推理时间数倍于 3D 模型。3D 基础模型中：

- **SAT（文本提示）**：DSC ≈ 0，文本-视觉对齐严重过拟合到结构影像，完全无法解析 PET 代谢信号
- **SAM-Med3D / nnInteractive（点提示）**：空间锚点有助于定位，但底层特征无法准确勾画 PET 边界
- **SegVol / VISTA3D**：表现有所改善但整体仍不足

SegAnyPET 在所有评估任务上（器官+病灶）一致显著优胜，归因于其在大规模 PET 数据上训练获得的领域特异性表征。

### 消融与关键发现

- **提示数量递增效果显著**：1 点 → 3 点 → 5 点，DSC 持续提升约 2–3 个百分点
- **零样本泛化**：在完全未见过的外部数据（不同中心、PET/MRI、PSMA 示踪剂）上依然保持鲁棒分割性能
- **临床实用性**：SegAnyPET 辅助交互工作流将标注时间减少 **82.4%** 和 **83.0%**（两位专家）
- **下游应用**：基于分割结果构建的全身代谢协方差网络具有高生物学保真度，验证了输出的临床可用性

## 亮点

- **首个 PET 分割基础模型**：填补了功能性 PET 影像在基础模型领域的关键空白
- **最大 PET 分割数据集**：PETWB-Seg11K 以 11K+ 扫描和 60K masks 远超现有数据集
- **一个模型替代多个专用网络**：单一模型在器官+病灶分割上达到甚至超越专用模型水平
- **强零样本跨域泛化**：跨中心、跨示踪剂、跨模态（PET/CT → PET/MRI）均表现鲁棒
- **临床可落地**：支持人机协作循环，标注效率提升超 80%

## 局限性

- **稀有疾病和示踪剂覆盖不足**：PETWB-Seg11K 虽大但某些罕见疾病、解剖区域仍欠缺代表性
- **离散病灶分割效率受限**：全身淋巴瘤等多发离散病灶需逐个点击提示，单次点击无法捕获所有病灶
- **病灶分割精度仍有提升空间**：定量指标显示肿瘤分割仍难以达到器官分割同等水平
- **缺乏文本提示**：当前仅支持点/掩码提示，未来融合放射学报告的多模态视觉-语言方向可能更高效

## 相关工作

- **SAM / MedSAM**：2D 通用/医学分割基础模型，逐切片处理 3D 数据效率低且空间不连续
- **SAM-Med3D / SegVol / SAT / VISTA3D / nnInteractive**：3D 医学分割基础模型，但训练数据以 CT/MRI 为主，在 PET 上泛化差
- **nnUNet**：自配置的任务特定分割框架，在充分监督下仍极具竞争力，但无法泛化到训练集外类别
- **TotalSegmentator / AbdomenAtlas**：大规模 CT 分割数据集与模型，均不涉及 PET 模态

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个专门面向 PET 功能影像的 3D 分割基础模型，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ — 数据规模大，内部+外部+跨模态+跨示踪剂验证全面，含临床实用性评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机充分，图表丰富
- 价值: ⭐⭐⭐⭐ — 对 PET 定量分析和临床工作流有直接推动作用，数据集和代码均开源

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_large_endoscopic_monocular_dataset_foundation_model_surgical.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis](sparse_task_vector_mixup_with_hypernetworks_for_efficient_knowledge_transfer_in_.md)

<!-- RELATED:END -->
