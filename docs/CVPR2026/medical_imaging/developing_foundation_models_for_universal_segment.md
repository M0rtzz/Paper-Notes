---
title: >-
  [论文解读] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography
description: >-
  [CVPR 2026][医学图像][基础模型] 构建了迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 分割掩码），并提出 SegAnyPET 基础模型，实现基于 prompt 交互的通用 PET 器官与病灶体积分割，在跨中心、跨示踪剂的零样本场景下表现优异。
tags:
  - CVPR 2026
  - 医学图像
  - 基础模型
  - PET成像
  - 通用分割
  - 3D分割
  - 可提示分割
---

# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**会议**: CVPR 2026  
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)  
**代码**: 无  
**领域**: 医学影像分割  
**关键词**: 基础模型, PET成像, 通用分割, 3D分割, 可提示分割

## 一句话总结

构建了迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 分割掩码），并提出 SegAnyPET 基础模型，实现基于 prompt 交互的通用 PET 器官与病灶体积分割，在跨中心、跨示踪剂的零样本场景下表现优异。

## 研究背景与动机

PET（正电子发射断层扫描）是核医学关键成像模态，可通过放射性示踪剂可视化体内代谢过程，在肿瘤学和神经学中具有不可替代的地位。然而 PET 图像分割面临多重困难：

1. **固有挑战**：PET 缺乏高对比度解剖边界信息，信噪比低，空间分辨率有限，使得器官/病灶勾画远比 CT/MRI 困难
2. **数据稀缺**：PET 数据采集和标注成本极高，公开数据集极少且范围狭窄（仅限特定肿瘤任务）
3. **现有基础模型失效**：SAM-Med3D、SegVol、SAT 等通用医学分割基础模型主要针对 CT/MRI 训练，直接迁移到 PET 效果极差（SAT 的 DSC 接近零）
4. **任务特定模型局限**：传统深度学习模型只能分割训练时见过的固定类别，面对新器官/新病灶需要重新标注和训练

## 方法详解

### 整体框架

SegAnyPET 采用类 SAM 的 3D 可提示分割架构，包含三个核心组件：

- **Image Encoder**：从输入 PET 体积提取离散 3D 特征嵌入
- **Prompt Encoder**：将用户提供的稀疏 prompt（如点）或密集 prompt（如粗糙掩码）通过固定位置编码和自适应嵌入层转换为紧凑的 prompt 嵌入
- **Mask Decoder**：融合图像特征和 prompt 嵌入，通过上采样和 MLP 生成最终分割输出

### 关键设计

1. **PETWB-Seg11K 数据集构建**：
    - 整合两个公开数据集（AutoPET、UDPET）和三个私有队列，共 11,041 例全身 3D PET 扫描和 59,831 个分割掩码
    - 涵盖多中心（全球多个临床中心）、多设备、多示踪剂（FDG/PSMA）、多疾病类型的真实世界变异
    - 精心划分内部验证集（同分布）和外部验证集（不同中心、不同癌症类型、不同示踪剂）用于严格评估

2. **3D 体积架构设计**：
    - 不同于 2D 模型逐层分割再堆叠的方式，SegAnyPET 直接在 3D 体积上操作，充分利用 PET 体积的层间上下文信息
    - 使用点 prompt 实现快速高效的 3D 交互，掩码 prompt 支持迭代精修

3. **双变体策略**：
    - **SegAnyPET**：在全部数据上训练的通用分割基础模型，提供广泛的器官和病灶覆盖与泛化能力
    - **SegAnyPET-Lesion**：在病灶中心数据上微调的专用变体，提升对小型异质病灶的敏感性和边界精度
    - 用户可根据临床需求选择通用模型或病灶专用模型

### 损失函数 / 训练策略

- 训练基于 PETWB-Seg11K 数据集，使用 prompt 工程策略进行 mask 生成
- 支持人在回路（human-in-the-loop）的交互式工作流：放射科医生可通过追加正/负点 prompt 迭代精修分割结果
- SegAnyPET-Lesion 通过在病灶中心数据上微调 SegAnyPET 获得

## 实验关键数据

### 主实验：与任务特定模型对比

SegAnyPET 作为通用模型，与多个在器官标注上专门训练的任务特定模型对比：

| 模型 | 类型 | 训练方式 | 器官分割性能 | 新目标能力 |
|------|------|----------|------------|-----------|
| nnU-Net | 任务特定 | 全监督 | 强竞争力 | ❌ 需重训 |
| STUNet | 任务特定 | 全监督 | 中等 | ❌ 需重训 |
| SwinUNETR | 任务特定 | 全监督 | 中等 | ❌ 需重训 |
| SegResNet | 任务特定 | 全监督 | 中等 | ❌ 需重训 |
| **SegAnyPET** | 通用基础 | Prompt | **可比/超越** | ✅ 零样本 |

关键结论：尽管 SegAnyPET 是通用模型而非专为器官分割训练，但在 5 个训练可见的目标器官上达到了与 nnU-Net 旗鼓相当甚至更优的性能，且无需按任务重训。

### 与分割基础模型对比

| 模型 | Prompt 类型 | PET 器官分割 | PET 病灶分割 |
|------|------------|-------------|-------------|
| SAM-Med3D | 点 prompt | 较差 | 较差 |
| SegVol | 点 prompt | 较差 | 较差 |
| SAT | 文本 prompt | DSC≈0 | DSC≈0 |
| nnIteractive | 点 prompt | 较差 | 较差 |
| VISTA3D | 点 prompt | 较差 | 较差 |
| **SegAnyPET** | 点/掩码 prompt | **最优** | **最优** |

关键发现：文本 prompt 模型（如 SAT）在 PET 上完全失效（DSC≈0），说明其文本-视觉对齐严重过拟合于结构成像的解剖特征。

### 消融与泛化实验

| 验证场景 | 分布偏移类型 | SegAnyPET 表现 |
|----------|------------|---------------|
| 内部验证 | 同分布 | 一致且可靠 |
| 外部-新癌种 | 未见疾病类型 | 鲁棒泛化 |
| 外部-PET/MRI | 成像架构变化 | 鲁棒泛化 |
| 外部-PSMA-PET | 新示踪剂 | 鲁棒泛化 |

临床效用验证：在淋巴瘤和肺癌场景中，SegAnyPET 辅助标注工作流分别为两位专家节省了 **82.37%** 和 **82.95%** 的标注时间。

### 关键发现

1. 现有通用医学分割基础模型（SAM-Med3D 等）在 PET 上表现极差，暴露了从结构成像到功能成像的巨大域差距
2. PET 特定的大规模数据训练是必要的——SegAnyPET 能学到域鲁棒的代谢表征
3. 单一 SegAnyPET 模型可有效替代多个任务特定网络
4. Prompt 驱动的交互设计使得模型能处理训练标签空间之外的目标，这对 PET 全身分析尤为关键

## 亮点与洞察

- **数据贡献突出**：PETWB-Seg11K 是迄今最大最全面的全身 PET 分割数据集，规模远超现有 PET 数据集
- **首创 PET 基础模型**：SegAnyPET 是首个专为 PET 成像设计的可提示分割基础模型，填补了功能成像领域的空白
- **零样本泛化**：在跨中心、跨示踪剂（FDG→PSMA）、跨成像架构（PET/CT→PET/MRI）等严格分布偏移下均表现稳健
- **临床价值实证**：不仅展示了分割精度，还验证了标注效率提升（>82% 时间节省）和全身代谢网络分析的下游应用
- **深刻的实验观察**：揭示了文本 prompt 模型在 PET 上完全失效的原因（跨模态对齐过拟合于结构成像）

## 局限与展望

1. **分散病灶的交互效率**：对全身分散的多发病灶（如淋巴瘤），点 prompt 需逐个病灶交互，效率受限
2. **罕见疾病/示踪剂不足**：数据集中某些罕见疾病和示踪剂仍然代表性不足
3. **病灶分割仍有提升空间**：定量指标显示病灶分割准确度仍有显著改善余地
4. **缺乏文本 prompt 支持**：多模态视觉-语言 PET 基础模型是重要的未来方向，可通过语义描述同时识别所有分散病灶
5. **推理效率**：3D 体积处理的计算开销未被充分讨论

## 相关工作与启发

- **SAM (Segment Anything)** 启发了可提示分割范式，但从 2D 自然图像到 3D PET 需要根本性的架构和数据调整
- **nnU-Net** 在充分任务特定监督下仍具竞争力，说明基础模型的优势在于灵活性而非在每个任务上绝对领先
- **AutoPET** 和 **UDPET** 是现有的 PET 分割数据集，但范围过窄，PETWB-Seg11K 显著扩展了规模和多样性
- **启发**：功能成像（PET/SPECT）和结构成像（CT/MRI）存在根本性域差距，通用基础模型不能简单迁移，需要模态特定的大规模训练

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_large_endoscopic_monocular_dataset_foundation_model_surgical.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)

<!-- RELATED:END -->
