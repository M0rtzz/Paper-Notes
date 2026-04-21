---
title: >-
  [论文解读] Unleashing the Power of Prompt-driven Nucleus Instance Segmentation
description: >-
  [ECCV 2024][医学图像][图像分割] 提出 PromptNucSeg 框架，通过训练一个 prompter 自动生成细胞核中心点 prompt，并微调 SAM 进行逐核分割，同时引入相邻核作为 negative prompt 解决重叠核分割问题，无需复杂后处理即在三个 benchmark 上达到 SOTA。
tags:
  - ECCV 2024
  - 医学图像
  - 图像分割
  - SAM
  - 提示学习
  - Histopathology
---

# Unleashing the Power of Prompt-driven Nucleus Instance Segmentation

**会议**: ECCV 2024  
**arXiv**: [2311.15939](https://arxiv.org/abs/2311.15939)  
**代码**: https://github.com/windygoo/PromptNucSeg (有)  
**领域**: 医学图像分割 / 细胞核实例分割  
**关键词**: Nucleus Instance Segmentation, SAM, Prompt Learning, Negative Prompts, Histopathology

## 一句话总结

提出 PromptNucSeg 框架，通过训练一个 prompter 自动生成细胞核中心点 prompt，并微调 SAM 进行逐核分割，同时引入相邻核作为 negative prompt 解决重叠核分割问题，无需复杂后处理即在三个 benchmark 上达到 SOTA。

## 研究背景与动机

细胞核实例分割是病理图像分析的基础任务，对癌症诊断、治疗规划至关重要。当前主流方法（bottom-up）先回归细胞核的代理图（如距离图、方向图），再通过复杂后处理（如 watershed）将像素分组为实例。这种范式的核心问题是：**后处理需要精心调参，对噪声敏感，阻碍实际应用**。

SAM（Segment Anything Model）以其强大的泛化能力和 promptable 特性引起广泛关注，但在细胞核分割中的潜力尚未充分发掘。现有工作要么仅复用 SAM 的 encoder 构建更好的回归模型（CellViT），仍需后处理；要么以 one-prompt-all-nuclei 方式做语义分割（SPPNet），缺乏实例信息且依赖手工 prompt。

核心 idea：采用 **one-prompt-one-nucleus** 范式，训练 prompter 自动为每个细胞核生成唯一的点 prompt，微调 SAM 输出对应的 mask，完全避免后处理。创新性地引入相邻核作为 negative prompt 解决重叠核的过分割问题。

## 方法详解

### 整体框架

PromptNucSeg 由两个独立训练的模型组成：
- **Nucleus Prompter**：从输入图像中自动预测每个细胞核的中心点坐标和类别
- **Segmentor（微调的 SAM）**：接收 prompter 生成的点 prompt，为每个核输出对应的分割 mask

推理流程：输入图像 → Prompter 预测所有核的中心点 prompt → 辅助分支过滤假阳性 → SAM 逐核生成 mask → NMS 去重 → 最终实例分割结果。

### 关键设计

1. **SAM 微调策略（Adapt SAM to Nucleus Segmentation）**:

    - 功能：用细胞核实例分割数据微调 SAM，使其适应医学图像域
    - 核心思路：对每个图像-标签对，随机选取 $Z$ 个核实例，从每个实例前景区域随机采样一个正点 prompt $p_z$，微调 SAM 预测该核的 mask
    - 前向过程：$\widetilde{\mathcal{O}}_z = \mathcal{M}(\mathcal{F}(x), \mathcal{P}(\{p_z\}), [\text{mask}], [\text{IoU}])$
    - 损失函数：$\mathcal{L}_{sam} = \omega \cdot \text{FL}(\widetilde{\mathcal{O}}_z, \mathcal{O}_z) + \text{DL}(\widetilde{\mathcal{O}}_z, \mathcal{O}_z) + \text{MSE}(\widetilde{\nu}, \nu)$，由 focal loss、dice loss 和 IoU 回归损失组成
    - 冻结 prompt encoder，更新 image encoder 和 mask decoder
    - 将输入分辨率从 1024×1024 降至 256×256，大幅减少 GPU 显存

2. **Nucleus Prompter（自动 prompt 生成）**:

    - 功能：自动预测每个细胞核的中心点坐标和类别，替代手工 prompt
    - 核心思路：受 P2PNet 启发，在输入图像上放置均匀 anchor points（间距 $\lambda$ 像素），通过特征金字塔提取多尺度特征，用 MLP 预测每个 anchor 的偏移量 $\delta_i$ 和分类 logit $q_i \in \mathbb{R}^{C+1}$
    - 匹配策略：通过二部图最大权匹配（Hungarian 算法）建立 anchor 到 ground-truth 核中心的一一映射，权重定义为 $w_{i,j} = q_i(c_j) - \alpha \|\hat{a}_i - b_j\|_2$，综合考虑分类置信度和位置距离
    - 训练损失：$\mathcal{L}_{prompter} = \mathcal{L}_{reg} + \mathcal{L}_{cls} + \mathcal{L}_{aux}$
        - 分类损失：$\mathcal{L}_{cls} = -\frac{1}{M}(\sum_{i=1}^N \log q_{\sigma(i)}(c_i) + \beta \sum_{a_i \in \mathcal{A}'} \log q_i(\varnothing))$
        - 回归损失：$\mathcal{L}_{reg} = \frac{\gamma}{N} \sum_{i=1}^N \|\hat{a}_{\sigma(i)} - b_i\|_2$
    - 设计动机：点 prompt 比 bounding box 更容易定位，且能更精确地分离接触的目标

3. **辅助任务与 Mask-aided Prompt Filtering**:

    - 功能：引入细胞核区域分割辅助任务，提升 prompter 对前景区域的感知能力
    - 核心思路：在 prompter 中增加一个简单的 mask head（Conv-BN-ReLU-Conv），从高分辨率特征 $P_2$ 预测核概率图 $\hat{S}$，用 focal loss 监督
    - 推理时利用预测的核概率图过滤假阳性 prompt：仅保留概率 > 0.5 的 prompt
    - 设计动机：prompter 训练仅涉及点标注和类别，辅助分割任务引入了核的大小、形态等丰富信息

4. **Negative Prompts 解决重叠核分割**:

    - 功能：将相邻核作为 negative prompt 输入 SAM，抑制重叠区域的过分割
    - 问题分析：对两个重叠核各用一个正 prompt 分割时，由于边界模糊会产生过分割 mask
    - 关键发现：**仅在推理时加 negative prompt 无效**，因为只用正 prompt 微调会导致模型对 negative prompt 的"灾难性遗忘"
    - 解决方案：在微调阶段就引入 negative prompt——对每个目标核，用其正 prompt $p_z$ 和 $K$ 个最近邻点 $\{n_{z,k}\}_{k=1}^K$ 作为 negative prompt 联合输入：$\widetilde{\mathcal{O}}_z = \mathcal{M}(\mathcal{F}(x), \mathcal{P}(\{p_z\} \cup \{n_{z,k}\}_{k=1}^K), [\text{mask}], [\text{IoU}])$
    - 推理时同样用 prompter 预测的最近 $K$ 个点作为 negative prompt

### 损失函数 / 训练策略

- Prompter 和 Segmentor **分别独立训练**，不做端到端优化
- Prompter 损失：$\mathcal{L}_{prompter} = \mathcal{L}_{reg} + \mathcal{L}_{cls} + \mathcal{L}_{aux}$
- Segmentor 损失：$\mathcal{L}_{sam} = \omega \cdot \text{FL} + \text{DL} + \text{MSE}$
- 推理时采用 sliding window（256×256 tile），有重叠区域 $\epsilon$ 保证核完整性，最后 NMS 去重

## 实验关键数据

### 主实验

**PanNuke 数据集（最具挑战性，19 种组织类型）**

| 方法 | bPQ (avg) | mPQ (avg) |
|------|-----------|-----------|
| HoVer-Net | 0.6596 | 0.4629 |
| CPP-Net | 0.6798 | 0.4847 |
| PointNu-Net | 0.6808 | 0.4957 |
| CellViT-H | 0.6793 | 0.4980 |
| **PromptNucSeg-H** | **0.6924** | **0.5123** |

超越前最优 bPQ +1.1, mPQ +1.4。

**Kumar & CPM-17 数据集**

| 方法 | Kumar AJI | Kumar PQ | CPM-17 AJI | CPM-17 PQ |
|------|-----------|----------|------------|-----------|
| PointNu-Net | 0.606 | 0.603 | 0.712 | 0.706 |
| CellViT-H | - | - | - | - |
| **PromptNucSeg-H** | **0.622** | **0.627** | **0.740** | **0.733** |

CPM-17 上 AJI 超前最优 +1.9，PQ 超前最优 +2.8。

### 消融实验

**各模块效果（CPM-17 数据集）**

| 配置 (FT/AUX/MAPF/NP) | AJI | PQ | 说明 |
|------------------------|-----|-----|------|
| 无微调 SAM | 0.319 | 0.223 | 原始 SAM 直接用 |
| FT only | 0.728 | 0.723 | 仅微调 |
| FT + AUX | 0.734 | 0.727 | +辅助分割任务 |
| FT + AUX + MAPF | 0.737 | 0.731 | +mask 过滤 |
| **FT + AUX + MAPF + NP** | **0.740** | **0.733** | 完整模型 |

**效率对比（PanNuke）**

| 方法 | Params (M) | MACs (G) | FPS | mPQ |
|------|-----------|----------|-----|-----|
| HoVer-Net | 37.6 | 150.0 | 7 | 0.4629 |
| CPP-Net | 122.8 | 264.4 | 14 | 0.4847 |
| PointNu-Net | 158.1 | 335.1 | 11 | 0.4957 |
| **PromptNucSeg-B** | **145.6** | **59.0** | **27** | **0.5095** |

计算量（MACs）减少 4-5 倍，推理速度提升 2-4 倍。

### 关键发现

- **Negative prompt 必须在训练和推理阶段同时使用**：仅推理时加 negative prompt 无效，因为灾难性遗忘；实验验证了 1 个 negative prompt 最优，2 个因噪声引入反而下降
- **辅助核区域分割任务有效**：AJI 提升 0.6%，同时生成的概率图用于 prompt 过滤，一举两得
- **Prompt 质量是瓶颈**：oracle 实验（使用 GT 中心点）性能显著高于实际性能，说明提升 prompter 精度是关键改进方向
- **分类由 prompter 负责效果更好**：相比让 SAM decoder 做分类，prompter 从全局视角分类更合理

## 亮点与洞察

- **范式创新**：用 one-prompt-one-nucleus 替代传统的代理图回归+后处理范式，简洁优雅
- **Negative prompt 设计精巧**：抓住了重叠核分割的核心难题，从 SAM 的 promptable 特性出发找到自然解法
- **辅助任务设计高效**：一个简单的分割头同时起到特征增强和推理过滤两个作用
- **实用性强**：无需后处理调参，计算量大幅降低，FPS 提升显著
- **可迁移思路**：negative prompt 的训练-推理一致性设计可推广到其他 SAM 变体的应用场景

## 局限与展望

- **Prompter 和 Segmentor 独立训练**：端到端训练可能带来进一步提升（论文提到初步尝试但未收敛）
- **依赖 SAM 的 ViT backbone**：模型参数量仍然较大（145M+），部署于边缘设备受限
- **Prompt 质量上限**：oracle 和实际性能的差距表明 prompter 精度是瓶颈，可探索更强的检测/定位方法
- **仅限 H&E 染色图像**：未验证在免疫组化等其他染色方式上的效果
- **Sliding window 策略**在超大图像上可能产生边界不一致问题

## 相关工作与启发

- **vs CellViT (Hörst et al., 2023)**: CellViT 复用 SAM encoder 但仍需代理图回归+后处理；PromptNucSeg 完全利用 SAM 端到端分割能力
- **vs SPPNet (Xu et al., 2023)**: SPPNet 用 one-prompt-all-nuclei 做语义分割，需手工 prompt 且无实例信息；PromptNucSeg 自动生成 prompt 且直接输出实例
- **vs Bottom-up 方法（HoVer-Net, StarDist 等）**: 传统方法依赖后处理且计算量大；PromptNucSeg 更高效，无需后处理
- **vs Mask R-CNN**: 都属 top-down，但 Mask R-CNN 用 bbox 表示核容易覆盖多个重叠核，且固定分辨率 mask 有量化误差；点 prompt 更精确

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 SAM 的 promptable 特性与核实例分割完美结合，negative prompt 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个 benchmark 全面对比，消融完整，效率分析到位
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述条理分明，图示直观
- 价值: ⭐⭐⭐⭐⭐ 范式创新，无需后处理的核分割方案对实际部署有重要价值

<!-- RELATED:START -->

## 相关论文

- [Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](pathologyknowledge_enhanced_multiinstance_prompt_learni.md)
- [iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection](../../ICML2025/medical_imaging/idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio.md)
- [The Four Color Theorem for Cell Instance Segmentation](../../ICML2025/medical_imaging/the_four_color_theorem_for_cell_instance_segmentation.md)
- [Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](../../CVPR2025/medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)
- [I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)

<!-- RELATED:END -->
