---
title: >-
  [论文解读] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
description: >-
  [CVPR 2026][红外-可见光融合] 提出首个在系数域（而非像素域）进行红外缺失条件下跨模态融合的框架：通过共享卷积字典建立 IR-VIS 统一原子空间，在系数域完成 VIS→IR 推理和自适应融合，配合冻结 LLM 提供弱语义先验进行热信息补全，在仅输入可见光图像的条件下达到接近双模态融合方法的性能。
tags:
  - CVPR 2026
  - 红外-可见光融合
  - 缺失模态
  - 卷积字典学习
  - 系数域推理
  - 大语言模型先验
---

# Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared

**会议**: CVPR 2026  
**arXiv**: [2603.08018](https://arxiv.org/abs/2603.08018)  
**代码**: https://github.com/harukiv/DCMIF (有)  
**领域**: 自监督学习  
**关键词**: 红外-可见光融合, 缺失模态, 卷积字典学习, 系数域推理, 大语言模型先验

## 一句话总结
提出首个在系数域（而非像素域）进行红外缺失条件下跨模态融合的框架：通过共享卷积字典建立 IR-VIS 统一原子空间，在系数域完成 VIS→IR 推理和自适应融合，配合冻结 LLM 提供弱语义先验进行热信息补全，在仅输入可见光图像的条件下达到接近双模态融合方法的性能。

## 研究背景与动机
红外-可见光（IR-VIS）图像融合对于监控、机器人和自动驾驶系统的鲁棒感知至关重要。现有方法（CNN、CNN-Transformer、GAN、扩散模型）都假设训练和推理时两种模态都可用。然而现实中红外模态经常缺失（如测试时仅有可见光相机）。

当红外缺失时，直观方案是在像素空间生成伪红外图像再融合。但像素空间生成存在严重问题：控制性差、可解释性弱、容易产生幻觉伪影和结构细节丢失。

核心矛盾：如何在红外缺失时稳定地恢复热信息并进行可解释的融合？本文的切入角度是：**不在像素空间生成红外，而是将两种模态映射到统一的字典-系数空间，在系数域完成推理和融合**，从而锚定数据一致性和先验约束在原子-系数层面。

## 方法详解

### 整体框架
完整 pipeline 为闭环的 **编码 → 转移 → 融合 → 重建**：
1. **JSRL（联合共享字典表示学习）**：学习 IR 和 VIS 共享的卷积字典 $\mathbf{D}$，将两种模态映射到统一原子空间
2. **VGII（可见光引导红外推理）**：在系数域将 VIS 系数转移为伪 IR 系数，并用冻结 LLM 提供弱语义先验进行单步闭环精炼
3. **AFRI（自适应表示推理融合）**：在原子级别通过窗口注意力和卷积混合融合 VIS 与推理 IR 系数，用共享字典重建最终图像

### 关键设计
1. **JSRL - 联合共享字典表示学习**:

    - 功能：学习一个跨模态共享字典 $\mathbf{D} \in \mathbb{R}^{B \times k \times k}$，使 VIS 和 IR 都能被表示为 $\mathbf{I} = \mathbf{D} * \mathbf{S}$
    - 核心思路：联合最小化两模态重建误差 + 系数先验 + 字典正则化：
    $\min_{\mathbf{D},\mathbf{S}_{vis},\mathbf{S}_{ir}} \frac{1}{2}\|\mathbf{I}_{vis} - \mathbf{D}*\mathbf{S}_{vis}\|_F^2 + \frac{1}{2}\|\mathbf{I}_{ir} - \mathbf{D}*\mathbf{S}_{ir}\|_F^2 + \lambda_1\varphi_1(\mathbf{S}_{vis}) + \lambda_2\varphi_2(\mathbf{S}_{ir}) + \lambda_3\phi(\mathbf{D})$
    - 通过模型驱动展开（model-driven unfolding）实现：交替进行数据一致性步骤（频域 Sherman-Morrison 公式求解）和近端更新步骤（CoeNet / DicNet 可学习代理）
    - 架构：N 个级联 IV-DLB（红外-可见字典学习块），每块含两个系数求解器 + 一个字典求解器，超参由 HypNet 自适应预测
    - 设计动机：共享字典建立了两种模态之间的原子级对应关系，为后续系数域推理提供可解释的统一表示空间

2. **VGII - 可见光引导红外推理**:

    - 功能：从 VIS 系数 $\tilde{\mathbf{S}}_{vis}$ 推理伪 IR 系数 $\mathbf{S}_{p\_ir}$
    - 核心思路：
        - 用冻结的 REN（表示编码网络，包含预训练 HeadNet + CSB + CoeNet）将 VIS 编码为系数
        - RIN（表示推理网络，encoder-decoder + multi-head attention）将 VIS 系数映射到伪 IR 系数
        - **LLM 弱语义先验精炼**：重建初始伪红外 $\mathbf{I}_{p\_ir}^{(0)}$，将 {VIS, 伪IR} 图像对 + 任务描述作为 prompt 输入冻结 LLM，提取文本特征 $\mathbf{F}_{text}$，通过 FiLM（Feature-wise Linear Modulation）调制系数：$\mathbf{S}_{fm} = \gamma \odot \tilde{\mathbf{S}}_{vis} + \beta$，再次通过 RIN 得到精炼系数
    - 损失函数：$\ell_{inf} = \ell_{int} + \ell_{reg} + \ell_{grad}$
        - 一致性损失 $\ell_{int}$：伪 IR 与真实 IR 在图像域和系数域的 L1 距离
        - 热正则化 $\ell_{reg}$：用归一化权重图强调热区域对齐
        - 梯度损失 $\ell_{grad}$：保持边缘一致性 $\|\nabla\mathbf{I}_{p\_ir} - \nabla\mathbf{I}_{vis}\|_1$
    - 设计动机：LLM 不生成像素，仅作为"语义评审员"提供通道级线性调制，轻量且可控；在系数域而非像素域推理，继承字典的可解释性

3. **AFRI - 自适应融合**:

    - 功能：在原子级别融合 VIS 系数和推理 IR 系数，重建最终图像
    - 核心思路：RFN（推理融合网络）通过两个级联的 Convolution-Attention Fusion 块，学习隐式原子级门控权重 $(\mathbf{W}_{vis}, \mathbf{W}_{p\_ir})$：$\mathbf{S}_f = \mathbf{W}_{vis} \odot \tilde{\mathbf{S}}_{vis} + \mathbf{W}_{p\_ir} \odot \mathbf{S}_{p\_ir}^{(1)}$
    - 融合损失：$\ell_f = \|\mathbf{I}_f - \max(\mathbf{I}_{p\_ir}, \mathbf{I}_{vis})\|_1 + \|\nabla\mathbf{I}_f - \max(\nabla\mathbf{I}_{p\_ir}, \nabla\mathbf{I}_{vis})\|_1$
    - 设计动机：逐元素 max 操作鼓励融合结果继承 IR 的热强度峰值和 VIS 的锐利结构边缘；门控在系数域操作，结构边缘的原子倾向 VIS，热语义的原子倾向 IR

### 损失函数 / 训练策略
三模块顺序训练：JSRL → VGII → AFRI。JSRL 在 MSRS 上训 1000 epoch，学到的字典可迁移到其他数据集；VGII 和 AFRI 各训 10 epoch。Adam 优化器，字典卷积核 5×5。两块 RTX 4090。

## 实验关键数据

### 主实验
| 方法 | 输入 | MSRS AG↑ | MSRS EN↑ | FLIR AG↑ | FLIR EN↑ | KAIST AG↑ |
|------|------|---------|---------|---------|---------|----------|
| CDDFuse | IR+VIS | 4.818 | 7.321 | 5.079 | 6.766 | 3.167 |
| EMMA | IR+VIS | 4.913 | 7.333 | 3.796 | 6.489 | 3.083 |
| DCEvo | IR+VIS | 4.858 | 7.298 | 4.585 | 6.763 | 3.229 |
| **Ours** | **仅VIS** | **5.037** | **7.188** | **4.518** | **6.639** | **4.414** |

关键发现：**仅用可见光输入的本文方法在 AG（平均梯度）等指标上甚至超越了部分需要双模态输入的 SOTA 融合方法。**

下游任务（M3FD 目标检测 YOLOv5）：本文 mAP=0.948 vs SAGE(双模态)=0.956，差距极小。
下游任务（FMB 语义分割 SegFormer-b5）：本文 mIoU=62.939 vs LRRNet(双模态)=62.942，基本持平。

### 消融实验
| 配置 | Dictionary | LLM | AG↑ | CE↓ | EI↑ | EN↑ | SF↑ |
|------|-----------|-----|-----|-----|-----|-----|-----|
| Model I（基线） | ✗ | ✗ | 3.320 | 1.452 | 45.531 | 6.058 | 9.238 |
| Model II | ✓ | ✗ | 4.363 | 1.046 | 48.351 | 6.578 | 11.936 |
| Model III | ✗ | ✓ | 4.256 | 0.619 | 48.154 | 6.423 | 11.175 |
| **Ours** | ✓ | ✓ | **4.518** | **0.596** | **48.784** | **6.639** | **12.554** |

### 关键发现
- 共享字典对性能提升贡献最大（Model I→II：AG +31%），验证了系数域范式的有效性
- LLM 调制提供额外的语义增强（CE 从 1.046 降到 0.596），特别是在亮度和对比度方面
- 两者互补，组合效果最佳
- 仅用 VIS 输入即可达到双模态方法的 90%+ 性能水平

## 亮点与洞察
- **范式创新**：首次提出系数域推理-融合方案处理红外缺失问题，避免像素空间生成的不稳定性
- **LLM 的巧妙使用**：不用 LLM 生成图像，仅用作语义级别的 FiLM 调制，极其轻量且有效
- **训练极简**：无需对抗训练或扩散采样，VGII 和 AFRI 各仅需 10 epoch
- **可解释性强**：所有计算在统一原子空间进行，字典原子提供直观的物理意义
- **闭环设计**：编码→推理→融合→重建全部在同一字典-系数空间中，保证表示一致性

## 局限与展望
- 共享字典在 MSRS 上训练后直接迁移，对域差异大的场景（如医疗红外）可能需要重训
- LLM 处理增加了推理延迟，实时场景需考虑效率
- 系数域推理的精度上限受字典容量限制，超大分辨率或细粒度热细节可能损失
- 仅验证了红外缺失场景，未探讨可见光缺失或其他多模态组合
- 方法假设 VIS 图像中包含足够的结构线索推理热信息，全黑场景可能失效

## 相关工作与启发
- 与 CDDFuse、EMMA 等双模态 SOTA 的关键区别：本文仅需单模态输入
- 模型驱动展开（DKSVD, Learned-CSC）为字典学习提供了理论基础
- FiLM 调制（源自条件生成）被创新性地用于 LLM 语义→系数空间调制
- 启发：系数域操作的可解释性范式可推广到其他缺失模态任务（如 MRI-CT 融合中模态缺失）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 系数域推理-融合范式+LLM弱先验的组合在该领域完全原创
- 实验充分度: ⭐⭐⭐⭐ 三个融合数据集+两个下游任务+完整消融，但缺少cross-dataset泛化分析
- 写作质量: ⭐⭐⭐⭐ 公式推导严谨，框架图清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 首次解决红外缺失融合，有实际应用前景，字典范式可推广

<!-- RELATED:START -->

## 相关论文

- [Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)

<!-- RELATED:END -->
