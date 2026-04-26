---
title: >-
  [论文解读] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback
description: >-
  [CVPR 2026][人体理解][虚拟试穿] 构建 VTON-QBench（62,688 张试穿图像、13,838 名合格标注者、431,800 条标注）并提出 VTON-IQA 无参考质量评估框架，通过非对称交错交叉注意力（ICA）模块联合建模服装保真度和人物保持度，实现与人类感知高度对齐的图像级质量预测。
tags:
  - CVPR 2026
  - 人体理解
  - 虚拟试穿
  - 无参考质量评估
  - 人类反馈对齐
  - 交错交叉注意力
  - VTON-QBench
---

# Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**会议**: CVPR 2026  
**arXiv**: [2603.13057](https://arxiv.org/abs/2603.13057)  
**代码**: [GitHub](https://github.com/litelightlite/VTON-IQA)  
**领域**: 图像生成评估 / 虚拟试穿  
**关键词**: 虚拟试穿, 无参考质量评估, 人类反馈对齐, 交错交叉注意力, VTON-QBench

## 一句话总结

构建 VTON-QBench（62,688 张试穿图像、13,838 名合格标注者、431,800 条标注）并提出 VTON-IQA 无参考质量评估框架，通过非对称交错交叉注意力（ICA）模块联合建模服装保真度和人物保持度，实现与人类感知高度对齐的图像级质量预测。

## 研究背景与动机

**领域现状**：虚拟试穿（VTON）在时尚电商中日益重要，给定人物图像和服装图像合成穿着效果是核心任务。从 GAN 到 U-Net 扩散模型再到 DiT，试穿质量持续提升。

**现有痛点**：

1. 真实场景中没有 ground-truth 图像（不可能让同一人穿目标服装拍照），SSIM/LPIPS 等有参考指标无法使用
2. FID/KID 只衡量数据集级分布相似性，无法反映单张图像质量
3. 现有 VTON 专用评估方法（VTONQA、VTBench、VTON-VLLM）要么数据集规模小（748 对）、要么未公开实现、要么缺乏大规模人类验证

**核心矛盾**：需要单张图像级、无参考、与人类感知对齐的质量评估，但现有工具均不满足。

**本文目标** 建立大规模人类标注基准 + 训练无参考质量预测模型。

**切入角度**：试穿质量本质上是两个方面的验证——(1) 服装是否忠实迁移，(2) 非目标区域是否保持。这天然需要跨图像交互建模。

**核心 idea**：构建大规模人类标注数据集，用非对称交叉注意力显式建模试穿图与服装/人物图的一致性。

## 方法详解

### 整体框架

三分支 Transformer 架构（基于 DINOv3 ViT-L/16），分别处理服装 $I_G$、人物 $I_P$ 和试穿结果 $I_V$。前 L/2 层独立提特征，后 L/2 层引入 ICA 模块进行跨图像交互。最终提取三个 [CLS] token，通过可学习加权余弦相似度融合，经 $\tanh$ 映射到 $[-1, 1]$ 输出质量分数。

### 关键设计

1. **VTON-QBench 数据集构建**

    - 14 个代表性 VTON 模型（含 GAN/U-Net 扩散/DiT/商业 API）生成 62,688 张试穿图像
    - 13,153 对服装-人物对（原始 6,981 对 + FLUX.1-dev LoRA 增广 6,172 对合成对，覆盖休闲/街头/正式/极简/复古风格）
    - 13,838 名合格标注者提供 431,800 条三级质量标注（自然/微不自然/不自然）
    - 两阶段标注清洗：(1) 5 道答案明确的检验题 + 行为过滤（>80% 相同答案或 >60% 与多数投票不一致者剔除），Krippendorff's α 从 0.286 提升到 0.550；(2) 删除 α ≤ 0.4 的问卷
    - 构建伪三元组用 Nano Banana Pro 做强模型生成参考图像，支持有参考指标对比

2. **交错交叉注意力（ICA）模块**

    - 在标准 Transformer 的 SA 和 MLP 之间插入交叉注意力层
    - **非对称交互设计**：试穿图 $V$ 与服装 $G$ 和人物 $P$ 双向交互：$\hat{X}_V^{(\ell)} = \tilde{X}_V^{(\ell)} + C_{V \leftarrow G}^{(\ell)} + C_{V \leftarrow P}^{(\ell)}$
    - 但 $G$ 和 $P$ 之间**不直接交互**，只通过 $V$ 连接：$\hat{X}_G^{(\ell)} = \tilde{X}_G^{(\ell)} + C_{G \leftarrow V}^{(\ell)}$
    - 这反映了质量评估以试穿图为中心——需验证 $V$ 是否保持了 $G$ 的服装属性和 $P$ 的非目标要素

3. **评分机制**

    - 三个 [CLS] token $c_G, c_P, c_V$，融合分数 $\tilde{s} = \alpha \frac{c_G^\top c_V}{\|c_G\|\|c_V\|} + (1-\alpha) \frac{c_P^\top c_V}{\|c_P\|\|c_V\|}$
    - $\alpha$ 为可学习标量，自适应平衡服装一致性和人物保持度的权重
    - 最终 $\hat{s} = \tanh(a\tilde{s} + b)$，约束在 $[-1,1]$

### 损失函数 / 训练策略

联合优化 Bradley-Terry 偏好学习和分数回归：

$$\mathcal{L}_\theta = -q_{ij} \log p_\theta - (1-q_{ij}) \log(1-p_\theta) + \sum_{k \in \{i,j\}} \|\Psi_\theta(I_G, I_P, I_{V_k}) - S_k\|_2^2$$

AdamW，lr=1e-4，batch size 16，早停策略（验证损失 3 epoch 无改善），单卡 A100 40GB，bfloat16 混合精度。

## 实验关键数据

### 主实验

| 方法 | ρ_SRCC | ρ_PLCC | R² | A_macro | A_micro | 无参考 |
|------|--------|--------|-----|---------|---------|--------|
| SSIM | — | 0.135 | — | 0.596 | 0.593 | ✗ |
| LPIPS | — | 0.387 | — | 0.701 | 0.695 | ✗ |
| DINOv3 (zero-shot) | — | 0.261 | — | 0.637 | 0.641 | ✓ |
| VTON-IQA w/o ICA | 0.617 | 0.615 | 0.372 | 0.722 | 0.747 | ✓ |
| **VTON-IQA** | **0.750** | **0.751** | **0.553** | **0.781** | **0.790** | ✓ |
| 人类 | 0.760 | 0.762 | 0.536 | 0.782 | 0.791 | — |

### 14 个 VTON 模型排名（Dress Code 数据集，VTON-IQA 分数）

| 排名 | 模型 | 分数 |
|------|------|------|
| 1 | Nano Banana Pro | 0.305 |
| 2 | GPT-Image-1.5 | 0.237 |
| 3 | FitDit | 0.219 |
| 4 | IDM-VTON | 0.141 |
| ... | ... | ... |
| 13 | HR-VITON (GAN) | -0.835 |
| 14 | VITON-HD (GAN) | -0.933 |

### 消融实验

| 配置 | ρ_SRCC | A_micro |
|------|--------|---------|
| DINOv3 zero-shot | — | 0.641 |
| + 微调（无 ICA） | 0.617 | 0.747 |
| + ICA | **0.750** | **0.790** |

ICA 模块贡献 +21.6% SRCC 提升和 +5.8% 微匹配准确率。

### 关键发现

- SSIM/LPIPS 在姿态/缩放变化时严重偏离人类判断；VTON-IQA 对全局变换鲁棒
- GPT-Image-1.5 在传统指标中被低估（因零样本模型常改变姿态/缩放），VTON-IQA 正确反映其高质量
- 人类 vs 模型：成对准确率几乎持平（0.782 vs 0.781），但相关性指标仍有差距
- DiT 模型整体优于 U-Net 扩散，GAN 方法大幅落后

## 亮点与洞察

- **数据集规模惊人**：62K 图像、13K 标注者、431K 标注，是目前 VTON 评估领域最大的人类标注数据集
- 严格的标注质量控制流程（Krippendorff's α 阈值、多阶段清洗）可在其他众包标注项目中复用
- ICA 的非对称设计巧妙编码了试穿质量评估的语义结构——以试穿图为中心
- 14 模型全面基准测试揭示了传统指标与感知质量的系统性偏差

## 局限与展望

- 仅限于工作室场景的标准试穿，未覆盖野外复杂背景/姿态
- 只输出标量分数，缺乏可解释的属性级反馈（如"领口不匹配"）
- 三级标注量表可能过于粗糙，细粒度 5-7 级评分可能更有用
- 未扩展到视频试穿或 3D 虚拟试穿场景

## 相关工作与启发

- **vs VTONQA**：数据集规模小（748 对 vs 13,153 对）且未公开；本文数据集和代码将开源
- **vs VTBench**：提供多维度诊断框架但不学习统一的质量评估模型
- **vs VTON-VLLM**：聚焦文本式批评而非定量预测
- 启发：无参考质量评估的思路可迁移到其他条件生成任务（图像编辑、风格迁移）的评估；ICA 的非对称交互设计可用于所有"验证生成结果是否保持输入条件"的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务定义和数据集构建的系统性工作，ICA 设计有新意但不算突破性
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个 VTON 模型、跨组泛化、人类对比、分类别评估，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建流程描述细致
- 价值: ⭐⭐⭐⭐⭐ 填补了 VTON 领域缺乏标准化评估基准的空白，有望成为社区标准工具

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile-vton_high-fidelity_on-device_virtual_try-on.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)

<!-- RELATED:END -->
