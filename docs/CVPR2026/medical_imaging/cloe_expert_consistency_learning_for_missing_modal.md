---
title: >-
  [论文解读] CLoE: Expert Consistency Learning for Missing Modality Segmentation
description: >-
  [CVPR 2026][医学图像][missing modality] 将缺失模态下的鲁棒性问题重新定义为决策级专家一致性控制，提出双分支一致性学习（全局MEC+区域REC）配合轻量门网络将一致性分数转化为模态可靠性权重，在BraTS 2020上15种缺失组合平均WT Dice达88.09%超越所有SOTA。
tags:
  - CVPR 2026
  - 医学图像
  - missing modality
  - consistency learning
  - expert fusion
  - reliability gating
  - brain tumor
---

# CLoE: Expert Consistency Learning for Missing Modality Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.09316](https://arxiv.org/abs/2603.09316)  
**代码**: 无  
**领域**: 医学图像分割 / 多模态学习  
**关键词**: missing modality, consistency learning, expert fusion, reliability gating, brain tumor

## 一句话总结

将缺失模态下的鲁棒性问题重新定义为决策级专家一致性控制，提出双分支一致性学习（全局MEC+区域REC）配合轻量门网络将一致性分数转化为模态可靠性权重，在BraTS 2020上15种缺失组合平均WT Dice达88.09%超越所有SOTA。

## 研究背景与动机

**领域现状**：多模态MRI分割（如脑肿瘤的T1/T1c/T2/FLAIR四模态）在训练时假设所有模态可用，采用U-Net/V-Net等编码器-解码器架构。临床中模态缺失（扫描中断、协议差异、质量问题）极为普遍。

**现有痛点**：(1) GAN合成缺失模态计算昂贵且引入幻觉；(2) HeMIS等算术融合在零填充时注意力机制失效；(3) RFNet的空间先验是被动的——指定看哪里但不知道哪个专家可信；(4) Mean Teacher等一致性学习被体积MRI中背景像素主导，全局一致≠前景小目标对齐。

**核心矛盾**：模态缺失不只是信息减少的问题，更导致模态专家间预测分歧加大——naive融合反而放大分歧而非消除分歧，尤其在临床关键的小前景结构上。

**本文目标** 如何将模态专家间的一致性量化为可靠性信号，并用该信号引导动态融合？

**切入角度**：将鲁棒性从表征层面提升到决策层面——不是学更好的特征，而是控制专家间谁说了算。

**核心 idea**：用专家预测间的余弦相似度量化一致性，分全局(MEC)和前景区域(REC)两层，再通过门网络将一致性分数转化为融合权重。

## 方法详解

### 整体框架

CLoE包含三个核心部分：(1) 并行模态编码器 $\Phi_m$ 提取多尺度特征；(2) 共享专家解码器 $D^{sep}$ 生成单模态预测 $p^{(m)}$；(3) ECL模块计算MEC/REC一致性分数，门网络映射为融合权重 $w_m$；(4) 加权融合后送入融合解码器 $D^{fuse}$ 输出最终分割。

### 关键设计

1. **模态专家一致性(MEC) + 区域专家一致性(REC)**
    - MEC：将可用专家的概率预测向量化后计算余弦相似度，对所有可用对取平均作为损失 $\mathcal{L}_{MEC} = \frac{1}{|\mathcal{P}|}\sum_{(a,b)}(1-\mathcal{S}(p^{(a)}, p^{(b)}))$
    - REC：通过轻量投影头从可用专家浅层特征聚合生成概率区域图 $r=\sigma(\pi(\frac{1}{|\mathcal{A}|}\sum f_1^{(m)}))$，用 $r$ 对预测加权后再计算余弦相似度
    - 设计动机：MEC约束全局分布对齐防止专家漂移，REC聚焦前景临床关键区域避免被背景主导

2. **一致性驱动动态门控**
    - 每个专家 $m$ 的全局一致性分数 $u_m$ 和区域一致性分数 $v_m$ 送入轻量门网络 $\mathcal{G}$
    - 输出可靠性logit $g_m = \mathcal{G}(u_m, v_m)$，对可用专家softmax归一化得融合权重 $w_m$
    - 多尺度特征按 $w_m$ 加权融合：$f_\ell = \sum w_m \odot f_\ell^{(m)}$
    - 设计动机：不仅用一致性做约束，更用一致性信号作为可靠性权重引导融合——一致的专家被信任，偏离的专家被抑制

### 损失函数 / 训练策略

总目标 $\mathcal{L}_{total} = \mathcal{L}_{seg} + \alpha \mathcal{L}_{ECL} + \beta \mathcal{L}_{contrast}$：
- $\mathcal{L}_{seg}$：融合预测的加权交叉熵+Dice损失
- $\mathcal{L}_{ECL}$：各专家独立监督(WCE+DL) + $\eta(\mathcal{L}_{MEC} + \lambda_{rec}\mathcal{L}_{REC})$
- $\mathcal{L}_{contrast}$：对比表征损失（SSIM对齐解剖内容 + Cosine聚类模态风格 + KL正则）
- Adam优化器，lr=0.0002, weight decay=0.0001, batch=1, 500 epochs, 输入112³ 3D patch

## 实验关键数据

### 主实验

| 数据集 | 指标 | CLoE | DC-Seg | M³AE | RFNet | HeMIS |
|---|---|---|---|---|---|---|
| BraTS2020 (15组合Avg) | WT Dice% | **88.09** | 87.54 | 86.90 | 86.98 | 75.10 |
| BraTS2020 (15组合Avg) | TC Dice% | **80.23** | 79.63 | 79.10 | 78.23 | 65.45 |
| BraTS2020 (15组合Avg) | ET Dice% | **65.06** | 65.00 | 61.70 | 61.47 | - |
| BraTS2020 Full | WT Dice% | **91.30** | 90.95 | 90.40 | 91.11 | 85.19 |
| MSD Prostate PZ Avg | Dice% | **80.12** | 79.59 | - | 77.35 | - |

### 消融实验

| 消融配置 | Avg Dice变化 | ET Dice变化 | 说明 |
|---------|------------|-----------|------|
| 去掉 REC | -1.98% | -3.41% | 区域一致性对小前景至关重要 |
| 去掉 Weight Fusion | -2.47% | -3.96% | 动态融合权重贡献最大 |
| 去掉 MEC | -0.70% | - | 全局一致性是精细调优角色 |
| 去掉 Gating Network | -0.47% | - | 参数化贡献有限但方向正确 |

### 关键发现

- REC和Weight Fusion是CLoE最关键的两个组件，对ET（临床最关键的增强肿瘤）提升最显著
- 在全模态设置下CLoE也不牺牲性能（WT 91.30%），证明一致性约束不引入退化
- 跨数据集泛化（BraTS→MSD Prostate）验证了框架通用性
- MedSAM即使给bbox提示也无法生成清晰肿瘤边界——专用多模态框架仍有价值

## 亮点与洞察

- 将鲁棒性重formulate为决策级一致性控制，视角清晰且直觉上合理
- REC用概率区域图自动识别前景关键区域，无需手动标注ROI，是解决"背景主导一致性"问题的优雅方案
- 一致性分数→可靠性权重的转化设计：不仅约束一致性，还将一致性信号循环利用为融合信号
- 在前景极小的ET分割上改进明显（+3.59% vs RFNet），证明REC对临床关键小目标有效

## 局限与展望

- 消融显示MEC和Gating各自独立贡献较小（-0.70%/-0.47%），设计空间可能有压缩余地
- 仅在2个数据集上验证，更多器官/更多模态组合有价值
- 极端缺失（仅剩1个模态）时无法做成对比较，一致性分数的意义退化
- 区域概率图r由浅层特征生成，训练早期可能不稳定
- 门网络结构极简（2维输入→1维输出），表达能力是否充分有待验证

## 相关工作与启发

- **vs DC-Seg**：DC-Seg通过VAE对比学习做潜在空间解耦，CLoE在此基础上增加决策级一致性控制和动态融合，二者互补
- **vs M³AE**：M³AE是大规模预训练方案（掩码自编码器），CLoE用更轻量框架超越了它
- **vs RFNet**：RFNet用区域感知先验但方式被动，CLoE的REC主动学习前景区域并量化专家可靠性
- 一致性→可靠性的框架可推广到其他多源信息融合场景（如自动驾驶多传感器融合）
- 区域一致性的前景聚焦思路对小目标检测/分割有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐ 一致性学习+动态门控的组合每个组件不算全新，但问题formulation清晰
- 实验充分度: ⭐⭐⭐⭐ 15种缺失组合全覆盖、跨数据集验证、消融细致
- 写作质量: ⭐⭐⭐ 方法描述清楚，论文篇幅有限部分细节需看算法框
- 价值: ⭐⭐⭐ 缺失模态分割是重要临床问题，改进幅度合理但不dramatic

<!-- RELATED:START -->

## 相关论文

- [MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)
- [Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
