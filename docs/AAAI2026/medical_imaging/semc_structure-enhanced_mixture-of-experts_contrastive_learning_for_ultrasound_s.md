---
title: >-
  [论文解读] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition
description: >-
  [AAAI 2026][医学图像][超声标准切面识别] 提出 SEMC 框架，通过**语义-结构融合模块（SSFM）**对齐浅层结构线索与深层语义表征，结合**混合专家对比识别模块（MCRM）**在多层特征上进行分层对比学习，提升超声标准切面识别的细粒度判别能力，并构建了新的肝脏超声数据集 LP2025。
tags:
  - AAAI 2026
  - 医学图像
  - 超声标准切面识别
  - 混合专家
  - 对比学习
  - 语义结构融合
  - 肝脏超声
---

# SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition

**会议**: AAAI 2026  
**arXiv**: [2511.12559](https://arxiv.org/abs/2511.12559)  
**代码**: https://github.com/YanGuihao/SEMC  
**领域**: 医学图像 / 超声影像  
**关键词**: 超声标准切面识别, 混合专家, 对比学习, 语义结构融合, 肝脏超声

## 一句话总结

提出 SEMC 框架，通过**语义-结构融合模块（SSFM）**对齐浅层结构线索与深层语义表征，结合**混合专家对比识别模块（MCRM）**在多层特征上进行分层对比学习，提升超声标准切面识别的细粒度判别能力，并构建了新的肝脏超声数据集 LP2025。

## 研究背景与动机

1. **领域现状**：超声标准切面识别对疾病筛查、器官评估和生物测量至关重要。深度学习方法（如 SonoNet）已取得一定进展，但识别性能仍有限。
2. **现有痛点**：(a) 超声图像类内差异大（不同采集角度/压力导致同一切面外观差异大）而类间差异小（不同切面视觉模式相似），需要细粒度判别；(b) 现有方法主要依赖深层语义特征，忽略浅层结构线索（如解剖边界、纹理），导致结构感知不足；(c) 对比学习通过数据增强构造正负对，难以捕捉超声图像固有的细粒度语义差异。
3. **核心矛盾**：超声图像的低对比度和模糊边界使得仅靠深层特征无法有效区分相似切面，而浅层特征虽含结构信息但语义水平低。
4. **本文目标**：如何融合多尺度结构信息增强模型的结构感知能力？如何在多层特征上进行有效的对比学习以提升类别可分性？
5. **切入角度**：将浅层结构线索通过自适应压缩-扩展对齐后与深层专家特征融合，并用 MoE 机制在多层融合特征上做分层对比学习。
6. **核心 idea**：结构感知的特征融合 + 专家引导的分层对比学习 = 更强的超声标准切面判别能力。

## 方法详解

### 整体框架

基于 ResNet 骨干，前三个 block 共享提取浅层特征 $\{F_1, F_2, F_3\}$，第四个 block 分为三条独立参数的深层专家路径 $\{D_1, D_2, D_3\}$。SSFM 模块将浅层特征与深层专家特征对齐融合，输出送入 MCRM 模块进行分层对比学习和分类。

### 关键设计

1. **语义-结构融合模块（SSFM）**

    - 功能：将浅层结构线索（边缘、纹理）与深层语义特征对齐融合，增强结构感知。
    - 核心思路：包含两个子模块——(a) **自适应压缩-扩展块（ACE）**：通过逐级步进深度卷积降采样和通道调整，将浅层特征 $\{F_1, F_2, F_3\}$ 对齐到深层特征的空间和通道维度，然后通过逐元素加法融合 $M_i = F_i' + D_i$（避免拼接带来的通道冗余）；(b) **结构感知多上下文块（SAMC）**：在通道维度做自适应注意力 $\mathbf{C}_i$（GAP+GMP + FC），在空间维度做空间注意力 $\mathbf{S}_i$（Mean+Max + Conv），然后用多尺度并行卷积提取多感受野特征，最终通道混洗压缩输出。
    - 设计动机：浅层特征保留了精细结构信息但分辨率和通道数与深层不匹配。ACE 轻量对齐避免了上采样/下采样信息损失；SAMC 的通道-空间双重注意力突出解剖相关区域。

2. **混合专家对比识别模块（MCRM）**

    - 功能：在多层特征上进行分层对比学习以增强类别可分性，同时通过 MoE 分类提升识别准确率。
    - 核心思路：包含两个分支——(a) **MoE 对比分支**：三个专家输出 $\{\mathbf{O}_1, \mathbf{O}_2, \mathbf{O}_3\}$ 中，$\mathbf{O}_1$ 作为锚点（query），$\mathbf{O}_2, \mathbf{O}_3$ 作为正样本键，结合动量记忆队列 $\mathcal{Q}$ 存储历史表征，联合优化监督对比损失 $\mathcal{L}_{sup}$ 和自监督对比损失 $\mathcal{L}_{self}$；(b) **MoE 识别分支**：使用 Gumbel-Softmax 门控机制自适应选择最相关专家，计算加权融合预测 $\mathbf{z}_{fused} = \sum_n w_n \cdot \mathbf{z}_n$，用交叉熵损失训练。两分支通过自适应网络预测样本相关的权重 $\alpha$ 进行端到端平衡训练。
    - 设计动机：数据增强生成的正负对无法逼近超声图像的细粒度语义差异。多专家多层特征的分层对比学习直接在分类相关的特征空间中构造更有信息量的对比对。

3. **LP2025 肝脏超声数据集**

    - 功能：填补肝脏超声标准切面公开数据集的空白。
    - 核心思路：包含 9,369 张高质量临床验证图像，覆盖 6 种标准切面（第一肝门、第二肝门、左叶、右叶、左门静脉矢状面、肝肾界面）+ 非标准面，由多位 5 年以上经验的高级超声医师独立标注并经多阶段质量控制。

### 损失函数 / 训练策略

$L_{total} = \alpha \cdot L_{moe} + (1-\alpha) \cdot L_{mc}$，其中 $\alpha = g(\mathbf{O})$ 由自适应网络根据样本难度动态调整。$L_{mc} = L_{sup} + \lambda L_{self}$。

## 实验关键数据

### 主实验

在 FPUS23 和 CAMUS 公开数据集上的结果：

| 方法 | FPUS23 Acc↑ | FPUS23 F1↑ | CAMUS Acc↑ | CAMUS F1↑ |
|------|------------|-----------|-----------|----------|
| Diffmic | 95.29 | 81.08 | 80.91 | 79.69 |
| Metaformer | 95.52 | 94.53 | 81.52 | 80.49 |
| Area | 95.20 | 94.40 | 81.59 | 80.88 |
| Supmin | 95.28 | 94.34 | 81.13 | 79.71 |
| **SEMC** | **95.78** | **95.06** | **82.13** | **80.93** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| w/o SSFM | 准确率下降 | 浅层结构信息缺失 |
| w/o ACE | 准确率下降 | 特征对齐不足 |
| w/o SAMC | 准确率下降 | 多上下文结构感知缺失 |
| w/o 对比分支 | 准确率下降 | 类内紧凑性/类间可分性减弱 |
| w/o 识别分支 | 准确率下降 | 分类能力减弱 |
| w/o 动量队列 | 对比学习效果减弱 | 负样本池不足 |

### 关键发现

- SSFM 的浅层结构融合对非头部类切面（如外观细微差异的切面）提升最大。
- 三专家设计比单专家或双专家效果好，但专家数继续增加收益递减。
- Gumbel-Softmax 门控比简单平均或 Softmax 更能适应样本级别的特征选择。
- 在自建 LP2025 数据集上的 7 类分类中，SEMC 同样取得最优性能。

## 亮点与洞察

- **浅层-深层多粒度融合**：ACE 的渐进下采样+通道调整是一种优雅的特征对齐方案，避免了暴力拼接的通道冗余。这种设计可以推广到其他需要多尺度特征融合的医学影像任务。
- **分层对比学习**：用多专家输出作为锚点和正键进行对比，比数据增强生成的对比对更有语义意义。
- **自适应损失平衡**：用网络预测$\alpha$自动平衡分类和对比损失，避免了手动调参。

## 局限与展望

- LP2025 数据集来自单一医院，跨中心泛化性未验证。
- 非标准面（NSP）样本量远大于标准面（4626 vs <1100），类不平衡处理策略描述不充分。
- 骨干仅用 ResNet，更强的骨干（如 Swin Transformer）是否进一步提升未探索。
- 方法主要适用于分类任务，扩展到切面质量评估或标准面检测（定位）需要额外设计。

## 相关工作与启发

- **vs SonoNet**：SonoNet 基于 VGG 仅用深层特征，SEMC 加入浅层结构融合显著提升细粒度辨别。
- **vs MoCo/SimCLR**：通用对比学习依赖增强生成正负对；SEMC 通过多专家生成更有语义意义的对。
- **vs Diffmic**：扩散模型在此任务上准确率尚可但 F1 远低于 SEMC，说明细粒度辨别仍需专门设计。

## 评分

- 新颖性: ⭐⭐⭐⭐ SSFM + MoE 对比学习组合设计有新意
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证+消融完整，但跨中心实验缺失
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示说明充分
- 价值: ⭐⭐⭐⭐ 对超声标准切面识别有临床应用价值，LP2025 数据集有贡献

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](../../CVPR2026/medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](../../CVPR2025/medical_imaging/enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)

<!-- RELATED:END -->
