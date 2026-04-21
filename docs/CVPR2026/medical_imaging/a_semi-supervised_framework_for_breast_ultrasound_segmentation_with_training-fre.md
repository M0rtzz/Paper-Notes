---
title: >-
  [论文解读] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement
description: >-
  [CVPR 2026][医学图像][半监督分割] 提出面向乳腺超声（BUS）图像分割的半监督框架，利用 GPT-5 生成外观描述 + Grounding DINO + SAM 免训练生成伪标签（APPG），结合双教师框架（静态+动态）通过不确定性-熵加权融合（UEWF）和自适应不确定性引导反向对比学习（AURCL）精炼标签，仅用 2.5% 标注即接近全监督性能。
tags:
  - CVPR 2026
  - 医学图像
  - 半监督分割
  - 乳腺超声
  - 伪标签
  - 双教师框架
  - 对比学习
  - SAM
  - Grounding DINO
---

# A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement

**会议**: CVPR 2026  
**arXiv**: [2603.06167](https://arxiv.org/abs/2603.06167)  
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: 半监督分割, 乳腺超声, 伪标签, 双教师框架, 对比学习, SAM, Grounding DINO

## 一句话总结

提出面向乳腺超声（BUS）图像分割的半监督框架，利用 GPT-5 生成外观描述 + Grounding DINO + SAM 免训练生成伪标签（APPG），结合双教师框架（静态+动态）通过不确定性-熵加权融合（UEWF）和自适应不确定性引导反向对比学习（AURCL）精炼标签，仅用 2.5% 标注即接近全监督性能。

## 研究背景与动机

### 1. 领域现状
乳腺超声（BUS）是乳腺癌筛查的重要影像手段，肿瘤的精确分割是计算机辅助诊断的基础。深度学习方法依赖大规模像素级标注，但医学图像的标注成本极高——需要专业放射科医生逐像素标注，耗时且昂贵。半监督学习（SSL）通过利用大量未标注数据+少量标注数据来缓解这一问题，但在 BUS 场景中面临特殊困难。

### 2. 痛点
BUS 图像的特殊性：(1) 肿瘤与周围组织对比度低，边界模糊；(2) 不同肿瘤形态差异大（椭圆形、圆形、分叶状）；(3) 超声固有的斑点噪声和伪影。这些因素导致 SSL 方法的核心假设——模型能从少量标注中学到可靠的伪标签——在 BUS 中严重受损。特别是在极少标注（如 2.5%）的场景下，伪标签质量极差，模型陷入确认偏差的恶性循环。

### 3. 核心矛盾
传统 SSL（如 Mean Teacher）依赖模型自身生成伪标签，但模型在极少标注下本身就不可靠，生成的伪标签噪声大，反过来进一步误导训练。这是"鸡和蛋"的困境：需要好的伪标签来训练好模型，但好模型的前提是有好的伪标签。

### 4. 要解决什么
(1) 在极少标注下获得高质量初始伪标签，打破冷启动困境；(2) 在训练过程中持续精炼伪标签，避免单一教师的确认偏差；(3) 增强模型对边界不确定区域的判别能力。

### 5. 切入角度
利用视觉-语言基础模型（GPT-5 + Grounding DINO + SAM）作为免训练的伪标签生成器，跳过模型冷启动阶段；再用双教师+不确定性感知融合来持续精炼。

### 6. 核心 idea
分三步解决极少标注下的 BUS 分割：(1) APPG 利用乳腺肿瘤的通用外观先验，转化为自然语言 prompt 驱动基础模型生成免训练伪标签；(2) 静态教师（伪标签 warmup 后冻结）和动态教师（EMA 更新）提供互补视角；(3) UEWF 按不确定性加权融合两教师输出，AURCL 通过反向对比学习专门强化边界判别。

## 方法详解

### 整体框架

框架包含三个阶段：(1) **APPG 伪标签生成**：GPT-5 描述 → Grounding DINO 定位 → SAM 分割，为所有未标注数据生成初始伪标签；(2) **Warmup 训练**：使用标注数据+伪标签数据训练模型至收敛，冻结为静态教师 $T^A$；(3) **双教师半监督训练**：$T^A$（冻结）和 $T^B$（EMA）同时为学生网络生成伪标签，经 UEWF 融合后指导学生训练，AURCL 进一步增强不确定区域的学习。

### 关键设计

#### 1. APPG（Appearance-Prompted Pseudo-Label Generation）

**功能**：利用乳腺肿瘤的通用外观知识，免训练为未标注 BUS 图像生成分割伪标签。

**核心思路**：乳腺超声中的肿瘤具有可预测的外观特征——表现为低回声（暗色）的区域，形状通常为椭圆形（oval）、圆形（round）或分叶状（lobulated）。GPT-5 将这些医学知识转化为自然语言描述（如 "dark oval region"、"dark round mass"、"dark lobulated area"），作为 text prompt 输入 Grounding DINO 进行目标检测，输出边界框（bounding box）。随后将边界框作为空间 prompt 传入 SAM（Segment Anything Model），由 SAM 输出像素级分割掩码作为伪标签。

**设计动机**：(1) 完全无需训练，不依赖任何标注数据，巧妙利用了 VLM 的零样本能力；(2) 乳腺肿瘤的外观特征具有通用性（所有 BUS 图像中肿瘤都是低回声暗区），适合用通用描述覆盖；(3) Grounding DINO 擅长开放词汇检测，SAM 擅长基于 prompt 的精确分割，两者组合天然适配。

#### 2. 双教师框架（Dual-Teacher）

**功能**：提供两个互补的伪标签源，避免单教师的确认偏差。

**核心思路**：
- **静态教师 $T^A$**：用 APPG 伪标签 + 少量真实标注完成 warmup 训练后，权重完全冻结。它编码了来自基础模型的初始分割知识，不受后续训练噪声影响，提供稳定的伪标签基线。
- **动态教师 $T^B$**：从相同初始化出发，通过 EMA（指数移动平均）持续跟踪学生模型的更新。它能捕捉训练过程中学到的新知识，适应数据分布的变化，但可能累积误差。
- 两个教师分别生成伪标签 $\hat{y}^A$ 和 $\hat{y}^B$，通过 UEWF 机制融合为最终伪标签 $\hat{y}^F$。

**设计动机**：单一 EMA 教师（如 Mean Teacher）在极少标注下容易陷入退化循环——错误伪标签 → 学生学偏 → EMA 教师也偏 → 更差伪标签。静态教师提供了独立于训练过程的"锚点"，打断这一循环。

#### 3. UEWF（Uncertainty-Entropy Weighted Fusion）

**功能**：根据两个教师各自的置信度，自适应加权融合其伪标签。

**核心思路**：对每个像素计算两教师预测的信息熵 $H^A$ 和 $H^B$。熵低意味着教师对该像素的预测更确定。用逆熵作为权重：

$$\hat{y}^F = w^A \cdot \hat{y}^A + w^B \cdot \hat{y}^B, \quad w^A = \frac{H^B}{H^A + H^B}, \quad w^B = \frac{H^A}{H^A + H^B}$$

即对某个像素，哪个教师更不确定（熵高），其权重就更低；哪个更确定（熵低），其权重更高。这样每个像素自适应选择更可靠教师的预测。

**设计动机**：不同教师在不同区域的可靠性不同——$T^A$ 可能在整体形状上更稳定，$T^B$ 可能在细节区域更精准。逆熵加权无需额外参数，计算简单，且天然逐像素自适应。

#### 4. AURCL（Adaptive Uncertainty-Guided Reverse Contrastive Learning）

**功能**：专门针对边界模糊等不确定区域，增强模型的判别能力。

**核心思路**：分四步：(1) **不确定性图计算**：对学生模型的预测经过多次 Monte Carlo Dropout，计算每个像素的方差作为不确定性度量，方差越大说明模型越不确定；(2) **低置信区域提取**：以不确定性阈值 $\tau$ 筛选出高不确定性区域，即模型"拿不准"的边界区域；(3) **概率反转（Reverse）**：对低置信区域的预测概率做反转（$1-p$），直觉是"如果模型不确定地预测某像素为前景，那反转后的表示更可能代表背景"；(4) **Patch 对比学习**：将反转后的低置信区域 patch 特征与高置信前景/背景 patch 特征做对比学习——拉近与对应类别的距离，推远与其他类别的距离。

**设计动机**：标准 SSL 对高置信区域利用充分，但对模糊边界区域几乎无能为力。AURCL 专门挖掘这些"难例"区域，通过反转操作巧妙将不确定性转化为可学习的信号。对比学习在特征空间约束边界区域的表示，补充像素级损失的不足。

### 损失函数 / 训练策略

- 有标注数据：标准分割监督损失（CE + Dice）
- 无标注数据：UEWF 融合伪标签的分割损失 + AURCL 对比损失
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_1 \mathcal{L}_{\text{unsup}} + \lambda_2 \mathcal{L}_{\text{AURCL}}$
- 学生模型为 U-Net 变体，$T^B$ 的 EMA 衰减率 $\alpha = 0.999$
- APPG 中每张图像使用3种外观描述生成候选框，NMS 去冗余后取最高置信度框

## 实验关键数据

### 主实验

在4个公开 BUS 数据集（BUSI、UDIAT、BUS-BRA、TN3K）上评估，标注比例 2.5%、5%、10%。

**关键结果（Dice %）**：

| 方法 | BUSI 2.5% | BUSI 5% | BUSI 10% | UDIAT 2.5% | UDIAT 5% | UDIAT 10% |
|------|-----------|---------|----------|------------|---------|-----------|
| Supervised-only | 51.2 | 60.8 | 69.4 | 53.7 | 63.2 | 71.5 |
| Mean Teacher | 58.6 | 66.3 | 73.8 | 60.4 | 68.1 | 75.2 |
| CPS | 59.1 | 67.0 | 74.1 | 61.2 | 69.3 | 75.8 |
| UniMatch | 62.4 | 69.5 | 76.2 | 64.0 | 71.8 | 78.1 |
| **Proposed** | **71.8** | **75.3** | **79.6** | **72.5** | **76.9** | **81.4** |
| Full Supervision | 80.2 | 80.2 | 80.2 | 82.1 | 82.1 | 82.1 |

关键发现：(1) 2.5% 标注下本文方法（71.8% Dice on BUSI）大幅超越最强基线 UniMatch（62.4%），提升 +9.4%；(2) 2.5% 标注性能已达全监督的 89.5%（71.8/80.2），5% 标注下达到 93.9%；(3) 在所有4个数据集、所有标注比例下均一致超越。

### 消融实验

**组件消融（BUSI 2.5% Dice）**：

| 配置 | Dice (%) |
|------|----------|
| Baseline（单教师 Mean Teacher） | 58.6 |
| + APPG 伪标签初始化 | 65.2 |
| + 双教师（简单平均） | 67.8 |
| + UEWF（替代简单平均） | 69.5 |
| + AURCL（完整方法） | 71.8 |

每个组件均有清晰贡献：APPG（+6.6%）> 双教师（+2.6%）> UEWF（+1.7%）> AURCL（+2.3%）。APPG 的免训练伪标签是最大贡献者。

### APPG 伪标签质量

APPG 生成的伪标签与真实标注的平均 Dice：BUSI 66.3%、UDIAT 68.7%。虽不完美，但远优于随机初始化的模型预测（~35-40%），为后续训练提供了强有力的起点。

## 亮点与洞察

1. **VLM 作为免费午餐**：利用 GPT-5 + Grounding DINO + SAM 的组合，将医学领域知识（乳腺肿瘤外观）零成本转化为初始伪标签，巧妙跳过冷启动困境
2. **静态+动态教师设计简洁有效**：静态教师作为锚点防止漂移，动态教师跟踪学习进度，UEWF 逐像素自适应融合——无需复杂架构改动
3. **AURCL 的反转操作有洞察**：将"不确定"转化为"反向确定"的思路巧妙，对比学习在特征空间构建边界感知表示
4. **极少标注下优势明显**：2.5% 标注（仅几张有标注）就达到接近全监督性能，这对标注资源极度稀缺的医学场景意义重大

## 局限与展望

1. **APPG 依赖外观先验的通用性**：对乳腺肿瘤有效，但并非所有病变都有统一的低回声外观，推广到其他器官/病变类型需重新设计 prompt
2. **GPT-5 + Grounding DINO + SAM 的部署成本**：虽是免训练但推理成本不低，特别是 GPT-5 的 API 调用成本
3. **伪标签质量上限**：APPG 的 ~67% Dice 仍有较大提升空间，SAM 对低对比度超声图像的分割精度有限
4. **未探索更强的分割骨干**：主要基于 U-Net，未测试 Swin-UNet、TransUNet 等更强架构
5. **对比学习超参数敏感性**：AURCL 中不确定性阈值 $\tau$ 和对比温度对性能的影响未充分分析

## 相关工作与启发

- **SSL 分割方法演进**：Mean Teacher → CPS（互学习）→ UniMatch（多视角一致性）→ 本文（VLM 伪标签 + 双教师），趋势是引入更强的先验来弥补标注不足
- **VLM 在医学图像中的应用**：MedSAM、SAMed 等微调 SAM 做医学分割，但需要目标域标注；本文的 APPG 完全免训练，更符合低标注场景需求
- **不确定性在 SSL 中的角色**：以往主要用不确定性做伪标签过滤（丢弃不确定样本），本文的 AURCL 反转思路将不确定区域变为可利用的正信号，可推广到其他 SSL 框架

## 评分

⭐⭐⭐⭐ 方案完整且各组件互补性强，APPG 利用 VLM 免训练生成伪标签的思路在极少标注医学场景下非常实用，4个数据集全面验证；不足在于 APPG 的外观先验通用性有限，且 VLM 推理成本未讨论。

<!-- RELATED:START -->

## 相关论文

- [Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

<!-- RELATED:END -->
