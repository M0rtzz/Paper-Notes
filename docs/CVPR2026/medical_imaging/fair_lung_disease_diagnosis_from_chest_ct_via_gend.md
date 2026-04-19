---
title: >-
  [论文解读] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning
description: >-
  [CVPR 2026 (PHAROS-AIF-MIH Workshop)][医学图像][公平性诊断] 在 ConvNeXt-Base 骨干上构建注意力 MIL 模型，用 GRL 对抗性消除扫描表示中的性别信息，配合 focal loss（$\gamma=2$）+ 标签平滑（$\varepsilon=0.1$）、子群过采样和 5-fold 集成，在 889 例胸部 CT 四类诊断中实现均值竞赛分数 0.685±0.030，女性 macro-F1（0.691）略高于男性（0.679），验证了 GRL 能有效闭合公平性差距。
tags:
  - CVPR 2026 (PHAROS-AIF-MIH Workshop)
  - 医学图像
  - 公平性诊断
  - 胸部CT
  - 多示例学习
  - 梯度反转层
  - 肺疾病分类
---

# Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning

**会议**: CVPR 2026 (PHAROS-AIF-MIH Workshop)  
**arXiv**: [2603.12988](https://arxiv.org/abs/2603.12988)  
**代码**: [GitHub](https://github.com/ADE-17/cvpr-fair-chest-ct)  
**领域**: 医学图像  
**关键词**: 公平性诊断, 胸部CT, 多示例学习, 梯度反转层, 肺疾病分类

## 一句话总结

在 ConvNeXt-Base 骨干上构建注意力 MIL 模型，用 GRL 对抗性消除扫描表示中的性别信息，配合 focal loss（$\gamma=2$）+ 标签平滑（$\varepsilon=0.1$）、子群过采样和 5-fold 集成，在 889 例胸部 CT 四类诊断中实现均值竞赛分数 0.685±0.030，女性 macro-F1（0.691）略高于男性（0.679），验证了 GRL 能有效闭合公平性差距。

## 研究背景与动机

**领域现状**：深度学习在胸部 CT 自动分析中取得巨大进展，可实现大规模肺恶性肿瘤和 COVID-19 筛查。然而公平性研究表明，模型容易编码并放大训练数据中的人口统计学偏差，对弱势群体产生系统性更差的诊断结果。

**现有痛点**：CVPR 2026 PHAROS-AIF-MIH 挑战赛数据集（889 例 CT：734 训练/155 验证，四类：Healthy/COVID-19/Adenocarcinoma/Squamous Cell Carcinoma）存在严重交叉不平衡——女性鳞癌仅 18 例 vs 男性 91 例，CT 深度从 <20 到 800+ 切片高度可变。竞赛指标为男女 macro-F1 均值 $P = \frac{1}{2}(\text{MacroF1}_\text{male} + \text{MacroF1}_\text{female})$，直接惩罚性别不公平。

**核心矛盾**：三个相互纠缠的挑战——(1) 体积信号稀疏：百余张切片中仅数张含病变，mean pooling 被健康切片淹没；(2) 人口统计学不平衡：女性鳞癌极度稀缺，标准训练严重不足；(3) 性别作为隐式捷径：即使不输入性别，模型可从体型和采集参数编码性别特征并与疾病共现统计量耦合。

**本文目标** 在端到端框架中同时应对信号稀疏、子群不平衡和性别编码，实现性别公平的四类肺疾病诊断。

**切入角度**：将 CT 视为切片 bag 用 MIL 自动选择信息切片，用 GRL 对抗性解耦性别，用公平性协议平衡子群。

**核心 idea**：注意力 MIL 聚合信息切片 + GRL 消除性别捷径 + 子群过采样闭合公平性差距。

## 方法详解

### 整体框架

输入 CT 体积（限制最多 $M=32$ 张切片），ConvNeXt-Base 提取每张切片的 $D$ 维嵌入，两层 MLP 注意力网络计算切片权重并加权求和得扫描级表示 $H$，随后分别送入 4 类疾病分类头和经 GRL 连接的性别对抗头（二分类），端到端联合训练。推理时 5-fold 全集成 + 水平翻转 TTA + OOF 阈值优化。

### 关键设计

1. **注意力 MIL 聚合**

    - 功能：从可变长度 CT 切片序列中学习哪些切片含诊断信息，加权聚合为扫描级表示
    - 核心思路：ConvNeXt-Base（去分类头）提取每张切片嵌入 $h_i = f_\text{enc}(x_i) \in \mathbb{R}^D$，两层 MLP 产生重要性分数 $s_i = a(h_i; \theta_a)$，softmax 归一化后加权 $H = \sum_i w_i h_i$。零填充位置施加 attention mask 屏蔽。训练时 $N>M$ 的体积随机采样，推理均匀采样保持空间覆盖
    - 设计动机：Mean pooling 被健康切片稀释信号，Max pooling 对伪影敏感。注意力机制作为两者的学习型折中，且不需要切片级标注

2. **GRL 对抗性别去偏**

    - 功能：从扫描表示中擦除性别预测信息，阻止模型利用性别作为诊断捷径
    - 核心思路：在 $H$ 上挂接 GRL + 两层 MLP 二分类器 $z_\text{gen} = c(\mathcal{R}_\lambda(H))$。前向恒等，反向梯度取反并缩放 $\lambda_\text{adv}$。总损失 $\mathcal{L} = \mathcal{L}_\text{disease} + \lambda_\text{adv} \cdot \mathcal{L}_\text{gender}$，性别头训练预测性别，反转梯度强迫骨干丢弃性别信息
    - 设计动机：骨干可从体型/采集参数等隐式编码性别特征。GRL 是最小侵入性的公平性约束——不改变主任务架构，仅增加对抗分支

3. **公平性训练协议**

    - 功能：多管齐下确保极端不平衡子群（女性鳞癌仅 18 例）不被忽略
    - 核心思路：(a) 按 (class, gender) 8 子群分层的 5-fold CV，保证每折含所有子群；(b) WeightedRandomSampler 大幅提升女性鳞癌采样权重，几乎每个 batch 均包含该子群；(c) 两阶段微调——前 5 epoch 冻结骨干只训练注意力和两个头（LR=1e-3），之后解冻骨干（骨干 LR=1e-5，头 LR=1e-4，cosine 退火）
    - 设计动机：单一策略在极端不平衡场景不够——过采样防崩塌，分层折保公平评估，两阶段让注意力先稳定

### 损失函数 / 训练策略

- 疾病损失：focal loss（$\gamma=2, \alpha=0.25$）+ 标签平滑（$\varepsilon=0.1$），$\tilde{p}_t = (1-\varepsilon)p_t + \varepsilon/C$
- 性别损失：二元交叉熵
- AdamW（$\beta_1=0.9, \beta_2=0.999$, WD=0.05）；梯度累积 $K=4$（等效 batch=16）；50 epoch；单卡 RTX A4000
- 推理：5-fold soft logit 投票 + 水平翻转 TTA；OOF per-class 阈值优化（dense grid $\mathcal{T} \subset [0.05, 0.95]$）

## 实验关键数据

### 主实验——Per-Fold 验证结果

| Fold | 竞赛分数 P | Male macro-F1 | Female macro-F1 | F1-腺癌 | F1-鳞癌 |
|------|-----------|---------------|-----------------|---------|---------|
| 0 | 0.698 | 0.673 | 0.722 | 0.807 | 0.258 |
| 1 | **0.727** | 0.754 | 0.699 | 0.796 | 0.378 |
| 2 | 0.674 | 0.658 | 0.690 | 0.692 | 0.500 |
| 3 | 0.688 | 0.743 | 0.634 | 0.803 | 0.303 |
| 4 | 0.637 | 0.565 | 0.709 | 0.681 | 0.389 |
| Mean±Std | 0.685±0.030 | 0.679±0.068 | 0.691±0.030 | 0.756±0.057 | 0.366±0.083 |

### OOF 全局集成结果

| 模型 | P | M-F1 | F-F1 | F1-A | F1-G | F1-Cov |
|------|---|------|------|------|------|--------|
| OOF Global Mean | 0.683 | 0.679 | 0.688 | 0.755 | 0.366 | 0.813 |
| OOF ± | 0.032 | 0.066 | 0.029 | 0.056 | 0.083 | 0.070 |

### 消融实验（定性路径）

| 设计选择 | 解决的挑战 | 改进 |
|---------|-----------|------|
| Mean → Max Pooling | 稀疏肿瘤信号被稀释 | 恢复对稀疏肿瘤切片正预测能力 |
| Max → Attention-MIL | 背景和边界切片噪声 | 学习动态忽略空肺区域，提升鲁棒性 |
| + 子群过采样 | 极端交叉稀缺（仅 18 例女性鳞癌） | 防止类别崩塌，大幅提升 Female macro-F1 |
| + GRL | 肿瘤特征与性别特征纠缠 | 闭合公平性差距（P=0.685，F-F1≈M-F1） |

### 关键发现

- GRL 成功解耦性别与肿瘤特征：Female macro-F1（0.691）略高于 Male（0.679），验证模型不再依赖性别偏差
- 鳞癌 F1 最低（0.366±0.083），根本约束是数据稀缺（仅 18 例女性鳞癌）而非方法缺陷
- 5-fold 集成 + TTA 有效缓解高方差折（如 Fold 4 的 0.637）的拖累
- OOF 阈值优化比直接 argmax 更稳健，全局竞赛分数 0.683 且无泄漏风险

## 亮点与洞察

- GRL 是极简但有效的公平性约束——不改变主任务架构，仅增加对抗分支。这种"最小侵入性公平性"可迁移到任何需要去偏的医学影像任务
- 极端子群不平衡下 WeightedRandomSampler + focal loss + 标签平滑的组合是可行补救——单一策略不足，多管齐下才能避免崩塌
- 两阶段微调（先稳定注意力头 → 再解冻骨干）对 MIL 训练稳定性至关重要
- OOF 阈值优化被低估——在小数据集上直接在验证集调阈值容易过拟合，OOF 提供无泄漏的全局估计

## 局限与展望

- 鳞癌 F1 仅 0.366±0.083，受限于 18 例女性鳞癌数据稀缺——作者建议用扩散模型生成合成 CT 增强稀有子群
- 消融为定性路径描述而非定量逐步表格，缺少去掉单个组件后的精确数值下降
- 仅考虑性别一种敏感属性，年龄、种族等其他公平性维度未涉及
- 每个体积仅采样 32 张切片，800+ 切片体积可能丢失关键病变
- 未使用 3D 卷积或 z 轴位置编码，忽略了切片间空间连续性
- 仅在单一挑战赛数据集（889 例）验证，外部泛化性未知

## 相关工作与启发

- **vs Ilse et al. (ICML 2018) Attention-MIL**：本文在其框架上增加 GRL 对抗分支和公平性协议，从弱监督聚合扩展到公平性感知诊断
- **vs Ganin & Lempitsky (2015) GRL**：原始用于域适应消除域特征，本文转用于消除性别特征实现人口统计学公平
- **vs 3D CT 分类（3D ResNet 等）**：本文用 2D backbone + MIL 聚合，更适合切片数高度可变场景，但牺牲 z 轴空间建模

## 评分

⭐⭐⭐

- **新颖性** ⭐⭐⭐：GRL 和 attention MIL 都是已有组件组合应用，缺乏架构层面原创
- **实验充分度** ⭐⭐⭐：消融为定性描述，仅单一挑战赛数据集
- **写作质量** ⭐⭐⭐⭐：方法描述清晰系统，公式完整，流程图直观
- **价值** ⭐⭐⭐：为医学 AI 公平性提供端到端方案模板，但受限于挑战赛报告深度

<!-- RELATED:START -->

## 相关论文

- [MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)
- [Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)

<!-- RELATED:END -->
