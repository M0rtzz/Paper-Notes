---
title: >-
  [论文解读] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][医学图像][PPFE] 在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "PPFE"
  - "肺癌筛查"
  - "低剂量CT"
  - "纵向定量分析"
  - "影像生物标志物"
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**会议**: CVPR 2026  
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)  
**代码**: 无  
**领域**: 医学影像分析  
**关键词**: PPFE, 肺癌筛查, 低剂量CT, 纵向定量分析, 影像生物标志物  

## 一句话总结

在两个独立大规模肺癌筛查队列中，利用深度学习自动分割量化PPFE纵向变化，首次验证其在筛查人群中的独立预后价值。

## 研究背景与动机

**胸膜实质弹力纤维增生症（PPFE）**是一种以上叶为主的罕见纤维化肺异常，在已确诊间质性肺病（如IPF、HP）中与死亡率增加相关。但既往研究集中在有明确纤维化诊断的患者群体，这些人本身就有较差预后。

**对于肺癌筛查人群——通常无症状或症状轻微——PPFE纵向进展的临床意义尚不明确**。这是一个重要的盲区：NHS预计到2028年将有100万人纳入年度CT筛查。如果能利用筛查已有的低剂量CT同时识别PPFE高风险个体，将极大提升筛查的附加临床价值。

**核心矛盾在于**：手动/半自动评估PPFE在大规模队列中不可行，而此前无人在筛查人群中用全自动定量方法系统验证过PPFE进展的预后意义。本文的切入点是：用深度学习做全自动分割+年化变化量计算，在两个独立队列中双重验证。

## 方法详解

### 整体框架

回顾性纵向研究设计：基于nnU-Net架构的深度学习模型自动分割基线和随访CT上的上肺PPFE病灶，计算年化体积变化量（ΔPPFE），通过分布阈值二分为进展性/非进展性，再用Cox比例风险模型评估与全因死亡率的独立关联。

### 关键设计

1. **自动PPFE分割模型**:

    - 功能：在低剂量CT上自动分割上肺区PPFE病灶体积
    - 核心思路：基于nnU-Net架构，在SUMMIT 100例手标注上5折交叉验证训练，Dice=0.91；只分割隆突以上区域并排除最顶端5mm以避免良性胸膜增厚混淆
    - 设计动机：手动分割在万人队列中不可行，限制分割范围减少假阳性

2. **ΔPPFE进展阈值定义**:

    - 功能：将连续的年化PPFE变化量二值化为进展性/非进展性
    - 核心思路：采用NLST基线PPFE体积标准差的一半（0.41 cm³/年）作为阈值，符合中等效应量的标准生物标志物方法论
    - 设计动机：在NLST中导出并直接应用于SUMMIT验证跨队列泛化性，NLST中5.4%为进展性，SUMMIT中1.5%

3. **多变量Cox回归分析**:

    - 功能：评估ΔPPFE与全因死亡率的独立关联
    - 核心思路：调整年龄、性别、吸烟史、身高、基线PPFE及交互项；SUMMIT额外调整FVC%和ILA
    - 设计动机：通过充分调整混杂因素隔离PPFE进展的独立预后贡献

### 损失函数 / 训练策略

分割模型使用Dice+交叉熵复合损失，SGD优化1000 epoch。统计分析框架包括Cox PH模型（生存分析）、负二项GLM（入院计数）、有序logistic回归（mMRC评分）。

## 实验关键数据

### 主实验

| 队列 | 调整变量 | HR | 95% CI | p值 |
|------|---------|-----|--------|-----|
| NLST | 多变量 | 1.25 | 1.01-1.56 | 0.042 |
| SUMMIT | 多变量 | 3.14 | 1.66-5.97 | <0.001 |
| SUMMIT | +FVC+ILA | 2.55 | 1.34-4.85 | 0.004 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 基线PPFE单独 | 不一致 | NLST中弱且不显著，SUMMIT中显著但弱于ΔPPFE |
| 调整ILA和FVC后 | HR仍显著 | ΔPPFE代表独立于经典UIP纤维化的生物过程 |
| 仅基线CT亚组 | IRR=1.24 | 基线PPFE与入院相关，但与mMRC和用药无关 |

### 关键发现

- 两队列Kaplan-Meier曲线均显示进展性PPFE组生存率显著降低（log-rank p<0.001）
- ΔPPFE与心血管事件（MACE5）无关——反映呼吸系统特异性风险而非系统性风险
- 进展性PPFE强烈关联更高呼吸入院率（IRR=2.79）、更多激素/抗生素使用（IRR=1.55）

## 亮点与洞察

- 在两个独立大规模筛查队列中双重验证且结果一致，自动定量使罕见病大队列流行病学研究可行。"利用已有筛查CT发现附加风险"的范式对公共卫生有直接价值。

## 局限与展望

- 回顾性观察研究，无法推断因果关系
- 低剂量CT分辨率低于诊断性HRCT，可能低估细微PPFE
- PPFE与ILA可能存在解剖重叠
- 进展阈值需在非筛查人群中外部验证

## 相关工作与启发

- **vs Gudmundsson et al. (2023)**: 在IPF/HP患者中验证PPFE进展与死亡关联，本文首次扩展到无症状筛查人群
- **vs Jacob et al. (2018)**: 手动/半自动PPFE评估，本文全自动深度学习在万人规模队列实现

## 评分

- 新颖性: ⭐⭐⭐ 方法学无创新，贡献在于临床发现
- 实验充分度: ⭐⭐⭐⭐⭐ 两个独立大队列、多终点、完整亚组分析
- 写作质量: ⭐⭐⭐⭐ 临床论文风格，结构清晰
- 价值: ⭐⭐⭐⭐ 对肺癌筛查有直接临床指导意义
---
title: >-
  [论文解读] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][医学图像][PPFE] 在NLST和SUMMIT两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量CT上PPFE的纵向变化（dPPFE），验证其与全因死亡率和呼吸系统发病率的独立关联。
tags:
  - CVPR 2026
  - 医学图像
  - PPFE
  - 肺癌筛查
  - 低剂量CT
  - 纵向定量分析
  - 影像生物标志物
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**会议**: CVPR 2026  
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)  
**代码**: 无  
**领域**: 医学影像分析  
**关键词**: PPFE、肺癌筛查、低剂量CT、纵向定量分析、影像生物标志物

## 一句话总结
在NLST（n=7980）和SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量CT上PPFE纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。

## 研究背景与动机

胸膜实质弹力纤维增生症（PPFE）是一种以上叶为主的罕见纤维化肺异常，在已确诊的间质性肺病（如IPF、HP）中已被证实与死亡率增加相关。既往研究主要集中在已确诊纤维化疾病中的PPFE，但在肺癌筛查人群中——通常无症状或症状轻微——PPFE纵向进展的临床意义尚不明确。NHS预计到2028年将有100万人纳入年度CT筛查，理解PPFE在筛查人群中的影响愈发重要。

核心挑战在于：在无症状的肺癌筛查人群中，低剂量CT上自动量化的PPFE纵向变化能否独立预测死亡率和呼吸系统不良结局？

## 方法详解

### 整体框架
回顾性纵向研究设计：基于nnU-Net的深度学习模型分割基线和随访CT上的PPFE病灶，计算年化变化量（$\Delta$PPFE），通过Cox比例风险模型评估与死亡率的独立关联。两个独立队列（NLST和SUMMIT）相互验证。

### 关键设计

1. **自动PPFE分割模型**: 基于nnU-Net架构，在SUMMIT的100例手工标注数据上进行5折交叉验证训练，Dice系数达到0.91。分割仅限于隆突以上区域，并排除最顶端5mm以避免良性胸膜增厚的混淆。模型使用Dice+交叉熵复合损失、SGD优化器训练1000个epoch。

2. **dPPFE进展阈值定义**: 采用NLST基线PPFE体积标准差的一半（0.41 $\text{cm}^3$/年）作为进展阈值，符合已建立的生物标志物方法学。该阈值在NLST中识别出5.4%的进展性个体，在SUMMIT中为1.5%。阈值在NLST中推导后直接应用于SUMMIT队列。

3. **多变量Cox回归分析**: 调整年龄、性别、吸烟史（包年）、身高、基线PPFE体积及交互项。SUMMIT额外调整FVC%预测值和ILA视觉评分。同时使用负二项GLM（住院次数）和有序logistic回归（mMRC评分）分析次要终点。

### 损失函数 / 训练策略
- 分割模型：Dice + 交叉熵复合损失，SGD优化器，训练1000 epoch
- 统计分析：Cox PH模型（生存分析）、负二项GLM（住院率）、有序logistic回归（呼吸困难评分）

## 实验关键数据

### 主实验

**Cox回归主要结果：**

| 队列 | 分析类型 | dPPFE的HR | 95% CI | p值 |
|------|----------|-----------|--------|-----|
| NLST | 多变量 | 1.25 | 1.01–1.56 | 0.042 |
| SUMMIT | 多变量 | 3.14 | 1.66–5.97 | <0.001 |
| SUMMIT | +FVC+ILA调整 | 2.55 | 1.34–4.85 | 0.004 |

**SUMMIT队列临床结局：**

| 结局指标 | IRR/OR | p值 |
|----------|--------|-----|
| 呼吸系统住院 | IRR=2.79 | <0.001 |
| 激素/抗生素使用 | IRR=1.55 | 0.010 |
| mMRC呼吸困难评分 | OR=1.40 | 0.055 |

两队列Kaplan-Meier曲线均显示进展性PPFE组生存率显著更低（log-rank p<0.001）。

### 消融实验

| 分析条件 | 发现 |
|----------|------|
| 基线PPFE单独预后 | 弱且不一致，dPPFE提供增量价值 |
| 调整ILA和FVC后 | dPPFE仍独立，代表与经典UIP型纤维化不同的生物过程 |
| dPPFE vs 心血管事件 | 与MACE5无关，说明反映呼吸系统而非系统性风险 |
| 仅基线CT亚组（n=2580） | 基线PPFE与住院相关（IRR=1.24），但与mMRC和用药无关 |

### 关键发现
- dPPFE在两个独立队列中一致性地独立预测全因死亡率
- 调整ILA和肺功能后效应仍然显著，表明PPFE代表独立的病理过程
- dPPFE与心血管事件无关，确认其特异性指向呼吸系统

## 亮点与洞察
- 两个独立大规模真实世界筛查队列（NLST+SUMMIT）双重验证，结果一致且稳健
- 自动定量+纵向变化使罕见病的大队列流行病学研究成为可能
- 直接关联临床可操作结果（住院率、用药、呼吸困难评分），方法学成熟
- 利用已有LDCT筛查数据即可识别高呼吸风险个体，无需额外检查

## 局限与展望
- 回顾性观察研究，无法推断因果关系
- 低剂量CT分辨率低于诊断性HRCT，可能低估细微PPFE
- PPFE与ILA可能存在解剖重叠，虽然统计调整后仍独立
- 进展阈值需在非筛查人群中进行外部验证
- 排除无随访人群可能引入生存偏倚

## 相关工作与启发
- **vs Gudmundsson et al. (2023)**: 在IPF/HP患者中验证PPFE进展与死亡关联，本文首次扩展到无症状筛查人群
- **vs Jacob et al. (2018)**: 手动/半自动评估PPFE预后，本文使用全自动深度学习方法实现大队列分析
- 自动定量影像生物标志物+纵向分析的范式可推广到其他筛查附带发现（如冠状动脉钙化、骨质疏松）

## 评分
- 新颖性: ⭐⭐⭐ 方法学无新意（nnU-Net + Cox），贡献在于临床发现
- 实验充分度: ⭐⭐⭐⭐⭐ 两个独立大队列、多终点、完整亚组分析
- 写作质量: ⭐⭐⭐⭐ 临床论文风格，结构清晰
- 价值: ⭐⭐⭐⭐ 对肺癌筛查临床实践有直接指导意义
---
title: >-
  [论文解读] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][医学图像][PPFE] 在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。
tags:
  - CVPR 2026
  - 医学图像
  - PPFE
  - 肺癌筛查
  - 低剂量CT
  - 纵向定量分析
  - 影像生物标志物
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**会议**: CVPR 2026  
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)  
**代码**: 无  
**领域**: 医学影像分析 / 纵向研究  
**关键词**: PPFE、肺癌筛查、低剂量CT、纵向定量分析、影像生物标志物  

## 一句话总结
在 NLST（n=7980）和 SUMMIT（n=8561）两个大规模肺癌筛查队列中，利用深度学习自动分割量化低剂量 CT 上 PPFE 的纵向变化（dPPFE），验证其与全因死亡率（HR=1.25/3.14）和呼吸系统发病率的独立关联。

## 背景与动机
胸膜实质弹力纤维增生症（PPFE）是一种以上叶为主的罕见纤维化肺异常，在确诊的间质性肺病中已被证实与死亡率增加相关。既往研究主要集中在 IPF/HP 等已确诊纤维化疾病中的 PPFE，但在肺癌筛查人群——通常无症状或症状轻微——中 PPFE 纵向进展的临床意义尚不明确。NHS 预计到 2028 年将有 100 万人纳入年度 CT 筛查，理解 PPFE 在筛查人群中的影响愈发重要。

## 核心问题
在无症状的肺癌筛查人群中，低剂量 CT 上自动量化的 PPFE 纵向变化能否独立预测死亡率和呼吸系统不良结局？

## 方法详解

### 整体框架
回顾性纵向研究：基于 nnU-Net 的深度学习模型分割基线和随访 CT 上的 PPFE 病灶，计算年化变化量（dPPFE），通过 Cox 比例风险模型评估与死亡率的独立关联。

### 关键设计
1. **自动 PPFE 分割模型**: 基于 U-Net 架构，在 SUMMIT 100 例手标注上 5 折交叉验证训练，Dice=0.91。只分割隆突以上区域，排除最顶端 5mm 以避免良性胸膜增厚混淆
2. **dPPFE 进展阈值**: 采用 NLST 基线 PPFE 体积标准差的一半（0.41 cm3/年）。NLST 中 5.4% 为进展性，SUMMIT 中 1.5%
3. **多变量 Cox 回归**: 调整年龄、性别、吸烟史、身高、基线 PPFE、交互项；SUMMIT 额外调整 FVC% 和 ILA 视觉评分

### 损失函数 / 训练策略
- 分割模型: Dice + 交叉熵复合损失，SGD 优化 1000 epoch
- 统计分析: Cox PH 模型（生存）、负二项 GLM（入院）、有序 logistic 回归（mMRC）

## 实验关键数据

| 队列 | dPPFE 的 HR | 95% CI | p 值 |
|------|------------|--------|------|
| NLST（多变量） | 1.25 | 1.01-1.56 | 0.042 |
| SUMMIT（多变量） | 3.14 | 1.66-5.97 | <0.001 |
| SUMMIT（+FVC+ILA） | 2.55 | 1.34-4.85 | 0.004 |

| 结局（SUMMIT） | IRR/OR | p 值 |
|----------------|--------|------|
| 呼吸系统入院 | IRR=2.79 | <0.001 |
| 激素/抗生素使用 | IRR=1.55 | 0.011 |
| mMRC 评分 | OR=1.40 | 0.055 |

- 两队列 Kaplan-Meier 曲线均显示进展性 PPFE 组显著生存率降低（log-rank p<0.001）
- dPPFE 与心血管事件（MACE5）无关，说明反映呼吸系统而非系统性风险
- 仅有基线 CT 的亚组: 基线 PPFE 与入院相关（IRR=1.24），但与 mMRC 和用药无关

### 消融实验要点
- 基线 PPFE 单独预后价值弱且不一致，dPPFE 提供增量价值
- 调整 ILA 和 FVC 后 dPPFE 仍独立——代表的生物过程与经典 UIP 型纤维化不同
- 排除无随访人群可能引入生存偏倚，但二次分析显示基线 PPFE 也携带部分风险信息

## 亮点
- 在两个独立大规模真实世界筛查队列中双重验证，结果一致且稳健
- 自动定量+纵向变化使罕见病的大队列流行病学研究成为可能
- 直接关联临床可操作结果（入院率、用药、呼吸困难评分）

## 局限与展望
- 回顾性观察研究，无法推断因果关系
- 低剂量 CT 分辨率低于诊断性 HRCT，可能低估细微 PPFE
- PPFE 与 ILA 可能存在解剖重叠
- 进展阈值需在非筛查人群中外部验证

## 与相关工作的对比
- vs Gudmundsson et al. (2023): 在 IPF/HP 患者中验证 PPFE 进展与死亡关联，本文首次扩展到无症状筛查人群
- vs Jacob et al. (2018): 手动/半自动评估 PPFE 预后，本文使用全自动深度学习方法实现大队列分析
- 技术上分割模型和统计框架都使用成熟已验证的工具

## 启发与关联
- 对肺癌筛查计划的附加价值：利用已有 LDCT，无额外检查即可筛出高呼吸风险个体
- 自动定量影像生物标志物+纵向分析的范式可推广到其他筛查附带发现

## 评分
- 新颖性: ⭐⭐⭐ 方法学无新意（nnU-Net + Cox），贡献在于临床发现
- 实验充分度: ⭐⭐⭐⭐⭐ 两个独立大队列、多终点、完整亚组分析
- 写作质量: ⭐⭐⭐⭐ 临床论文风格，结构清晰
- 价值: ⭐⭐⭐⭐ 对肺癌筛查临床实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[ICML 2026\] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](../../ICML2026/medical_imaging/auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)

</div>

<!-- RELATED:END -->
