---
title: >-
  [论文解读] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
description: >-
  [CVPR 2026][医学图像][scleral imaging] 提出ScleraGluNet多视角深度学习框架，通过五方向巩膜血管成像结合多分支CNN+MRFO特征精炼+Transformer跨视角融合，实现93.8%代谢状态三分类精度和MAE=6.42 mg/dL的空腹血糖连续估计。
tags:
  - CVPR 2026
  - 医学图像
  - scleral imaging
  - noninvasive glucose estimation
  - multiview learning
  - Transformer
  - MRFO
---

# Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging

**会议**: CVPR 2026  
**arXiv**: [2603.12715](https://arxiv.org/abs/2603.12715)  
**代码**: 无  
**领域**: 医学图像分析 / 无创检测  
**关键词**: scleral imaging, noninvasive glucose estimation, multiview learning, transformer fusion, MRFO

## 一句话总结
提出ScleraGluNet多视角深度学习框架，通过五方向巩膜血管成像结合多分支CNN+MRFO特征精炼+Transformer跨视角融合，实现93.8%代谢状态三分类精度和MAE=6.42 mg/dL的空腹血糖连续估计。

## 研究背景与动机

**糖尿病监测的痛点**：全球糖尿病患者预计到2045年达7.83亿。现有金标准检测（FPG/OGTT/HbA1c）均依赖血液采样，频繁检测不便且有感染风险。CGM虽减少了指尖采血频率，但仍需皮下传感器植入且成本较高。迫切需要非侵入式监测方案。

**巩膜血管的独特优势**：与需要专业设备的视网膜成像不同，巩膜/结膜微血管可通过低成本前节段相机直接可视化。慢性高血糖导致巩膜微血管重塑（血管口径变化、迂曲度增加、分支模式改变、灌注密度下降），这些改变在OCTA等研究中已被证实。

**现有方法的不足**：(1) 单视角采集仅覆盖有限巩膜区域，丢失区域特异性血管异常信息；(2) 不同象限的微血管重塑具有非均匀性，需要多方向成像全面捕获；(3) 缺乏同时处理分类和回归的多任务架构。

## 方法详解

### 整体框架
标准化多方向巩膜图像采集（5方向×每人5张）→ ROI提取+CLAHE+Frangi血管增强预处理 → 5个独立参数CNN分支特征提取 → MRFO特征精炼去冗余 → Transformer跨视角自注意力融合 → 分类头(3类softmax) + 回归头(FPG估计)。

### 关键设计

1. **多方向标准化采集协议**：
    - 功能：从正前方(straight)、上方(superior)、下方(inferior)、鼻侧(nasal)、颞侧(temporal)5个注视方向采集巩膜照片
    - 核心思路：不同象限展现不同的糖尿病相关微血管改变——颞侧和鼻侧可能有非对称重塑、上下方有不同灌注特征。多方向采集确保保留区域特异性细节
    - 设计动机：先前研究已证实糖尿病引起的结膜/巩膜内微血管病变具有空间非均匀性。单视角必然丢失关键诊断信息

2. **MRFO特征精炼 + Transformer跨视角融合**：
    - 功能：5个CNN分支输出的特征拼接后，先用蝠鲼觅食优化(MRFO)算法选择最优特征子集消除冗余，再用Transformer自注意力建模跨象限远程血管关联
    - 核心思路：MRFO是生物启发式特征选择算法，通过群智能搜索在高维特征空间中识别相关且非冗余的特征子集。Transformer的自注意力捕获多象限间的细微血管模式关联（如跨区域非对称重塑、跨象限一致性退化）
    - 设计动机：5分支拼接产生高维冗余特征，直接输入Transformer效率和效果均不佳。MRFO先筛选后Transformer融合的级联设计可理解为"先去噪再关联"

3. **多任务学习的双头输出**：
    - 功能：同时进行三分类（正常/受控糖尿病/高血糖糖尿病）和连续FPG值回归
    - 核心思路：分类头使用softmax输出3类概率，回归头直接输出FPG估计值(mg/dL)。两个任务共享底层特征表示，通过互补学习信号提升整体性能
    - 设计动机：分类和回归从不同角度利用血管特征——分类关注类间边界特征，回归关注连续变化趋势。联合训练通过多任务正则化效应防止过拟合

### 损失函数 / 训练策略
- 复合损失：Cross-Entropy（分类）+ MSE（血糖回归），按经验调优权重
- Subject-wise 5-fold交叉验证：同一受试者所有5张图像严格归入同一fold，防止数据泄漏
- Adam优化器，超参在每fold验证集上经验调优
- Bootstrap重采样1000次计算95%置信区间
- 排除标准：眼表疾病、活动性感染、近期手术等，确保巩膜成像质量

## 实验关键数据

### 主实验

| 指标 | 分类结果 | 回归结果 |
|------|---------|---------|
| 总体准确率 | 93.8% (CI: 91.8-95.4%) | - |
| 正常组准确率 | 94.0% (141/150) | - |
| 受控组准确率 | 92.1% (129/140) | - |
| 高血糖组准确率 | 93.5% (145/155) | - |
| AUC (正常/受控/高血糖) | 0.971 / 0.956 / 0.982 | - |
| MAE | - | 6.42 mg/dL |
| RMSE | - | 7.91 mg/dL |
| 相关系数 r | - | 0.983 |
| R² | - | 0.966 |
| Bland-Altman偏差 | - | +1.45 mg/dL (LOA: -8.33~+11.23) |

数据集规模：445人×5张=2225张巩膜图像（正常150/受控140/高血糖155）。

### 消融实验

| 配置 | 分类准确率 | MAE (mg/dL) |
|------|-----------|-------------|
| 单视角CNN | 较低 | 较高 |
| 多视角CNN (无MRFO/Transformer) | 提升 | 下降 |
| + MRFO特征精炼 | 进一步提升 | 进一步下降 |
| + Transformer融合 (完整) | **93.8%** | **6.42** |

### 关键发现
- 多方向采集对分类和回归均有显著提升，验证了巩膜血管异常的区域异质性假设
- 误分类主要发生在相邻代谢状态之间（正常↔受控），符合血糖水平的连续谱特征
- 5-fold各fold准确率范围92.8%-94.6%，稳定性良好
- Bland-Altman分析显示与实验室测量有较好一致性，95%点在±11 mg/dL内

## 亮点与洞察
- **新颖的无创检测途径**：利用巩膜微血管（而非视网膜）进行代谢状态评估，成像设备要求低，有远程医疗潜力
- 多方向采集协议的设计有生理学依据——不同象限的巩膜血管对高血糖的响应确实不同
- MRFO+Transformer的级联"去冗余→建关联"设计在多视角融合场景中很有参考价值

## 局限与展望
- 单中心数据集（长沙爱尔眼科），445人样本量有限，泛化性存疑
- 排除了眼表疾病患者，但实际临床中糖尿病患者常伴随眼表异常
- 未与现有无创方法（PPG、热成像）做直接对比
- 采集标准化依赖手动ROI提取，自动化程度不够
- 分类为三类的粒度较粗，更细粒度的血糖分段可能更有临床价值

## 相关工作与启发
- **视网膜成像的糖尿病AI**：大量工作通过眼底图像预测心血管代谢状态和HbA1c，本文将类似思路扩展到更易获取的巩膜血管
- **MRFO算法**：蝠鲼觅食优化在生物医学特征选择中的应用日增，本文验证了其在多视角特征去冗余中的有效性
- 启发：眼前节成像结合AI可能开辟低成本、无创、即时的代谢监测新赛道

## 评分
- 新颖性: ⭐⭐⭐ 巩膜多方向血管成像估血糖的思路新颖，但架构创新有限
- 实验充分度: ⭐⭐⭐ 有详细的分类/回归/Bland-Altman分析，但单中心小样本且缺少与其他无创方法对比
- 写作质量: ⭐⭐⭐ 临床背景阐述充分，但方法描述部分行文冗余
- 价值: ⭐⭐⭐ 概念验证阶段，需多中心验证和自动化改进才有临床转化可能

<!-- RELATED:START -->

## 相关论文

- [Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](../../CVPR2025/medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)
- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)

<!-- RELATED:END -->
