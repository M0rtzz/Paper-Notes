---
title: >-
  [论文解读] Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy
description: >-
  [AAAI 2026][医学图像][知识蒸馏] 提出 PaGKD，一个无需配对样本的组级知识蒸馏框架，通过组级原型蒸馏（GKD-Pro，用共享的病变查询Transformer提取模态不变语义原型）和组级密集蒸馏（GKD-Den，用激活图引导的语义关系交叉注意力实现密集空间对齐），突破传统NBI→WLI跨模态蒸馏对配对数据的依赖，在四个临床数据集上AUC分别提升3.3%/1.1%/2.8%/3.2%。
tags:
  - AAAI 2026
  - 医学图像
  - 知识蒸馏
  - 跨模态学习
  - 白光内窥镜
  - 窄带成像
  - 无配对
  - 组级蒸馏
  - 消化道病变分类
---

# Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy

**会议**: AAAI 2026  
**arXiv**: [2601.09209](https://arxiv.org/abs/2601.09209)  
**代码**: [Huster-Hq/PaGKD](https://github.com/Huster-Hq/PaGKD)  
**领域**: 医学图像 / 内窥镜  
**关键词**: 知识蒸馏, 跨模态学习, 白光内窥镜, 窄带成像, 无配对, 组级蒸馏, 消化道病变分类

## 一句话总结

提出 PaGKD，一个无需配对样本的组级知识蒸馏框架，通过组级原型蒸馏（GKD-Pro，用共享的病变查询Transformer提取模态不变语义原型）和组级密集蒸馏（GKD-Den，用激活图引导的语义关系交叉注意力实现密集空间对齐），突破传统NBI→WLI跨模态蒸馏对配对数据的依赖，在四个临床数据集上AUC分别提升3.3%/1.1%/2.8%/3.2%。

## 研究背景与动机

**领域现状**：内窥镜是消化道（GI）癌症早期检测的关键工具。白光成像（WLI）是临床标准模态，但窄带成像（NBI）通过光谱滤波增强血管和黏膜细节，提供更优的病变可视性和分类性能。然而NBI在常规部署中经常不可用或未被使用。

**现有痛点**：
   - **WLI分类性能较差**：WLI图像缺乏NBI的精细血管和黏膜信息，独立分类效果受限
   - **配对数据稀缺且昂贵**：现有跨模态蒸馏方法（ADD、CPC-Trans、PolypsAlign）都要求同一病变的配对NBI-WLI图像，但这类数据采集困难
   - **海量非配对数据未利用**：大量NBI和WLI图像独立采集自不同病变和患者，现有方法无法利用
   - **实例级蒸馏的语义失配**：当对非配对图像做实例级对齐时，个体病变图像只呈现疾病的部分特征，导致跨模态特征不兼容

**核心矛盾**：NBI知识对WLI分类有帮助，但现有蒸馏方法要求配对数据→大量非配对数据被浪费→模型无法充分学习→WLI分类性能受限。

**本文目标** 利用丰富的非配对NBI和WLI数据实现有效的跨模态知识蒸馏，提升WLI-only的病变分类性能。

**切入角度**：不再对齐单个图像，而是将同类病变图像组织成"组"进行组级蒸馏——组内多张图像提供更完整的疾病表征，缓解个体样本的偏差和噪声。

**核心 idea**：组级蒸馏 = 原型级全局语义对齐（GKD-Pro）+ 密集级局部空间对齐（GKD-Den），无需图像级配对。

## 方法详解

### 整体框架

PaGKD 由三部分组成：
1. **预训练冻结的NBI分类器**（教师）
2. **可训练的WLI分类器**（学生）
3. **两个组级知识蒸馏模块**：GKD-Pro 和 GKD-Den

每次训练迭代中，构建同类图像组 $\mathcal{G}_c^{mod}$，其中 $c$ 是类别，$mod \in \{WLI, NBI\}$。每张图像独立通过对应分类器提取特征图 $\mathcal{F}_c^{mod} \in \mathbb{R}^{N_c \times d \times h \times w}$，展平聚合为统一表示。

### 关键设计一：组级原型知识蒸馏（GKD-Pro）

**目标**：提取模态不变的病变语义原型，在全局高层进行跨模态对齐。

**病变查询Transformer（LR-QFormer）**：
- 设计一组共享的可学习病变查询 $\mathcal{Q} \in \mathbb{R}^{N_q \times d}$（$N_q=12$，远小于特征序列长度 $L_c$）
- 查询在所有组和模态间共享，是类别和模态无关的"病变概念锚点"
- T层Transformer块中，查询先做自注意力（SA），再与组特征做交叉注意力（CA）：
$$\mathcal{Q}_{t,c}^{mod} = \text{CA}(\text{SA}(\mathcal{Q}_{t-1,c}^{mod}),\ \mathcal{F}_c^{mod} + \mathbf{E}_{pos})$$
- 经过T层迭代，每个查询累积了特定疾病属性的模态特定证据

**组级对比损失**：
- 相似度定义：同索引查询的余弦相似度平均
$$S_{\mathcal{Q}_c^{WLI}, \mathcal{Q}_{c'}^{mod'}} = \frac{1}{N_q}\sum_{i=1}^{N_q} \frac{(\mathbf{q}_{i,c}^{WLI})^\top \mathbf{q}_{i,c'}^{mod'}}{\|\mathbf{q}_{i,c}^{WLI}\| \|\mathbf{q}_{i,c'}^{mod'}\|}$$
- 对称对比目标：拉近同类WLI-NBI原型，推远不同类原型

### 关键设计二：组级密集知识蒸馏（GKD-Den）

**目标**：在局部空间细粒度上对齐跨模态特征，补充GKD-Pro遗漏的空间细节。

**语义关系生成**：
1. 对每个组的特征计算CAM（类激活图），经像素自适应精细化增强空间一致性
2. 用双阈值方案（$\tau_1=0.3, \tau_2=0.7$）将CAM离散化为：背景(0)、病变(1)、歧义(∅)
3. 构建语义关系矩阵 $\mathbf{R}_c \in \{0, -\infty\}^{L_c \times L_c}$：
    - 当两位置在各自CAM中有相同且非歧义的标签时为0（允许注意力通过）
    - 否则为 $-\infty$（阻断注意力）

**语义关系引导的交叉注意力（SRCA）**：
$$\mathbf{A} = \text{Softmax}\left(\mathbf{R}_c + \frac{(\mathcal{F}_c^{WLI} W_q)(\mathcal{F}_c^{NBI} W_k)^\top}{\sqrt{d/4}}\right)$$

- 关系矩阵 $\mathbf{R}_c$ 作为注意力掩码引导空间重构
- 双向重构：NBI→WLI 和 WLI→NBI
- 双向一致性损失：$\mathcal{L}_{den}$ 约束重构特征与原始特征的L2距离

### 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{pro} + \mathcal{L}_{den} + \mathcal{L}_{cls}$$

其中 $\mathcal{L}_{cls}$ 是WLI分类器的交叉熵损失。推理时仅保留WLI分类器。

## 实验

### 数据集

| 数据集 | 类别 | 配对 | 非配对WLI | 非配对NBI | 类型 |
|--------|------|------|-----------|-----------|------|
| PICCOLO | 3类 | 1,055 | 1,065 | 214 | 结直肠息肉 |
| PolypSet | 2类 | 165 | 450 | 450 | 结直肠息肉 |
| IH-Polyp | 2类 | 556 | 3,730 | 921 | 结直肠息肉 |
| IH-GC | 2类 | 264 | 469 | 303 | 胃癌 |

### 主实验结果（vs. 8个SOTA方法）

**PICCOLO数据集（3类）**：

| 方法 | 训练数据 | Acc | F1 | AUC |
|------|----------|-----|-----|-----|
| ADD (配对) | $\mathcal{D}_p$ | 79.1 | 76.4 | 83.8 |
| ADD (配对+非配对) | $\mathcal{D}_p + \mathcal{D}_{unp}$ | 77.9 | 74.2 | 84.2 |
| **PaGKD (非配对)** | $\mathcal{D}_{unp}$ | 80.8 | 78.8 | 86.6 |
| **PaGKD (全部)** | $\mathcal{D}_p + \mathcal{D}_{unp}$ | **81.9** | **81.1** | **90.1** |

关键观察：
1. **PaGKD仅用非配对数据**（80.8/78.8/86.6）已超越所有使用配对数据的SOTA方法
2. **加入非配对数据反而伤害现有CDC方法**：ADD从83.8→84.2（仅AUC微升），CPC-Trans从86.6→87.2，改善甚微
3. **PaGKD全数据**在四个数据集上AUC分别提升至少3.3%、1.1%、2.8%、3.2%

**四个数据集AUC汇总**：

| 方法 | PICCOLO | PolypSet | IH-Polyp | IH-GC |
|------|---------|----------|----------|-------|
| NBI分类器（上界） | 86.9 | 97.6 | 87.0 | 86.3 |
| 最强基线 | 87.2 | 93.7 | 82.8 | 81.4 |
| **PaGKD** | **90.1** | **94.7** | **85.1** | **84.0** |

### 消融实验

**核心组件消融**：

| GKD-Pro | GKD-Den | PICCOLO AUC | IH-GC AUC |
|---------|---------|-------------|-----------|
| ✗ | ✗ | 71.2 | 66.9 |
| ✓ | ✗ | 83.5 | 75.5 |
| ✗ | ✓ | 85.0 | 77.3 |
| ✓ | ✓ | **90.1** | **84.0** |

- 两个模块单独都有显著提升（+12.3/+13.8 和 +8.6/+10.4）
- 联合使用进一步提升（互补而非冗余）

**子组件消融**：
- 去掉LR-QFormer（用平均池化替代）：PICCOLO AUC从83.5降至85.7→78.3
- 去掉SRCA（用标准交叉注意力）：AUC从85.0降至83.2
- 去掉双向一致性（仅单向NBI→WLI）：AUC从90.1降至87.8

**组级 vs 图像级蒸馏**：

| 蒸馏级别 | GKD-Pro AUC | GKD-Den AUC | 联合AUC |
|----------|-------------|-------------|---------|
| 图像级 | 78.6 | 79.9 | 84.3 |
| **组级** | **83.5** | **85.0** | **90.1** |

组级蒸馏在所有设置下显著优于图像级（PICCOLO上联合AUC：90.1 vs 84.3），验证了组级聚合能缓解非配对数据的噪声。

### 关键发现

1. 组级蒸馏是利用非配对数据的关键——图像级蒸馏在非配对数据上引入噪声和语义失配
2. GKD-Pro（全局语义）和GKD-Den（局部空间）互补——二者联合的提升显著大于单独使用
3. 仅使用非配对数据的PaGKD已可与使用配对数据的最强基线匹配
4. 非配对数据不仅不伤害PaGKD，反而提供了更多的训练信号
5. LR-QFormer的共享查询设计确保了跨模态跨类别的一致病变概念提取

## 亮点与洞察

1. **范式突破**：从"必须配对"到"无需配对"的跨模态蒸馏——这解锁了现实中大量闲置的非配对临床数据
2. **组级聚合的智慧**：单张病变图像只展示疾病的片段，组级聚合提供更完整、更鲁棒的疾病表征——这一思路可推广至其他医学多模态任务
3. **共享查询作为语义锚点**：LR-QFormer中的共享可学习查询就像一套"疾病属性词典"，在不同模态和类别之间建立统一的语义参照系
4. **CAM引导的密集对齐**：利用CAM确定语义对应关系来引导交叉注意力，比全局对齐更精细、比像素对齐更鲁棒
5. **实验设计公平**：将CDC方法也训练在配对+非配对数据上进行对比，公正地展示了现有方法在非配对数据上的退化

## 局限性

1. **组构建策略**：如何在batch中组织同类图像组需要根据数据集类别分布调参
2. **计算开销**：组级操作的空间复杂度与组大小成正比（$L_c = N_c \cdot h \cdot w$），大组可能内存受限
3. **仅验证二/三分类**：消化道病变的细分类（如多种息肉亚型）未测试
4. **骨干限制**：仅用ResNet-50作骨干，未探索ViT等更强架构
5. **类别假设**：要求非配对数据有类别标签——无标签的非配对数据仍无法利用
6. **CAM质量依赖**：GKD-Den的语义关系生成依赖CAM质量，弱分类器可能产生噪声CAM

## 相关工作

- **跨模态独立分类（CIC）**：SSL-CPCD (自监督patch-image聚类), SSL-WCE (自适应聚合注意力), FFCNet (傅里叶变换降噪)
- **跨模态蒸馏分类（CDC）**：
    - PolypsAlign: 判别器+对比损失全局对齐
    - CPC-Trans: Transformer交叉注意力patch/全局对齐
    - ADD: 像素级图像到图像蒸馏（当前最强配对方法）
    - SAMD: 语义注意力蒸馏
- **知识蒸馏**：FitNets, RKD, CRD, PKT, Hinton原始KD

## 评分与推荐

⭐⭐⭐⭐⭐ (5/5)

- 创新性: ⭐⭐⭐⭐⭐ — 无配对跨模态蒸馏是重要范式创新
- 实验: ⭐⭐⭐⭐⭐ — 四个数据集、8个SOTA对比、充分消融、公平设置
- 写作: ⭐⭐⭐⭐⭐ — 逻辑严谨，问题动机和方法设计环环相扣
- 实用性: ⭐⭐⭐⭐⭐ — 解决真实临床痛点，提供代码，推理仅需WLI分类器

<!-- RELATED:START -->

## 相关论文

- [Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/medical_imaging/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](../../CVPR2026/medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)

<!-- RELATED:END -->
