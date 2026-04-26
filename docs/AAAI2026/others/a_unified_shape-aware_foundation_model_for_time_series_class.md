---
title: >-
  [论文解读] UniShape: A Unified Shape-Aware Foundation Model for Time Series Classification
description: >-
  [AAAI 2026][时间序列分类] 提出UniShape——首个面向时间序列分类(TSC)的形状感知基础模型：通过多尺度子序列(shape)自适应聚合的shape-aware adapter捕获类别判别性时序模式，结合原型预训练模块在实例级和shape级联合学习可迁移的shapelet表示；在189万样本上预训练后，128个UCR数据集上达到0.8708平均准确率，超越所有基线。
tags:
  - AAAI 2026
  - 时间序列分类
  - 基础模型
  - Shapelet
  - 多尺度
  - 原型学习
---

# UniShape: A Unified Shape-Aware Foundation Model for Time Series Classification

**会议**: AAAI 2026  
**arXiv**: [2601.06429](https://arxiv.org/abs/2601.06429)  
**代码**: [https://github.com/qianlima-lab/UniShape](https://github.com/qianlima-lab/UniShape)  
**领域**: 时间序列分类 / 基础模型  
**关键词**: 时间序列分类, 基础模型, Shapelet, 多尺度, 原型学习

## 一句话总结

提出UniShape——首个面向时间序列分类(TSC)的形状感知基础模型：通过多尺度子序列(shape)自适应聚合的shape-aware adapter捕获类别判别性时序模式，结合原型预训练模块在实例级和shape级联合学习可迁移的shapelet表示；在189万样本上预训练后，128个UCR数据集上达到0.8708平均准确率，超越所有基线。

## 研究背景与动机

现有时间序列基础模型主要面向预测(forecasting)任务，而分类任务有本质差异：预测关注趋势和季节性的连续延续，分类则需要识别固定长度样本中的判别性局部模式(shapelets)。预测型基础模型直接用于分类效果不佳。同时，现有TSC方法多在小规模单域数据集上训练，跨域泛化能力有限。此外，shapelet作为最具可解释性的分类特征，其多尺度特性（判别性子序列可能以不同长度和位置出现）在基础模型中尚未被有效建模。

## 方法详解

### 整体框架

UniShape采用预训练+微调范式：(1) Shape-Aware Adapter将变长子序列编码为shape tokens并通过注意力池化聚合为class tokens；(2) 原型预训练模块在实例级和shape级联合做对比学习；(3) 微调时class token经分类头输出预测。

### 关键设计

1. **Shape-Aware Adapter**：对输入时序用Q个不同尺度的滑动窗口(W_q∈{64,32,16,8,4})提取多尺度子序列，每个子序列经归一化+1D CNN编码为shape token，再通过注意力池化(attention pooling)自适应加权聚合为class token。采用从粗到细的层级融合策略，上一尺度的class token前置到下一尺度的token序列中，实现跨尺度信息传递
2. **原型预训练模块**：维护每个类别的可学习原型向量，通过指数滑动平均动态更新。实例级对比(class token ↔ 类原型)捕捉全局判别特征；shape级对比(高置信shape tokens ↔ 类原型)建模局部判别模式。无标签样本用最近原型分配伪标签
3. **多尺度可解释性**：注意力池化的权重α直接反映每个shape的判别重要性，提供shapelet级的可解释性——在ECGFiveDays数据集上正确高亮了延迟T波区间，在GunPoint上定位了运动过冲区间

### 损失函数 / 训练策略

- **预训练损失** = 原型对比损失(实例级+shape级) + MoCo v3自监督对比损失
- shape级损失权重λ=0.01，温度τ控制对比学习的尖锐度
- 预训练仅需10%标注数据即可达到接近全标注的效果
- 预训练30 epochs，batch size 2048；微调300 epochs，cross-entropy + shape对比辅助损失(μ=0.01)

## 实验关键数据

### 主实验（128 UCR数据集，全监督）

| 方法 | 类型 | 参数量 | 平均准确率 | 平均排名 |
|------|------|--------|-----------|---------|
| UniShape | FM | 3.1M | **0.8708** | **2.71** |
| Mantis | FM | 8.7M | 0.8441 | 5.21 |
| NuTime | FM | 2.4M | 0.8353 | 6.68 |
| MR-H | NDL | - | 0.8621 | 3.97 |
| SoftShape | DS | 472K | 0.8388 | 5.89 |
| MOMENT | FM | 341M | 0.7020 | 12.10 |

### 零样本特征提取（30个额外数据集）

| 方法 | 平均准确率 | 平均排名 |
|------|-----------|---------|
| UniShape | **0.7262** | **3.07** |
| Mantis | 0.7052 | 3.67 |
| NuTime | 0.6917 | 3.53 |
| RandomForest | 0.6930 | 3.77 |

### 消融实验

- 预训练数据规模越大效果越好（UCR 60K → ALL 1.89M准确率持续提升）
- 10%标注 vs 100%标注预训练差异统计不显著(P=0.20)，少量标签足够
- shape-aware adapter + 原型预训练均有独立贡献，移除任一组件均导致显著下降

### 关键发现

- 预测型基础模型(GPT4TS, MOMENT, UniTS)在TSC上大幅低于非深度学习方法，说明任务特异性设计至关重要
- UniShape仅3.1M参数即超越341M参数的MOMENT，参数效率极高
- 可解释性分析表明注意力权重与领域专家认定的shapelet区间高度一致

## 亮点与洞察

- 首次明确指出"预测型基础模型不适合分类"，并给出针对性解决方案
- shape-aware adapter的多尺度设计优雅且计算友好，共享参数处理所有尺度
- 原型学习以很少的标签就能捕捉类别结构，对半监督/少样本场景特别有价值
- 注意力权重作为shapelet的可解释性机制，在医疗时序等领域有实际应用价值

## 局限与展望

- 仅覆盖单变量时间序列分类，多变量场景需要额外设计
- 固定的5个尺度(4-64)可能不适合所有领域的最佳shapelet长度
- 预训练数据集通过插值统一到512长度，可能丢失超长序列的信息
- 零样本准确率仍有提升空间（0.73 vs 监督的0.87）

## 相关工作与启发

- Shapelet学习从传统穷举搜索→梯度优化→基础模型预训练的演进路径值得关注
- 原型对比学习可推广到其他需要类别感知预训练的领域（如少样本图像分类）
- 多尺度注意力池化的设计思想可迁移到时序预测基础模型
- MoCo v3的动量对比学习框架在时序数据上同样有效
- Rocket/MiniRocket等非深度学习方法仍然是极强的基线(0.85+)，基础模型需要明显超越才有说服力

## 预训练数据构建

| 数据源 | 样本数 | 特点 |
|--------|--------|------|
| UCR Archive | ~60K | 128个单变量分类数据集 |
| UEA Archive | ~1.39M | 多变量→channel-independent拆分 |
| 额外数据 | ~0.44M | 8个常用时序数据集 |
| **合计** | **1.89M** | 统一插值到长度512 |

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 首个面向TSC的shape-aware基础模型 |
| 技术深度 | 4 | 多尺度adapter+原型学习设计精巧 |
| 实验充分性 | 5 | 158个数据集，16个基线，全面消融 |
| 写作质量 | 4 | 动机清晰，方法阐述流畅 |
| 实用价值 | 4 | 参数高效+可解释+跨域迁移 |

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](../../ICML2025/others/time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)
- [\[AAAI 2026\] A Mind Cannot Be Smeared Across Time](a_mind_cannot_be_smeared_across_time.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)

<!-- RELATED:END -->
