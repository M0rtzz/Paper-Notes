---
title: >-
  [论文解读] MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection
description: >-
  [CVPR 2026][neuroevolution] 提出 MyoVision 智能手机透射成像框架和 NEATBoost-Attention 神经进化优化集成模型，用于低成本实时鸡胸肉肌病（木质胸、意面肉）三分类检测。 木质胸 (WB) 和意面肉 (SM) 是严重影响家禽肉质的结构性肌病，当前检测依赖主观手工评估或昂贵的…
tags:
  - "CVPR 2026"
  - "neuroevolution"
  - "ensemble learning"
  - "mobile imaging"
  - "food quality inspection"
  - "transillumination"
---

# MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection

**会议**: CVPR 2026  
**arXiv**: [2604.13456](https://arxiv.org/abs/2604.13456)  
**代码**: 无  
**领域**: 计算机视觉应用  
**关键词**: neuroevolution, ensemble learning, mobile imaging, food quality inspection, transillumination

## 一句话总结

提出 MyoVision 智能手机透射成像框架和 NEATBoost-Attention 神经进化优化集成模型，用于低成本实时鸡胸肉肌病（木质胸、意面肉）三分类检测。

## 研究背景与动机

木质胸 (WB) 和意面肉 (SM) 是严重影响家禽肉质的结构性肌病，当前检测依赖主观手工评估或昂贵的实验室级成像系统。这些缺陷主要表现为内部结构异常而非表面可见特征，给自动化检测带来挑战。现有自动化方法依赖高光谱成像、近红外光谱等专业硬件，成本高且部署受限。研究动机是用消费级智能手机实现非破坏性、低成本的多类肌病分类。

## 方法详解

### 整体框架

MyoVision 想用一部智能手机就完成原本要靠高光谱/近红外专业设备的鸡胸肉肌病检测。它走的是一条"成像→手工特征→集成分类"的三段流水线：先用手机白光透射拍 14-bit RAW 图，再从图里抽 16 维结构纹理描述符（梯度统计、频域纹理响应、密实组织特征）得到一份紧凑的表格特征，最后交给 NEATBoost-Attention 集成模型做木质胸/意面肉/正常三分类。集成这一段又拆成两层：NEAT 进化搜索自动找出两个互补学习器（LightGBM 与 AttentionMLP）的超参，再把它们的预测概率加权融合成最终结果。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["智能手机透射成像<br/>白光穿透 + 14-bit RAW"] --> B["16 维结构描述符提取<br/>梯度统计 / 频域纹理 / 密实组织"]
    B --> C["NEAT 神经进化优化<br/>进化拓扑+权重生成候选超参"]
    C -->|10 个超参| D["LightGBM<br/>特征交互 + 非线性边界"]
    C -->|6 个超参| E["AttentionMLP<br/>特征注意力重加权描述符"]
    D --> F["加权概率融合集成<br/>Nelder-Mead 优化融合权重"]
    E --> F
    F --> G["三分类输出<br/>正常 / 木质胸 / 意面肉"]
```

### 关键设计

**1. 智能手机透射成像：让光穿过肉、把内部结构变成可测信号**

木质胸和意面肉的病变在表面看不出来，问题出在内部结构，这正是普通拍照失效的原因。MyoVision 用宽带白光穿透鸡胸肉，记录 2D 空间上的积分衰减模式——这种衰减聚合了组织密度、纤维化硬化以及肌肉中液体重分布的变化，等于把"看不见的内部"编码进一张透射图。采集时用 14-bit RAW 保留最大动态范围，避免压缩丢掉暗部细节。

**2. NEAT 神经进化优化：让算法自己进化出超参，绕开小数据上的手动调参**

数据集只有 336 个样本，手动网格搜索既费力又容易过拟合。MyoVision 用 NEAT 算法同时进化网络拓扑和权重，每个基因组编码一个小网络、输出一组候选超参，自动发现 LightGBM 的 10 个超参和 AttentionMLP 的 6 个超参。进化通过变异、交叉和物种形成推进，相当于把"调参"交给演化搜索，省掉人工试错。

**3. 加权概率融合集成：让两个互补学习器各管一摊再合议**

单一模型在这种小样本表格特征上不稳。MyoVision 让 LightGBM 负责特征交互和非线性决策边界，AttentionMLP 通过特征注意力对输入描述符重加权、突出最具判别力的纹理，二者输出的概率再做加权融合得到最终预测；融合权重用 Nelder-Mead 单纯形法直接优化。两个学习器视角互补，集成后比任一单模更鲁棒。

### 损失函数 / 训练策略

- 适应度评估用分层交叉验证的加权 F1 分数
- 训练数据用 SMOTE 做类别平衡
- 进化通过变异、交叉和物种形成操作推进

## 实验关键数据

### 主实验

| 方法 | 测试准确率 | F1 分数 |
|------|-----------|---------|
| 传统 ML 基线 | 较低 | 较低 |
| 深度学习基线 | 较低 | 较低 |
| NEATBoost-Attention | **82.4%** | **0.83** |
| 高光谱成像系统 | ~同等 | ~同等 |

在 336 个鸡胸肉样本上达到 82.4% 测试准确率（F1=0.83），匹配成本高出数个数量级的高光谱系统。

### 关键发现

- NEAT 进化的超参数配置优于网格搜索和随机搜索
- 特征注意力机制有效识别了最具判别力的纹理描述符
- 透射成像能捕获表面成像无法获取的内部结构信息

## 亮点与洞察

- 用消费级设备实现与专业实验室设备可比的检测性能，极具实用价值
- NEAT 自动架构搜索避免了小数据集上的手动调参困境
- 多模态研究平台设计（RAW + LiDAR + SAM + ChatGPT）前瞻性强

## 局限与展望

- 336 个样本数据集偏小，泛化能力需更多数据验证
- 82.4% 准确率在工业部署中可能仍不够
- 透射成像受环境光影响需要标准化采集条件

## 相关工作与启发

- NEAT 在食品质量评估领域的首次探索，开辟了新方向
- 透射成像原理可推广到其他农产品内部缺陷检测
- 小样本表格数据上的神经进化优化思路值得借鉴

## 评分

5/10 — 有趣的跨领域应用，但数据规模有限，核心方法创新性一般。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[ACL 2026\] Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays](../../ACL2026/others/qayyem_a_real-time_platform_for_scoring_proficiency_of_arabic_essays.md)
- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](../../AAAI2026/others/lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)

</div>

<!-- RELATED:END -->
