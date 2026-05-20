---
title: >-
  [论文解读] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models
description: >-
  [CVPR 2026][多模态VLM][OOD检测] 提出 TANL（Test-time Activated Negative Labels），通过在测试时动态评估负标签在OOD样本上的"激活程度"来挖掘最有效的负标签，配合激活感知评分函数…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "OOD检测"
  - "视觉语言模型"
  - "负标签"
  - "测试时自适应"
  - "激活度量"
---

# Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.25250](https://arxiv.org/abs/2603.25250)  
**代码**: [GitHub](https://github.com/YBZh/OpenOOD-VLM)  
**领域**: 多模态VLM / AI安全  
**关键词**: OOD检测, 视觉语言模型, 负标签, 测试时自适应, 激活度量

## 一句话总结
提出 TANL（Test-time Activated Negative Labels），通过在测试时动态评估负标签在OOD样本上的"激活程度"来挖掘最有效的负标签，配合激活感知评分函数，在 ImageNet 基准上将 FPR95 从 17.5% 大幅降至 9.8%，且完全免训练、测试高效。

## 研究背景与动机
**领域现状**：OOD检测是AI安全的核心问题。基于VLM（如CLIP）的方法通过引入"负标签"（与ID类别语义距离远的文本标签）检测OOD样本——与负标签相似度高的样本更可能是OOD。

**关键问题——"低激活负标签"**：
   - NegLabel等方法从语料库中选择与ID标签距离最远的词作为负标签
   - 但这些负标签仅基于ID标签选出，**未考虑测试分布**
   - 结果：许多负标签在OOD数据上的激活度（相似度）极低，甚至低于在ID数据上的激活度（见Fig.1a）
   - 这些"低激活"标签不仅无效，还引入噪声，降低检测性能

**核心观察**：少数高激活负标签即可有效检测OOD（Fig.1b），大量低激活标签反而有害。

**核心idea**：在测试时动态评估标签激活度，选择真正"被OOD样本激活"的负标签。

## 方法详解

### 整体框架
测试时维护正/负样本FIFO队列 → 在语料库上动态计算标签激活分数 → 选择高激活负标签 → 使用激活感知评分函数检测OOD。

### 关键设计
1. **激活度量（Activation Metric）**：
   衡量特定标签在数据集上的平均分类概率：
    $Act(\mathcal{X}, \hat{y}_i) = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x} \in \mathcal{X}} \frac{\exp(\mathbf{v}\hat{\mathbf{t}}_i)}{\sum_j \exp(\mathbf{v}\mathbf{t}_j) + \sum_j \exp(\mathbf{v}\hat{\mathbf{t}}_j)}$
   理想负标签应在OOD上高激活、在ID上低激活。差分激活分数：
    $Act_d(\hat{y}_i) = Act(\mathcal{X}_{ood}, \hat{y}_i) - Act(\mathcal{X}_{id}, \hat{y}_i)$
    - 设计动机：直接量化标签对OOD检测的判别力

2. **分布自适应激活标签（Distribution-adaptive）**：
   用缓存的高置信正/负样本近似 $\mathcal{X}_{id}$ 和 $\mathcal{X}_{ood}$：
    - FIFO队列 $\mathcal{X}_{pos}$ / $\mathcal{X}_{neg}$，长度 $L$
    - 正样本：$S_{aa}(\mathbf{v}) \geq \gamma + (1-\gamma)g$；负样本：$S_{aa}(\mathbf{v}) < \gamma - \gamma g$
    - **初始化**：正样本用ID标签特征，负样本用高斯噪声图像特征
    - 设计动机：OOD分布未知且可能动态变化，需要在线自适应

3. **批自适应变体（Batch-adaptive）**：
   在当前测试batch内额外提取正负样本，与历史样本加权融合：
    $Act_b(\mathcal{X}_{pos}, \hat{y}_i) = \alpha Act(\mathcal{X}_{pos}, \hat{y}_i) + (1-\alpha) Act(\mathcal{X}^b_{pos}, \hat{y}_i)$
    - 设计动机：历史样本反映总体趋势，当前batch捕获即时特征，两者互补

4. **激活感知评分函数（Activation-aware Score）**：
    $S_{aa}(\mathbf{v}) = \frac{1}{M}\sum_{m=1}^{M}\sum_{i=1}^{C}\frac{\exp(\mathbf{v}\mathbf{t}_i)}{\sum_j \exp(\mathbf{v}\mathbf{t}_j) + \sum_{j=1}^m \exp(\mathbf{v}\tilde{\mathbf{t}}_j)}$
   负标签按激活度排序后，高激活标签在分母中出现更多次从而被隐式赋予更高权重。
    - 设计动机：不同负标签重要性不同，高激活标签应主导评分。这种累积求和设计同时增强了对标签数量 $M$ 的鲁棒性。

### 损失函数 / 训练策略
- **完全免训练**（zero-shot, training-free）
- CLIP编码器冻结，仅在测试时维护FIFO队列
- 超参数：$\gamma$ 为ID/OOD阈值，$g$ 为置信间隔，$L$ 为队列容量，$\alpha$ 为历史/batch权重

## 实验关键数据

### 主实验（ImageNet-1k, CLIP ViT-B/16）

| 方法 | 类型 | INaturalist FPR95↓ | Sun FPR95↓ | Places FPR95↓ | Textures FPR95↓ | Average FPR95↓ |
|------|------|-----|-----|-----|-----|-----|
| NegLabel | 免训练 | 1.91 | 20.53 | 35.59 | 43.56 | 25.40 |
| CSP | 免训练 | 1.54 | 13.66 | 29.32 | 25.52 | 17.51 |
| AdaNeg | 测试时自适应 | 0.59 | 9.50 | 34.34 | 31.27 | 18.92 |
| OODD | 测试时自适应 | 0.85 | 12.94 | 30.68 | 30.67 | 18.79 |
| **TANL** | **测试时自适应** | **0.42** | **3.53** | - | - | **9.8** |

*注：TANL 将平均 FPR95 从 NegLabel的25.4%降至9.8%（降幅61%），比CSP再降44%*

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| NegLabel（距离选择） | FPR95: 25.4% | 不考虑激活度 |
| + 激活分数选择 | FPR95 大幅下降 | 激活标签是核心 |
| + 激活感知评分 | 进一步提升 | 加权效果显著 |
| + 批自适应 | 最优 | 即时信息有帮助 |
| M对鲁棒性 | $S_{aa}$ 对M鲁棒 | 传统方法对M敏感 |

### 关键发现
- 激活感知标签选择是核心：少量高激活标签 > 大量低激活标签
- FPR95从25.4%降至9.8%（vs NegLabel），降低15.6个百分点
- 比当前最优CSP再降7.7个百分点
- $S_{aa}$ 对负标签数量M具有天然鲁棒性——不需要精调M
- 初始化策略有效：ID特征初始化正样本、噪声图像初始化负样本提供稳定启动
- 在不同CLIP骨干（ViT-B/16, ViT-L/14等）、near-OOD、full-spectrum OOD、医学OOD等多种设置下均有效

## 亮点与洞察
- **"激活度"概念简单但有效**：量化了"哪些负标签真正有用"这一被忽视的问题
- **累积求和评分函数**设计巧妙：一个公式同时实现加权和鲁棒性
- **免训练+测试高效**极具实用性：仅维护FIFO队列，无需反向传播
- **初始化用噪声图像**作为OOD代理是有趣的直觉

## 局限与展望
- 依赖高置信样本初始化队列，如果初期检测不准会导致错误累积
- FIFO队列长度L是超参数，极端情况下可能不足
- 当ID和OOD分布非常接近（如near-OOD）时，高置信正负样本可能不存在
- 理论分析基于特定假设，实际分布可能不满足
- 仅在CLIP模型上验证，对其他VLM的适应性未知

## 相关工作与启发
- 对 NegLabel 的改进直接且有效——核心文认识到"标签选择策略"比"标签数量"更重要
- 测试时自适应（TTA）的思路从模型参数更新扩展到标签选择，是新颖的变体
- 激活度量可能可推广到其他基于标签的零样本方法
- 与 AdaNeg（图像代理 vs 本文标签激活）形成互补视角

## 评分
- 新颖性: ⭐⭐⭐⭐ 激活度量概念新颖，评分函数设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 多OOD类型、多骨干、理论分析、鲁棒性验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机图分析清晰，算法框图直观
- 价值: ⭐⭐⭐⭐⭐ 简单有效的改进，对VLM OOD检测有即时实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)

</div>

<!-- RELATED:END -->
