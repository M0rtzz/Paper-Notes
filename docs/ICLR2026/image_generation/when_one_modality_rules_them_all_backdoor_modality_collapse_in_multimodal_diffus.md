---
title: >-
  [论文解读] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models
description: >-
  [ICLR 2026][图像生成][backdoor attack] 首次揭示并系统研究多模态扩散模型中的"后门模态坍缩"现象——多模态后门攻击中后门效果退化为仅依赖单一模态（通常是文本），提出TMA和CTI两个基于Shapley值的新指标量化模态贡献和跨模态交互，发现"赢者通吃"动态和负交互。
tags:
  - ICLR 2026
  - 图像生成
  - backdoor attack
  - 模态坍缩
  - 多模态扩散
  - Shapley值
  - 触发器交互
---

# When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2603.06508](https://arxiv.org/abs/2603.06508)  
**代码**: 无  
**领域**: AI安全/多模态扩散模型  
**关键词**: backdoor attack, 模态坍缩, 多模态扩散, Shapley值, 触发器交互

## 一句话总结
首次揭示并系统研究多模态扩散模型中的"后门模态坍缩"现象——多模态后门攻击中后门效果退化为仅依赖单一模态（通常是文本），提出TMA和CTI两个基于Shapley值的新指标量化模态贡献和跨模态交互，发现"赢者通吃"动态和负交互。

## 研究背景与动机

**领域现状**：扩散模型的后门攻击已成为重要研究方向。现有工作（BadDiffusion、VillanDiffusion等）已展示在单模态和多模态场景中注入后门的可行性。

**现有痛点**：(1) 直觉假设同时攻击多个模态应产生更强后门，但缺乏验证；(2) 受多模态学习中"模态坍缩"启发，后门攻击中可能存在同样问题；(3) 现有评估仅关注整体攻击成功率，未分解各模态贡献。

**核心矛盾**：高攻击成功率可能掩盖一个关键事实——后门实际上仅依赖于部分模态。这意味着防御者可能低估了攻击的简单性（只需操纵文本prompt即可触发）。

**本文目标**：多模态扩散后门中是否存在模态坍缩？如何量化各模态的后门贡献和跨模态交互？

**切入角度**：利用合作博弈论的Shapley值框架，将模态视为"玩家"，后门偏移量视为"收益"，进行精确的贡献分解。

**核心 idea**：通过Shapley值分解各模态对后门激活的边际贡献（TMA），并通过超可加性检验量化跨模态交互（CTI），揭示"赢者通吃"的坍缩特性。

## 方法详解

### 整体框架
二模态（图像+文本）设定下，模态集 $\mathcal{M} = \{I, T\}$。对每个样本计算4种coalition的value function $v(\emptyset), v(\{I\}), v(\{T\}), v(\{I,T\})$，基于此精确计算Shapley值。

### 关键设计

1. **Per-example值函数**:

    - 对每个样本和coalition $S$，计算触发器分数和正常分数的差值
    - $v(S) = s_{\text{tr}}(S) - s_{\text{nr}}(S) = \cos(\mathbf{z}_S, \mathbf{z}_{\text{tr}}) - \cos(\mathbf{z}_S, \mathbf{z}_{\text{cl}})$
    - 使用CLIP嵌入空间的余弦相似度
    - 衡量输出更接近后门目标还是干净参考

2. **Trigger Modality Attribution (TMA)**:

    - $\phi_I = \frac{1}{2}(v(\{I\}) - v(\emptyset)) + \frac{1}{2}(v(\{I,T\}) - v(\{T\}))$
    - $\phi_T = \frac{1}{2}(v(\{T\}) - v(\emptyset)) + \frac{1}{2}(v(\{I,T\}) - v(\{I\}))$
    - 二模态精确计算（4次评估，无需蒙特卡洛近似）
    - Efficiency axiom：$\phi_I + \phi_T = v(\{I,T\}) - v(\emptyset)$

3. **Cross-Trigger Interaction (CTI)**:

    - $\mathcal{I} = v(\{I,T\}) - v(\{I\}) - v(\{T\}) + v(\emptyset)$
    - $\mathcal{I} > 0$：超可加性合作（真正的协同）
    - $\mathcal{I} < 0$：干扰/冗余
    - 数据集级别聚合：$\bar{\mathcal{I}} = \frac{1}{|\mathcal{D}_{\text{val}}|} \sum \mathcal{I}(x)$

4. **实验设置**:

    - 三对触发器：White-box+mignneko、Eyeglasses+anonymous、Stop-sign+latte coffee
    - 两种投毒协议：OR（三个等大小子集分别投毒）、AND（仅联合投毒）
    - 三种投毒比例：1%、5%、10%
    - 模型：InstructPix2Pix，数据集：CelebA

## 实验关键数据

### TMA和CTI关键结果

| 触发器对 | 协议 | 比例 | $\bar{\phi}_I$ (TMA-I) | $\bar{\phi}_T$ (TMA-T) | $\bar{\mathcal{I}}$ (CTI) |
|----------|------|------|------------------------|------------------------|---------------------------|
| White-box+mignneko | OR | 5% | 0.0060 | 0.9743 | -0.0089 |
| White-box+mignneko | AND | 5% | 0.0045 | 0.9532 | -0.0086 |
| Eyeglasses+anonymous | OR | 5% | 0.1200 | 0.7376 | -0.2174 |
| Eyeglasses+anonymous | AND | 5% | 0.1063 | 0.8907 | -0.2185 |
| Stop-sign+latte coffee | OR | 5% | 0.0043 | 0.9280 | -0.0094 |
| Stop-sign+latte coffee | AND | 5% | 0.0048 | 1.0033 | -0.0101 |

### 关键发现

1. **模态主导性（Modality Dominance）**：
    - 文本模态TMA几乎总是 > 0.7，图像模态TMA几乎总是 < 0.15
    - White-box+mignneko 5% AND：$\bar{\phi}_T = 0.9532$ vs $\bar{\phi}_I = 0.0045$→文本几乎完全接管
    - 后门本质上退化为单模态文本后门

2. **负交互（Negative Interaction）**：
    - 所有配置的CTI均为负值或接近零
    - 最严重：Eyeglasses+anonymous的CTI达到-0.22
    - 联合触发不产生协同效应，反而存在干扰

3. **投毒协议的影响**：
    - AND投毒理论上应更强调联合触发，但TMA仍显示文本主导
    - OR投毒的模态分布更均匀（因为有单模态投毒子集），但文本仍主导

4. **投毒比例的影响**：
    - 1%-10%范围内，坍缩模式一致
    - 文本主导特性不随投毒比例变化

5. **安全启示**：
    - 攻击者只需操纵文本prompt（如添加稀有token）即可触发后门
    - 图像端触发器基本多余——降低了攻击部署门槛
    - 防御应优先关注文本模态的异常检测

## 亮点与洞察
- **首次揭示的重要安全现象**：后门模态坍缩改变了我们对多模态后门威胁模型的理解
- **理论工具引入恰当**：Shapley值作为唯一满足效率/对称/虚player/可加性的归因方法，是分解模态贡献的最佳选择
- **负交互的反直觉发现**：挑战了"多模态=更强攻击"的假设，对防御设计有重要指导
- **实用安全启示**：防御者应意识到，高ASR的多模态后门可能仅需文本侧检测

## 局限与展望
- 仅在InstructPix2Pix上验证，更大/更新的多模态扩散模型待测试
- 二模态设定，三+模态场景中的Shapley计算需要蒙特卡洛近似
- 未提出防御方案，仅做分析（但分析本身对设计防御有重要价值）
- 可以探索模态坍缩的形成动态（训练过程中何时发生）

## 相关工作与启发
- **vs VillanDiffusion**：他们评估整体ASR但不分解模态贡献，可能得出"多模态更强"的错误结论
- **vs 多模态学习中的模态坍缩**：首次将此概念从训练优化迁移到后门攻击领域
- **vs 强弱模态平衡方法**：提示了后门场景中可能也需要类似的模态平衡策略

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 后门模态坍缩是全新概念，TMA/CTI指标的提出有据可循
- 实验充分度: ⭐⭐⭐⭐ 三组触发器×两种协议×三种比例的全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义和形式化清晰，博弈论框架应用自然
- 价值: ⭐⭐⭐⭐ 对AI安全社区的多模态后门研究有重要方向性指引

<!-- RELATED:START -->

## 相关论文

- [Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)
- [All-in-One Slider for Attribute Manipulation in Diffusion Models](../../CVPR2026/image_generation/all-in-one_slider_for_attribute_manipulation_in_diffusion_models.md)
- [Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective](../../AAAI2026/image_generation/enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)
- [When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](../../CVPR2026/image_generation/when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)

<!-- RELATED:END -->
