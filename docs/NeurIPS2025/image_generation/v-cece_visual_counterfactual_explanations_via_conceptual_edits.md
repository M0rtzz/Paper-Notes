---
title: >-
  [论文解读] V-CECE: Visual Counterfactual Explanations via Conceptual Edits
description: >-
  [NeurIPS 2025][图像生成][反事实解释] V-CECE提出首个系统性揭示人类与神经网络分类器语义理解差异（explanatory gap）的黑盒视觉反事实解释框架，通过WordNet知识图谱+匈牙利算法保证编辑集最优性，用Stable Diffusion执行概念级编辑，核心发现是CNN分类器的语义推理与人类严重不对齐（需5+步编辑），而LVLM（Claude 3.5 Sonnet）与人类高度一致（仅需2-3步）。
tags:
  - NeurIPS 2025
  - 图像生成
  - 反事实解释
  - 概念编辑
  - 黑盒
  - 知识图谱
  - 扩散模型
---

# V-CECE: Visual Counterfactual Explanations via Conceptual Edits

**会议**: NeurIPS 2025  
**arXiv**: [2509.16567](https://arxiv.org/abs/2509.16567)  
**代码**: [项目页面](https://nickspanos55.github.io/vcece)  
**领域**: 可解释 AI / 反事实解释 / 扩散模型  
**关键词**: 反事实解释, 概念编辑, 黑盒, 知识图谱, 扩散模型

## 一句话总结

V-CECE提出首个系统性揭示人类与神经网络分类器语义理解差异（explanatory gap）的黑盒视觉反事实解释框架，通过WordNet知识图谱+匈牙利算法保证编辑集最优性，用Stable Diffusion执行概念级编辑，核心发现是CNN分类器的语义推理与人类严重不对齐（需5+步编辑），而LVLM（Claude 3.5 Sonnet）与人类高度一致（仅需2-3步）。

## 研究背景与动机

**领域现状**：反事实解释（counterfactual explanation）是可解释AI的重要工具——通过"如果改变X，分类结果会如何"来揭示模型的决策依据。现有方法分为白盒（需梯度访问）和黑盒（无需内部访问）两类。

**现有痛点**：现有反事实图像生成方法存在三大问题：(1)编辑分散不可解释（ACE、DiME等产生人类难以理解的像素级改变），(2)过度依赖训练来引导生成（白盒方法需天级训练），(3)最关键的是——所有语义反事实方法都假设分类器以人类语义水平推理，但从未验证这个假设。

**核心矛盾**：人类和神经网络分类器"理解语义"的方式是否一致？如果不一致，用人类可理解的概念编辑去解释CNN的分类决策本身就是误导性的。这个问题比不可解释的对抗编辑更危险，因为它引入了虚假的可解释性。

**本文目标** 两个递进问题：(1)分类器的决策过程能否用人类级别的语义来解释？(2)如果能，翻转分类标签所需的最少语义编辑是什么？

**切入角度**：将反事实解释分解为"解释"和"生成"两个独立阶段——先用知识图谱计算最优语义编辑集（与模型无关），再用冻结的扩散模型执行编辑（避免训练偏差），最后通过分类结果验证效果。

**核心 idea**：用知识图谱保证编辑最优性，用冻结扩散模型保持公平性，从而系统性地测量人类与模型的语义理解差距。

## 方法详解

### 整体框架

两阶段pipeline：(1)解释阶段：给定源类L和目标类L*的概念集合，在WordNet知识图谱上用匈牙利算法求解最小代价编辑集E（包含插入I、删除D、替换S操作）；(2)生成阶段：按三种策略之一排序编辑，用GroundingDINO+SAM定位目标区域，用Stable Diffusion v1.5 Inpainting执行编辑，每步后检查分类器是否翻转标签。

### 关键设计

1. **最优编辑保证（知识图谱+匈牙利算法）**:

    - 功能：计算从类L到类L*的最小语义编辑集
    - 核心思路：替换代价 = WordNet上两概念间的最短路径距离，用Dijkstra计算。构建二部图匹配问题——源概念和目标概念作为两侧节点，边权为替换代价，添加虚拟节点模拟插入/删除操作（代价为到根节点的距离）。用匈牙利算法求解最小权匹配，时间复杂度 $O(mn\log n)$
    - 设计动机：提供确定性的最优性保证，不同于之前方法的启发式编辑选择。不可执行的编辑可赋无穷代价自动排除

2. **三种编辑排序策略**:

    - 功能：在最优编辑集E中决定执行顺序，尽早翻转标签
    - 核心思路：(1)**Local Edits**：LVLM每步观察当前图像+剩余编辑，选择下一个操作（每步更新图像防止逻辑不一致）；(2)**Global Edits**：统计所有图像上每个编辑的出现频率，按Importance分数排序——分数公式为 $(|I_{s_j^*}| - |D_{s_i}| + |S_{s_i \to s_j^*}| - |S_{s_j^* \to s_i}|) / |e \in E|$，捕捉分类器的系统性偏差；(3)**Local-Global**：对特定图像选择local编辑子集，按global重要性排序
    - 设计动机：Local利用图像上下文但忽略分类器偏差，Global利用偏差但忽略场景细节，Local-Global平衡两者

3. **冻结扩散模型执行编辑**:

    - 功能：执行概念级图像编辑同时保持公平评估
    - 核心思路：使用Stable Diffusion v1.5 Inpainting（冻结参数，零训练），DPM++ 2M SDE采样器40步，GroundingDINO+SAM生成概念掩码，仅对掩码区域修补。用LVLM（Claude 3.5 Sonnet）确定最佳放置位置和背景填充
    - 设计动机：故意不在目标数据集上训练扩散模型，因为训练会引入数据偏差导致虚假有利的反事实图像。冻结模型保证偏差一致，评估结果的差异完全来自分类器行为

### 损失函数 / 训练策略

无训练——V-CECE是完全即插即用的框架，所有模块（分类器、扩散模型、LVLM）都以黑盒方式使用，零训练。

## 实验关键数据

### 主实验

BDD100K（自动驾驶场景分类，Stop/Move）：

| 方法 | FID↓ | CMMD↓ | SR↑ | Avg\|E\|↓ | 训练 |
|------|------|-------|------|-----------|------|
| ACE l1 (白盒) | 1.02 | - | 99.9% | - | 天 |
| TIME (黑盒) | 51.5 | - | 81.8% | - | 小时 |
| V-CECE+DenseNet Local | 90.42 | 1.101 | 88.9% | 4.77 | N/A |
| V-CECE+Claude3.5 Global | **45.22** | **0.427** | 97.8% | **2.65** | N/A |
| V-CECE+Claude3.5 L-G | 42.76 | 0.364 | **98.1%** | 2.44 | N/A |

### 消融实验

人类评估——模型所需编辑步数 vs 人类认为合理的步数：

| 分类器 | Avg\|E\| 模型 | Avg\|E\| 人类 | 视觉正确率(%) |
|--------|-------------|-------------|-------------|
| DenseNet | 5.22 | 2.21 | 59.71 |
| ConvNext | 7.35 | 2.27 | 34.24 |
| EfficientNet | 5.96 | 2.66 | 30.17 |
| Claude 3 Haiku | 2.91 | 1.88 | 69.58 |
| Claude 3.5 Sonnet | 2.19 | 1.33 | 81.20 |
| Claude 3.7 Sonnet | 2.50 | 1.37 | 79.98 |

### 关键发现

- **CNN与人类存在显著语义鸿沟**：DenseNet需要5.22步编辑才能翻转，而人类认为仅需2.21步即可。且DenseNet翻转时59.7%图像已出现视觉伪影，说明它依赖的不是语义变化而是像素分布变化
- **LVLM与人类高度一致**：Claude 3.5 Sonnet仅需2.19步（接近人类的1.33步），且81.2%图像视觉正确，其语义理解与人类基本对齐
- **CNN的决策具有随机性**：重要性最高的概念仅0.16-0.23，且有35-55个重要概念，说明CNN没有一致的语义依赖模式。LVLM相反，最重要概念达0.37-0.40，仅27-31个重要概念
- **思维链反而有害**：Claude 3.7开启thinking后需要更多编辑步骤（3.78 vs 3.03），FID反而更差，印证了CoT在视觉任务上可能有害的研究发现

## 亮点与洞察

- **问题定义精准**：将反事实解释拆分为"语义对齐度验证"和"最小编辑计算"两个递进问题，这比之前只做第二步的方法更根本。如果分类器不在语义层面推理，用语义反事实解释它就是误导
- **冻结模型保证公平性**：故意不训练扩散模型的设计非常巧妙——避免了数据偏差污染评估结果，使得分类器之间的差异可以真实反映其语义理解能力
- **LVLM-as-classifier的新用法**：将LVLM当作分类器来解释是可行的且语义对齐度高，这为解释黑盒商业模型提供了新范式

## 局限与展望

- **人类评估规模有限**：当前人类调查规模较小，统计检验力和精度有限，结果应视为初步洞察
- **扩散模型的生成质量限制**：Stable Diffusion v1.5 inpainting在多步编辑后产生伪影是不可避免的，这混淆了"分类器的语义不对齐"和"图像质量下降导致的分类变化"
- **知识图谱的语义粒度**：WordNet的概念粒度固定且不完整，可能遗漏对分类重要的视觉概念
- **仅在BDD100K和Visual Genome验证**：需要扩展到医学影像等高风险领域验证通用性
- 改进方向：引入白盒生成模型做对比、扩大人类评估规模并评估评分者间一致性、测试新一代扩散模型（SD3/Flux）

## 相关工作与启发

- **vs ACE/DiME（白盒反事实）**：白盒方法SR高达99.9%但依赖梯度和训练，编辑不可解释；V-CECE是黑盒零训练，编辑人类可理解
- **vs Dervakos/Dimitriou（语义反事实）**：前人工作需要12+步编辑且不生成图像，V-CECE仅需2-3步且生成可视化结果
- **vs TIME（黑盒反事实）**：TIME需要训练且不提供语义编辑，V-CECE以LVLM为分类器时FID和SR均优于TIME

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示人类与模型的语义理解差距，问题定义很有价值
- 实验充分度: ⭐⭐⭐⭐ 覆盖CNN/ViT/LVLM多种分类器，包含人类评估，但人评规模偏小
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，问题动机阐述精准，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 揭示的explanatory gap对XAI领域有根本性意义，改变了对反事实解释的理解

<!-- RELATED:START -->

## 相关论文

- [LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)
- [Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)

<!-- RELATED:END -->
