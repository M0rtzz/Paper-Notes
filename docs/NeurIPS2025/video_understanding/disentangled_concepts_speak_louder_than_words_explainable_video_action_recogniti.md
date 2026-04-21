---
title: >-
  [论文解读] Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition
description: >-
  [NeurIPS 2025 (Spotlight)][视频理解][可解释视频动作识别] 提出DANCE框架，通过将动作解释解耦为运动动态、物体和场景三类概念，实现结构化和运动感知的可解释视频动作识别。
tags:
  - NeurIPS 2025 (Spotlight)
  - 视频理解
  - 可解释视频动作识别
  - 概念瓶颈模型
  - 运动解耦
  - 姿态序列
  - 概念发现
---

# Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2511.03725](https://arxiv.org/abs/2511.03725)  
**代码**: [有](https://jong980812.github.io/DANCE/)  
**领域**: 视频理解 / 可解释AI  
**关键词**: 可解释视频动作识别, 概念瓶颈模型, 运动解耦, 姿态序列, 概念发现

## 一句话总结

提出DANCE框架，通过将动作解释解耦为运动动态、物体和场景三类概念，实现结构化和运动感知的可解释视频动作识别。

## 研究背景与动机

视频动作识别模型在性能上取得了巨大进步，但其决策过程仍然不透明。现有可解释方法存在明显局限：

**显著性方法**（Saliency Tubes、GradCAM等）：产生纠缠的解释，无法区分模型究竟依赖运动还是空间上下文

**语言方法**（LLM生成概念描述）：能描述物体和场景，但难以表达运动动态——运动属于**默会知识**（tacit knowledge），即直觉理解但难以言语化的知识

从认知科学角度，人类感知动作时会分别分析两个因素：
- **时间动态**：运动如何随时间展开
- **空间上下文**：周围的物体和场景

因此，理想的视频XAI应显式解耦时间动态与空间上下文，但现有方法都未做到这一点。

## 方法详解

### 整体框架

DANCE基于前置（ante-hoc）概念瓶颈设计，在预训练视频骨干编码器和最终分类器之间插入概念层。预测流程：

**输入视频 → 视频特征 → 三类概念激活（运动动态、物体、场景） → 动作预测**

三类概念各有独立的概念层参数 $W_C = [W_C^m; W_C^o; W_C^s]$，确保概念类型之间的显式解耦。

### 关键设计

**1. 运动动态概念（Motion Dynamics Concepts）**

核心创新：用**人体姿态序列**定义运动概念，而非文本描述。

- **关键片段选择**：通过关键帧检测提取视频中最具信息量的短片段
- **姿态序列提取**：对每个关键片段用2D姿态估计器逐帧提取姿态 $P_i^s \in \mathbb{R}^{L \times J \times 2}$
- **聚类发现概念**：将所有训练视频的姿态序列聚合，使用FINCH聚类算法发现代表性运动模式
- **概念标注**：通过聚类归属自动生成二值标签 $c_k^m = I(\sum_s a_{i,s,k})$

优势：姿态序列提供与外观无关的运动表示，用户可直观理解动作如何随时间展开。

**2. 物体和场景概念（Object & Scene Concepts）**

- 使用GPT-4o查询每个动作类别相关的物体和场景
- 通过视觉-语言双编码器（InternVid）自动生成伪标签
- 物体伪标签：$\tilde{c}_i^o = E_T(\mathcal{O}) E_V(V_i)$

**3. 概念瓶颈架构**

- 冻结预训练视频骨干（VideoMAE），仅训练概念层和分类器
- 概念层将视频特征投射到概念空间，获得激活值 $z = [z_m; z_o; z_s]$
- 分类器基于概念激活预测动作

### 损失函数 / 训练策略

分两阶段训练：

**阶段1：概念层训练**
- 运动动态概念：二值交叉熵损失（因为运动标签是多标签的）
$$\mathcal{L}_m = -\frac{1}{M_m}\sum_{k=1}^{M_m}[c_k^m\log\sigma_k(z_m) + (1-c_k^m)\log(1-\sigma_k(z_m))]$$
- 物体/场景概念：余弦立方损失（cosine cubed loss），强调方向对齐

**阶段2：分类层训练**
- 冻结概念层，用交叉熵损失+稀疏正则化训练最终线性分类器
$$\mathcal{L}_{cls} = -\frac{1}{K}\sum_k y_k\log\hat{y}_k + \lambda[(1-\alpha)\frac{1}{2}\|W_A\|_F + \alpha\|W_A\|_{1,1}]$$
- L1正则化促进权重稀疏，提高可解释性

## 实验关键数据

### 主实验

**表1：视频动作识别性能（Top-1 Accuracy %）**

| 方法 | KTH | Penn Action | HAA-100 | UCF-101 |
|:---:|:---:|:---:|:---:|:---:|
| 无可解释性基线 | 89.7 | 97.8 | 73.5 | 88.4 |
| CBM + UCF-101属性 | - | - | - | 86.8 |
| LF-CBM + 纠缠语言概念 | 87.4 | 96.3 | 66.5 | 85.5 |
| LF-CBM + 解耦语言概念 | 89.9 | 97.7 | 65.3 | 83.7 |
| **DANCE** | **91.1** | **98.1** | **70.7** | **87.5** |

关键发现：
- DANCE在KTH和Penn Action上**超越**无可解释性基线（+1.4和+0.3）
- 在HAA-100和UCF-101上仅有轻微下降（-2.8和-0.9）
- 相比使用语言概念的CBM，DANCE在所有数据集上一致领先

**用户研究结果（图6）**

| 对比方法 | DANCE更好 | 差不多 | 对方更好 |
|:---:|:---:|:---:|:---:|
| vs GPT-4o概念CBM | >70% | ~20% | <10% |
| vs VTCD（显著性方法） | >70% | ~20% | <10% |
| vs 专家定义概念 | >70% | ~15% | <15% |

运动动态概念可解释性评分：本文方法 **4.3/5**，语言方法 2.3/5，专家概念 3.4/5。

### 消融实验

**跨域模型编辑实验（图10）**

在UCF-101→UCF-101-SCUBA（严重域偏移）场景下：
- 通过调整3个类别的概念权重，准确率从 77.7% 提升到 **82.0%**（+4.3%）
- 无需重新训练

**样本级干预（图9）**

- 去激活不相关的场景概念（如"Table Tennis Club"）可将错误预测纠正为正确预测
- 展示了DANCE支持细粒度、透明的预测控制

**时间方向灵敏度检查（图7）**

- 正向视频预测为"Bowing FullBody"，反向视频预测为"Burpee"
- 验证模型确实依赖运动动态概念而非仅靠空间上下文

### 关键发现

1. **更清晰的概念带来更好的性能**：DANCE使用姿态序列代替语言描述运动，在所有数据集上一致优于语言概念方法
2. **可解释性与性能不一定矛盾**：在KTH和Penn Action上甚至提升了性能
3. **运动动态概念最直观**：用户研究中89.7%参与者给出4或5分（满分5分）
4. **支持无需重训练的模型调试**：通过概念权重编辑可在域偏移下恢复性能

## 亮点与洞察

1. **用姿态序列表示运动概念是关键创新**：绕过了运动的"默会知识"难题——不用语言描述运动，而是直接可视化姿态序列
2. **完全自动化的概念发现**：运动概念通过聚类发现，物体/场景概念通过LLM提取，无需人工标注
3. **前置可解释设计**：不是事后解释，而是模型本身通过概念做预测，保证解释的忠实性
4. **实用的模型调试能力**：概念权重的可编辑性使得模型调试和域适应变得简单直接

## 局限与展望

1. 依赖2D姿态估计器的质量，估计不准确会影响运动概念质量
2. 仅适用于以人为中心的动作识别，对非人类动作（如自然现象）不适用
3. 线性概念层可能限制了概念间复杂交互的建模能力
4. 概念数量（尤其是运动概念数量）依赖聚类超参数选择
5. UCF-101上仍有约0.9%的性能损失，大规模数据集上的可扩展性待验证

## 相关工作与启发

- **概念瓶颈模型（CBM）** [Koh et al., 2020]：本文的概念层设计基础
- **Label-Free CBM** [Oikarinen et al., 2023]：用LLM自动发现概念，DANCE的物体/场景概念发现基于此
- **VTCD** [Kowal et al., 2024]：基于优化的视频概念发现，需额外计算开销
- **Saliency Tubes** [Stergiou et al., 2019]：3D显著性方法，但解释纠缠

## 评分

- **新颖性**: ★★★★★ — 运动动态概念的定义和发现方式是视频XAI的开创性贡献
- **技术深度**: ★★★★☆ — 概念瓶颈框架成熟，创新主要在概念定义和发现流水线
- **实验充分性**: ★★★★★ — 4个数据集、用户研究、消融、模型编辑等全方位评估
- **写作质量**: ★★★★★ — 图表精美，故事线清晰，Spotlight论文实至名归
- **实用性**: ★★★★☆ — 模型调试和编辑功能具有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](../../ECCV2024/video_understanding/sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)
- [Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](empower_words_dualground_for_structured_phrase_and_sentencel.md)
- [Referring Atomic Video Action Recognition](../../ECCV2024/video_understanding/referring_atomic_video_action_recognition.md)

<!-- RELATED:END -->
