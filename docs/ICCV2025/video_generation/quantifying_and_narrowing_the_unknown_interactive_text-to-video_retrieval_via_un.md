---
title: >-
  [论文解读] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization
description: >-
  [ICCV 2025][文本视频检索] 本文提出UMIVR框架，显式量化文本视频检索中的三种不确定性——文本歧义（语义熵）、映射不确定性（JS散度）和帧不确定性（时序质量帧采样），基于量化的不确定性自适应生成澄清问题，迭代精炼查询，在MSR-VTT-1k上经10轮交互达到69.2% R@1。
tags:
  - ICCV 2025
  - 文本视频检索
  - 不确定性量化
  - 交互式检索
  - 语义熵
  - 帧质量采样
---

# Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization

**会议**: ICCV 2025  
**arXiv**: [2507.15504](https://arxiv.org/abs/2507.15504)  
**代码**: [GitHub](https://github.com/bingqingzhang/umivr)  
**领域**: others  
**关键词**: 文本视频检索, 不确定性量化, 交互式检索, 语义熵, 帧质量采样

## 一句话总结
本文提出UMIVR框架，显式量化文本视频检索中的三种不确定性——文本歧义（语义熵）、映射不确定性（JS散度）和帧不确定性（时序质量帧采样），基于量化的不确定性自适应生成澄清问题，迭代精炼查询，在MSR-VTT-1k上经10轮交互达到69.2% R@1。

## 研究背景与动机

1. **领域现状**: 文本视频检索（TVR）从注意力机制发展到视觉-语言预训练，CLIP4Clip、HunYuan等方法不断推进性能。交互式检索系统通过澄清问题精炼用户意图。
2. **现有痛点**: TVR面临三类不确定性：(1)文本歧义——查询模糊/不完整/多义；(2)映射不确定性——清晰查询仍可能匹配多个相似视频；(3)帧不确定性——运动模糊/失焦等低质量帧遮蔽关键视觉线索。现有交互式方法依赖启发式问题生成，未显式量化这些不确定性。
3. **核心矛盾**: 不确定性的类型不同，需要不同的干预策略，但现有方法用同质化的策略处理异质不确定性。
4. **本文目标**: 如何显式量化不同类型不确定性并据此自适应生成最有效的澄清问题？
5. **切入角度**: 将三种不确定性分别映射到可计算的数学度量（语义熵、JS散度、图像质量评估），全部无需训练。
6. **核心 idea**: 用语义熵量化文本歧义、JS散度量化映射不确定性、时序质量帧采样缓解帧不确定性,据此分级自适应生成澄清问题。

## 方法详解

### 整体框架
UMIVR使用统一的VideoLLaVA架构同时实现视频检索、字幕生成、视频问答和澄清问题生成。离线预处理(TQFS帧采样+字幕生成)→在线交互(不确定性量化→自适应问题生成→用户反馈→查询精炼→检索)。

### 关键设计

1. **文本歧义分数（TAS, Text Ambiguity Score）**:
    - 功能: 量化查询文本的语义不确定性
    - 核心思路: 对检索库中视频生成字幕并编码，给定查询x检索top-K最相似字幕，聚类为$M$组$\{c_j\}$。计算聚类概率$p(c_j|x)$为该组相似度占比。语义熵: $SE(x) = -\sum_{j=1}^{M} p(c_j|x) \log p(c_j|x)$，归一化到$[0,1]$。高TAS意味着查询语义分散。
    - 设计动机: 相比token级启发式方法，语义熵在语义层面聚合后计算，不因词汇变体导致虚高。聚类将近义表达合并为同一语义簇。

2. **映射不确定性分数（MUS, Mapping Uncertainty Score）**:
    - 功能: 量化查询与候选视频间相似度分布的尖锐程度
    - 核心思路: 取top-k相似度分数，先做均值中心化和平方归一化得到分布$p$: $p_i = \frac{\max(s_i - \bar{s}, 0)^2}{\sum \max(s_j - \bar{s}, 0)^2}$。定义理想one-hot分布$q$（第1位为1）。计算JS散度: $MUS(x) = \frac{JSD(p \| q)}{JSD_{\max}}$。高MUS意味着分数分布平坦，难以区分最相关视频。
    - 设计动机: JS散度有界且对称，比KL散度更稳健。平方归一化突出高置信候选，抑制低置信噪声。

3. **时序质量帧采样器（TQFS）**:
    - 功能: 即插即用地选择高质量且时序分散的视频帧
    - 核心思路: 三步策略——(1)低帧率均匀采样得$N$帧；(2)用无参考图像质量评估$Q(\cdot)$（如拉普拉斯方差或BRISQUE）评分每帧；(3)将视频分为$M$个时间段，每段选最高质量帧$F_m^* = \arg\max_{F_i \in \mathcal{I}_m} Q(F_i)$；(4)对候选帧提取语义嵌入，K-means聚类后每簇选最高质量帧，按时间排序得最终$K$帧。
    - 设计动机: 均匀采样可能包含模糊帧。TQFS兼顾视觉清晰度和语义多样性，可作为任何TVR模型的插件使用。

### 损失函数 / 训练策略
UMIVR是**无训练**框架——TAS、MUS、TQFS均为推理时计算的度量。统一VideoLLaVA架构在检索/字幕/QA上使用预训练权重。自适应问题生成策略：TAS高→开放式澄清问题（要求描述外观/活动）；TAS低+MUS高→针对性区分问题（利用候选视频元信息）；两者都低→丰富性问题（进一步充实描述）。

## 实验关键数据

### 主实验

| 方法 | R@1↑ | R@5↑ | Hit@1↑ | MnR↓ |
|--------|------|------|----------|------|
| HunYuan (非交互SOTA) | 62.9 | 84.5 | 62.9 | 9.3 |
| UMIVR round 0 | 43.1 | 66.1 | 43.1 | 22.4 |
| UMIVR round 3 | 61.3 | 84.1 | 68.9 | 8.1 |
| UMIVR round 6 | 65.9 | 87.7 | 76.0 | 5.9 |
| UMIVR round 10 | **69.2** | **89.0** | **80.0+** | — |

3轮交互即超越非交互SOTA，6轮超越所有方法。

### 消融实验

| 配置 | R@1 (round 3) | 说明 |
|------|---------|------|
| Full UMIVR | 61.3 | 完整框架 |
| w/o TAS | 下降 | 无文本歧义感知 |
| w/o MUS | 下降 | 无映射不确定性感知 |
| w/o TQFS | 下降 | 均匀帧采样 |
| TQFS as plug-in (CLIP4Clip) | 提升 | TQFS可即插即用增强其他模型 |

### 关键发现
- 3轮交互即可超越大多数非交互方法，显示不确定性驱动提问的高效性
- TQFS作为独立模块可提升CLIP4Clip等基线4-5%的R@1
- UMIVR还可直接扩展到交互式文本-图像检索场景
- 用户响应可通过VideoQA模块模拟，也可接入真实用户

## 亮点与洞察
- 三种不确定性的显式分离和独立量化是核心创新：不同类型用不同数学工具
- TAS(语义熵)+MUS(JS散度)的分层决策树简洁优雅
- TQFS的即插即用特性使其有独立价值
- 用单一VideoLLaVA替代传统多模型集成，显著简化系统架构

## 局限与展望
- VideoLLaVA作为基础模型的检索性能起点较低（round 0仅43.1）
- 模拟用户响应可能与真实用户偏差
- 10轮交互在实际应用中可能过多，需要探索更少轮次的上限
- 语义熵的聚类数$M$选择影响TAS精度

## 相关工作与启发
- **vs PlugIR**: 基于ChatGPT的混合云方案，开销大
- **vs TAM/UATVR**: 仅考虑单一不确定性类型
- **vs D2V**: 交互式基线但不显式量化不确定性

## 评分
- 新颖性: ⭐⭐⭐⭐ 不确定性分类与量化的结合具有理论优雅性
- 实验充分度: ⭐⭐⭐⭐⭐ MSR-VTT/AVSD/MSVD/ActivityNet四数据集+TQFS迁移实验
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法论严谨
- 价值: ⭐⭐⭐⭐ 无训练、即插即用的不确定性量化对交互式检索有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](../../CVPR2025/video_generation/video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)
- [The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](../../CVPR2025/video_generation/the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)
- [Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](../../ACL2025/video_generation/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)
- [RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](../../CVPR2025/video_generation/interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)

<!-- RELATED:END -->
