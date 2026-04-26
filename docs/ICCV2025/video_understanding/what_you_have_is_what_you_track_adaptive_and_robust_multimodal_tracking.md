---
title: >-
  [论文解读] What You Have is What You Track: Adaptive and Robust Multimodal Tracking
description: >-
  [ICCV 2025][视频理解][多模态跟踪] 提出FlexTrack——首个系统研究**时序性不完整多模态数据**下跟踪性能的框架，通过异构MoE融合机制（HMoE）实现自适应计算复杂度，配合视频级masking训练策略，在9个基准上取得SOTA，完整模态提升2.6%，缺失模态场景提升10.2%。
tags:
  - ICCV 2025
  - 视频理解
  - 多模态跟踪
  - 缺失模态
  - Mixture-of-Experts
  - 视频级Masking
  - 自适应复杂度
---

# What You Have is What You Track: Adaptive and Robust Multimodal Tracking

**会议**: ICCV 2025  
**arXiv**: [2507.05899](https://arxiv.org/abs/2507.05899)  
**代码**: 即将公开  
**领域**: 视频理解 / 目标跟踪 / 多模态融合  
**关键词**: 多模态跟踪, 缺失模态, Mixture-of-Experts, 视频级Masking, 自适应复杂度

## 一句话总结

提出FlexTrack——首个系统研究**时序性不完整多模态数据**下跟踪性能的框架，通过异构MoE融合机制（HMoE）实现自适应计算复杂度，配合视频级masking训练策略，在9个基准上取得SOTA，完整模态提升2.6%，缺失模态场景提升10.2%。

## 研究背景与动机

多模态跟踪（如RGB+深度、RGB+热红外、RGB+事件相机）通过互补信息提升遮挡、光照变化等场景下的鲁棒性。然而现实中**传感器同步失败**是常态：

### 核心问题：时序性模态缺失

不同传感器的**曝光时间和帧率差异**使得完美同步极为困难。现有数据集（如LasHeR、DepthTrack）假设模态总是成对存在，但这在实际应用中不成立——模态可能在时间窗口内间歇性缺失。

### 现有方法的局限

**ViPT、UnTrack**：统一架构但完全不考虑缺失情况

**IPT**：首个处理缺失模态的方法，但为每种缺失模式设计**独立的prompt策略**，架构僵化

**核心反思**：当模态缺失时，模型是否应维持与完整模态相同的计算复杂度？——答案是否定的

FlexTrack的核心理念：**跟踪器应根据数据可用性动态调整计算资源**——缺失越多、启用越简单的专家；数据越完整、启用越复杂的专家。

### 与现有方法的对比

| 方法 | 统一架构 | 统一参数 | 缺失模态 | 自适应复杂度 |
|------|---------|---------|---------|------------|
| IPT | ✗ | ✗ | ✓ | ✗ |
| ViPT | ✓ | ✗ | ✗ | ✗ |
| UnTrack | ✓ | ✓ | ✗ | ✗ |
| **FlexTrack** | **✓** | **✓** | **✓** | **✓** |

## 方法详解

### 整体框架

FlexTrack接受RGB和辅助模态（深度/热红外/事件，统称X）的视频片段和搜索区域作为输入，包含两个核心组件：

1. **HMoE融合模块**：推理时动态调整复杂度
2. **视频级Masking策略**：仅训练时使用，增强时序鲁棒性

### 异构Mixture-of-Experts融合 (HMoE-Fuse)

与传统MoE中所有专家**同构**（相同架构和大小）不同，HMoE的每个专家具有**不同的隐藏层维度** $2^d$，其中 $d \in \{2, \ldots, D-1\}$。

**路由机制**：与token级路由不同，HMoE将**整个视频片段**路由到专家，确保时序一致性：

$$g_n = \begin{cases} \text{Softmax}(G(T_v))_n, & \text{if } G(T_v)_n \in \text{Top-K}(G(T_v)) \\ 0, & \text{otherwise} \end{cases}$$

$$T_y^1 = \sum_{n=1}^{M} g_n E_n(T_v)$$

其中 $K=2$，输出为激活专家的加权和。

**直觉**：当模态完整（信息量大）时，门控函数倾向于激活**复杂专家**（高维隐藏层）；当模态缺失（信息量少）时，倾向于激活**简单专家**——实现自适应计算复杂度。

HMoE-Fuse的完整流程还包括线性注意力变换：

$$T_y^2 = T_v W_1 (T_v W_2)^T$$
$$T_y^3 = T_y^1 (T_y^2)^T$$
$$T_y^4 = \text{Softmax}(T_v W_3) T_y^3$$
$$T_y^5 = T_y^4 W_4$$

### 多模态视频级Masking策略

现有masking方法不适合多模态跟踪：

- **随机/MAE式masking**：破坏空间完整性和时序连续性
- **VideoMAE tube masking**：保持时序一致但同一空间位置在所有帧持续缺失
- **关键洞察**：跟踪任务同时需要**空间完整性**（理解目标外观）和**时序连续性**（捕捉动态变化）

FlexTrack的解决方案（Algorithm 1）：

1. **搜索区域masking**：从5种预定义模式中随机选择（两种模态都保留/RGB保留-X缺失/X保留-RGB缺失等），以概率3/5保持完整
2. **视频片段masking**：以概率 $\alpha$ 触发，对每个clip独立随机选择模态缺失模式（完整/仅RGB/仅X）
3. **核心保证**：每个时刻至少保留一种模态的完整空间信息

### 损失函数

$$\mathcal{L}_{all} = \lambda_1 \mathcal{L}_{aux} + \lambda_2 \mathcal{L}_{cls} + \lambda_3 \mathcal{L}_{l1} + \lambda_4 \mathcal{L}_{GIoU}$$

其中 $\mathcal{L}_{aux} = \mathcal{L}_{balance} + \mathcal{L}_{important}$（来自Switch Transformer），确保专家负载均衡。默认 $\lambda_1=1, \lambda_2=5, \lambda_3=2, \lambda_4=1$。

## 实验

### 完整模态结果

**表2：RGB-Event跟踪（VisEvent）**

| 方法 | P | AUC |
|------|---|-----|
| ViPT | 75.8 | 59.2 |
| UnTrack | 75.5 | 58.9 |
| STTrack | 78.6 | 61.9 |
| **FlexTrack** | **81.4** | **64.1** |

**表3：RGB-Thermal跟踪**

| 方法 | LasHeR AUC | RGBT234 MSR |
|------|-----------|------------|
| ViPT | 52.5 | 61.7 |
| STTrack | 60.3 | 66.7 |
| SUTrack | 59.5 | 69.5 |
| **FlexTrack** | **62.0** | **69.9** |

**表4：RGB-Depth跟踪（DepthTrack）**

| 方法 | F-score | Re | Pr |
|------|---------|----|----|
| SUTrack | 65.1 | 65.7 | 64.5 |
| STTrack | 63.3 | 63.4 | 63.2 |
| **FlexTrack** | **67.0** | **66.9** | **67.1** |

在9个完整模态基准上全面SOTA，平均超越前最佳方法**2.6%**。

### 缺失模态结果

在IPT创建的缺失模态基准上（random/switched/prolonged missing），FlexTrack相对前最佳方法提升**10.2%**，验证了HMoE自适应机制的有效性。

### 消融实验

视频级masking的贡献：不使用masking时，模型在缺失模态测试中性能大幅下降，证明训练时模拟缺失是必要的。HMoE的异构设计：统一大小专家的性能低于异构设计，说明不同复杂度专家的分工确实发生了。

## 亮点与洞察

1. **首个系统性研究**：时序性不完整多模态跟踪是真实且重要的问题，此前被数据集设计掩盖
2. **HMoE的"不等复杂度"设计**直觉清晰：信息少→简单处理，信息多→复杂处理。模型不仅适应缺失率，还能自适应场景复杂度
3. **视频级masking的三重保障**：空间完整性+时序连续性+至少一种模态可用
4. 统一框架处理RGB-D、RGB-T、RGB-Event三种模态组合，**联合训练**进一步提升泛化

## 局限性

1. HMoE的专家数量和维度设计依赖超参数，对不同模态组合可能需要调整
2. 训练时的masking策略（5种搜索区域模式、3种clip模式）是手工设计的，理想情况下应从数据分布中自适应学习
3. 仅验证了双模态场景（RGB+1），对三种及以上模态的扩展性未知
4. 推理时masking关闭，但实际部署中模态缺失是在线发生的——模型的实时自适应能力需更多验证

## 相关工作

- **多模态跟踪**: ViPT, UnTrack, SDSTrack, OneTracker, STTrack
- **缺失模态**: IPT, FuseMoE, Flex-MoE
- **MoE**: DeepSeekMoE, Switch Transformer

## 评分

- 创新性：⭐⭐⭐⭐ — 异构MoE+视频级masking的组合是多模态跟踪的新范式
- 实用性：⭐⭐⭐⭐⭐ — 直接面向传感器不完美的现实场景，工程价值高
- 实验充分度：⭐⭐⭐⭐⭐ — 9个基准、完整+缺失两种设置、详细消融
- 写作质量：⭐⭐⭐⭐ — 问题定义清晰，masking策略的伪代码直观

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ICCV 2025\] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions](umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_of_multi_modal_llms_via_token_merging_and_pruning.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)

<!-- RELATED:END -->
