---
title: >-
  [论文解读] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?
description: >-
  [ACL 2025][多模态][视觉数学推理] 本文提出We-Math基准，首次通过将复合数学问题按知识概念分解为子问题，引入IK/IG/CM/RM四维指标来层次化评估LMM的推理过程（而非仅看最终结果），揭示了LMM普遍存在知识不足（IK）问题，且GPT-4o是首个从IK阶段迈入知识泛化（IG）阶段的模型。
tags:
  - ACL 2025
  - 多模态
  - 视觉数学推理
  - benchmark
  - 知识概念分解
  - 过程评估
  - 大规模多模态模型
---

# We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?

**会议**: ACL 2025  
**arXiv**: [2407.01284](https://arxiv.org/abs/2407.01284)  
**代码**: [We-Math/We-Math](https://github.com/We-Math/We-Math)  
**领域**: 多模态VLM  
**关键词**: 视觉数学推理, benchmark, 知识概念分解, 过程评估, 大规模多模态模型

## 一句话总结
本文提出We-Math基准，首次通过将复合数学问题按知识概念分解为子问题，引入IK/IG/CM/RM四维指标来层次化评估LMM的推理过程（而非仅看最终结果），揭示了LMM普遍存在知识不足（IK）问题，且GPT-4o是首个从IK阶段迈入知识泛化（IG）阶段的模型。

## 研究背景与动机
视觉数学推理是多模态大语言模型（LMM）的基础能力之一，受到广泛关注。然而，现有benchmark存在核心问题：

**结果导向的评估方式**：MathVista、MathVerse等基准仅关注最终答案的正确性，忽略了推理过程中的内在问题，导致反直觉的评估结论（如MathVista发现LMM在大学级别问题上表现优于小学级别）

**无法诊断推理能力的真正缺陷**：正确答案不代表模型真正掌握了推理方法，错误答案也不代表模型缺乏基础知识

本文提出两个关键问题：
- Q1：正确答案是否真正反映了LMM准确推理该类问题的能力？
- Q2：错误答案是否意味着LMM缺乏推理过程中的基础知识？

切入角度：借鉴人类数学学习方式——通过逐步掌握和泛化知识概念来解决复杂问题。将复合问题分解为基于知识概念的子问题，通过对比子问题和原始问题的结果来诊断LMM的推理缺陷。

核心 idea：**基于知识概念的问题分解 + 四维推理评估指标，从"做对/做错"走向"为什么做对/做错"**。

## 方法详解

### 整体框架
We-Math的构建与评估流程：
1. 构建层次化知识结构（5层67个知识概念）
2. 收集并标注6.5K视觉数学问题
3. 将复合问题按知识概念分解为独立子问题
4. 同时用LMM解答子问题和原始问题
5. 根据结果组合进行四维分类评估

### 关键设计

1. **层次化数据集构建**:

    - 将数学问题分为5大类：平面图形、立体图形、图形变换与运动、位置与方向、测量
    - 进一步分解为12种典型问题和67个终端知识概念
    - 严格控制每个终端节点包含10-40个样本，确保数据均衡
    - 所有6.5K问题来源于公开权威数学网站，3名专家标注知识概念，交叉验证确保标注一致性
    - 按解题步数分为一步、两步和三步问题

2. **基于知识的问题分解**:

    - 对于含M个知识概念的复合问题，由专家逐步分解为M个一步子问题
    - 每个子问题只涉及单一知识概念
    - 通过递归计算条件c_i^m来保持逻辑连贯性：c_i^m = c_i^{m-1} + a_i^{m-1}
    - 约束条件：最后一个子问题的题目和答案等于原始问题的题目和答案
    - 选取1.5K高质量多知识概念问题进行分解

3. **四维推理评估指标（核心贡献）**:

    - **IK（知识不足）**：部分子问题错误 + 复合问题错误。模型对单一知识概念掌握不足
    - **IG（泛化不足）**：所有子问题正确 + 复合问题错误。模型掌握了单个知识但难以综合运用
    - **CM（完全掌握）**：所有子问题正确 + 复合问题也正确。结果可靠且准确
    - **RM（死记硬背）**：部分子问题错误 + 复合问题正确。违背人类逻辑推理直觉
    - 建立推理能力层次：IK < IG < CM
    - 引入宽松/严格两种RM判定标准
    - 最终推理置信度得分：Score_avg = α·S_IK + β·S_IG + S_CM（默认α=0, β=0.5）

4. **知识概念增强策略（KCA）**:

    - 针对IK问题的启发式解决方案
    - 为67个知识概念创建知识卡片，内容来源于欧几里得《几何原本》、Wikipedia和教科书
    - 每张卡片包含精确定义和关键知识提示
    - 在推理时将相关知识卡片提供给LMM

### 损失函数 / 训练策略
本文为评估基准，不涉及模型训练。评估采用：
- testmini集：1740样本（1215一步 + 360两步 + 165三步）
- 统一转换为多选题格式，使用正则匹配预测
- 引入额外的"不确定"选项防止模型从选项推断答案

## 实验关键数据

### 主实验

| 数据集 | 指标 | GPT-4o | LLaVA-NeXT-110B | InternVL-Chat-V1.5 |
|--------|------|------|----------|------|
| S1(一步) | Accuracy | 72.84% | 53.74% | 49.38% |
| S2(两步) | Accuracy | 58.06% | 36.94% | 30.56% |
| S3(三步) | Accuracy | 43.64% | 31.52% | 28.48% |

### 四维指标（严格指标）

| 模型 | AVG↑ | IK↓ | IG↓ | CM↑ | RM↓ |
|------|---------|------|------|------|------|
| GPT-4o | 42.86% | 31.24% | 15.24% | 35.24% | 34.16% |
| LLaVA-NeXT-110B | 19.24% | 50.29% | 14.48% | 12.00% | 65.95% |
| LLaVA-1.6-7B | 3.33% | 78.29% | 2.48% | 2.10% | 89.11% |

### KCA效果

| 模型 | 无KCA (AVG strict) | 有KCA (AVG strict) | 说明 |
|------|---------|------|------|
| 所有LMM均值 | - | - | KCA显著缓解IK问题但对IG改善有限 |

### 关键发现
- **知识概念数与性能负相关**：GPT-4o准确率从S1的72.84%骤降至S3的43.64%，其他模型降幅更大
- **IK是LMM最大的薄弱环节**：几乎所有LMM都显示严重的知识不足问题，小模型尤为突出（LLaVA-1.6-7B和DeepSeek-VL-1.3B超过350个IK问题）
- **GPT-4o率先进入知识泛化阶段**：其IK问题相比LLaVA-NeXT-110B减少19.05%，但IG比例最高，说明核心挑战从知识获取转向知识泛化
- **RM问题普遍存在**：多数开源LMM在宽松指标下仍有约25%的RM比例，G-LLaVA-13B高达35.98%。GPT-4o在宽松指标下RM<2%
- **参数压缩潜力大**：Phi3-Vision-4.2B和MiniCPM-Llama3-V 2.5在更小参数量下展现出与LLaVA-NeXT-72B相当的性能
- **LMM擅长计算但在精细测量上困难**：角度和长度的测量、单位换算是痛点

## 亮点与洞察
- 四维评估指标设计精巧，从"知识不足→泛化不足→完全掌握"建立了清晰的推理能力层次
- "死记硬背"（RM）指标揭示了一个反直觉现象：模型能解复合题却解不了子问题，质疑了当前LMM是否真正具备数学推理能力
- 严格/宽松两种RM标准的设计考虑了模型不稳定性的影响
- KCA策略验证了知识补充对缓解IK问题的有效性，但泛化能力需要更根本的改进
- 错误分析发现视觉错误集中在特定概念（如"角度理解"），指出需要增强视觉编码器的细粒度测量能力

## 局限与展望
- benchmark主要覆盖初等数学，高等数学和更复杂的推理类型未涉及
- 问题分解目前依赖人工专家标注，扩展性有限。可探索利用LLM辅助自动分解
- 四维指标中RM的定义较为二元化，缺乏对"部分RM"情况的连续度量
- 仅评估了选择题格式，未覆盖开放式回答的推理评估
- 可进一步研究如何从IG阶段提升到CM阶段的训练策略

## 相关工作与启发
- MathVista、MathVerse等结果导向benchmark：We-Math通过知识分解提供了过程级评估的互补视角
- GSM8K、MATH等文本推理benchmark：We-Math将类似评估思想扩展到视觉数学推理
- 与AR-MCTS论文形成互补：We-Math作为评估工具，AR-MCTS作为解决方案
- 课程学习思想的应用：从子问题到复合问题的渐进评估与课程学习理念一致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 四维推理评估指标是全新贡献，基于知识概念的问题分解评估方式开创先河
- 实验充分度: ⭐⭐⭐⭐⭐ 17个LMM的全面评估，覆盖闭源和开源模型，多维度分析深入
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，图表丰富，但部分公式与文字描述有冗余
- 价值: ⭐⭐⭐⭐⭐ 改变了多模态数学推理评估的范式，四维指标和知识结构对社区有长期参考价值

<!-- RELATED:START -->

## 相关论文

- [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](../../ECCV2024/multimodal_vlm/mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)
- [The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math?](../../ECCV2024/multimodal_vlm/mathverse_does_your_multimodal_llm_truly_see_the_diagrams_in.md)
- [OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)

<!-- RELATED:END -->
