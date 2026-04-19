---
title: >-
  [论文解读] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension
description: >-
  [CVPR 2026][多模态VLM][指代表达理解] 提出VIRO框架，在神经符号推理管道中嵌入算子级验证机制（不确定性验证+逻辑验证），解决组合推理中的级联错误和强制预测问题，实现无目标场景的稳健检测，零样本设置下在gRefCOCO无目标分割和标准REC基准上达到SOTA。
tags:
  - CVPR 2026
  - 多模态VLM
  - 指代表达理解
  - 神经符号推理
  - 零样本学习
  - 开放词汇检测
  - CLIP验证
---

# VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension

**会议**: CVPR 2026  
**arXiv**: [2601.12781](https://arxiv.org/abs/2601.12781)  
**代码**: https://github.com/ml-postech/VIRO-neuro-symbolic-reasoning-with-verification (有)  
**领域**: 多模态VLM  
**关键词**: 指代表达理解, 神经符号推理, 算子级验证, 零样本学习, 无目标检测

## 一句话总结
VIRO在神经符号REC管道中嵌入轻量算子级验证机制（CLIP不确定性验证+空间逻辑验证），使每个推理步骤能自我验证并在无目标时提前终止，在零样本设置下以61.1%平衡准确率大幅超越组合推理baselines，同时保持0.3%以下的程序失败率和高效推理速度。

## 研究背景与动机

1. **领域现状**：指代表达理解（REC）旨在根据自然语言描述定位图像中的目标区域。近年来，基于LLM和VLM的神经符号方法通过将查询分解为结构化程序并逐步执行，实现了可解释推理和强大的零样本泛化能力。

2. **现有痛点**：现有组合推理管道假设每一步中间结果都是正确的，但实际上开放词汇检测器（OVD）经常产生高置信度的假阳性（FP）。这些错误会沿推理链级联传播，在没有目标的场景中尤其严重——系统被迫从假阳性中选择一个作为答案（"强制预测"问题）。

3. **核心矛盾**：现有方法缺乏中间推理步骤的验证机制。一方面，OVD生成的候选框可能是视觉或语义相似的假阳性；另一方面，空间关系推理也可能在不满足约束时仍然输出结果。此外，很多系统将大型多模态LLM放在推理内循环中，导致严重的延迟问题，且程序生成与执行紧耦合，每张图都需重新生成推理程序。

4. **本文目标** (a) 如何在推理步骤中嵌入验证以防止级联错误？(b) 如何在无目标场景中正确"弃权"而非强制预测？(c) 如何在保持准确性的同时提高效率和可扩展性？

5. **切入角度**：在每个推理算子内部集成轻量级验证模块——利用CLIP进行不确定性验证过滤OVD假阳性，利用几何测试进行逻辑验证检查空间关系是否成立。

6. **核心 idea**：在神经符号推理管道中让每个算子"先执行再验证"，验证不通过就返回空集并提前终止，从而实现稳健的无目标检测。

## 方法详解

### 整体框架

VIRO采用两阶段解耦管道：(1) 预执行阶段：LLM将自然语言查询翻译为由验证推理算子（VRO）组成的符号程序 $P = (o_1, o_2, \dots, o_T)$，并通过程序验证器确保语法正确；(2) 执行阶段：程序解释器在图像上逐步执行算子序列，每个算子执行后进行自我验证，如果验证不通过返回空集并立即终止整个管道。输出要么是定位到的边界框，要么是空集（表示无目标）。

### 关键设计

1. **验证推理算子（VROs）**:

    - 功能：定义有限算子集合作为推理基本构件，涵盖四类——识别算子（FIND、PROPERTY）、绝对空间算子（LOCATE、SIZE、ORDER）、相对空间算子（FIND_DIRECTION、FIND_NEAR、FIND_INSIDE）、终止算子（RESULT）
    - 核心思路：每个算子不仅执行推理动作，还自我验证执行结果。如果验证条件不满足，算子返回 $\varnothing$，触发整个管道的提前终止。这将REC输出形式化为 $Y = B$（有目标）或 $Y = \varnothing$（无目标）
    - 设计动机：通过在算子级别而非管道末端进行验证，能在最早的出错点阻断级联错误，同时实现高效的提前退出

2. **不确定性验证（UV）— FIND算子内**:

    - 功能：过滤OVD产生的高置信度假阳性候选框
    - 核心思路：对每个OVD候选框 $B_j$，裁剪对应图像区域 $I_j$，预定义 $K$ 个常见类别作为负锚点。计算CLIP验证分数 $S(l|I_j) = \frac{1}{K}\sum_{k=1}^K \frac{\exp(\text{sim}(I_j, l)/\tau)}{\exp(\text{sim}(I_j, l)/\tau) + \exp(\text{sim}(I_j, c_k)/\tau)}$，即候选标签与负锚点的平均二分类概率。分数低于阈值 $\delta_l$ 的候选框被过滤。为应对CLIP对不同标签的内在偏差，使用ImageNet做逐标签阈值校准
    - 设计动机：OVD（如GroundingDINO）在开放词汇场景下容易对视觉/语义相似但不正确的目标产生高置信度检测。CLIP作为二分类判别器计算开销极小，却能有效过滤这类假阳性

3. **逻辑验证（LV）— FIND_DIRECTION算子内**:

    - 功能：验证候选目标是否真正满足指定的空间关系约束
    - 核心思路：对所有输入候选进行几何测试，检查每个目标候选是否相对于参考目标满足指定的空间方向关系。不满足则返回空集
    - 设计动机：在空间推理步骤中，即使前面的FIND算子通过了验证，空间关系可能并不成立（例如"大象左边的人"但图中人并不在大象左边），通过逻辑验证可进一步过滤错误

4. **程序生成与验证**:

    - 功能：将自然语言查询可靠地转换为可执行符号程序
    - 核心思路：使用LLM（Qwen2.5-72B-Instruct-AWQ）通过few-shot prompting生成程序 $P = \text{LLM}(Q|m)$，然后通过程序验证器检查语法正确性。如果验证失败，提供简洁的诊断反馈触发LLM自我修正。受限的算子空间（固定符号结构 vs. ViperGPT的开放Python代码）大幅降低了运行时错误
    - 设计动机：LLM偶尔产生语法错误的程序，受限的算子集合+结构化验证循环使得程序失败率降至0.3%以下

5. **解耦式架构**:

    - 功能：程序生成与执行解耦，实现高效的1-query-N-images场景
    - 核心思路：VIRO只为每个查询生成一次程序，然后在所有 $N$ 张图像上复用，推理时间为 $T_{\text{total}} = T_{\text{pre}} + N \times T_{\text{exec}}$。相比之下，HYDRA和NAVER每张图都需重新生成程序
    - 设计动机：在机器人视觉搜索等实际应用中，同一查询需要在大量图像上执行，解耦设计使延迟线性增长而非乘法增长

## 实验关键数据

### 主实验

| 数据集/分割 | 指标 | VIRO | 之前SOTA (组合推理) | 提升 |
|--------|------|------|----------|------|
| gRefCOCO+RefCOCO TestA | Balanced Acc. | 61.1% | 35.2% (HYDRA) | +25.9 |
| gRefCOCO TestA | TNR (N-acc) | 50.2% | 7.5% (HYDRA) | +42.7 |
| RefCOCO TestA | TPR (Acc@0.5) | 71.9% | 66.7% (ViperGPT) | +5.2 |
| RefCOCO TestA | 程序失败率 | 0.07% | 3.45% (ViperGPT) | 大幅降低 |
| RefCOCO TestA | 执行延迟 | 0.71s | 1.49s (ViperGPT) | 2.1× 更快 |
| RefEgo (全帧) | ACC@0.5+n | 51.9% | 23.0% (ViperGPT) | +28.9 |

### 消融实验

| 配置 | Balanced Acc. | TNR | TPR | 说明 |
|------|---------|------|------|------|
| Detector-only | 40.0% | 22.8% | 57.1% | 仅OVD |
| + Operators | 56.8% | 38.9% | 74.6% | +组合推理算子 |
| + LV | 57.0% | 39.3% | 74.6% | +逻辑验证 |
| + UV (fixed) | 58.8% | 43.1% | 74.4% | +不确定性验证(固定阈值) |
| + UV (adaptive) | 61.1% | 50.2% | 71.9% | +自适应阈值(完整模型) |

### 关键发现
- 组合推理算子本身贡献最大（Balanced Acc +16.8），UV验证对无目标检测提升显著（TNR +11.3），但存在TPR-TNR trade-off
- 自适应阈值比固定阈值在TNR上提升7.1%但TPR下降2.5%，反映精度-召回权衡
- 在1-query-N-images场景中，VIRO和ViperGPT因解耦架构而优于HYDRA/NAVER，但VIRO执行延迟更低
- CLIP backbone选择：ViT-H/14比ViT-L/14 TPR高3.1%但执行延迟增加29%

## 亮点与洞察
- **算子级验证是关键创新**：不是在整个管道末端验证，而是在每个推理步骤内部"执行+验证"，这是同类方法中首次实现的设计，使得错误在源头被捕获
- **CLIP作为轻量二分类验证器**：巧妙利用CLIP的判别能力（而非通常的检索能力），通过与负锚点集合的逐一比较计算验证分数，计算开销极小但效果显著
- **无目标检测作为"弃权"而非"分类"**：通过空集返回机制自然实现，无需额外的无目标检测训练，这个设计理念可迁移到任何需要"拒绝回答"的视觉推理系统

## 局限与展望
- TPR和TNR存在固有trade-off——提升无目标检测能力会牺牲有目标情况下的准确率（TPR从74.6%降到71.9%）
- 依赖固定的算子集合，面对复杂查询时覆盖能力有限（如涉及动作、时间等更复杂语义时可能需扩展算子集）
- CLIP验证使用ImageNet校准阈值，对领域外数据的泛化能力有待验证
- 逻辑验证目前仅限于简单几何测试，更复杂的空间关系（遮挡、相对大小等）可能需更强的推理机制

## 相关工作与启发
- **vs ViperGPT**: ViperGPT生成开放Python代码但缺乏验证，程序失败率3.45%；VIRO用受限算子+验证器将失败率降到0.07%，同时TPR更高
- **vs HYDRA/NAVER**: 这两者将程序生成与执行紧耦合，导致每张图重新生成推理程序，且依赖大型多模态LLM，延迟和失败率都远高于VIRO
- **vs 监督式REC (GREC-UNINEXT)**: VIRO在无目标检测上接近甚至超越需要无目标标注训练的监督方法，展示了零样本方法的潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ 算子级验证+弃权机制是组合推理REC中的重要创新，但整体思路较直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多个标准REC基准+无目标基准+视频基准+效率+可扩展性+全面消融
- 写作质量: ⭐⭐⭐⭐ 清晰的问题定义和方法阐述，图表丰富
- 价值: ⭐⭐⭐⭐ 在实际应用中（机器人搜索、无目标检测）有重要意义，填补了组合推理方法的关键空白

<!-- RELATED:START -->

## 相关论文

- [Efficient Vision-Language Reasoning via Adaptive Token Pruning](../../NeurIPS2025/interpretability/efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](../../NeurIPS2025/interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)
- [SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)
- [DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration](../../ECCV2024/interpretability/detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)

<!-- RELATED:END -->
