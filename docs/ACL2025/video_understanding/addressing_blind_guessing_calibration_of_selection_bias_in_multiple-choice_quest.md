---
title: >-
  [论文解读] Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models
description: >-
  [ACL 2025][视频理解][选择偏差] 首次系统性研究视频语言模型（VLM）在多选题回答中的选项选择偏差问题，通过任务分解分析偏差来源，提出BOLD后处理校准技术，在减少偏差的同时提升模型性能。
tags:
  - "ACL 2025"
  - "视频理解"
  - "选择偏差"
  - "视频语言模型"
  - "多选题"
  - "去偏校准"
  - "MCQA"
---

# Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models

**会议**: ACL 2025  
**arXiv**: [2410.14248](https://arxiv.org/abs/2410.14248)  
**代码**: [github.com/ologin/BOLD](https://github.com/ologin/BOLD)  
**领域**: 视频理解  
**关键词**: 选择偏差, 视频语言模型, 多选题, 去偏校准, MCQA

## 一句话总结

首次系统性研究视频语言模型（VLM）在多选题回答中的选项选择偏差问题，通过任务分解分析偏差来源，提出BOLD后处理校准技术，在减少偏差的同时提升模型性能。

## 研究背景与动机

多选题问答（MCQA）因其评估的透明性和便利性，被广泛用于衡量视频语言模型的推理能力。然而，VLM在回答MCQA时存在严重的**选择偏差（selection bias）**——模型倾向于根据训练中观察到的位置模式不成比例地偏好某些选项位置，而不是基于内容进行真正的推理。

虽然选择偏差在文本LLM中已有较多研究，在图像-文本VLM中也有一些工作，但在**视频语言模型**中几乎完全未被探索。视频语言模型由于依赖复杂的视觉输入、需要时空推理能力，具有独特的挑战。现有的去偏方法（如选项打乱）需要对所有排列组合进行多次推理，计算成本极高。

本文的核心贡献在于：(1) 首次全面分析VLM-MCQA中的选择偏差；(2) 提出一种计算效率高的后处理去偏方法BOLD。

## 方法详解

### 整体框架

方法分为两个阶段：首先通过11种数据集修改设定系统性地分析偏差在不同VLM和数据集上的表现模式；然后提出基于任务分解的BOLD（Bias Optimisation Leveraging Decomposition）去偏方法。

### 关键设计

1. **系统性偏差分析（11种数据修改设定）**: 将MCQA分解为视频、问题、答案选项三个核心组件，设计了11种修改设定：
   
   **视频修改**：正确帧（仅提供答案所在帧）、空帧（黑帧）
   
   **问题修改**：重述问题（Llama3改写5次随机选1）、空问题（空字符串）
   
   **答案修改**：选项打乱、正确答案置于每个位置、正确答案固定+打乱、添加空选项、所有选项相同、所有选项都正确、空选项

   关键发现：准确率相关设定（正确帧、改写、打乱）的选项分布与默认设定几乎一致，说明模型更依赖选项位置而非内容；准确率无关设定（相同选项、空设定）清晰暴露了偏差模式。

2. **偏差公平性指标适配**: 将公平性研究中的指标适配到VLM-MCQA评估：

    - **F1_std**：各选项F1分数的标准差，高值表示对不同选项性能不一致
    - **Recall_std**：各选项召回率的标准差，高值说明特定位置的正确答案更容易被识别
    - **JS_std**：预测分布与真实分布之间Jensen-Shannon距离的标准差，检测分布不一致性

3. **BOLD去偏方法**: 核心思路是将观测到的预测分布 $P_o$ 分解为先验偏差分布 $P_p$ 和去偏分布 $P_d$：
   
    $P_o(d_i|T) = \frac{1}{Z_T} P_p(d_i|T) \times P_d(d_i|T)$
   
   **偏差估计**：通过三种"攻击"使任务变为ill-defined（移除视频/问题/选项），此时理想模型应均匀选择。由此估计先验偏差：
   
    $\tilde{P}_p(d_i|T) = softmax\left(\sum_j P_p(d_i|A_j(T))\right)$
   
   **全局先验计算**：从数据集中采样 $K = k \times ||D||$ 个样本（$k=0.5$最优），对每个样本分别进行三种攻击，平均得到全局先验。
   
   **去偏推断**：
    $P_d(d_i|T) = softmax\left(\log P_o(d_i|T) - \log \tilde{P}_p(d_i)\right)$

4. **Weighted_BOLD扩展**: 对三种分解的先验引入可学习权重 $w_j$，通过5折交叉验证和COBYLA优化器优化权重，以更精确地估计偏差方向。权重约束为 $0 \leq w_i \leq 1$ 或 $|w_i| \leq 1$。

### 损失函数 / 训练策略

BOLD是后处理方法，不涉及模型重训练。Weighted_BOLD的权重优化使用COBYLA约束优化算法，目标是最小化验证集上的Recall_std。

## 实验关键数据

### 主实验

| 模型 | 数据集 | Accuracy | F1 Mean |
|------|--------|----------|---------|
| Video-LLaMA | NExT-QA | 40.85 | 40.85 |
| Video-LLaVA | NExT-QA | 49.96 | 49.81 |
| SeViLA | NExT-QA | 63.78 | 63.88 |
| Video-LLaMA | STAR | 36.59 | 31.86 |
| Video-LLaVA | Perception Test | 40.73 | 35.69 |
| SeViLA | Video-MME | 39.85 | 39.82 |

### 消融实验

**BOLD/Weighted_BOLD去偏效果（k=0.5, Video-LLaVA）：**

| 数据集 | 方法 | Acc提升 | F1提升 | Recall_std↓ | F1_std↓ |
|--------|------|---------|--------|-------------|---------|
| NExT-QA | BOLD | +3.72% | +3.82% | -18.63% | -18.42% |
| NExT-QA | W-BOLD | +4.39% | +4.49% | -22.02% | -19.83% |
| STAR | BOLD | +7.01% | +12.37% | -26.85% | -27.15% |
| STAR | W-BOLD | +7.94% | +13.69% | -30.15% | -28.03% |
| Perc. Test | W-BOLD | +3.02% | +10.58% | -28.26% | -30.53% |

### 关键发现

- **偏差模式因模型而异**：Video-LLaMA强烈偏好a1且忽略后面的选项；Video-LLaVA对a1的偏好更强；SeViLA最鲁棒，在NExT-QA上接近均匀20%分布
- **去偏同时提升性能**：这一发现颇为出乎意料，BOLD不仅减少偏差指标，还在大多数情况下提升了Accuracy和F1
- **Weighted_BOLD优于BOLD**：权重化的分解先验能更精确地捕捉偏差结构
- **SeViLA的鲁棒性**：由于采用argmax函数和专门的QA训练，SeViLA本身偏差较小，去偏效果也相应较有限
- **k=0.5是最优样本比例**：不需要在全部数据上估计偏差，50%的样本即可获得良好效果
- **偏差具有跨数据集一致性**：同一模型在不同数据集上展现相似但非完全相同的偏差模式

## 亮点与洞察

- **首次系统性研究**：填补了视频MCQA选择偏差研究的空白，11种修改设定的实验设计非常全面
- **去偏即性能提升**：打破"减少偏差=牺牲准确率"的直觉，说明偏差确实在阻碍真实推理
- **计算高效**：相比传统打乱方法需要$n!$次排列推理，BOLD只需3次额外推理（空视频/空问题/空选项）加50%的样本
- **强理论基础**：将偏差分解为三个正交平面上的投影，数学框架清晰
- **使用非常规选项名**：用a0/a1/a2代替A/B/C以减少token偏差的干扰，实验设计细节体现了严谨性

## 局限与展望

- 仅测试了3个VLM模型，未涵盖最新的大规模VLM（如GPT-4o、Gemini等闭源模型）
- 假设选项之间相互独立、无逻辑重叠，在某些实际基准中可能不完全成立
- Weighted_BOLD需要交叉验证来优化权重，增加了一定的实现复杂度
- SeViLA上的去偏效果有限甚至略有下降，说明方法对本身偏差较小的模型适用性有限
- 未探索偏差与模型规模、训练数据分布之间的关系
- 可以进一步研究偏差在微调阶段的产生机制

## 相关工作与启发

- PriDe（Zheng et al., 2024a）通过打乱分离偏差和无偏结果，但计算开销大
- Zhang et al.（2024）对图像MCQA切断视觉输入来分析偏差，本文扩展到视频并增加了多平面分解
- Wang et al.（2024a）通过数据增强解决偏差，但需要增加更多QA对
- 本文的分解思路可以推广到其他多模态任务的公平性分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次研究视频VLM的MCQA偏差，分解去偏方法有理论创新
- 实验充分度: ⭐⭐⭐⭐⭐ 11种设定×3个模型×4个数据集，分析极为详尽
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，数学公式化规范
- 价值: ⭐⭐⭐⭐ 对MCQA评估的可靠性提出了重要质疑，去偏方法实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](../../CVPR2025/video_understanding/question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](../../CVPR2025/video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->
