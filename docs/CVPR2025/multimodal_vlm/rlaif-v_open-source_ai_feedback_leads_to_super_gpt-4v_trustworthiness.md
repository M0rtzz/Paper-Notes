---
title: >-
  [论文解读] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness
description: >-
  [CVPR 2025][多模态][AI反馈] RLAIF-V 提出一套完全基于开源MLLM的反馈对齐框架，通过去混淆的候选回复生成策略和分治式反馈标注方法来产生高质量偏好数据，并结合DPO迭代训练与自反馈推理时扩展，使7B模型幻觉率降低80.7%，12B模型仅用自身反馈即超越GPT-4V的可信度。
tags:
  - CVPR 2025
  - 多模态
  - AI反馈
  - 幻觉抑制
  - 偏好学习
  - 开源对齐
  - 推理时扩展
---

# RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness

**会议**: CVPR 2025  
**arXiv**: [2405.17220](https://arxiv.org/abs/2405.17220)  
**代码**: https://github.com/RLHF-V/RLAIF-V (有)  
**领域**: 多模态VLM  
**关键词**: AI反馈, 幻觉抑制, 偏好学习, 开源对齐, 推理时扩展

## 一句话总结

RLAIF-V 提出一套完全基于开源MLLM的反馈对齐框架，通过去混淆的候选回复生成策略和分治式反馈标注方法来产生高质量偏好数据，并结合DPO迭代训练与自反馈推理时扩展，使7B模型幻觉率降低80.7%，12B模型仅用自身反馈即超越GPT-4V的可信度。

## 研究背景与动机

MLLM存在严重的"自信地产生错误内容"的幻觉问题。基于人类反馈的RLHF依赖昂贵的人工标注且覆盖面有限；而现有RLAIF方法（如Silkie）依赖GPT-4V等闭源模型提供反馈，本质上是蒸馏闭源模型能力，不可持续且受限于闭源模型的天花板。

核心矛盾有二：(1) **反馈来源的不可行性**——社区缺乏关于如何用能力相当的开源模型构建高质量反馈的基础知识，简单替换闭源标注器为弱开源模型会导致反馈质量断崖式下降；(2) **推理时扩展的缺失**——现有工作聚焦偏好学习阶段而忽略推理阶段的反馈利用，盲目增加推理计算预算并不能提升性能。

RLAIF-V的核心idea：通过"去混淆采样"精准暴露回复间的可信度差异，再通过"分治"将复杂的回复评估拆解为简单的原子声明验证任务，使开源模型也能产生人类级别质量的反馈。

## 方法详解

### 整体框架

RLAIF-V包含四个阶段：(1) 去混淆候选回复生成——同一输入用不同随机种子多次采样产生候选回复；(2) 分治式反馈标注——将每个回复拆分为原子声明，逐个用开源MLLM验证可信度；(3) 迭代偏好学习——周期性更新反馈数据进行DPO训练；(4) 自反馈推理时扩展——利用对齐后模型自身作为奖励函数，结合长度归一化进行Best-of-N选择。

### 关键设计

1. **去混淆候选回复生成 (Deconfounded Response Generation)**:
    - 功能：消除偏好对中文本风格等混淆因素，精准暴露可信度差异
    - 核心思路：对同一输入 $x$（图像+提示），使用模型以相同解码参数、不同随机种子进行 $n$ 次采样解码，生成候选回复 $\{y_1, y_2, \cdots, y_n\}$。由于回复采自同一分布，它们共享相似的文本风格和语言模式
    - 设计动机：传统方法（如RLHF-V）中 $y_w$ 和 $y_l$ 来自不同来源（人工标注vs模型生成），包含大量非鲁棒浅层模式差异（如文字风格、用词习惯），模型可能学到这些捷径而非真正的可信度判断。去混淆采样使训练聚焦于内容层面的可信度差异

2. **分治式反馈标注 (Divide-and-Conquer Annotation)**:
    - 功能：将困难的"完整回复质量评估"拆解为简单的"原子声明验证"任务
    - 核心思路：**分(Divide)**——用LLM将回复 $y$ 拆分为原子声明 $\{c_1, c_2, \cdots, c_m\}$（排除观点和主观内容）。**治(Conquer)**——将每个声明转化为极性问题（如"图中时钟是否显示11:20?"），让开源MLLM输出 $p_{yes}$ 和 $p_{no}$ 概率。**合(Combine)**——统计 $p_{no} > p_{yes}$ 的声明数 $n_{rej}$，用 $-n_{rej}$ 作为回复总分。分数更高表示错误更少
    - 设计动机：开源MLLM在回答简单的yes/no问题时能力远强于评估复杂完整回复。通过任务分解，大幅降低了对标注器模型能力的要求。实验表明该方法达到96.7%的人类一致率，超过GPT-4V的VL-Feedback(92.3%)

3. **自反馈推理时扩展 (Self-Feedback for Inference-time Scaling)**:
    - 功能：利用DPO对齐后的模型自身作为奖励函数，在推理时进一步提升可信度
    - 核心思路：DPO训练目标隐式定义了奖励函数 $r(y) = \beta \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$。但直接用会偏好短回复，因此引入长度归一化：$r(y) = \frac{\beta}{T} \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$，其中 $T$ 是回复长度。基于该奖励进行Best-of-N选择
    - 设计动机：DPO目标公式化本身会导致对短回复的偏好，因为短回复的总累积分更容易为正。长度归一化通过平均化Token级分数来消除这种偏差。实验显示归一化后平均回复长度差异从-7.7词变为+3.9词

### 损失函数 / 训练策略

- 使用标准**DPO损失**进行偏好学习，$\beta=0.1$，学习率 $5\times10^{-7}$
- **迭代训练**：共4轮迭代，每轮4个epoch。每轮用最新模型 $M_i$ 重新生成候选回复并标注偏好数据 $D_i$，解决DPO的分布偏移问题
- 每轮使用4k条指令收集反馈，数据来源涵盖MSCOCO、ShareGPT-4V、MovieNet、VQA v2等多样数据集
- 总训练成本：7B模型数据收集48h+训练6h，12B模型数据收集50h+训练8h（8×A100 80G）

## 实验关键数据

### 主实验（可信度）

| 模型 | Object HalBench Rsp.↓ | Object HalBench Men.↓ | MHumanEval Rsp.↓ | AMBER Acc. | AMBER F1 |
|------|----------------------|----------------------|------------------|-----------|---------|
| LLaVA 1.5 7B (基线) | 54.5 | 27.8 | 67.1 | 73.5 | 77.7 |
| + RLAIF-V 7B | **10.5** (↓80.7%) | **5.2** (↓81.3%) | **44.5** | 76.8 | 84.5 |
| + RLAIF-V 7B BoN | **6.8** | **3.8** | **39.7** | - | - |
| OmniLMM 12B (基线) | 19.4 | 10.9 | 52.7 | 86.5 | 89.5 |
| + RLAIF-V 12B (self) | **4.5** (↓76.8%) | **2.9** (↓73.4%) | **35.6** | 88.0 | 90.9 |
| GPT-4V | 13.6 | 7.3 | 45.9 | 83.4 | 87.4 |

### 消融实验

| 配置 | ObjHal Rsp.↓ | ObjHal Men.↓ | AMBER Acc. | 说明 |
|------|-------------|-------------|-----------|------|
| RLHF-V（人工标注反馈） | 28.5 | 12.3 | 76.4 | 人类标注但局限于特定模型分布 |
| RLAIF-V | **10.1** | **4.7** | **80.1** | 开源AI反馈显著优于人类标注 |
| RLAIF-V w/o 去混淆 | 25.7 | 11.8 | 73.3 | 去混淆策略至关重要 |
| RLAIF-V w/o 分治 | - | - | 73.5 | 人类一致率仅66.7% vs 96.7% |
| VL-Feedback (GPT-4V反馈) | 37.9 | 21.0 | 72.8 | RLAIF-V开源反馈优于GPT-4V反馈 |

### 关键发现

- **开源反馈可超越人类标注和GPT-4V反馈**：RLAIF-V用开源模型生成的反馈数据训练效果优于RLHF-V的人工标注反馈和Silkie的GPT-4V反馈，关键在于去混淆策略提升了学习效率
- **自对齐潜力**：12B模型仅用自身作为标注器，对齐后在Object HalBench和MHumanEval上大幅超越GPT-4V，证明开源MLLM具有自我改进的潜力
- **反馈可泛化**：用OmniLMM收集的RLAIF-V反馈数据可有效训练LLaVA 1.5 7B/13B、MiniCPM-V等完全不同的模型，泛化性强
- **推理时扩展有效**：RLAIF-V奖励在LLaVA 1.5和Qwen-VL-Chat上持续提升可信度，且长度归一化有效解决偏好短回复的问题

## 亮点与洞察

- **范式突破**：从"用强模型教弱模型"转变为"用同级模型互相改进"甚至"模型自我改进"，对MLLM对齐领域有深远影响
- **分治策略的普适性**：将复杂评估拆解为原子验证的思路不限于幻觉检测，可推广到事实核查、推理验证等场景
- **DPO隐式奖励的深度利用**：大多数偏好学习工作仅用DPO做训练，忽略了DPO训练后模型本身就是一个奖励函数这一事实，RLAIF-V充分挖掘了这一潜力
- **去混淆采样的精妙**：用同一模型同参数不同种子采样，消除了风格差异这一最大混淆因素，是提升反馈学习效率的关键

## 局限与展望

- 迭代训练成本仍然较高（数据收集约50小时），难以快速迭代
- 分治策略依赖LLM准确拆分原子声明，在复杂推理场景中拆分准确性可能下降
- Best-of-N推理扩展需要采样多个候选回复（16-32个），推理成本线性增加
- 论文主要关注幻觉抑制，对逻辑推理、复杂问题解答等能力的影响分析不深

## 相关工作与启发

- **vs RLHF-V**: RLHF-V依赖人工标注纠正性反馈，数据来源有限且成本高。RLAIF-V用AI反馈在同等数据量下效果更好（ObjHal 10.5% vs 12.2%），关键在于数据可按需生成
- **vs Silkie/VL-Feedback**: Silkie用GPT-4V收集反馈，RLAIF-V用开源模型达到更高的人类一致率(96.7% vs 92.3%)和更好的训练效果，证明"蒸馏闭源"不是唯一路径
- **vs LLaVA-Critic**: LLaVA-Critic训练专门的评估器模型，RLAIF-V则直接利用DPO训练的隐式奖励，两者在偏好信号生成上是互补的方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 去混淆采样和分治式标注是高度原创的设计，自反馈推理扩展开辟了新方向
- 实验充分度: ⭐⭐⭐⭐⭐ 6个Benchmark、多维消融(去混淆/分治/迭代/泛化/推理扩展)、多模型验证，极其充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但表格较密集，部分符号定义需要反复查看
- 价值: ⭐⭐⭐⭐⭐ 完全开源的MLLM自对齐方案，为社区提供了摆脱闭源模型依赖的可行路径

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](../../CVPR2026/multimodal_vlm/the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)
- [\[CVPR 2025\] Compositional Caching for Training-free Open-vocabulary Attribute Detection](compositional_caching_for_training-free_open-vocabulary_attribute_detection.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](../../ACL2025/multimodal_vlm/unsolvable_problem_detection.md)

<!-- RELATED:END -->
