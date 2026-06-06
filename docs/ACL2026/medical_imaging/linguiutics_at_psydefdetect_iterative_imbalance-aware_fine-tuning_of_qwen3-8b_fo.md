---
title: >-
  [论文解读] LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification
description: >-
  [ACL 2026 / BioNLP 2026][医学图像][心理防御机制] 这篇 PsyDefDetect 参赛系统通过 Qwen3-8B QLoRA、少数类词法增强、分组 5-fold 交叉验证、OOF logit bias 和多种子融合…
tags:
  - "ACL 2026 / BioNLP 2026"
  - "医学图像"
  - "心理防御机制"
  - "类别不均衡"
  - "QLoRA"
  - "分组交叉验证"
  - "后处理校准"
---

# LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification

**会议**: ACL 2026 / BioNLP 2026  
**arXiv**: [2606.00647](https://arxiv.org/abs/2606.00647)  
**代码**: https://github.com/Shefwef/LingIUTics-PsyDefDetect-BIONLP26  
**领域**: 临床 NLP / 心理健康文本分类 / 类别不均衡学习  
**关键词**: 心理防御机制, 类别不均衡, QLoRA, 分组交叉验证, 后处理校准

## 一句话总结
这篇 PsyDefDetect 参赛系统通过 Qwen3-8B QLoRA、少数类词法增强、分组 5-fold 交叉验证、OOF logit bias 和多种子融合，把心理防御机制 9 分类的官方 macro F1 提升到 0.3917，排名 21 支队伍中的第 4。

## 研究背景与动机
**领域现状**：PsyDefDetect 2026 任务要求把心理支持对话中的 seeker utterance 分类到 DMRS 框架下的 9 个 psychological defense levels。该任务对临床 NLP 和心理健康对话系统有价值，因为防御机制能够反映用户如何处理压力、焦虑和冲突。

**现有痛点**：数据极度不均衡。论文给出的合并训练集有 1,864 条样本，其中 Level 7 High-Adaptive 占 51.9%，Level 8 Unclear 只有 1.5%，Level 7 与 Level 8 的比例约为 34.6 倍。官方指标是 macro F1，因此单纯优化 accuracy 会让模型坍缩到多数类。

**核心矛盾**：小型 encoder 和 zero-shot LLM 都难以识别稀有心理防御类别；直接 fine-tune 大模型又会因为类别不均衡和验证泄漏风险导致 leaderboard 泛化差。系统需要同时解决模型容量、少数类召回、验证可靠性和后处理校准。

**本文目标**：作者希望构建一个对少数类友好的 Qwen3-8B 微调系统，在不泄漏 dialogue group 的前提下获得可信 OOF 信号，并用后处理恢复稀有类 recall。

**切入角度**：论文采用迭代式工程路线：先尝试 MentalBERT、MentalRoBERTa、DeBERTa、RoBERTa 和 zero-shot LLM，发现 rare classes 仍为 0 或极低；随后转向 Qwen3-8B QLoRA，并逐步加入 weighted CE、label smoothing、round-robin augmentation、grouped CV、logit bias 和 ensemble。

**核心 idea**：在极端长尾临床文本分类中，模型容量只是必要条件，真正决定 leaderboard 的是 leakage-safe validation、少数类数据构造和面向 macro F1 的后处理校准。

## 方法详解
系统由三大阶段组成：数据预处理与少数类增强、两套 grouped 5-fold QLoRA 训练、OOF 校准与多种子概率融合。每个组件都针对一个早期失败模式：encoder 容量不足、单 fold 泛化差、Level 7 多数类吸引过强、Level 8 等稀有类 recall 近零。

### 整体框架
输入由三部分组成：DMRS Label Guide、最近 30 轮对话上下文和输出指令。模型只需输出 0 到 8 的整数标签。训练数据是 PsyDefConv 的 train+validation 合并，共 1,864 条训练样本，另有 472 条测试样本；源对话数为 200。作者使用 dialogue_id 做 grouped stratified 5-fold，确保同一对话及其增强样本不会跨 fold。

模型层面，系统用 Qwen3-8B 作为 base model，4-bit NF4 量化后通过 QLoRA 微调。推理后处理层面，先在 Anchor OOF 预测上搜索 class-specific logit bias，再把 Anchor 和 Seed-A 两套 5-fold 模型的测试概率按 30/70 融合，并用 $\tau_7=0.69$ 的多数类保护门决定是否应用 minority rerouting。

### 关键设计
1. **面向长尾的 Qwen3-8B QLoRA 架构**:

	- 功能：提供足够的语义容量，区分临床上相近的心理防御类别。
	- 核心思路：使用 4-bit NF4 + double quant 把 Qwen3-8B 的峰值显存从约 32GB 降到约 8GB；LoRA 作用于 q/k/v/o/gate/up/down/score，rank/alpha 为 128/256，dropout 为 0.1，可训练参数约 31M，占 0.4%。
	- 设计动机：BERT-family encoder 最高 validation macro F1 只有 0.314，且 Classes 3、5、8 多次为 0；更大的生成式模型提供了更强语境理解能力，但必须用 PEFT 控制硬件成本。

2. **round-robin 少数类词法增强与 grouped CV**:

	- 功能：提高稀有类样本覆盖，同时避免增强样本泄漏到不同 fold。
	- 核心思路：对 Levels 2、3、4、5、8 进行 $k=3$ round-robin lexical mutation，模式包括 contraction + hedging、style shift + filler、hesitation markers；只改 seeker utterance，不改上下文。增强后少数类从 28-84 条提升到 65-252 条。
	- 设计动机：防御机制标签依赖 utterance 中的心理信号，不能用过强 paraphrase 改写破坏标签；grouped CV 保证同源对话和增强样本在同一 fold，论文报告 0 leaked dialogues。

3. **OOF bias、Seed-A 融合和 $\tau_7$ 保护门**:

	- 功能：在不牺牲多数类 precision 的情况下恢复 minority recall。
	- 核心思路：在 OOF 预测上随机搜索约 22,000 个 bias vector，预测规则为 $\hat{y}=\arg\max_c[\log p_c+\delta_c]$；$\delta_7<0$ 抑制多数类，$\delta_8>0$ 提升 Unclear。测试时用 $p_{blend}=0.30p_{anchor}+0.70p_{seedA}$，若 $p_{blend,7}\geq0.69$ 则锁定 Level 7，否则应用 bias rerouting。
	- 设计动机：raw probabilities 仍偏向 Level 7。宏 F1 需要少数类召回，但直接强行 reroute 会伤害多数类；保护门将“确定的多数类”和“模糊样本”分开处理。

### 损失函数 / 训练策略
训练使用 inverse-square-root class weighting，权重公式为 $w_c=(1/\sqrt{n_c})/\sum_i(1/\sqrt{n_i})$，例如 $w_8=1.67$、$w_5=1.29$、$w_7=0.28$。同时使用 label smoothing $\epsilon=0.05$，防止 Level 7 过早 logit saturation。优化器为 AdamW，学习率 $1.2\times10^{-4}$，weight decay 0.01，cosine annealing，8% warmup，per-device batch size 2，gradient accumulation 8，有效 batch size 16，gradient clip 0.3，每 fold 10 epochs，最大序列长度 1024，bf16，硬件为 NVIDIA RTX 3090 Ti 24GB。

## 实验关键数据

### 主实验
最终系统在官方 positive-class leaderboard 上 macro F1 为 0.3917，排名 4/21。相较任务论文中的 Ministral-8B fine-tuned baseline 31.48 macro F1，提升 +7.7 绝对点，约 +24.4% 相对提升。

| 系统 | Acc. (%) | Macro F1 (%) |
|------|----------|--------------|
| GPT-5 zero-shot (task paper) | 52.75 | 19.53 |
| Gemini 2.5 Pro zero-shot | 56.36 | 25.99 |
| DeepSeek-V3.2 zero-shot (CoT) | 55.72 | 26.17 |
| Llama 3.1-8B fine-tuned | 62.92 | 30.51 |
| InternLM3-8B fine-tuned | 63.98 | 30.53 |
| Ministral-8B fine-tuned (SOTA) | 64.83 | 31.48 |
| Qwen3-8B LoRA baseline | 54.45 | 24.91 |
| Qwen3-8B LoRA + grouped CV + bias tuning | 58.43 | 35.48 |
| Qwen3-8B LoRA + SeedA ensemble + v2decode | 64.19 | 39.17 |

### 消融实验
消融显示，单个组件都不是银弹，但组合起来形成稳定提升。最终从 0.249 增至 0.392。

| 配置 | Macro F1 | 说明 |
|------|----------|------|
| R0: 1-fold, rr=64, no weighting | 0.249 | Qwen3-8B 早期 baseline |
| + 5-fold CV, rr=128 | 0.284 | 增大 LoRA rank 并引入 5-fold |
| + Weighted CE + label smoothing | 0.329 | 抑制多数类坍缩 |
| + Grouped-clean 5-fold | 0.355 | 对话级分组，降低 OOF-LB gap |
| + Data augmentation (RR-k3) | 0.355 | 数字未进一步提升，但辅助少数类稳定 |
| + Seed-A blend (30/70) + v2 decode | 0.392 | 最终提交策略 |

### 关键发现
- grouped-clean augmented run 的 OOF macro F1 为 0.3716，5 个 fold 的 macro F1 分别为 0.3804、0.3701、0.3899、0.3553、0.3326。
- per-class OOF 中 Level 8 “Unclear” 通过增强和 bias tuning 从近零提升到 F1=0.797；Level 7 High-Adaptive 仍保持 F1=0.709。
- 最终 blended system 的 per-label macro 汇总为 precision 0.431、recall 0.436、F1 0.426；官方 leaderboard 正类 macro F1 为 0.3917。
- Level 4 Minor Image-Distorting 和 Level 5 Neurotic 仍较难，F1 约 0.254 和 0.278，论文认为它们与多数类语言重叠较高。
- grouped CV 将 OOF-leaderboard gap 从 9.6 点降到 1.7-4.5 点，使后处理阈值调优更可信。

## 亮点与洞察
- 这篇系统论文的核心不是“换大模型就赢”，而是把长尾分类里的验证、增强、损失和解码全部串起来。Qwen3-8B baseline 只有 24.91 macro F1，真正拉升来自 grouped CV、weighted loss 和后处理。
- round-robin 词法增强非常克制，只改 seeker utterance 的表面形式，保留上下文和心理信号。对临床 NLP 来说，这比大幅 paraphrase 更稳，因为防御机制往往依赖微妙措辞。
- $\tau_7$ gate 是工程上很实用的设计：当模型对多数类非常确定时不强行校准，当多数类置信度不足时才把 logit bias 用于少数类 rerouting。
- 论文用完整 run log 展示从 R0 到 R10 的迭代路径，对 shared task 系统复现很友好，也能帮助读者理解哪些失败推动了后续设计。

## 局限与展望
- OOF bias vector 和 decode rule 是针对 PsyDefDetect 数据集校准的，迁移到新领域必须重新估计。
- grouped CV 能减少增强泄漏，但不能完全排除由相似对话主题或模板引起的泛化风险。
- 硬件限制使作者只做 8B 级别 PEFT，没有探索更大模型或更强 instruction-tuned clinical LLM。
- 数据增强只使用表面词法变换，未来可尝试更可靠的 paraphrase augmentation 或 label-preserving dialogue context augmentation。
- Level 4/5 等心理机制边界仍模糊，可能需要专家知识、更细粒度标签说明或 ordinal / hierarchy-aware loss。

## 相关工作与启发
- **vs BERT-family encoders**: MentalBERT、MentalRoBERTa、DeBERTa 和 RoBERTa 在稀有类上遇到容量瓶颈，本文用 Qwen3-8B 提供更强上下文理解。
- **vs zero-shot LLM**: Qwen3-8B、Llama 3.1-8B 和 Ministral-8B zero-shot 只有约 8-16% macro F1，说明单靠任务定义 prompt 不足以学习 DMRS 标签。
- **vs 普通 cross-entropy fine-tuning**: Ministral-8B fine-tuned 虽有 64.71% accuracy，但 macro F1 只有 14.74，说明 accuracy 在长尾心理分类中会误导模型选择。
- **对后续工作的启发**: 类似医疗或心理健康文本任务可以复用“grouped CV + 少数类保守增强 + OOF bias + majority gate”的范式，而不是只追求更大模型。

## 评分
- 新颖性: ⭐⭐⭐ 系统工程创新多于算法创新，但组合设计贴合任务痛点。
- 实验充分度: ⭐⭐⭐⭐ run log、消融、per-class 和官方对比都比较完整。
- 写作质量: ⭐⭐⭐⭐ 迭代过程清楚，表格信息密集但实用。
- 价值: ⭐⭐⭐⭐ 对长尾临床 NLP shared task 和小显存 QLoRA 参赛系统很有借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](../../AAAI2026/medical_imaging/coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)

</div>

<!-- RELATED:END -->
