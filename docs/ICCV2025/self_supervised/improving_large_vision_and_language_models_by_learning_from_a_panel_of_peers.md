---
title: >-
  [论文解读] Improving Large Vision and Language Models by Learning from a Panel of Peers
description: >-
  [ICCV2025][自监督学习][自我改进] 提出 Panel-of-Peers (PoP) 学习框架，利用多个性能相近的 LVLM 互相生成候选答案、互相评分、构建偏好数据，并通过 SimPO 迭代自我改进，在 15 个基准上将平均分从 48% 提升至 57%，无需人工标注数据。
tags:
  - "ICCV2025"
  - "自监督学习"
  - "自我改进"
  - "偏好对齐"
  - "同伴学习"
  - "LVLM"
  - "奖励建模"
---

# Improving Large Vision and Language Models by Learning from a Panel of Peers

**会议**: ICCV2025  
**arXiv**: [2509.01610](https://arxiv.org/abs/2509.01610)  
**代码**: -  
**领域**: 自监督  
**关键词**: 自我改进, 偏好对齐, 同伴学习, LVLM, 奖励建模  

## 一句话总结

提出 Panel-of-Peers (PoP) 学习框架，利用多个性能相近的 LVLM 互相生成候选答案、互相评分、构建偏好数据，并通过 SimPO 迭代自我改进，在 15 个基准上将平均分从 48% 提升至 57%，无需人工标注数据。

## 研究背景与动机

大型视觉-语言模型（LVLM）在多种任务上展现了强大能力，但其进一步提升仍面临挑战：

**人工标注成本高**：传统对齐方法依赖人工偏好数据，标注昂贵且难以规模化

**机器生成数据质量有限**：利用 GPT-4V 等基础模型生成数据，性能上限受限于教师模型

**自监督偏好数据的幻觉问题**：模型自己生成和评估的数据容易引入幻觉

作者受人类**课堂协作学习**的启发：学生先学基础知识，然后通过完成练习题和同伴讨论来加深理解。类似地，一组性能相近的 LVLM（"同伴"）可以互相学习，通过彼此生成和评估答案来提升集体能力。与传统的教师-学生蒸馏不同，PoP 的核心是**同伴间的互相反馈和学习**，每个模型既是生成者又是评估者。

## 方法详解

### 整体框架

PoP 的核心流程分为两个交替进行的阶段：

1. **候选响应生成**：面板中的每个模型对同一组 image-question 对生成候选回答
2. **数据创建与微调**：各模型互相评分→构建偏好数据集→用偏好优化算法微调

这两个阶段交替执行，形成迭代自我改进循环。

### 奖励建模

面板中的每个模型 $\pi_i$ 充当评委，对其他模型 $\pi_j$ 的输出 $\mathbf{y}_j$ 进行评分。评分基于五个维度：有用性、正确性、详细程度、连贯性和复杂性，每项 1-5 分（Likert scale），归一化到 [0,1] 范围。

最终的集成奖励通过均值投票聚合每个评委的评分：

$$R_\mu(\mathbf{y}_j) = \frac{1}{M} \sum_{i=1}^{M} R_i(\mathbf{y}_j)$$

关键设计：评分基于模型自身的内部知识，不需要参考答案。

### 候选响应生成

每个模型可以通过 Best-of-N 采样生成多个候选回答并选择最优：

$$\mathbf{y}^* = \arg\max_{\mathbf{y}^{(n)} \in Y_N} R(\mathbf{y}^{(n)})$$

标准 PoP 每个模型采样 15 个候选（PoP），简化版每个模型仅采样 1 个（st-PoP）。采样后进行拒绝采样，仅保留奖励分 ≥ 0.85 的正例，且正负例间维持 0.75 的奖励边距。

### 偏好数据构建与迭代训练

选择最高和最低累积奖励的响应分别作为偏好和非偏好回答。使用 SimPO 作为偏好优化目标：

$$\mathcal{L}_{\text{SimPO-PoP}} = -\mathbb{E}_{\mathcal{D}_t}\left[\log\sigma\left(\frac{\beta}{|\mathbf{y}^{(n)}|}\pi_{\theta_t}(\mathbf{y}^{(n)}) - \frac{\beta}{|\mathbf{y}^{(1)}|}\pi_{\theta_t}(\mathbf{y}^{(1)}) - \gamma\right)\right]$$

选择 SimPO 的原因：不需要参考模型、隐式奖励直接与生成指标对齐、引入 $\gamma$ 边距帮助分离偏好响应。

**迭代机制**：每次迭代后重新初始化面板为上一轮的检点，重新生成、评分、微调。总共执行 3 次迭代。

### 实验设置

- **面板成员**：LLaVA-1.5 架构 + Mistral-7B / Llama3-8B / Vicuna-7B
- **训练数据**：从 Cambrian-7M 随机采样 1M 图像（不含 GT 答案），每次迭代产生约 300K 偏好样本
- **评估**：15 个基准覆盖 General VQA、Knowledge、Chart&OCR、Hallucination、Vision Centric

## 实验结果

### 主实验：与偏好优化方法对比（基于 LLaVA-1.5-Vicuna-7B）

| 方法 | 附加数据 | MMB | SEED-B | MM-Vet | SciQA | POPE |
|------|---------|-----|--------|--------|-------|------|
| LLaVA-1.5-7B | - | 64.3 | 58.6 | 30.5 | 66.8 | 85.9 |
| +RLHF | 10k | 63.4 | 58.1 | 31.1 | 65.8 | 81.5 |
| +CSR | 17k | 65.4 | 60.3 | 33.9 | 70.7 | 87.0 |
| +SeVa | 8k | 65.6 | 65.8 | 37.2 | 67.5 | 86.7 |
| +STIC | 6k | 65.3 | 66.2 | 32.6 | 67.4 | 85.8 |
| **+PoP-iter1** | **300k** | **68.7** | **67.9** | 34.1 | 71.2 | 87.0 |
| **+PoP-iter3** | **900k** | **72.5** | **68.8** | 35.0 | **86.4** | 87.0 |

PoP-iter3 在 SciQA 上提升 19.6 个绝对百分点（从 66.8% 到 86.4%），在 MMB 上提升 8.2 个百分点。

### 15 基准迭代进展

| 迭代 | PoP-Mistral | PoP-Vicuna | PoP-LLaMA3 |
|------|------------|------------|------------|
| Iter 0 | 47.7 | 48.0 | 48.7 |
| Iter 1 | 54.3 | 53.1 | 55.6 |
| Iter 2 | 55.7 | 55.9 | 57.3 |
| Iter 3 | 56.4 | 56.7 | **58.2** |

LLaMA3 成员从 48.7% 提升到 58.2%（+9.5 绝对百分点），证明迭代自我改进的有效性。3 轮后趋于饱和。

### 关键消融发现

- **绝对评分 vs 相对评分**：绝对评分（每个模型独立打分）优于相对评分（同时比较所有回答），后者因提示过长导致上下文丢失
- **SFT vs SimPO**：在 PoP（多次采样）设置下两者接近；但 SimPO 在减少幻觉和提升 vision-centric 任务上更具优势
- **同伴学习新技能**：一个缺乏 OCR 能力的 "OCR-Dumb" 模型通过 PoP 从具有 OCR 能力的同伴处习得该技能，验证跨模型知识迁移
- **PoP vs 直接 SFT 900K GT 数据**：PoP-Vicuna 达到 57.0，而用 900K 真实答案 SFT 仅达 54.0，说明同伴反馈合成数据优于静态标注数据

## 亮点与洞察

1. **无需人工标注的自我改进**：整个流程不使用任何 ground truth 答案，仅利用图像和问题，模型从彼此的互评中学习
2. **同伴反馈优于 GT 数据**：900K PoP 合成数据的训练效果（57.0）超过同量 GT 数据的 SFT（54.0），这是反直觉但深刻的发现
3. **知识跨模型迁移**：PoP 能使缺乏特定能力的模型从同伴处习得该能力（如 OCR），展现了协作学习的潜力
4. **方法的通用性**：对面板大小、模型容量无限制，可随未来前沿模型无缝扩展

## 局限性

- 计算成本高：收集偏好数据约 80 小时，每个模型微调约 10 小时（8×A100 80GB）
- 需要多个基线模型（至少 3 个），增加了初始投入
- 每次迭代需要大量数据采样（从 1M 图像中生成 900K 偏好对），3 轮迭代后趋于饱和
- 评估提示的设计对最终效果有显著影响，但缺乏系统的提示选择方法论

## 相关工作

- **LVLM 对齐**：LLaVA-RLHF（10K 人工交互），POVID/SeVa（注入错误模拟幻觉），CSR（CLIPScore 排序），STIC（增强+DPO）
- **自我改进 LLM**：Self-Rewarding（LLM 同时做奖励模型和生成器），SPIN（双人博弈框架）
- **模型即评委**：PoLL（弱模型面板可产生与强模型相当的人类对齐评分），Prometheus-Vision

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 4 |
| 实验充分度 | 5 |
| 写作清晰度 | 4 |
| 实用价值 | 4 |
| 总评 | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](../../ICML2025/self_supervised/large_language_models_to_diffusion_finetuning.md)
- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
