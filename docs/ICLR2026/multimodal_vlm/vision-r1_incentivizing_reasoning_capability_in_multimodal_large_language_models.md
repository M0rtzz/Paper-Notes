---
title: >-
  [论文解读] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
description: >-
  [ICLR 2026][多模态][多模态推理] 提出Vision-R1，通过Modality Bridging构建200K高质量多模态CoT数据进行冷启动初始化，再用渐进思维抑制训练(PTST)策略结合GRPO强化学习，在7B参数规模达到与OpenAI O1接近的多模态数学推理能力。
tags:
  - ICLR 2026
  - 多模态
  - 多模态推理
  - 强化学习
  - 思维链
  - GRPO
  - 冷启动初始化
---

# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2503.06749](https://arxiv.org/abs/2503.06749)  
**代码**: [GitHub](https://github.com/Osilly/Vision-R1)  
**领域**: 多模态VLM  
**关键词**: 多模态推理, 强化学习, 思维链, GRPO, 冷启动初始化

## 一句话总结

提出Vision-R1，通过Modality Bridging构建200K高质量多模态CoT数据进行冷启动初始化，再用渐进思维抑制训练(PTST)策略结合GRPO强化学习，在7B参数规模达到与OpenAI O1接近的多模态数学推理能力。

## 研究背景与动机

DeepSeek-R1成功展示了纯RL可以在LLM中激发复杂推理能力（如自我反思、质疑），但这一成功能否迁移到多模态LLM (MLLM)？

作者首先尝试直接用RL训练MLLM（命名Vision-R1-Zero），发现关键困难：

**RL直接训练无法激发复杂推理**：由于缺乏大规模高质量多模态推理数据，模型生成不出复杂CoT

**现有多模态CoT数据质量不足**：缺乏自我反思、质疑等人类认知过程，只是格式化的"伪CoT"

**冷启动后的过度思考问题**：用CoT数据SFT后，模型生成过长的推理链，但正确推理集中在短链中，导致RL训练难以优化

## 方法详解

### 整体框架

两阶段流程：（1）构建Vision-R1-cold数据集 → 冷启动SFT得到Vision-R1-CI；（2）渐进思维抑制训练(PTST) + GRPO强化学习 → 最终Vision-R1。

### 关键设计

1. **Modality Bridging构建多模态CoT数据**: DeepSeek-R1能生成人类般的复杂推理，但是纯文本模型无法直接处理图像。解决方案分三步：

    - 输入图文对到MLLM生成"伪CoT"（含图像描述+推理过程），暴露更多视觉细节
    - 将"伪CoT"与原图文对再次输入MLLM获取详细描述——实现**模态桥接**
    - 将纯文本描述送入DeepSeek-R1获取高质量复杂CoT
   最终经规则过滤得到200K条Vision-R1-cold数据集，其中"Wait"出现585K次（vs LLaVA-CoT的2.3K次），自我反思特征显著

2. **过度思考优化问题(Overthinking)**: 冷启动后模型倾向于所有问题都生成极长推理链，但正确答案多集中在短链中。直接用16K长度RL训练会导致模型生成更长但错误的推理，性能下降。

3. **Progressive Thinking Suppression Training (PTST)**: 分阶段训练，早期严格限制推理长度（如4K token），迫使模型在短空间内学会正确推理。随训练推进逐步放宽（如8K），让模型自主学习用更复杂推理解决更难问题。具体设置：Stage 1用4K×16 (长度×采样数)，Stage 2用8K×8，总采样量×长度在各阶段保持恒定。使用**硬格式结果奖励函数(HFRRF)**：仅当格式正确**且**答案正确时才给奖励1，否则为0。

### 损失函数 / 训练策略

GRPO目标函数（带PTST）：

$$J_{\text{GRPO}}^{(s)}(\theta) = \mathbb{E}\left[\frac{1}{G_s}\sum_{i=1}^{G_s}\min\left(\frac{\pi_\theta(o_i^{(s)}|q)}{\pi_{\theta_{\text{old}}}(o_i^{(s)}|q)}A_i^{(s)}, \text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)A_i^{(s)}\right) - \beta D_{\text{KL}}\right]$$

其中 $\varepsilon=0.2$, $\beta=10^{-2}$，优势估计 $A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$。

冷启动阶段用标准SFT在Vision-R1-cold上训练基座模型（Qwen2.5-VL）。

## 实验关键数据

### 主实验

| 模型 | 参数量 | MathVista | MathVerse | MM-Math | DynaMath | 均分 |
|------|--------|-----------|-----------|---------|----------|------|
| OpenAI O1 | - | 73.9 | - | - | - | - |
| GPT-4o | - | 63.8 | 37.6 | 31.8 | 64.9 | - |
| Qwen2.5-VL-7B | 7B | 68.1 | 46.7 | 34.1 | 50.7 | 49.9 |
| Qwen2.5-VL-72B | 72B | 73.5 | 51.3 | 45.6 | 61.2 | 57.9 |
| **Vision-R1-7B** | **7B** | **73.5** | **52.4** | **40.2** | **56.3** | **55.6** |
| **Vision-R1-32B** | **32B** | **76.4** | **62.1** | **55.3** | **65.6** | **64.9** |
| **Vision-R1-72B** | **72B** | **78.2** | **63.2** | **59.3** | **66.4** | **66.8** |

Vision-R1-7B vs 基座Qwen2.5-VL-7B：GEO +13.4, ALG +10.3, GPS +16.4, MathVista整体 +5.4

### 消融实验

| 方法 | Cold Start | GRPO | PTST | 平均推理长度 | 均分(MathVista/MathVerse/MM-Math) |
|------|-----------|------|------|------------|------|
| Vision-R1-Zero | ✗ | ✓ | ✗ | 1285 | 50.7 |
| Vision-R1-CI | ✓ | ✗ | ✗ | 3566 | 44.5 |
| Vision-R1-Long | ✓ | ✓ | ✗ | 3107 | 47.7 |
| **Vision-R1** | **✓** | **✓** | **✓** | **2057** | **55.4** |

| PTST配置 | Stage1 | Stage2 | MathVista | 均分 | 说明 |
|----------|--------|--------|-----------|------|------|
| 固定16K | 16K×4 | 16K×4 | 70.3 | 47.7 | 早期无约束严重过思考 |
| 固定4K | 4K×16 | 4K×16 | 72.6 | 54.3 | 有效但限制了复杂推理 |
| **PTST 2阶段** | **4K×16** | **8K×8** | **73.5** | **55.4** | 最优，渐进放宽 |
| PTST 3阶段 | 4K×16 | 6K×12 → 8K×8 | 73.0 | 55.1 | 额外阶段无显著增益 |

### 关键发现

- **7B打败70B**: Vision-R1-7B在MathVista上达73.5%，仅低于OpenAI O1 0.4%，超越Qwen2.5-VL-72B
- **RL直接训练不够**: Vision-R1-Zero仅50.7均分，无法激发有效推理
- **冷启动必不可少但不够**: CI模型44.5均分（严重过度思考），必须配合PTST
- **PTST简单有效**: 两阶段(4K→8K)即达最优，额外阶段无益，说明策略稳健
- **数据质量关键**: Vision-R1-cold中"Wait"出现586K次 vs LLaVA-CoT仅2.3K次，自我反思标记频率高2个数量级
- 在Llama-3.2-11B-V上验证跨模型泛化：Vision-R1-cold SFT在所有基准上超越LLaVA-CoT和Mulberry

## 亮点与洞察

- 首次系统探索R1式RL在MLLM上的应用，清晰揭示了直接RL、冷启动、PTST各自的作用
- Modality Bridging巧妙解决了DeepSeek-R1不能处理图像的限制
- PTST策略洞察深刻：先学会"正确思考"再"复杂思考"，类比人类学习规律
- 仅用10K数据做RL即可获得~6%平均提升，数据效率极高
- "Aha moment"在MLLM中首次被观察到（如自我纠正和反思）

## 局限与展望

- RL训练仅使用数学数据，对通用推理任务的泛化有待验证
- PTST的阶段数和长度设置目前靠经验，缺乏理论指导
- Modality Bridging存在信息损失风险（视觉→文本转换）
- 32B和72B版本使用了额外数据，与7B不完全可比
- cold-start数据规模(200K)可能是瓶颈，更大规模数据的收益待探索

## 相关工作与启发

- 与DeepSeek-R1对应的多模态版本，指明了MLLM推理增强的可行路径
- PTST思想可应用到其他需要控制生成长度的RL场景
- Modality Bridging方法可推广到其他需要纯文本LLM处理多模态数据的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将R1式推理范式成功迁移到MLLM，PTST策略原创
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准(MathVista/MathVerse/MM-Math/DynaMath)、多规模(7B/32B/72B)、丰富消融
- 写作质量: ⭐⭐⭐⭐ 论述流畅，问题驱动的叙事结构好，但部分符号较密
- 价值: ⭐⭐⭐⭐⭐ 7B参数达到O1水平的多模态推理能力，对社区有重大启发意义

<!-- RELATED:START -->

## 相关论文

- [Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)
- [Topo-R1: Detecting Topological Anomalies via Vision-Language Models](../../CVPR2025/multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)

<!-- RELATED:END -->
