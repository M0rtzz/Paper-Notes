---
title: >-
  [论文解读] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
description: >-
  [ICLR 2026][机器人][mllm-as-verifier] 本文发现多模态大语言模型（MLLM）作为 agent 行为验证器时存在严重的"同意偏差"（agreement bias）——系统性地过度认可 agent 行为，并提出 Self-Grounded Verification（SGV）方法，通过两步生成（先提取行为先验、再条件化验证）缓解该偏差，在 web 导航、桌面操作和机器人操控任务中将失败检测率提升最高 25pp、准确率提升 14pp。
tags:
  - ICLR 2026
  - 机器人
  - mllm-as-verifier
  - agreement-bias
  - self-grounded-verification
  - agent-evaluation
  - robotics
---

# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification

**会议**: ICLR 2026  
**arXiv**: [2507.11662](https://arxiv.org/abs/2507.11662)  
**代码**: [项目主页](https://github.com/GT-RIPL/SGV)  
**领域**: robotics  
**关键词**: mllm-as-verifier, agreement-bias, self-grounded-verification, agent-evaluation, robotics  

## 一句话总结

本文发现多模态大语言模型（MLLM）作为 agent 行为验证器时存在严重的"同意偏差"（agreement bias）——系统性地过度认可 agent 行为，并提出 Self-Grounded Verification（SGV）方法，通过两步生成（先提取行为先验、再条件化验证）缓解该偏差，在 web 导航、桌面操作和机器人操控任务中将失败检测率提升最高 25pp、准确率提升 14pp。

## 背景与动机

1. **验证器是 AI 进步的核心引擎**：从围棋到代码推理，搜索+验证器范式推动了多项突破，但开放式任务（如网页操作、机器人抓取）缺乏形式化的成功判定标准，难以构建可靠的自动验证器。

2. **MLLM 被寄望充当通用验证器**：凭借广泛的世界知识、人类偏好对齐和多模态推理能力，MLLM 理论上适合对 agent 轨迹做打分/评判，已被用于轨迹筛选、Reflexion 自我改进、在线监督等多种场景。

3. **Agreement bias 问题普遍而严重**：作者在 13+ 模型家族、28+ 评估模板上发现，MLLM 系统性倾向给 agent 行为打高分——TNR（真阴性率）低至 50%，即一半失败轨迹被错误判为成功。更令人担忧的是，MLLM 会生成 CoT 来"合理化"其错误判断。

4. **现有 test-time scaling 技术无法解决**：CoT、SoM、majority voting、reasoning model 等主流方法均无法有效缓解偏差，甚至 sampling 可能因幻觉而加剧问题。

5. **偏差根源在知识提取瓶颈**：MLLM 实际上拥有正确的行为先验（在只给部分信息时能描述正确行为），但在面对完整轨迹时这些先验被"覆盖"，这与预训练和 RLHF 的已知局限一致。

6. **下游应用受损严重**：agreement bias 会污染自我改进管道（Reflexion）、在线监督反馈、行为克隆数据筛选等依赖 MLLM 判断的应用，导致 agent 无法获得纠正信号。

## 方法详解

### 核心思想：Self-Grounded Verification（SGV）

SGV 是一个轻量、零样本的两步验证框架，核心理念是**先让 MLLM 在信息受限条件下自由提取行为先验，再以这些先验为锚点对完整轨迹进行条件化验证**。

### 第一步：先验生成（Prior Extraction）

给定任务 $q$ 和部分轨迹信息 $\tau_{u:v}$（如初始截图），让 MLLM 生成关于期望行为的广泛先验 $\hat{k}$：

$$\hat{k}_{q,u:v} = g\left(\prod_{i=1}^{n} P(y_i | y_{<i}, \tau_{u:v}, C, q)\right)$$

由于只看到部分信息，模型能更自由地探索其概率分布，提取与任务相关且不受待评估数据干扰的知识。

### 第二步：条件化验证（Grounded Verification）

以自生成的先验为条件，MLLM 对完整轨迹进行推理评估：

$$r_{\text{SGV}}(\tau_t, C, q) = h\left(\prod_{i=1}^{n} P(y_i | y_{<i}, q, \tau_{p:t}, C, \hat{k}_q)\right)$$

先验充当"公正裁判"的参考基线，使模型不再盲目顺从轨迹中呈现的信息。

### 关键设计选择

- **评分模板**：采用三级 Likert 量表（SUCCESS / PARTIAL SUCCESS / FAILURE），实验证明比二元判断更能减少偏差
- **轨迹表示**：截图-动作序列对，可选 Set-of-Marks 标注增强视觉定位
- **通用性**：SGV 可叠加在任何 MLLM 上，无需微调，兼容 reasoning model，甚至能让非推理模型达到推理模型水平

## 实验结果

### 实验设置

- **环境**：VisualWebArena（910 任务，web 导航）、OSWorld（369 任务，桌面操作）、robomimic（机器人操作，tool-hang 任务）
- **模型**：14 个模型覆盖 GPT-5/o4、Gemini 2.5、Qwen3-235B、Llama-4 等
- **Agent**：VWA 用 Gemini-2.5-Flash ReAct agent（SR=47%），OSWorld 用 UI-TARS-1.5（SR=22%），robomimic 用 diffusion policy

### 表1：离线验证性能（VWA + OSWorld 合并）

| 模型 | Acc (无SGV) | TNR (无SGV) | Acc (SGV) | TNR (SGV) | Acc↑ | Bias↓ |
|------|------------|------------|----------|----------|------|-------|
| GPT-5 (T) | 81 | 78 | 86 | 87 | +5 | -6 |
| GPT-o4 (T) | 78 | 71 | 84 | 82 | +6 | -6 |
| GPT-4.1 Mini | 60 | 40 | 74 | 65 | +14 | -20 |
| Gemini-2.5-Flash (T) | 74 | 64 | 82 | 78 | +8 | -15 |
| Qwen3-235b (T) | 66 | 53 | 77 | 71 | +11 | -12 |
| Llama-4-Maverick | 60 | 44 | 65 | 54 | +5 | -7 |

SGV 在所有模型上一致提升 TNR（最高+25pp）和准确率（最高+14pp），弱模型受益最大。

### 表2：下游任务——在线监督与自我改进

| 方法 | VWA 全部 | VWA S/C/R | OSWorld | robomimic SR |
|------|---------|-----------|---------|-------------|
| Base Agent | 45 | 50/35/48 | 22 | 24 |
| + Verifier 无SGV | 46 | 52/36/49 | 24 | 16 |
| + Verifier SGV | **54** | 56/43/58 | **27** | **32** |

SGV 在 VWA 上提升 9pp（20% 相对），OSWorld 提升 5pp（22%），robomimic 提升 8pp（33%）。VWA 达到新 SOTA，超越此前最佳 20pp。值得注意的是，无 SGV 的验证器在 robomimic 上反而降低了性能（24→16），说明 agreement bias 在机器人任务中危害尤其严重。

## 亮点与创新

- **问题定义精准**：首次系统性定义并量化 agreement bias，跨 13+ 模型家族验证其普遍性和对下游应用的实际危害
- **方法极其简洁**：SGV 是零样本、无训练的两步 prompting 方法，极易集成到现有 pipeline
- **评估全面**：覆盖离线评估和两种下游应用（自我改进+在线监督），使用细粒度指标而非仅报告准确率
- **发现深刻**：揭示了推理模型同样受 agreement bias 影响，SGV 能在推理模型上额外提升 6-11pp

## 局限性

- **未彻底消除偏差**：SGV 缓解但不根除 agreement bias，剩余失败多源于基础模型视觉感知与语言整合能力不足
- **计算开销增加**：两步调用使 token 消耗增至 1.5-2.2 倍，在大规模场景下需权衡成本
- **先验质量受限**：先验生成依赖 MLLM 自身能力，若模型对任务领域知识不足，先验质量难以保证
- **环境覆盖有限**：仅在 web/桌面/机器人三类环境验证，更复杂的开放世界场景（如自动驾驶）待探索

## 相关工作对比

### vs. Pan et al. (2024) — GPT-4V 评估器

Pan et al. 使用 GPT-4V 加 benchmark 特定 rubric 做二元判断，被后续多项工作沿用。本文指出二元评分放大 agreement bias，且即使提供人工 rubric 也无法解决（Tab.3 第 6 行 Acc 仅 66%）。SGV 不需要人工 rubric 即超越其效果（Acc 76+），更具扩展性。

### vs. Reasoning Models（DeepSeek-R1, GPT-o1/o4）

推理模型通过 RL 训练生成思维链，理论上应更擅长验证。但实验表明推理模型仍受 agreement bias 影响（GPT-o1 TNR 仅 62%）。SGV 能额外提升推理模型 6-11pp，说明偏差根源在知识提取瓶颈而非推理能力本身。

### vs. Majority Voting / Tree Search

Majority voting 依赖输出分布的众数，但 agreement bias 导致分布本身偏斜（失败轨迹仅 48% 概率采到正确判断），投票无法修正系统性偏差。SGV 从根本上改变条件分布，而非在偏斜分布上做聚合。

## 评分

- ⭐⭐⭐⭐⭐ 创新性：首次形式化 agreement bias 并给出简洁有效的解法
- ⭐⭐⭐⭐⭐ 实验充分度：14 个模型、3 个环境、28+ 模板、离线+下游评估，极为全面
- ⭐⭐⭐⭐ 写作质量：结构清晰论证严密，但数学符号偏多，部分段落较密集
- ⭐⭐⭐⭐ 实用价值：SGV 即插即用对 agent 系统有直接帮助，但 token 开销需关注

<!-- RELATED:START -->

## 相关论文

- [Visual Planning: Let's Think Only with Images](visual_planning_lets_think_only_with_images.md)
- [On the Generalization Capacities of MLLMs for Spatial Intelligence](on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [Token Taxes: Mitigating AGI's Economic Risks](token_taxes_mitigating_agis_economic_risks.md)
- [Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)
- [JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)

<!-- RELATED:END -->
