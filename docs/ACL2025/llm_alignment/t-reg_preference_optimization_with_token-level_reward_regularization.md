---
title: >-
  [论文解读] T-REG: Preference Optimization with Token-Level Reward Regularization
description: >-
  [ACL 2025][LLM对齐][偏好优化] T-REG 提出了一种 token 级奖励正则化方法，利用 LLM 的对比提示自生成 token 级奖励信号，将其作为弱监督来引导 DPO 隐式学习到的 token 级奖励分配，在 Alpaca Eval 2 和 Arena-Hard 上分别超过 DPO 最多 3.8% 和 4.4%。
tags:
  - ACL 2025
  - LLM对齐
  - 偏好优化
  - token级奖励
  - 信用分配
  - DPO正则化
  - 对比提示
---

# T-REG: Preference Optimization with Token-Level Reward Regularization

**会议**: ACL 2025  
**arXiv**: [2412.02685](https://arxiv.org/abs/2412.02685)  
**代码**: https://github.com/wzhouad/T-REG  
**领域**: 对齐RLHF  
**关键词**: 偏好优化, token级奖励, 信用分配, DPO正则化, 对比提示

## 一句话总结

T-REG 提出了一种 token 级奖励正则化方法，利用 LLM 的对比提示自生成 token 级奖励信号，将其作为弱监督来引导 DPO 隐式学习到的 token 级奖励分配，在 Alpaca Eval 2 和 Arena-Hard 上分别超过 DPO 最多 3.8% 和 4.4%。

## 研究背景与动机

1. **领域现状**：RLHF（基于人类反馈的强化学习）是对齐 LLM 与人类偏好的主流方法。DPO 等直接对齐算法通过偏好数据优化策略模型，避免了训练额外的奖励模型。这些方法通常使用**序列级奖励**——即对整个输出给一个总体评分。

2. **现有痛点**：序列级奖励本质上是稀疏信号。一段几百甚至上千 token 的输出中，不同 token 对最终质量的贡献是不均匀的。用单一的序列级奖励来训练，模型很难学会哪些 token 真正重要（即 token 级信用分配问题）。

3. **核心矛盾**：现有 token 级 RLHF 方法（如 RTO、SePO、TLCR）要么依赖 AI 标注器生成 token 级奖励（质量不可靠），要么用信用分配模型重新分配序列奖励（需要额外训练），要么直接用这些噪声较大的 token 级奖励来做 PPO 优化（对噪声敏感）。

4. **本文目标** 如何在偏好优化中有效利用 token 级奖励，同时不依赖外部标注且对噪声鲁棒？

5. **切入角度**：作者观察到 DPO 本身就隐式学习了 token 级奖励（即 $\beta \log \frac{\pi(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$），但缺乏直接的 token 级引导。同时 LLM 本身具备自我优化能力（self-refinement），可以通过对比提示"自生成" token 级奖励。

6. **核心 idea**：不直接用自动标注的 token 级奖励去优化策略，而是将其作为**弱监督正则化项**，引导 DPO 隐式学习到的 token 级奖励与自生成奖励对齐。

## 方法详解

### 整体框架

输入是偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$（prompt + 偏好/非偏好输出对）。训练过程分两步：（1）利用对比提示通过参考模型自生成 token 级奖励；（2）在 DPO 损失基础上加入 token 级奖励正则化项进行训练。输出是一个经过偏好优化的策略模型。

### 关键设计

1. **Token 级奖励正则化（核心创新）**:

    - 功能：在 DPO 的序列级偏好优化目标上，增加一个正则化项来约束模型学到的 token 级奖励
    - 核心思路：定义 DPO 隐式学到的 token 级奖励 $r_{\text{token}}$ 与外部自生成的 token 级奖励 $\hat{r}_{\text{token}}$ 之间的相似度 $\text{sim}(y_t) = r_{\text{token}}(y_t) \cdot \hat{r}_{\text{token}}(y_t)$，最大化二者在所有 token 上的对齐。化简后正则化项为加权语言建模损失 $\mathcal{L}_{\text{reg}} = -\sum_t \beta \hat{r}_{\text{token}}(y_t) \log \pi(y_t|x,y_{<t})$，即对正奖励 token 增加概率、对负奖励 token 降低概率
    - 设计动机：不像 RTO/SePO 那样直接用噪声较大的 token 级奖励做 PPO/选择优化，而是以弱监督方式"引导"DPO 本身的 token 级信用分配，兼顾序列级一致性和 token 级精细度

2. **对比提示自生成 Token 级奖励**:

    - 功能：无需额外模型训练，仅用参考模型通过两次前向传播生成所有 token 的奖励
    - 核心思路：设计两个对比修订提示 $x_{\text{better}}$（要求将输出改成更好的）和 $x_{\text{worse}}$（要求改成更差的），分别从"helpful, correct, coherent, concise"和"unhelpful, incorrect, incoherent, verbose"四个维度引导。token 级奖励定义为 $\hat{r}(y_t) = \sigma(\log \frac{\pi_{\text{eval}}(y_t|x_{\text{better}}, y_{<t})}{\pi_{\text{eval}}(y_t|x_{\text{worse}}, y_{<t})}) - 0.5$，通过 sigmoid 裁剪到 $[-0.5, 0.5]$ 范围
    - 设计动机：利用 LLM 对输出的"好坏感知"能力，高质量 token 在 better prompt 下概率更高，在 worse prompt 下更低，差值自然反映 token 质量

3. **序列级梯度权重平衡**:

    - 功能：防止正则化项在某些序列上主导梯度，确保与 DPO 损失平稳协调
    - 核心思路：引入序列权重 $w = \sigma(r_{\text{DPO}}(x,y_l) - r_{\text{DPO}}(x,y_w))$（从 DPO 梯度中提取），对正则化损失进行加权。最终损失为 $\mathcal{L}_{\text{DPO-REG}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot w \cdot (\mathcal{L}_{\text{REG}}(x,y_w) + \mathcal{L}_{\text{REG}}(x,y_l))$
    - 设计动机：DPO 梯度本身已包含序列级权重信息；当 DPO 梯度大时（模型还没学好），给正则化更大权重；反之减小，避免过拟合 token 级噪声

### 损失函数 / 训练策略

最终损失 $\mathcal{L}_{\text{DPO-REG}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot w \cdot (\mathcal{L}_{\text{REG}}(y_w) + \mathcal{L}_{\text{REG}}(y_l))$，其中 $\alpha \in \{0.1, 0.25, 0.5\}$ 进行搜索。训练采用 on-policy 近似设定，用参考策略采样 5 个输出后由 ArmoRM 评分选最好最差组成偏好对。

## 实验关键数据

### 主实验

| 数据集 | 指标 | T-REG (DPO-REG) | DPO | 提升 |
|--------|------|-----------------|-----|------|
| Alpaca Eval 2 (Llama-3-8B) | LC Win Rate | 50.8% | 47.0% | +3.8% |
| Alpaca Eval 2 (Gemma-2-9B) | LC Win Rate | 74.5% (SimPO-REG) | 73.5% (SimPO) | +1.0% |
| Arena-Hard (Llama-3-8B) | Win Rate | 40.3% | 35.9% | +4.4% |
| Arena-Hard (Gemma-2-9B) | Win Rate | 64.2% (SimPO-REG) | 63.0% (SimPO) | +1.2% |

### 消融实验

| 配置 | Alpaca Eval 2 LC WR | Arena-Hard WR | 说明 |
|------|---------------------|---------------|------|
| DPO-REG (完整) | 50.8% | 40.3% | 完整模型 |
| DPO-SFT on $y_w$ | 46.0% | 32.7% | 全 token SFT 反而大幅下降 |
| Static weighting | 48.0% | 35.1% | 去掉序列权重后接近 DPO |
| DPO reward 做正则化 | 49.8% | 36.9% | Arena-Hard 比自生成差 3.4% |
| DPO baseline | 47.0% | 35.9% | 基线 |

### 关键发现

- **选择性正则化 vs 全 token SFT**：在所有 token 上做 SFT（DPO-SFT）反而降性能，因为 $y_w$ 中也包含低质量 token；T-REG 只增强高奖励 token 的概率，效果好得多
- **序列权重至关重要**：去掉序列级加权后，正则化项无法与 DPO 梯度平衡，效果退化到接近 baseline
- **自生成奖励优于 DPO 派生奖励**：在 Arena-Hard 上自生成 token 级奖励比用 DPO 隐式奖励做正则化好 3.4%
- **可兼容其他偏好优化算法**：SimPO-REG 同样在 SimPO 基础上获得一致提升
- **Case study 定性分析**：T-REG 能正确给格式不匹配的 token 分配负奖励，DPO 则容易误判

## 亮点与洞察

- **"弱监督 + 正则化"的 token 级信用分配范式**：不直接用噪声信号做主要优化目标，而是用它来"引导"模型自身学到的 token 级表示——这个设计理念非常巧妙，对信号质量的容忍度高
- **对比提示的自持性**：不需要外部更强的模型或额外训练，仅通过构造相反方向的提示在同一模型上做两次前向传播即可获得 token 级奖励，计算开销极小
- **序列级梯度权重**：从 DPO 梯度中直接提取权重来平衡两个目标，避免引入额外超参数

## 局限与展望

- 对比提示的质量依赖于 LLM 对"好坏"感知的能力，对于非常弱的基座模型可能效果有限
- token 级奖励目前没有定量评估 benchmark，只有定性 case study
- 只在 instruction following 任务上验证，没有测试 reasoning/coding 等场景
- $\alpha$ 超参数仍需搜索，可以探索自适应策略

## 相关工作与启发

- **vs RTO (Zhong et al.)**: RTO 先用 DPO 训练得 token 级奖励再做 PPO，两阶段流程复杂且对 token 级奖励噪声敏感；T-REG 单阶段联合优化，更简洁鲁棒
- **vs SePO (Yang et al.)**: SePO 选择高奖励 token 子集做偏好优化，丢弃了大量信息；T-REG 保留所有 token 但用奖励值做加权
- **vs TDPO (Zeng et al.)**: TDPO 从 MDP 角度推导 token 级 DPO，不使用显式 token 级奖励监督；T-REG 额外引入自生成奖励作为引导信号

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心idea（弱监督正则化）简洁但有效，不算颠覆性但非常 practical
- 实验充分度: ⭐⭐⭐⭐ 两个 benchmark + 详细消融 + case study + 不同基座模型
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，motivation 逻辑链条紧凑
- 价值: ⭐⭐⭐⭐ 方法简洁、通用性好（可插入 DPO/SimPO），工业界可直接采用

<!-- RELATED:START -->

## 相关论文

- [TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [MPO: Multilingual Safety Alignment via Reward Gap Optimization](mpo_multilingual_safety_alignment.md)
- [ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)

<!-- RELATED:END -->
