---
title: >-
  [论文解读] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding
description: >-
  [AAAI 2026][视觉定位] 发现 CoT 推理在视觉定位任务中可能适得其反，提出 CuRPO（Curriculum-based Relative Policy Optimization），利用 CoT 长度和 gIoU 奖励作为数据复杂度指标进行课程式 RL 训练，在 RefCOCO 上比 Visual-RFT 提升最高 +12.52 mAP。
tags:
  - AAAI 2026
  - 视觉定位
  - 课程学习
  - GRPO
  - 思维链
  - 视觉语言模型
---

# Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding

**会议**: AAAI 2026  
**arXiv**: [2511.13924](https://arxiv.org/abs/2511.13924)  
**代码**: [github.com/qyoung-yan/CuRPO](https://github.com/qyoung-yan/CuRPO)  
**领域**: 强化学习  
**关键词**: 视觉定位, 课程学习, GRPO, 思维链, 视觉语言模型

## 一句话总结

发现 CoT 推理在视觉定位任务中可能适得其反，提出 CuRPO（Curriculum-based Relative Policy Optimization），利用 CoT 长度和 gIoU 奖励作为数据复杂度指标进行课程式 RL 训练，在 RefCOCO 上比 Visual-RFT 提升最高 +12.52 mAP。

## 研究背景与动机

### 问题背景

视觉定位（Visual Grounding）要求模型根据文本描述在图像中定位目标对象，输出边界框坐标。近期 CoT 提示与 RL 的结合（如 Visual-RFT 使用 GRPO）在多种视觉推理任务中取得了成功。

### 三个反直觉发现

本文通过系统实验揭示了三个重要的反直觉现象：

**发现一：CoT 推理反而降低视觉定位性能**。要求模型在输出边界框前先生成 CoT 推理过程，不仅没有提升反而降低了定位精度。如图 1 所示，CoT 引导的模型因误解文本语境产生了错误的框，而不使用 CoT 的模型反而定位正确。在 40 个样本下，无 CoT 模型的 mIoU 达到 35.6，而有 CoT 模型需要 239 个样本才达到 34.4。

**发现二：CoT 越长 ≈ 任务越困难**。通过统计分析发现 CoT 长度与 gIoU 奖励呈显著负相关：
- Pearson 相关系数：-0.4395（$p = 1.04 \times 10^{-12}$）
- Spearman 等级相关：-0.4268（$p = 5.34 \times 10^{-12}$）
- Kendall's Tau：-0.2981（$p = 6.81 \times 10^{-12}$）

理论解释：将 CoT 每步成功率建模为 $p_c < 1$，则 $C$ 步推理链的总成功概率 $\text{Pr}(\text{success}) = \prod_{c=1}^{C} p_c$ 随步数指数衰减。

**发现三：更多数据 ≠ 更好性能**。增大训练集规模并不总能提升性能——使用 CoT 的模型甚至出现性能振荡和停滞，而不使用 CoT 的模型则随数据增加持续提升。

### 核心动机

既然 CoT 长度是任务难度的可靠指标，且训练数据的复杂度顺序影响学习效果，那么可以利用 CoT 长度和奖励信号来构建课程式训练策略，让模型先学简单再学难。

## 方法详解

### 整体框架

CuRPO 是一个将课程学习（Curriculum Learning）与 GRPO 强化学习结合的训练框架。核心流程：计算每个样本的复杂度指标 → 按难度排序分组 → 按从易到难的顺序分阶段训练 → 每阶段内使用 GRPO 优化策略。

### 关键设计

#### 1. **奖励函数设计**：gIoU + 格式奖励

采用广义 IoU（gIoU）替代标准 IoU，因为当预测框与真值框不重叠时标准 IoU 为零，无法提供有用梯度。

$$\text{gIoU}(A, B) = \text{IoU}(A, B) - \frac{C - (A \cup B)}{C}$$

其中 $C$ 是包含 $A$ 和 $B$ 的最小外接矩形面积。gIoU 范围为 $[-1, 1]$，线性缩放到 $[0, 2]$ 以减少过度负反馈。

总奖励：$R_d = R_{\text{visual}} + R_{\text{format}}$

- $R_{\text{visual}}$：基于 gIoU 的定位准确度奖励
- $R_{\text{format}}$：确保输出格式正确的奖励

#### 2. **课程训练策略**：基于 CoT 长度的难度排序

具体流程：

1. 为每个训练样本用预训练 VLM 生成多个 CoT（通常 8 个），计算平均长度
2. 按 CoT 长度从短到长排序所有样本
3. 在每个 CoT 长度区间（间隔 50 token）内再按奖励值排序
4. 分阶段训练：
    - **初始阶段**：仅使用短 CoT 样本（最简单），学习基础视觉推理模式
    - **中间阶段**：逐步引入中等长度 CoT，接触中等复杂度场景
    - **高级阶段**：加入长 CoT 样本，挑战复杂推理任务

排序策略探索了三种变体：
- **CuRPO (Length)**：按 CoT 长度排序
- **CuRPO (Reward)**：按 gIoU 奖励排序
- **CuRPO (Random)**：随机顺序（作为基线验证课程学习本身的有效性）

#### 3. **GRPO 训练目标**：组相对策略优化

给定查询 $q$，策略 $\pi_\theta$ 生成 $G$ 个候选输出 $\{o_i\}$，计算组归一化优势：

$$A_i = \frac{r_i' - \mu'}{\sigma'}, \quad \mu' = \frac{1}{G}\sum_{j=1}^{G} r_j', \quad \sigma' = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j' - \mu')^2}$$

最终 GRPO 目标为裁剪代理损失 + KL 散度正则化：

$$L_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\min(c_i A_i, \text{clip}(c_i, 1-\epsilon, 1+\epsilon)A_i) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $c_i = \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}$ 是新旧策略的概率比。

### 损失函数 / 训练策略

- 基座模型：Qwen2-VL-2B
- 基线：在 "with CoT" 设置下零课程微调的 Qwen2-VL-2B
- 关键设计：模型**不显式输出 CoT**，直接输出边界框坐标（CoT 仅在训练前用于衡量数据难度）
- 评估指标：mIoU 和 mAP

## 实验关键数据

### 主实验

**LISA 数据集**：

| 方法 | 模型 | 训练样本数 | mIoU |
|------|------|-----------|------|
| SFT | Qwen2-VL-2B | 239 | 29.7 |
| GroundingDINO | X-Decoder | 239 | 28.5 |
| Visual-RFT | Qwen2-VL-2B | 239 | 34.4 |
| **CuRPO (Ours)** | Qwen2-VL-2B | **50** | **37.4 (+3.0)** |
| **CuRPO (Ours)** | Qwen2-VL-2B | 200 | **38.7 (+4.3)** |
| **CuRPO (Ours)** | Qwen2-VL-2B | 239 | **38.4 (+4.0)** |

仅用 50 个样本就超过了 Visual-RFT 用全部 239 个样本的效果！

**RefCOCO 系列数据集 (mAP)**：

| 数据集 | Qwen2-VL-2B | Visual-RFT | CuRPO (Random) | CuRPO (Length) | CuRPO (Reward) |
|--------|------------|------------|----------------|----------------|----------------|
| RefCOCO (val) | 11.57 | 21.28 | 33.09 (+11.81) | **33.80 (+12.52)** | 32.64 (+11.36) |
| RefCOCO (test) | 10.70 | 20.38 | 29.92 (+9.54) | **31.42 (+11.04)** | 27.89 (+7.51) |
| RefCOCO+ (val) | 13.72 | 18.41 | 26.82 (+8.41) | 26.18 (+7.77) | **26.85 (+8.44)** |
| RefCOCO+ (test) | 16.11 | 20.90 | 24.34 (+3.44) | **25.10 (+4.20)** | 23.55 (+2.65) |
| RefCOCOg (val) | 14.89 | 23.39 | 27.98 (+4.59) | 29.27 (+5.88) | **32.65 (+9.26)** |

### 消融实验

| 排序策略 | RefCOCO (val) mAP | 特点 |
|---------|-------------------|------|
| 无课程（Visual-RFT） | 21.28 | 基线 |
| CuRPO (Random) | 33.09 | 甚至随机课程也有大幅提升 |
| CuRPO (Length) | **33.80** | 简单数据集上最优 |
| CuRPO (Reward) | 32.64 | 复杂描述数据集（RefCOCOg）上最优 |

### 关键发现

1. **课程学习本身就很有效**：即使是随机排序的 CuRPO (Random) 也比 Visual-RFT 高出 +11.81 mAP（RefCOCO val），说明课程式 RL 训练的价值
2. **Length 排序适合简单数据集**（RefCOCO/RefCOCO+），**Reward 排序适合复杂描述数据集**（RefCOCOg）
3. **少样本场景下优势更大**：仅 50 个样本时 CuRPO 就大幅超过用全部数据的基线
4. **显式 CoT 生成一致劣于不生成 CoT**：这一结论在各种数据规模和排序策略下均成立
5. **按类别分析**：在 chair、bed、toilet 等难以精确定位的类别上改进最为显著

## 亮点与洞察

- **发现 CoT 在视觉定位中的"过度思考"问题**是重要的实证贡献——挑战了"CoT 总是有益"的流行假设
- **将 CoT 长度作为难度代理变量**的想法简洁高效，无需额外标注任务难度
- **只用 50 个样本超越全量数据训练的 SOTA**，展示了课程学习在数据效率上的巨大优势
- CuRPO 的设计是**正交的**——不修改模型架构，任何 VLM + GRPO 的组合都可以直接应用

## 局限与展望

- 仅在 Qwen2-VL-2B 上验证，更大模型（7B/72B）上的效果未知
- CoT 长度作为难度代理可能在其他任务（如 VQA、图像描述）中不一定成立
- 课程阶段的划分（CoT 长度区间=50 token）是手动设定的，自适应划分可能更优
- 目前不支持实例分割任务（只做 bounding box detection）
- 排序策略的选择依赖于数据集特性，缺乏自动选择机制

## 相关工作与启发

- 与 FASTCURL 类似思路：后者用提示长度作为推理复杂度代理进行课程式 RL 训练
- Visual-RFT 是最直接的基线：使用 GRPO + gIoU 奖励训练 VLM 做视觉定位
- 跟 CoT 长度自适应（如 SelfBudgeter）互补：本文发现某些任务不需要 CoT，SelfBudgeter 则调节 CoT 长度
- 启发：在视觉推理类任务中，数据难度/顺序的重要性可能被严重低估

## 评分

- **新颖性**: ⭐⭐⭐⭐ — CoT 长度作为难度指标的洞察新颖，但课程学习和 GRPO 本身不新
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集验证充分，但仅在 2B 模型上实验偏小
- **写作质量**: ⭐⭐⭐⭐ — 三个发现的呈现清晰有说服力，动机驱动
- **价值**: ⭐⭐⭐⭐⭐ — 实际提升巨大（+12.52 mAP），且少样本效果突出，实用价值高
