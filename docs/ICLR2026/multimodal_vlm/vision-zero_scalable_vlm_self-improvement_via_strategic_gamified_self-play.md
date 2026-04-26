---
title: >-
  [论文解读] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play
description: >-
  [多模态] 提出 Vision-Zero，首个无标注的游戏化自博弈框架，通过"谁是卧底"式视觉推理游戏实现 VLM 的可扩展自进化，结合 Iterative-SPO 训练算法在推理、图表理解和视觉中心任务上超越基于人工标注数据的 SOTA 方法。
tags:
  - 多模态
---

# Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play

- **会议**: ICLR 2026
- **arXiv**: [2509.25541](https://arxiv.org/abs/2509.25541)
- **代码**: [GitHub](https://github.com/Vision-Zero-AI/Vision-Zero)
- **领域**: 多模态视觉语言模型
- **关键词**: VLM, Self-Play, Reinforcement Learning, Zero-Shot, Gamification, Self-Improvement

## 一句话总结

提出 Vision-Zero，首个无标注的游戏化自博弈框架，通过"谁是卧底"式视觉推理游戏实现 VLM 的可扩展自进化，结合 Iterative-SPO 训练算法在推理、图表理解和视觉中心任务上超越基于人工标注数据的 SOTA 方法。

## 研究背景与动机

当前 VLM 训练面临两个核心瓶颈：

**数据稀缺**：多模态标注成本极高（COCO Attributes: $60,480/200K 物体；Ego4D: >250K 标注小时）

**知识天花板**：模型能力受人类标注上限约束，无法发现超越人类经验的策略

**自博弈**（Self-Play）已在围棋（AlphaGo）、电竞（OpenAI Five）等领域证明可突破知识天花板。但将自博弈扩展到 VLM 面临挑战：需要同时考虑视觉和语言模态，设计满足技能对齐、难度可扩展、多样性和低数据需求的游戏环境。

**Vision-Zero 的设计理念**：灵感来自社交推理游戏"谁是卧底"，平民观察真实图像、卧底接收空白输入，通过交互式策略博弈让模型自主生成训练数据。

## 方法详解

### 游戏环境

**角色设定**：$n_c$ 个平民（观察真实图像 $I_c$）+ 1 个卧底（接收空白图像 $I_s$）

**两阶段博弈**：

**线索阶段**（Clue Stage）：
- 每个玩家根据角色和观察提供语言线索
- 卧底必须仅从平民线索推断隐藏图像内容并伪装
- 平民需提供准确线索同时最小化信息泄露给卧底

**决策阶段**（Decision Stage）：
- 平民分析所有线索+自己的图像，投票识别卧底
- 卧底不参与投票
- 支持 "n/a"（不确定）回答

### 无标注、领域无关的数据输入

训练仅需任意图像，实验验证三类数据：
- **CLEVR 数据**：2000 张自动渲染图像（4-6 随机物体）
- **图表数据**：1000 张 ChartQA 图像
- **真实世界数据**：1000 张 ImgEdit 图像

### Iterative Self-Play Policy Optimization (Iterative-SPO)

**线索阶段 - 自博弈优化**：

**零和奖励**：

$$r_s^{clue} = -\beta(v_s - \bar{v}_c), \quad r_{c_j}^{clue} = \frac{\beta}{n_c}(v_s - \bar{v}_c) - \lambda(v_{c_j} - \bar{v}_c)$$

卧底与平民奖励之和为零，收到更多票数的获得更低奖励。

**角色优势估计 (RAE)**：缓解信息不对称导致的胜率失衡：

$$A_k^{clue} = r_k^{clue} - b_k, \quad b_s = \alpha b_s + (1-\alpha) r_s^{clue}$$

**线索阶段目标函数**：

$$\mathcal{L}^{clue}(\theta) = -\mathbb{E}\left[\frac{1}{n}\sum_{k \in \mathcal{K}} A_k^{clue} \log \pi_\theta^k(u_k | I_k, h)\right] + \tau_{clue} \cdot D_{KL}(\pi_\theta^k \| \pi_{ref}^k)$$

**决策阶段 - RLVR 优化**：

**离散奖励**：正确识别卧底 +1，回答 n/a -0.5，错误 -1

**组归一化 GRPO 目标**：

$$\mathcal{L}^{dec}(\theta) = -\mathbb{E}\left[\frac{1}{n_c}\sum_{i=1}^{n_c} A_{c_i}^{dec} \log q_\theta(\hat{s}_{c_i} | H)\right] + \tau_{dec} \cdot D_{KL}(q_\theta \| q_{ref})$$

**交替训练**：通过滞后阈值切换阶段：
- 决策 → 线索：当 $\bar{acc}_t \geq \tau_{acc}^\uparrow$ 且 $\bar{na}_t \leq \tau_{na}^\downarrow$（卧底太容易被发现时，增加线索阶段难度）
- 线索 → 决策：当 $1 - \bar{acc}_t \geq \tau_{err}^\uparrow$ 或 $\bar{na}_t \geq \tau_{na}^\uparrow$（卧底太难识别时，加强决策训练）

### 优势分析

1. **领域无关**：利用图像差异进行博弈，不依赖特定图像类型
2. **同时增强多种能力**：推理、空间理解、视觉理解、OCR
3. **成本极低**：无需人工标注，用 ChatGPT/NanoBanana 快速生成数据

## 实验

### 推理和数学任务

| 方法 | MathVista | MathVision | WeMath | MathVerse | LogicVista | Avg |
|------|-----------|------------|--------|-----------|------------|-----|
| Qwen2.5-VL-7B | 68.2 | 25.4 | 36.1 | 49.0 | 47.2 | 41.1 |
| MM-Eureka-7B | 73.0 | 26.9 | 36.2 | 50.3 | 42.9 | 42.9 |
| ViGaL-S+R | 71.9 | 27.5 | 36.9 | 52.4 | 46.5 | 43.0 |
| **VZ (CLEVR)** | 72.2 | **28.4** | **39.2** | **53.2** | **49.8** | **44.3** |
| **VZ (Real)** | **73.1** | 28.5 | **40.1** | 52.1 | 50.8 | **44.5** |

Vision-Zero 在仅使用无标注数据的情况下，超越了所有使用人工标注数据的基线。

### 图表理解和视觉中心任务

Vision-Zero (Chart) 在 ChartXiV、FunctionQA 等图表任务上显著提升，在 MMVP、BLINK 等视觉中心任务上也有增益。

### 训练动态

- 胜率（平民 vs 卧底）在训练中持续上升
- 线索长度（token 数）随训练增长，模型学会更详细地描述和推理
- Iterative-SPO 有效避免了纯自博弈的过早收敛

### 消融实验

| 消融 | MathVista | MathVision |
|------|-----------|------------|
| 仅线索阶段 | 70.8 | 27.1 |
| 仅决策阶段 | 71.5 | 27.6 |
| Iterative-SPO | **73.1** | **28.5** |

交替训练的效果显著优于单阶段训练。

### 与 Gobang 的对比

在 MathVision 上：Vision-Zero 提升 +3%（100轮），Gobang 无提升，证明视觉推理游戏的泛化能力。

## 亮点

1. **零人类参与**：完全不需要人工标注或人类反馈
2. **领域无关输入**：CLEVR、图表、自然图像均有效
3. **Iterative-SPO 理论优雅**：交替自博弈+RLVR避免局部均衡
4. **超越标注基线**：无标注方法超越昂贵人工标注训练的 SOTA
5. **多能力同时提升**：推理、图表理解、视觉中心任务全面增益

## 局限性

1. 游戏中固定角色数（$n_c + 1$），未探索更复杂的多角色设定
2. "谁是卧底"游戏的策略空间是否充分覆盖了所有视觉推理能力存疑
3. 卧底使用空白图像而非相似图像，与原版"谁是卧底"有偏差
4. Iterative-SPO 的阈值超参数需手动设定
5. 在部分视觉中心任务上（如 RealWorldQA）提升有限

## 相关工作

- **LLM 自博弈**：SPIRAL (Liu et al., 2025) 用棋类游戏增强推理；Absolute Zero (Zhao et al., 2025) 在数学/编码上 SOTA
- **VLM 后训练**：R1-OneVision, MM-Eureka, VLAA-Thinker 使用 RLVR + 人工标注
- **游戏化 VLM**：ViGaL (Xie et al., 2025) 用蛇/旋转游戏训练但需收集游戏数据
- **自博弈理论**：AlphaGo (Silver et al., 2017), TD-Gammon (Tesauro, 1995)

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首个 VLM 无标注游戏化自博弈框架
- **实用性**: ⭐⭐⭐⭐⭐ — 极低成本、领域无关、即插即用
- **清晰度**: ⭐⭐⭐⭐ — 框架清晰但公式较多
- **意义**: ⭐⭐⭐⭐⭐ — 开辟了 VLM 自进化的新范式

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)

<!-- RELATED:END -->
