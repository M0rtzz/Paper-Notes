---
title: >-
  [论文解读] FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning
description: >-
  [NeurIPS 2025][医学图像][公平性] 提出 FairGRPO，一种层级式公平强化学习算法，通过自适应重要性加权（基于群体表示量和任务难度）解决临床 AI 中的人群表现差异问题，在 7 个临床数据集（280K样本，5种模态）上将预测平价降低 27.2%、F1 提升 12.49%，并发布首个公平性优化的临床 VLLM——FairMedGemma-4B。
tags:
  - NeurIPS 2025
  - 医学图像
  - 公平性
  - 强化学习
  - GRPO
  - 临床推理
  - 视觉语言模型
  - 人口统计偏差
---

# FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.19893](https://arxiv.org/abs/2510.19893)  
**代码**: 有（匿名链接，含模型权重 FairMedGemma-4B）  
**领域**: 医学公平性 / 临床推理  
**关键词**: 公平性, 强化学习, GRPO, 临床推理, 视觉语言模型, 人口统计偏差

## 一句话总结

提出 FairGRPO，一种层级式公平强化学习算法，通过自适应重要性加权（基于群体表示量和任务难度）解决临床 AI 中的人群表现差异问题，在 7 个临床数据集（280K样本，5种模态）上将预测平价降低 27.2%、F1 提升 12.49%，并发布首个公平性优化的临床 VLLM——FairMedGemma-4B。

## 研究背景与动机

**医学 AI 偏差问题**：临床数据集大量偏向多数群体（按种族/性别/年龄/社会经济地位），导致 AI 系统在少数群体上性能显著下降

**反馈循环**：常规优化自然偏好充分表示的群体（贡献更多梯度更新、主导损失景观），形成恶性循环——模型越来越专注于多数群体，少数群体性能停滞甚至退化

**现有公平方法不适用**：Group DRO 等方法针对判别模型的固定输出空间设计，不能直接应用于生成式多步推理过程；数据增强/重加权/后处理校准在 VLLM 中效果有限

**RL 训练放大偏差**：通过 RL 进行推理训练会继承并放大训练数据中的偏差，而 RL 中的公平性在医学推理领域尚未被探索

**人口统计标签缺失**：实际临床数据中人口统计标签常常不完整或不可用，进一步加大公平性优化难度

## 方法详解

### 整体框架

FairGRPO 在标准 GRPO 基础上引入层级式公平缩放机制，分三个阶段：

**Stage 1: 标准 GRPO 归一化**

对于 prompt $q$ 在迭代 $t$ 时生成的响应组 $G_{(q,t)}$，每个响应 $o_{(q,i,t)}$ 获得奖励 $r_{(q,i,t)}$，归一化分数为：

$$s_{(q,i,t)} = \frac{r_{(q,i,t)} - \hat{\mu}_{G_{(q,t)}}}{\hat{\sigma}_{G_{(q,t)}} + \varepsilon}$$

**Stage 2: 群体发现**
- **显式群组**：有标签的人口统计属性（年龄、性别）
- **隐式群组**：当标签缺失时，为每个未标记 prompt 构建特征向量 $\mathbf{v}_q \in \mathbb{R}^{|G_{(q,t)}|}$（每维=一个 rollout 的原始奖励），用 K-means 聚类发现潜在群组（肘部法自动定簇数）
- 基于奖励的表示直接捕获任务级难度模式，比 CNN/ViT 嵌入计算效率更高

**Stage 3: 基于人口统计群体的奖励缩放**

计算两级温度因子：

$$T_{(g,t)} = \sqrt{N_{(g,t)}} \cdot \bar{r}_{(g,t)}, \quad T_{(\gamma,g,t)} = \sqrt{N_{(\gamma,g,t)}} \cdot \bar{r}_{(\gamma,g,t)}$$

其中 $N$ 为样本数量，$\bar{r}$ 为平均原始奖励。逆温度缩放：

$$s_{(q,i,t)}^{\text{scaled}} = \frac{s_{(q,i,t)}}{\max(T_{(g_{(q,t)},t)} \cdot T_{(\gamma_{(q,t)},g_{(q,t)},t)}, \varepsilon)}$$

**关键直觉**：少数群体（$N$ 小）或困难群体（$\bar{r}$ 低）→ 温度因子小 → 缩放后信号放大；多数群体信号被衰减。

### 训练目标

保持 GRPO 的策略梯度形式 + 裁剪重要性采样：

$$J_{\text{FairGRPO}}(\theta) = \mathbb{E}_{q,o}\left[\sum_{k=1}^{n_o} \min\left(\varphi_k(\theta)\hat{A}^{\text{FairGRPO}}, \text{clip}(\varphi_k(\theta), 1\pm\varepsilon)\hat{A}^{\text{FairGRPO}}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

- **奖励设计**：简单准确度奖励（正确=1，错误=0）
- **基座模型**：Qwen-2.5-VL-7B 和 MedGemma-4B
- **统一微调**：7 个临床数据集同时训练，无数据集特定适配
- **硬件**：4 × NVIDIA H200 GPU

## 实验关键数据

### 数据集配置

7 个公开临床数据集，5 种模态，共 280.2K 样本：

| 数据集 | 样本数 | 模态 | 人口统计 |
|--------|--------|------|----------|
| CheXpert | 212K | 胸部 X 光 | 年龄、性别 |
| Hemorrhage | 2.5K | CT | 年龄、性别 |
| VinDr-Mammo | 20K | 乳腺摄影 | 年龄 |
| ISIC-2020 | 33K | 皮肤镜 | 年龄、性别 |
| HAM10000 | 10K | 皮肤镜 | 年龄、性别 |
| PAD-UFES-20 | 2.3K | 皮肤镜 | 年龄、性别 |
| COVID-BLUES | 362 | 超声 | 年龄 |

### MedGemma-4B 上的主要结果

| 方法 | PP↓ | EOD↓ | σ_F1↓ | ΔF1↓ | F1↑ | Acc↑ | F1_ES↑ |
|------|-----|------|-------|------|-----|------|--------|
| REINFORCE++ | 20.99 | 8.749 | .0518 | .1033 | .2978 | 78.60 | .2831 |
| RLOO | 23.68 | 10.37 | .0600 | .1170 | .3047 | 80.62 | .2875 |
| GRPO | 22.42 | 6.476 | .0418 | .0795 | .3123 | 80.02 | .2998 |
| GRPO+RS | 23.76 | 6.664 | .0433 | .0835 | .2843 | 80.76 | .2725 |
| GRPO+DRO | 16.04 | 7.367 | .0447 | .0871 | .3271 | 81.19 | .3009 |
| FairGRPO_ND | 25.15 | 11.56 | .0547 | .1067 | **.3513** | 79.23 | .3331 |
| **FairGRPO** | **11.67** | **6.663** | **.0383** | **.0721** | .3218 | **81.83** | **.3100** |

### 关键数字

- **PP 降低 27.2%**：FairGRPO (11.67) vs 最佳公平基线 Group DRO (16.04)
- **F1 提升 12.49%**：FairGRPO_ND (.3513) vs GRPO (.3123)
- **25/33 群体改善**：在 33 个人口统计子群中，FairGRPO 在 25 个群体上优于 GRPO
- **75+ 人群显著提升**：PAD-UFES-20 上 75+ 群体准确率提升 73.08%
- **运行时开销**：奖励计算 < 总训练时间的 0.1%，几乎无额外开销

### 训练动态分析

- FairGRPO 在训练过程中**持续改善公平性**（F1 差异单调下降）
- GRPO 的公平性随训练进行**反而恶化**（F1 差异扩大）
- FairGRPO 扩展了 Pareto 前沿——在性能-公平性权衡上全面优于 GRPO

### 定性分析

- 84 岁女性皮肤镜图像：FairGRPO 正确识别不规则边界、中央坏死、色素模式 → 正确诊断基底细胞癌；GRPO 幻觉不存在的特征 → 误诊 AKIEC
- 老年女性乳腺摄影：FairGRPO 正确识别高密度阴影并评级 BI-RADS 2；GRPO 低估严重性 → 误分类为 BI-RADS 1

## 亮点

- ⭐⭐⭐⭐ **首创性**：首个将公平性优化融入 critic-free RL 训练的临床 VLLM 方法
- ⭐⭐⭐⭐ **隐式群组发现**：无需人口统计标签即可通过聚类发现潜在弱势群组，解决临床数据标签缺失的实际问题
- ⭐⭐⭐⭐ **规模验证**：7 数据集 × 5 模态 × 280K 样本 × 2 个基座模型的大规模验证
- ⭐⭐⭐ **零额外开销**：奖励计算 < 0.1% 训练时间，即插即用
- ⭐⭐⭐ **开源贡献**：发布 FairMedGemma-4B，首个公开的公平性优化临床 VLLM

## 局限与展望

1. **人口统计维度有限**：仅考虑年龄和性别，未涵盖种族/社会经济地位等重要维度
2. **年龄分组粗糙**：25 年为一档的分组可能掩盖组内差异
3. **奖励设计简单**：二元准确度奖励（0/1）可能无法捕获临床推理质量的细微差异
4. **交叉群体未探索**：未分析交叉群体（如"老年+女性"）的公平性表现
5. **模型规模受限**：仅在 4B/7B 模型上验证，更大模型的行为可能不同

## 总体评价 ⭐⭐⭐⭐

解决了一个重要且被忽视的问题——RL 训练中的公平性。层级温度缩放的设计直觉清晰且实现简洁。隐式群组发现是特别有价值的贡献，因为临床数据中人口统计标签常常不可用。大规模多数据集验证增强了说服力。主要不足是人口统计维度单一和奖励设计过于简单。FairMedGemma 的发布对社区有实际贡献。
- 75+ 岁人群在 PAD-UFES-20 上准确度提升 73.08%

## 亮点与洞察

1. **首个公平 RL 算法用于临床 VLLM**：将公平性作为基础优化目标而非后处理，开创了新范式
2. **无标签公平优化**：通过基于奖励模式的聚类发现隐式群组，解决了医学数据中常见的人口统计标签缺失问题
3. **Pareto 前沿改善**：FairGRPO 同时提升性能和公平性，而非在两者间做权衡
4. **定性分析有深度**：展示了 FairGRPO 训练的模型在诊断推理质量上的提升（减少幻觉，更准确识别关键特征）

## 局限与展望

- 目前仅评估视觉-语言任务，未覆盖更多医学模态（如时间序列、EHR）
- 公平性维度限于年龄和性别，缺少种族/社经地位等因素
- 缺乏 FairGRPO 收敛性质的理论分析
- 研究原型，不应直接用于临床决策

## 相关工作与启发

- 与 Group DRO 的关键区别：DRO 为判别模型设计，FairGRPO 专为生成式多步推理的 RL 场景设计
- 与重采样方法互补：重采样操作数据分布，FairGRPO 操作梯度信号
- 启发：基于奖励模式的聚类方法可推广到其他需要公平性的 RL 场景（如自动驾驶中的长尾场景）

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个将公平性融入 VLLM 的 RL 训练的方法）
- 实验充分度: ⭐⭐⭐⭐⭐ （7 数据集 × 5 模态 × 2 基座模型，多维度评估）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机充分）
- 价值: ⭐⭐⭐⭐ （解决了重要的医学 AI 公平性问题）O 对所有 prompt 一视同仁，忽视其来源域和人口统计表示度。

**隐式群体发现**（针对无人口统计标签的样本）：

- 为每个无标签 prompt 构建特征向量 $\mathbf{v}_q \in \mathbb{R}^{|G_{(q,t)}|}$，每个维度为一个 rollout 的原始奖励
- 例如：一个胸部X光 prompt 生成5个 rollout 的奖励 [0.2, 0.8, 0.7, 0.9, 0.3]
- 使用 **K-means 聚类**将奖励分布相似的 prompt 分组
- 通过 **Elbow 方法**自动确定最优聚类数
- 核心优势：计算高效（维度=rollout数，而非CNN/ViT高维特征），且直接捕获任务特定的难度模式

**层次化温度缩放**：

$$T_{(g,t)} = \sqrt{N_{(g,t)}} \cdot \bar{r}_{(g,t)}, \quad T_{(\gamma,g,t)} = \sqrt{N_{(\gamma,g,t)}} \cdot \bar{r}_{(\gamma,g,t)}$$

其中 $N$ 为样本数，$\bar{r}$ 为平均原始奖励。**逆温度缩放**使得少数群体/困难群体获得放大的学习信号：

$$s^{\text{scaled}}_{(q,i,t)} = \frac{s_{(q,i,t)}}{\max(T_{(g_{(q,t)},t)} \cdot T_{(\gamma_{(q,t)},g_{(q,t)},t)}, \varepsilon)}$$

最后重归一化到零均值单位方差。

### 损失函数 / 训练策略

训练目标保留 GRPO 的 PPO 风格裁剪重要性采样：

$$J_{\text{FairGRPO}}(\theta) = \mathbb{E}_{q,o}\left[\sum_{k=1}^{n_o} \min\left(\varphi_k(\theta)\hat{A}^{\text{FairGRPO}}, \text{clip}(\varphi_k(\theta), 1\pm\varepsilon)\hat{A}^{\text{FairGRPO}}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

**奖励设计**：简单的正确性奖励——正确答案得1分，错误得0分。

**训练配置**：4×NVIDIA H200 GPU，在7个数据集上同时进行多任务统一微调。

## 实验关键数据

### 主实验

**数据集规模**：7个公开数据集，5种临床模态（X光/CT/皮肤镜/乳腺X光/超声），共280.2K样本。

| 方法 | PP↓ | EOD↓ | F1↑ | Acc↑ | F1_ES↑ |
|------|-----|------|-----|------|--------|
| GRPO (MedGemma) | 22.42 | 6.476 | .3123 | 80.02 | .2998 |
| GRPO+DRO | 16.04 | 7.367 | .3271 | 81.19 | .3009 |
| **FairGRPO (FairMedGemma)** | **11.67** | **6.663** | .3218 | **81.83** | **.3100** |
| FairGRPO_ND (无标签版) | 25.15 | 11.56 | **.3513** | 79.23 | .3331 |

FairGRPO 在 MedGemma 上将PP降低27.2%（vs. 最佳公平性基线 DRO），EOD提升23.8%。

| 方法 | PP↓ | EOD↓ | F1↑ | F1_ES↑ |
|------|-----|------|-----|--------|
| GRPO (Qwen-2.5-VL) | 11.39 | 9.091 | .2550 | .2437 |
| **FairGRPO** | 16.80 | **5.546** | **.2647** | **.2588** |

在 Qwen-2.5-VL 上，EOD降低15.7%，最大F1差距减小28.9%。

### 消融实验

**FairGRPO_ND**（完全无人口统计标签）的表现：

- 最大准确率差距改善10.81%，准确率标准差改善13.38%
- F1提升12.49%（可能因隐式聚类更好地对齐下游任务）
- 证明即使无demographic信息,仅靠latent group discovery也能改善公平性

**训练动态分析**（Fig 2）：

- FairGRPO 的 F1差异持续低于 GRPO，且差距随训练增加
- FairGRPO 扩展了性能-公平性的 Pareto 前沿
- 运行时开销可忽略：优势计算不到总训练时间的0.1%

### 关键发现

- FairGRPO 在33个人口统计子群中的25个表现优于GRPO（Fig 3）
- 在 CheXpert 上，女性F1提升24.4%，男性提升34.4%
- 在 PAD-UFES-20 上，75+患者提升6.33%，51-75岁提升3.68%
- 定性分析：FairGRPO 减少了对少数群体的幻觉（hallucination），改善了诊断推理链

## 亮点与洞察

1. **首创性**：第一个在 critic-free RL 训练中针对 VLLM 解决公平性的工作
2. **无标签也管用**：隐式群体发现通过奖励向量聚类，无需人口统计标签也能改善公平性
3. **计算高效**：基于奖励的特征表示只需 rollout 维度的向量，远低于传统视觉特征
4. **性能与公平性兼顾**：不存在"以多数群体性能换公平性"的tradeoff，多数群体也有提升
5. **Pareto 前沿扩展**：FairGRPO 在整个训练过程中提供更优的性能-公平性权衡点

## 局限与展望

- 人口统计分组仅涉及年龄和性别，缺少种族、社会经济等更多维度
- Elbow 方法确定聚类数的鲁棒性未充分验证
- 简单的二值奖励（正确/错误）可能限制了更细粒度的公平性优化
- 未探索与其他 RL 算法（如RLOO、REINFORCE++）集成FairGRPO的可能性
- 仅在视觉-语言任务上验证，可扩展到其他医学模态（如EHR、时间序列）

## 相关工作与启发

- **vs. Group DRO**：DRO 为判别模型设计，FairGRPO 首次将公平性引入 critic-free RL
- **vs. Resampling**：重采样是静态方法，FairGRPO 是动态自适应的
- **启发**：奖励向量聚类的思路可推广到其他需要发现latent subgroup的RL场景
- **关键insight**：在RL训练中，标准方法（GRPO/RLOO）的公平性会随训练**恶化**，而FairGRPO使其**持续改善**

## 评分

- 新颖性: ⭐⭐⭐⭐ (首个面向VLLM的公平RL方法，隐式群体发现设计巧妙)
- 实验充分度: ⭐⭐⭐⭐⭐ (7个数据集5种模态，2个VLLM架构，多维公平性指标)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，动机到方法到实验逻辑通顺)
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)
- [CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)

<!-- RELATED:END -->
