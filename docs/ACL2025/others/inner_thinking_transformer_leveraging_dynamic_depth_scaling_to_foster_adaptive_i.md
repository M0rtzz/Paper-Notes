---
title: >-
  [论文解读] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking
description: >-
  [ACL 2025][动态深度] 提出 Inner Thinking Transformer (ITT)，通过自适应 token 路由和残差思维连接，在不增加参数的情况下为关键 token 动态分配更多计算步骤，实现隐式深度推理，162M 参数即可达到 466M Transformer 96.5% 的性能。
tags:
  - ACL 2025
  - 动态深度
  - 自适应计算
  - Transformer
  - 隐式推理
  - token级路由
---

# Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking

**会议**: ACL 2025  
**arXiv**: [2502.13842](https://arxiv.org/abs/2502.13842)  
**代码**: 无  
**领域**: 其他  
**关键词**: 动态深度, 自适应计算, Transformer架构, 隐式推理, token级路由

## 一句话总结

提出 Inner Thinking Transformer (ITT)，通过自适应 token 路由和残差思维连接，在不增加参数的情况下为关键 token 动态分配更多计算步骤，实现隐式深度推理，162M 参数即可达到 466M Transformer 96.5% 的性能。

## 研究背景与动机

大语言模型在参数受限条件下面临性能瓶颈，尤其是处理需要复杂推理的关键 token 时。现有方法如 Test-Time Scaling（慢思考）通过推理搜索分配更多计算，但受限于关键 token 的准确生成，在小模型中尤其容易出现灾难性推理失败。层共享、递归、隐式推理等方法也未能灵活提升模型对关键 token 的推理能力。

作者通过分析 GPT-2 在 AQuA 数据集上的梯度核范数（GNN）发现了两个关键现象：
- **简单样本**：GNN 在早期层（L0-L2）指数衰减，中间层（L3-L10）稳定低于 3
- **困难样本**：GNN 在所有 12 层持续振荡，在 L3、L5、L7、L9 处出现突然尖峰

这表明困难 token 在模型各层面临架构或参数限制导致的优化困难，启发了"内在思考"（Inner Thinking）的概念——将每一层的变换视为一个隐式推理步骤。

## 方法详解

### 整体框架

ITT 将层计算重新定义为隐式思考步骤，核心包含三个组件：
1. **自适应 Token 路由（ATR）**：动态选择需要深度思考的关键 token
2. **残差思维连接（RTC）**：迭代累积各步骤结果以精炼表示
3. **思维步骤编码（TSE）**：区分不同推理阶段

ITT 层以固定间隔插入在原始模型层之间，使用语言建模交叉熵损失统一优化所有参数。

### 关键设计

1. **内在思考步骤（Inner Thinking Step）**：将单个 token 的生成分解为一系列内部思考步骤 $X^{(t)} = f^{(t)}(x^{(t-1)})$，支持两种场景——提前退出（中间步骤已足够好）和性能不足（所有步骤后仍不够）

2. **残差思维连接（RTC）**：核心创新，通过累积残差连接迭代精炼表示。最终输出为所有步骤输出的加权累加：$x^{(t)} = \sum_{i=1}^{t}(f(x^{(i-1)}) \odot \phi^{(i)})$，其中 $\phi^{(i)}$ 为可学习的思维位置编码。相比直接循环，RTC 不仅实现更深层思考，还能有效度量和组合各步骤结果

3. **自适应 Token 路由（ATR）**：通过线性权重预测器为每个 token 生成重要性分数，使用百分位阈值 $P_\rho$ 选择最关键的 token 进行深度处理。被选中的 token 经过加权变换，未选中的保留原始表示。路由权重参与梯度传播

4. **思维步骤编码（TSE）**：可学习的位置编码 $\phi^{(t)}$，用于区分不同思考步骤并衡量各步骤的重要性

### 损失函数 / 训练策略

- 使用标准语言建模交叉熵损失 $\mathbb{L} = \mathbb{L}_{\text{CE}}$
- ITT 层以固定间隔替换原始模型的每隔一层
- 训练 50B token（50000 步），学习率 3e-4
- 训练时使用固定路由模式（如 70% token 参与），推理时可弹性调整
- 理论证明：RTC 将单步优化扩展为多步优化，每步误差以因子 $c$ 递减，确保稳定高效收敛，避免梯度消失或爆炸

## 实验关键数据

### 主实验

| 模型配置 | 参数量 | FLOPs | 平均准确率 | 对比 |
|---------|--------|-------|-----------|------|
| LLaMA2-162M | 162M | 1.88 | 40.4 | 基线 |
| ITT ×4-162M | 162M | 3.29 | 42.1 | +1.7% |
| LLaMA2-230M | 230M | 2.87 | 41.8 | - |
| ITT ×4-230M | 230M | 3.41 | 43.9 | +2.1% |
| LLaMA2-466M | 466M | 4.92 | 43.6 | - |
| ITT ×4-466M | 466M | 5.84 | 45.3 | +1.7% |

ITT ×4-162M 在 11 个基准上超越 230M Transformer，达到 466M Transformer 96.5% 的性能。

### 消融实验

| 配置 | Eval PPL | 说明 |
|------|---------|------|
| ITT ×4 完整 | 10.25 | 基线 |
| 去掉 RTC | 11.02 (+0.77) | 最重要组件 |
| 去掉 ATR | 10.44 (+0.19) | 影响效率 |
| 去掉 TSE | 10.56 (+0.22) | 丢失步骤信息 |
| LLaMA2-162M | 11.13 (+1.36) | 原始基线 |

### 弹性推理实验

| 选择比例 | FLOPs | PPL |
|---------|-------|-----|
| 90%, 90%, 90% | 4.42 | 10.27 |
| 70%, 70%, 90% | 4.04 | 10.21 (最优) |
| 70%, 70%, 70% (训练) | 3.85 | 10.52 |
| 50%, 50%, 50% | 3.29 | 10.47 |

### 关键发现

- **数据效率**：ITT 仅用 56.8% 的训练数据即可匹配 LLaMA2-162M 的性能，节省 43.2% 训练预算
- **计算效率**：3 步思考仅需 Loop 84% 的计算量，4 步时降至 70%
- **弹性思考**：推理时可灵活调整 token 选择比例，实现性能-效率平衡
- **路由可视化**：约 30%-50% token 接受迭代思考，任务关键 token（动词、语义关键点）更可能多步思考；连续步骤间展现互补思考模式

## 亮点与洞察

1. **概念创新**：将 Transformer 层计算重新解读为"内在思考步骤"，巧妙连接了隐式推理与动态计算分配
2. **参数效率极高**：不增加任何参数即可显著提升性能，162M 模型可达 466M 模型水平
3. **弹性推理**：训练完成后可灵活调整计算分配，适应不同部署场景
4. **路由的互补性**：模型自发学会了"深度思考"与"广度补偿"交替的策略
5. **理论支撑**：证明了多步优化相比单步映射更容易收敛

## 局限与展望

1. 训练时使用固定路由模式，可能限制对多样 token 复杂度的动态适应
2. 实验仅在 162M-466M 参数规模验证，大规模模型可能出现新的架构交互
3. RTC 在反向传播时引入额外内存开销，需要工业部署优化
4. 思维步骤编码较简单，更复杂的时序建模可能进一步增强推理深度
5. 与 CoT 等显式推理方法的结合尚未充分探索

## 相关工作与启发

- **递归计算**：包括 LSTM、Universal Transformer、Loop Transformer 等深度递归方案
- **动态计算分配**：MoE、Early Exit、Parameter Sharing 等减少冗余计算
- 本文的 token 级深度动态分配思路可启发更精细的计算资源管理，特别是与 MoE 结合的可能性
- RTC 机制类似于迭代优化中的残差学习，可推广到其他需要逐步细化的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 将层计算视为思考步骤的概念新颖，但动态路由思路借鉴了已有工作
- 实验充分度: ⭐⭐⭐⭐ 三个规模的全面评估、详细消融、弹性推理分析，但缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，叙事流畅，图表丰富直观
- 价值: ⭐⭐⭐⭐ 在参数受限场景下的实用性强，弹性推理特性对部署有价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](tree-of-debate_multi-persona_debate_trees_elicit_critical_thinking_for_scientifi.md)
- [\[ACL 2025\] Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)
- [\[ECCV 2024\] Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](../../ECCV2024/others/adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)
- [\[ACL 2025\] ConceptCarve: Dynamic Realization of Evidence](conceptcarve_dynamic_realization_of_evidence.md)

<!-- RELATED:END -->
