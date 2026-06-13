---
title: >-
  [论文解读] Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis
description: >-
  [ACL2025][LLM效率][Mixture-of-Experts] 提出跨层级知识归因算法，系统解析 MoE 模型中共享专家与路由专家的"基础-精炼"协作框架，揭示 MoE 相比稠密模型实现 31% 更高的逐层效率，并通过语义驱动路由机制（注意力头-专家相关性 r=0.68）和专家阻断实验验证了架构深度对鲁棒性的决定性影响。
tags:
  - "ACL2025"
  - "LLM效率"
  - "Mixture-of-Experts"
  - "知识归因"
  - "可解释性"
  - "专家协作"
  - "稀疏路由"
---

# Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis

**会议**: ACL2025  
**arXiv**: [2505.24593](https://arxiv.org/abs/2505.24593)  
**作者**: Junzhuo Li, Bo Wang, Xiuze Zhou, Peijie Jiang, Jia Liu, Xuming Hu (HKUST(GZ), Ant Group)
**领域**: 其他  
**关键词**: Mixture-of-Experts, 知识归因, 可解释性, 专家协作, 稀疏路由

## 一句话总结

提出跨层级知识归因算法，系统解析 MoE 模型中共享专家与路由专家的"基础-精炼"协作框架，揭示 MoE 相比稠密模型实现 31% 更高的逐层效率，并通过语义驱动路由机制（注意力头-专家相关性 r=0.68）和专家阻断实验验证了架构深度对鲁棒性的决定性影响。

## 研究背景与动机

Mixture-of-Experts (MoE) 通过稀疏激活子集专家降低计算开销，但其可解释性——特别是异构设计（如共享专家模块）中专家如何协作处理和精炼知识——仍是未解难题。

**现有方法的不足：**
- 现有知识归因方法（Knowledge Neurons、Transformer FFN 分析等）针对稠密模型设计，无法捕捉 MoE 中动态路由-专家的交互
- 缺乏对异构 MoE 架构的对比研究，共享专家的功能假设（"通用特征提取器"还是"冗余备份"）缺乏实证验证
- 跨层级的专家协作机制不透明，阻碍了 MoE 模型的系统优化

**核心问题：** 专家在 MoE 模型中如何协同处理和精炼知识？共享专家与路由专家各扮演什么角色？架构深度如何影响鲁棒性？

## 方法详解

### 跨层级知识归因算法

将稠密模型的神经元级归因方法扩展到 MoE，同时分析宏观架构行为和微观专家贡献。

**MoE 专家神经元的重要性评分：**

对于第 $l$ 层专家 $\mathcal{E}_j$ 的神经元 $\mathbf{v}^l_{\mathcal{E}_j}$，重要性评分定义为加入该神经元后对目标 token 预测对数概率的增益。其中 $g^l_{i,j}$ 为门控概率，$\mathbf{u}^l$ 为注意力输出后的中间表征。该评分衡量每个专家神经元对最终预测的影响，即该模块在推理过程中引入的增益。

### 实验设计

**MoE 模型：** Qwen 1.5-MoE (24层, 64专家+4共享)、OLMoE (16层, 64专家)、Mixtral-8x7B (32层, 8专家)
**稠密模型对照：** Qwen 1.5-7B (32层)、Llama-7B (32层)、Mistral-7B (32层)
**评估指标：** HIT@10、MRR（知识预测任务）、逐层效率（FFN Gain / 层数）

### "基础-精炼"协作框架

通过消融实验验证共享专家和路由专家的角色分工：

1. **共享专家 = 通用基础处理器**：负责实体识别、句法解析等跨领域基础任务
2. **路由专家 = 领域精炼器**：在共享专家提供基础表征后，进行领域特定的属性关联（如将"Canada"映射到"Ottawa"）
3. **语义驱动路由**：注意力头与专家选择存在强时间相关性（r=0.68, p<0.001），注意力头主动引导专家选择

### 三阶段处理模式（以 Qwen 1.5-MoE 为例）

- **早期（1-13层）**：专家并行初始化，提取基础特征，贡献 6.1% 总增益
- **中期（14-19层）**：动态路由激活专家筛选（top-4 门控），贡献 43.5% 总增益
- **后期（20-24层）**：共享专家与路由专家协作精炼，贡献 50.4% 总增益

### 因果验证

三项干预实验确认注意力头因果性地驱动专家选择：
- 抑制关键注意力头 → top-4 专家门控概率下降 54%，MRR 下降 29%
- 强制激活正确专家 → 恢复 85% 原始 MRR
- 积分梯度路径分析 → 28% 专家激活归因可直接追溯到特定注意力头

## 实验关键数据

### Table 1: 稠密模型与 MoE 模型的效率对比

| 模型 | HIT@10 | MRR | FFN Gain | Attn Gain | Peak Gain 位置 | 逐层效率 |
|------|--------|-----|----------|-----------|---------------|---------|
| Llama-7B | 0.90 | 0.70 | 5.16 | 4.03 | 77.6% | 0.161 |
| Qwen 1.5-7B | 0.79 | 0.62 | 6.49 | 4.14 | 90.2% | 0.203 |
| Mistral-7B | 0.88 | 0.71 | 6.74 | 2.96 | 83.2% | 0.211 |
| **Qwen 1.5-MoE** | 0.85 | 0.63 | **7.36** | 3.18 | 84.8% | **0.307** |
| **OLMoE** | 0.83 | 0.64 | 4.98 | 4.40 | 84.6% | **0.311** |
| **Mixtral-8x7B** | 0.90 | 0.73 | 6.79 | 3.03 | 83.3% | 0.212 |

**发现：** MoE 模型逐层效率显著优于稠密模型。Qwen 1.5-MoE 用 24 层达到 0.307 效率，比 32 层的 Qwen 1.5-7B（0.203）高 **51%**。OLMoE 16 层即达 0.311 最高效率。MoE 模型中 FFN Gain 占主导，说明专家网络是知识处理核心。

### Table 5: 专家阻断实验——鲁棒性对比（MRR）

| 任务 | 模型 | 原始 | 阻断Top1 | 阻断Top5 | 阻断Top10 |
|------|------|------|---------|---------|----------|
| name_birthplace | Qwen 1.5-MoE | 0.85 | 0.84 | 0.83 | 0.81 |
| name_birthplace | OLMoE | 0.82 | 0.80 | 0.80 | 0.60 |
| country_capital | Qwen 1.5-MoE | 0.71 | 0.68 | 0.68 | **0.40** |
| country_capital | OLMoE | 1.00 | 1.00 | 0.76 | 0.76 |
| country_language | Qwen 1.5-MoE | 0.94 | 0.92 | 0.92 | 0.92 |
| country_language | OLMoE | 0.96 | 0.92 | 0.87 | **0.68** |
| fruit_inside_color | Qwen 1.5-MoE | 0.74 | 0.68 | 0.63 | 0.60 |
| fruit_inside_color | OLMoE | 0.76 | 0.60 | 0.52 | **0.41** |
| object_superclass | Qwen 1.5-MoE | 0.83 | 0.82 | 0.80 | 0.79 |
| object_superclass | OLMoE | 0.80 | 0.75 | 0.70 | **0.66** |

**发现：** 深层 Qwen 1.5-MoE 在通用任务上表现出强鲁棒性（name_birthplace 阻断 Top10 仅下降 4.7%），但核心敏感任务（country_capital）下降 43%。浅层 OLMoE 在多数任务上退化严重：fruit_inside_color 下降 46%、country_language 下降 29%。这验证了架构深度与共享专家对冗余设计的关键作用。

### Table 2: Qwen 1.5-MoE 消融实验

| 配置 | HIT@10 | MRR |
|------|--------|-----|
| 仅 Top1 路由专家 | 0 | 0 |
| 仅 Top2 路由专家 | 0 | 0 |
| 仅共享专家 | 0.03 | 0.01 |
| 共享 + Top1 | 0.82 | 0.59 |
| 共享 + Top2 | 0.83 | 0.63 |
| 共享 + Top4（默认） | 0.85 | 0.63 |

**发现：** 单独激活任何类型的专家均导致严重退化，证明复杂知识任务必须依赖共享专家与路由专家的协同。存在"有效专家阈值"——核心知识主要由少量专家捕获，增加更多专家收益递减。

## 亮点

- **首个 MoE 可解释性系统框架**：提出跨层级归因算法，弥补了稀疏 MoE 架构在可解释性方面的空白，可同时分析宏观架构行为和微观专家贡献
- **"基础-精炼"协作范式**：通过消融实验严格验证共享专家作为通用处理器、路由专家作为领域精炼器的角色分工，并通过因果干预证实注意力头驱动专家选择
- **任务敏感性洞察**：区分了"核心敏感任务"（地理推理，需集中专业知识）和"分布容忍任务"（物体属性，可利用广泛参与），为 MoE 设计提供定量指导
- **可操作的设计原则**：深层 MoE 应在早期层部署共享专家保障冗余，晚期层分配路由专家进行精炼；浅层 MoE 需在专家通用性和路由适应性间取得平衡

## 局限性

- 分析仅限 7B 参数规模的静态路由 MoE，更大规模（100B+）或完全动态路由 MoE 是否展现相同专业化模式尚未验证
- 未涵盖检索增强型 MoE（如 Monet），其外部知识耦合引入的混杂因素超出分析范围
- 知识归因方法依赖对数概率变化，对多步推理或组合推理等复杂场景的解释力有待探索
- 实验数据集以事实性知识预测为主，对需要更高层抽象推理的任务（如数学推理、代码生成）的泛化性未充分评估

## 相关工作

- **MoE 架构**：从 Jacobs et al. (1991) 的经典框架到 Switch Transformer (Fedus et al., 2021)、DeepSeekMoE (Dai et al., 2024)、Qwen-MoE (Team, 2024)，MoE 通过稀疏路由平衡效率与性能，但可解释性研究滞后
- **知识归因**：Transformer FFN 层作为键值存储 (Geva et al., 2021)、知识神经元 (Dai et al., 2022)、知识电路 (Yao et al., 2024)、神经元级归因 (Yu & Ananiadou, 2024) → 均针对稠密模型，无法处理 MoE 的动态路由和专家协作
- **本文定位**：首次系统性地将知识归因从稠密模型扩展到异构 MoE 架构，填补了稀疏路由模型在可解释性方面的空白

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个系统性的 MoE 知识归因框架，"基础-精炼"协作范式和语义驱动路由机制是有意义的发现
- 实验充分度: ⭐⭐⭐⭐ — 3 个 MoE + 3 个稠密模型对比，消融实验+因果干预+专家阻断，多角度验证充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，分析层次分明，但数学符号较多，部分实验细节需翻阅附录
- 价值: ⭐⭐⭐⭐ — 为 MoE 架构设计提供了可操作的原则（深度-冗余-任务敏感性），对理解和优化 DeepSeek-V3 等大规模 MoE 有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)
- [\[ACL 2025\] Synergistic Weak-Strong Collaboration by Aligning Preferences](synergistic_weak-strong_collaboration_by_aligning_preferences.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)

</div>

<!-- RELATED:END -->
