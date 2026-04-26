---
title: >-
  [论文解读] How LLMs Learn to Reason: A Complex Network Perspective
description: >-
  [ICLR 2026][RLVR] 本文从复杂网络视角提出"稀疏概念网"理论来统一解释RLVR训练中四个令人困惑的现象（V形响应长度、两阶段学习曲线、灾难性遗忘、策略坍塌），揭示它们都源于平均度约为2的稀疏推理图的拓扑自组织，并据此设计Annealed-RLVR算法在数学推理基准上超越标准RLVR。
tags:
  - ICLR 2026
  - RLVR
  - 概念网络
  - 稀疏图
  - 灾难性遗忘
  - 退火算法
---

# How LLMs Learn to Reason: A Complex Network Perspective

**会议**: ICLR 2026  
**arXiv**: [2509.23629](https://arxiv.org/abs/2509.23629)  
**代码**: https://anonymous.4open.science/r/CoNet-83A4  
**领域**: LLM推理 / 强化学习  
**关键词**: RLVR, 概念网络, 稀疏图, 灾难性遗忘, 退火算法

## 一句话总结
本文从复杂网络视角提出"稀疏概念网"理论来统一解释RLVR训练中四个令人困惑的现象（V形响应长度、两阶段学习曲线、灾难性遗忘、策略坍塌），揭示它们都源于平均度约为2的稀疏推理图的拓扑自组织，并据此设计Annealed-RLVR算法在数学推理基准上超越标准RLVR。

## 研究背景与动机

**领域现状**：RLVR（Reinforcement Learning with Verifiable Rewards）用于训练LLM的推理能力，代表性工作如DeepSeek-R1。但RLVR训练表现出四个令人困惑的现象：(1) 快速提升后长期平台期的两阶段学习曲线；(2) 正确答案长度先变短后变长的V形轨迹；(3) SFT后的灾难性遗忘；(4) 策略多样性坍塌。

**现有痛点**：现有解释各自孤立——平台期被归因于熵耗尽，V形被归因于冗余推理删减后的自我反思出现，灾难性遗忘被视为目标不匹配问题。缺乏统一框架将四个现象联系到共同的底层机制。

**核心矛盾**：直接从LLM的高维隐空间构建微观推理图极其困难，阻碍了对RLVR动力学结构根源的直接研究。

**本文目标**：提供一个统一的物理框架，将四个RLVR现象追溯到共同的拓扑自组织过程。

**切入角度**：受重整化群思想启发，不在token级别分析完整推理图，而是在语义级别研究粗粒化的"概念网"——一个平均度约为2的稀疏网络。使用简化的Concept Network Model (CoNet)作为计算显微镜验证。

**核心 idea**：提出并验证中心假设——RLVR训练后形成的概念网是一个平均度 $\langle k \rangle \approx 2$ 的稀疏网络。这种以树状结构为主的拓扑高效但脆弱，统一解释了V形曲线（从局部技能岛优化到全局网络集成时路径必然变长）、灾难性遗忘（关键"主干"边被切断导致子树不可达）和策略坍塌（叶节点的相变式学习累积导致探索冻结）。

## 方法详解

### 整体框架
(1) 在DeepSeek-R1-Distill-Qwen-1.5B上复现RLVR的四个现象；(2) 用最小化CoNet模型（将推理抽象为图遍历问题）再现这些现象；(3) 利用CoNet的透明性分析微观拓扑机制；(4) 据此设计Annealed-RLVR干预算法。

### 关键设计

1. **稀疏概念网假设 ($\langle k \rangle \approx 2$)**:

    - 功能：统一解释RLVR的四个宏观现象
    - 核心思路：V形曲线的下降段对应独立"技能岛"的并行局部优化（删除冗余路径），上升段对应技能岛合并为全局概念网时，稀疏结构迫使路径变长（平均测地距离随网络增长而增加）。灾难性遗忘：$\langle k \rangle = 2$ 意味着主干边是唯一连接，SFT覆写了这些关键分支点的权重就切断整个子树。策略坍塌：叶节点的相变式学习（从探索到利用的sharp transition）累积导致全局多样性丧失
    - 设计动机：复杂系统中涌现行为通常不依赖微观细节而由大尺度组织决定；稀疏图提供了最简洁的解释框架

2. **CoNet计算显微镜**:

    - 功能：提供可追溯的最小模型验证理论
    - 核心思路：CoNet将LLM的"语义状态"映射为固定随机图中的抽象节点，"逻辑过渡"映射为可学习的概率边。学习过程退化为图遍历问题，但惊人地再现了LLM的V形曲线和两阶段学习等宏观行为
    - 设计动机：直接分析LLM不可行，CoNet作为renormalized proxy让微观分析成为可能

3. **Annealed-RLVR算法**:

    - 功能：突破RLVR的拓扑瓶颈提升推理性能
    - 核心思路：在"最大挫折态"（技能岛竞争最激烈、对应V形曲线底部）时精确插入短暂SFT"加热"步骤——仅对准确率极低(<0.1)但有正确解的问题施加SFT，然后恢复RLVR"冷却"。类比模拟退火：加热打破局部最优，冷却引导到更优全局配置
    - 设计动机：最大挫折态是探索多样性的峰值（Figure 6b），恰好是SFT干预的最佳时机——此时技能岛尚未固化为全局网络，对扰动最鲁棒

### 损失函数 / 训练策略
使用GRPO（Group Relative Policy Optimization）作为RLVR算法。Annealed-RLVR在检测到reward曲线膝点和V形底部时触发SFT加热（几十步），然后恢复标准RLVR。

## 实验关键数据

### 主实验

| 方法 | 训练集(512题) | Minerva(OOD) | AIME 2024/2025(OOD) |
|---|---|---|---|
| 标准RLVR | 基线 | 基线 | 基线 |
| **Annealed-RLVR** | **更优** | **更优** | **更优** |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| RLVR无干预 | 基线，后期策略坍塌 | 标准方法的固有问题 |
| 错误时机的SFT干预 | 灾难性遗忘 | 时机至关重要 |
| 最大挫折态SFT干预 | **最优** | 理论预测的最佳时机 |

### 关键发现
- CoNet（最小模型）和1.5B LLM展现出惊人一致的宏观动力学，支持"涌现行为不依赖微观细节"
- 概念网的平均度确实稳定在约2，直接验证了核心假设
- 灾难性遗忘后的快速恢复证实了"拓扑局部损伤"解释——知识未被擦除，只是变得不可达
- Annealed-RLVR在in-distribution和OOD基准上都超越标准RLVR

## 亮点与洞察
- **物理学视角的大一统理论**：用一个简洁的拓扑假设（$\langle k \rangle \approx 2$）统一解释四个独立现象，展示了跨学科思维的力量
- **最大挫折态=最佳探索时刻**：揭示了看似性能最差的时刻恰好是探索多样性最高的时刻，这个洞察对所有使用RLVR的研究者都有指导意义
- **从解释到处方**：不仅提出理论框架，还直接据此设计了可验证的优化算法，理论→实践的闭环完整

## 局限与展望
- CoNet是高度简化的模型，与真实LLM在规模和机制上差距巨大
- 核心假设（$\langle k \rangle \approx 2$）缺乏直接从LLM内部提取推理图的验证
- 仅在1.5B模型上验证，更大规模模型的适用性有待确认
- 退火时机的检测（V形底部/reward膝点）在实践中可能不总是清晰

## 相关工作与启发
- **vs DeepScaleR/DeepSeek-R1**: 提供RLVR训练动力学的理论解释，而非新的训练方法
- **vs 学习曲线分析工作**: 将宏观现象追溯到拓扑结构而非统计/优化视角
- 稀疏网络自组织的框架可能推广到理解其他涌现能力（如ICL、工具使用）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用复杂网络理论统一解释RLVR现象，视角全新
- 实验充分度: ⭐⭐⭐⭐ CoNet验证充分，LLM验证覆盖多个基准但模型规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 叙事出色，从现象→理论→算法的逻辑链完整优雅
- 价值: ⭐⭐⭐⭐⭐ 对理解LLM推理学习机制有深远影响，Annealed-RLVR有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](../../CVPR2026/reinforcement_learning/reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)

<!-- RELATED:END -->
