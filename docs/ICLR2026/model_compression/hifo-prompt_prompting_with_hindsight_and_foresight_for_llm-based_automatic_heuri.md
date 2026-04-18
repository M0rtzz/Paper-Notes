---
title: >-
  [论文解读] HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design
description: >-
  [ICLR 2026][模型压缩][自动启发式设计] 提出 HiFo-Prompt 框架，通过 Hindsight（回顾式知识池）和 Foresight（前瞻式进化导航器）两个协同模块提升 LLM 驱动的自动启发式设计（AHD），在 TSP 和 FSSP 等任务上显著超越现有方法。
tags:
  - ICLR 2026
  - 模型压缩
  - 自动启发式设计
  - LLM+进化计算
  - 知识管理
  - 探索-利用平衡
  - 组合优化
---

# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design

**会议**: ICLR 2026  
**arXiv**: [2508.13333](https://arxiv.org/abs/2508.13333)  
**代码**: [GitHub](https://github.com/Challenger-XJTU/HiFo-Prompt)  
**领域**: 模型压缩  
**关键词**: 自动启发式设计, LLM+进化计算, 知识管理, 探索-利用平衡, 组合优化

## 一句话总结
提出 HiFo-Prompt 框架，通过 Hindsight（回顾式知识池）和 Foresight（前瞻式进化导航器）两个协同模块提升 LLM 驱动的自动启发式设计（AHD），在 TSP 和 FSSP 等任务上显著超越现有方法。

## 研究背景与动机
LLM + 进化计算（EC）的范式（如 FunSearch、EoH）已展现出以 LLM 作为高层语义变异算子来自动设计启发式算法的潜力。但现有方法面临两个根本性挑战：

**缺乏全局自适应引导**：现有方法多依赖局部或反应式信号——ReEvo 仅对单个候选进行反思，MCTS-AHD 将探索-利用权衡被动嵌入搜索结构。这些局部控制无法对种群停滞或多样性崩溃等系统性问题做出主动干预。另一种方式如 EvoTune 直接微调 LLM 权重，但计算代价高昂且使知识不可解释。

**知识衰退（Knowledge Decay）**：成功的设计策略往往纠缠在具体代码实现中，当父代被淘汰后，底层逻辑也随之丢失。系统无法实现累积学习，反复重新发现相似概念。

核心idea：将 LLM 从"代码生成器"提升为"符号元优化器"，赋予其分层控制能力——Foresight 观察种群动态以引导宏观策略，Hindsight 从精英个体中蒸馏可复用的设计原则。

## 方法详解

### 整体框架
HiFo-Prompt 通过引导式Prompt合成（Guided Prompt Synthesis）构建每轮的LLM提示，融合三部分：基础Prompt策略（遗传算子等价物）、Hindsight 模块（历史知识注入）、Foresight 模块（当前策略指令）。

### 关键设计

1. **Hindsight 模块：自演化的 Insight Pool**:

    - 功能：从成功的启发式代码中蒸馏抽象的设计原则（insights），构建持久的知识库
    - 核心思路：包含三个阶段——(1) *洞见提取与准入*：每代结束从精英个体中提取设计原则，用 Jaccard 相似度阈值 $\theta_{\text{novelty}}$ 去重；(2) *洞见检索与信用分配*：选取效用最高的 top-$s$ 个洞见注入Prompt，效用函数平衡有效性、使用惩罚和新近度奖励：$U(k_i, t) = E_i(t) - w_u \log(N_i(t)+1) + B_r(t, t_i^{\text{last}})$；(3) *自适应裁剪*：当池容量超限时淘汰最低驱逐分数的洞见
    - 信用分配使用分段函数将种群相对性能映射为信用信号：超越最优时 $g_{\text{eff}} = 0.8 + 0.2\tilde{\rho}$，高于均值时 $0.2 + 0.6\tilde{\rho}$，低于均值时 $-0.3 + 0.5\tilde{\rho}$，通过 EMA 更新效用分数
    - 设计动机：将暂时的进化成功转化为可复用的知识资产，解决知识衰退问题

2. **Foresight 模块：进化导航器 (Evolutionary Navigator)**:

    - 功能：实时监控种群动态，在探索/利用/平衡三种模式间切换
    - 核心思路：维护两个互斥计数器 $C_{\text{prog}}$（进步）和 $C_{\text{stag}}$（停滞）跟踪性能趋势，同时计算表型多样性 $\Delta_p(t)$——衡量种群中所有算法描述文本的非重复对比例。基于阈值的规则策略选择进化体制：$\theta_{\text{explore}}$（停滞或多样性低时）、$\theta_{\text{exploit}}$（持续进步时）、$\theta_{\text{balance}}$（其他情况）
    - 多样性度量采用精确字符串匹配而非嵌入相似度，避免语义平滑导致的细微逻辑差异被忽略
    - 设计动机：作为"语言梯度"的符号替代品，提供可解释的全局控制策略

3. **基础Prompt策略**:

    - 功能：提供LLM等价的遗传算子
    - 包含初始化策略 I1、重组策略（E1综合多父代生成新结构、E2抽象共性产生变体）和变异策略（M1结构修改、M2参数调优、M3简化防过拟合）

### 训练策略
种群大小8，CO任务运行8代、BO任务4代。LLM使用 Qwen2.5-Max。Insight Pool 容量30，Jaccard阈值0.7，检索top-3，EMA率0.3，停滞阈值3，进步阈值2，多样性阈值0.3。

## 实验关键数据

### TSP Step-by-step Construction 主实验
| 方法 | TSP50 Gap(%) | TSP100 Gap(%) | TSP200 Gap(%) |
|------|-------------|---------------|---------------|
| LKH3 | 0.000 | 0.000 | 0.000 |
| EoH | 12.820 | 15.361 | 16.658 |
| ReEvo | 10.239 | 12.577 | 14.890 |
| MCTS-AHD | 10.642 | 12.521 | 13.510 |
| **HiFo-Prompt** | **6.625** | **8.582** | **8.877** |

### TSP Guided Local Search
| 方法 | TSP100 Gap(%) | TSP200 Gap(%) | TSP500 Gap(%) |
|------|-------------|---------------|---------------|
| EoH | 0.026 | 0.453 | 2.037 |
| ReEvo | 0.049 | 0.424 | 2.090 |
| **HiFo-Prompt** | **—** | **—** | **—** |

### 关键发现
- TSP step-by-step 上 HiFo-Prompt 将 Gap 从 ~13% 降至 ~8%，相对改进约40%
- Insight Pool 的信用分配机制有效引导知识演化，避免低效洞见持续占用资源
- Foresight 的自适应策略切换对避免早熟收敛至关重要
- 在 Bayesian Optimization 任务上表现竞争力和可靠性

## 亮点与洞察
- "代码与思维解耦"理念新颖——独立更新和评价insights而非直接进化代码，大幅降低评估成本
- Insight Pool 的生命周期管理（提取→检索→信用分配→裁剪）设计精巧，类比强化学习中稀疏奖励的处理
- 探索-利用的显式控制通过自然语言"设计指令"实现，是参数调优的符号替代方案

## 局限与展望
- 种群动态的判断依赖多个手工阈值（停滞=3、进步=2、多样性=0.3），缺乏自适应调节
- 实验基模型仅为 Qwen2.5-Max，不同LLM对Prompt策略的响应可能差异显著
- Insight Pool 中洞见的语义相似度仅用 Jaccard 度量，可能遗漏语义相近但词汇不同的重复洞见

## 相关工作与启发
- **vs EoH**: EoH缺乏知识持久化和全局控制，HiFo-Prompt通过Insight Pool和Navigator弥补这两个不足
- **vs ReEvo**: ReEvo仅对单个候选反思，HiFo-Prompt对整个种群进行宏观监控

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 代码-思维解耦和符号元优化的idea非常新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖TSP/BPP/FSSP/BO多任务，但缺少更大规模实例
- 写作质量: ⭐⭐⭐⭐ 方法描述详细清晰，公式丰富
- 价值: ⭐⭐⭐⭐ 为LLM驱动的自动算法设计提供了系统性框架
