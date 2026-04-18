---
title: >-
  [论文解读] DPO Meets PPO: Reinforced Token Optimization for RLHF
description: >-
  [ICML 2025][LLM对齐][RLHF] 本文提出 Reinforced Token Optimization (RTO)，将 RLHF 建模为 token 级别的 MDP（而非句子级 bandit），利用 DPO 隐式地提取 token-wise 奖励信号后用 PPO 进行策略优化，在 AlpacaEval 2 上比 PPO 高 7.5 分、在 Arena-Hard 上高 4.1 分，且仅需 1/8 数据量即可达到 PPO 级别性能。
tags:
  - ICML 2025
  - LLM对齐
  - RLHF
  - DPO
  - PPO
  - token-level reward
  - MDP 建模
---

# DPO Meets PPO: Reinforced Token Optimization for RLHF

**会议**: ICML 2025  
**arXiv**: [2404.18922](https://arxiv.org/abs/2404.18922)  
**代码**: https://github.com/zkshan2002/RTO (有)  
**领域**: LLM对齐/RLHF  
**关键词**: RLHF, DPO, PPO, token-level reward, MDP 建模

## 一句话总结
本文提出 Reinforced Token Optimization (RTO)，将 RLHF 建模为 token 级别的 MDP（而非句子级 bandit），利用 DPO 隐式地提取 token-wise 奖励信号后用 PPO 进行策略优化，在 AlpacaEval 2 上比 PPO 高 7.5 分、在 Arena-Hard 上高 4.1 分，且仅需 1/8 数据量即可达到 PPO 级别性能。

## 研究背景与动机
**领域现状**：RLHF 是对齐 LLM 的核心技术，经典流程为"先训练句子级奖励模型，再用 PPO 优化"。ChatGPT、Claude 等闭源模型都依赖此范式。

**现有痛点**：PPO 在开源实现中效果不佳。深层原因是 RLHF 被建模为 bandit 问题——整句作为一个 action，奖励是句子级的。但 PPO 本身是为多步 MDP 设计的，需要每步都有奖励。现有实现中，句子级奖励只分配给最后一个 token，其余 token 得到零奖励（稀疏奖励问题）。

**核心矛盾**：PPO 需要密集的逐步奖励信号才能高效学习，但人类偏好标注天然是句子级的，难以直接获取 token 级反馈。

**本文目标**：（1）建立 RLHF 的 token-level MDP 理论框架；（2）找到从偏好数据中提取 token-wise 奖励的实用方法；（3）改进 PPO 在 RLHF 中的表现。

**切入角度**：发现 DPO 训练的模型天然隐含了 token 级奖励信息（$r^*(s_h, a_h) = \beta \log \frac{\pi_{dpo}(y_h|x, y_{1:h-1})}{\pi_{ref}(y_h|x, y_{1:h-1})}$），可以提取出来指导 PPO 训练。

**核心 idea**：DPO "secretly" 提供了 token-wise 奖励，将其作为密集奖励信号融入 PPO 训练（DPO 用于奖励学习 + PPO 用于策略优化 = RTO）。

## 方法详解

### 整体框架
RTO 是一个两阶段框架：
1. **Token-wise 奖励学习**：用 DPO 在偏好数据上训练策略 $\pi_{dpo}$，以此提取每个 token 的隐式奖励
2. **Token-wise 奖励优化**：将 DPO 提取的 token 级奖励结合 KL 正则化项和可选的句子级奖励，用 PPO 进行在线策略优化

### 关键设计

1. **MDP 建模 RLHF（从 Bandit 到 MDP）**:

    - 功能：将 RLHF 重新建模为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \rho, H)$
    - 状态 $s_h = (x, y_{1:h-1})$：prompt + 已生成 token
    - 动作 $a_h = y_h$：当前生成的 token
    - 转移 $\mathcal{P}$：确定性（因为 LLM 生成过程是拼接）
    - 奖励 $r(s_h, a_h)$：token 级奖励
    - 设计动机：精确捕捉 LLM 的自回归特性，实现更细粒度的奖励分配
    - **理论优势（Proposition 3.2）**：在确定性 MDP 中，用 token 级奖励找到最优响应需要 $A^{\min\{\xi+1,H\}}$ 样本，而句子级奖励需要 $A^H$ 样本（指数级差距）

2. **DPO 隐式 Token-wise 奖励提取**:

    - 功能：从 DPO 训练的策略中提取每个 token 的奖励信号
    - 关键推导：在 MDP + 确定性转移下，Bradley-Terry 偏好模型等价于：
    $\mathbb{P}(\tau^1 \succ \tau^2) = \sigma\left(\sum_{h=1}^{H} \beta \log \frac{\pi_\beta^*(a_h^1|s_h^1)}{\pi_{ref}(a_h^1|s_h^1)} - \sum_{h=1}^{H} \beta \log \frac{\pi_\beta^*(a_h^2|s_h^2)}{\pi_{ref}(a_h^2|s_h^2)}\right)$
    - 这恰好与 DPO 的学习目标一致，因此 DPO 学到的策略 $\pi_{dpo}$ 是 $\pi_\beta^*$ 的近似
    - Token-wise 奖励定义为：$r^*(s_h, a_h) = \beta \log \frac{\pi_{dpo}(y_h|x, y_{1:h-1})}{\pi_{ref}(y_h|x, y_{1:h-1})}$
    - 这是本文最核心的洞察：**"Your DPO model is secretly a token-wise reward model"**

3. **RTO 奖励函数设计**:

    - 功能：组合 token 级 DPO 奖励、KL 正则化和可选的句子级奖励为最终奖励
    - 关键公式（RTO reward, Eq 4.7）：
        - 对 $h \leq H-1$：$r_{rto} = \beta_1 \log \frac{\pi_{dpo}(y_h|...)}{\pi_{ref}(y_h|...)} - \beta_2 \log \frac{\pi(y_h|...)}{\pi_{ref}(y_h|...)}$
        - 对 $h = H$：额外加上 $\beta_3 \cdot r_{MLE}(x, y_{1:H})$
    - $\beta_1$ 控制 DPO 奖励强度，$\beta_2$ 控制 KL 正则化，$\beta_3$ 控制句子级奖励
    - 设计动机：句子级奖励 $r_{MLE}$ 防止响应过长或过短；DPO 奖励作为 reward shaping 提供密集信号

### 损失函数 / 训练策略
- 第一阶段：标准 DPO 损失训练 $\pi_{dpo}$
- 第二阶段：使用 Eq 4.7 计算的 $r_{rto}$ 作为 PPO 的奖励信号，执行标准 PPO 更新
- 句子级奖励模型 $r_{MLE}$ 可以比策略模型和 DPO 模型小得多（如 1B），使总计算成本接近标准 RLHF
- 实践中 $\beta_3 = 1$，$\beta_2$ 按标准 PPO 配置选取，$\beta_1$ 设小以避免 DPO 奖励主导

## 实验关键数据

### 主实验（AlpacaEval 2 & Arena-Hard）
| 方法 | AE LC↑ | AE WR↑ | AH SC↑ | AH WR↑ |
|------|--------|--------|--------|--------|
| SFT | 13.22 | 8.58 | 9.2 | 8.9 |
| DPO | 17.40 | 12.23 | 13.2 | 13.8 |
| R-DPO | 18.34 | 12.03 | 14.2 | 14.1 |
| SimPO | 25.46 | 20.20 | 14.5 | 15.2 |
| TDPO | 20.13 | 11.97 | 13.2 | 12.3 |
| PPO | 19.47 | 12.89 | 16.2 | 15.6 |
| **RTO** | **27.00** | **22.45** | **20.3** | **21.4** |

### 消融实验（奖励粒度 & Reward Shaping）
| 配置 | AE LC↑ | AE WR↑ | AH SC↑ | 说明 |
|------|--------|--------|--------|------|
| RTO（逐 token）| 27.00 | 22.45 | 20.3 | 密集奖励，最优 |
| Semi-RTO（逐句）| 23.77 | 19.17 | 19.0 | 句分隔符级奖励 |
| DDPO（EoS token）| 21.09 | 13.06 | 13.1 | 稀疏奖励，最差 |
| RS-PPO（reward shaping）| 27.52 | 21.69 | 19.2 | 验证 DPO 奖励是 shaping |

### 关键发现
- **RTO >> PPO**：AlpacaEval 2 LC +7.5 分，Arena-Hard SC +4.1 分
- **密集奖励 >> 稀疏奖励**：逐 token 明显优于逐句优于 EoS-only
- **DPO 奖励的角色是 reward shaping**：RS-PPO（总奖励等于 $r_{MLE}$ 但用 DPO 做 shaping）与 RTO 性能接近，说明 DPO 奖励的核心价值在于重分配奖励信号而非改变总奖励
- **卓越的数据效率**：RTO 用 1/8 数据即达 PPO 性能，且持续扩展——PPO 早饱和而 RTO 不断提升

## 亮点与洞察
- **设计优雅**：将 DPO 和 PPO 各自的优势有机结合——DPO 提供 token 级奖励信号，PPO 提供在线策略优化能力
- **理论扎实**：从 MDP vs Bandit 的样本复杂度分析到 RTO 的次优性保证，理论部分完整且有实际指导意义
- **核心洞察精彩**："DPO 模型隐含了 token-wise 奖励"这一观察虽然事后看很自然，但将其系统化地用于 PPO 训练是重要贡献
- 数据效率优势意味着 RTO 在数据稀缺场景下也能工作，拓宽了适用范围

## 局限与展望
- DPO 模型作为 $\pi_\beta^*$ 的近似可能不准确，尤其在分布外区域
- 仍需训练一个额外的 DPO 模型，增加了总计算量（虽然论文声称整体成本相当）
- 实验主要在 Llama-3-8B 上验证，对更大模型的效果仍待确认
- 可探索的方向：用其他直接偏好学习算法（如 IPO、KTO）替代 DPO 提取 token 级奖励

## 相关工作与启发
- 与 Rafailov et al. (2024) 的并行工作共享"DPO 隐含 token 级奖励"的核心洞察，但应用方向不同（本文用于 PPO，他们用于搜索）
- Token-level DPO (TDPO) 和 SimPO 属于直接偏好学习的改进，而 RTO 属于 PPO 改进路线
- 为"DPO + PPO 混合"这一新的 RLHF 范式奠定了基础
- 后续工作 (Cui et al., 2025; Yin et al., 2025) 将 RTO 应用于推理和聊天任务验证了方法的广泛适用性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
