---
title: >-
  [论文解读] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models
description: >-
  [ACL 2026][目标检测][扩散语言模型] 提出 AHD（Anchor-based History-stable Decoding），一种无需训练的即插即用动态解码策略，通过动态锚点回溯历史轨迹判定扩散LLM中跨块稳定token，实现早期解锁，在BBH上减少80%解码步数的同时提升3.67%性能。
tags:
  - ACL 2026
  - 目标检测
  - 扩散语言模型
  - 半自回归解码
  - 跨块稳定token
  - 动态锚点
  - 推理加速
---

# Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.08964](https://arxiv.org/abs/2604.08964)  
**代码**: [GitHub](https://github.com/zs1314/AHD)  
**领域**: LLM推理加速  
**关键词**: 扩散语言模型, 半自回归解码, 跨块稳定token, 动态锚点, 推理加速

## 一句话总结
提出 AHD（Anchor-based History-stable Decoding），一种无需训练的即插即用动态解码策略，通过动态锚点回溯历史轨迹判定扩散LLM中跨块稳定token，实现早期解锁，在BBH上减少80%解码步数的同时提升3.67%性能。

## 研究背景与动机

**领域现状**：扩散大语言模型（dLLMs）如LLaDA已成为自回归LLM的有力替代。半自回归（Semi-AR）解码被广泛采用——将输出序列分为多个block从左到右顺序解码，每个block内用扩散迭代去噪。

**现有痛点**：Semi-AR解码中存在严重的"块边界延迟"问题——许多token在对应block解码之前就已经收敛到最终值并保持稳定，但被迫等到所属block轮次才能解码。这些"跨块稳定token"的延迟解码不仅浪费了大量解码步数，还因为压制了局部区域的辐射效应导致性能下降。

**核心矛盾**：如何准确识别跨块稳定token？现有方法（基于置信度/熵的单步判断）不可靠——(1) 已稳定token仍可能出现局部波动导致误判；(2) 标准解码中历史信息是孤立的，每步预测仅依赖上一步。

**本文目标**：打破Semi-AR解码的块边界约束，通过早期解锁跨块稳定token同时提升效率和性能。

**切入角度**：三个关键洞察——(1) 朴素前瞻解码不可靠（局部波动）；(2) token稳定性与收敛趋势高度相关（绝对稳定趋势）；(3) 历史信息在标准解码中被孤立。因此需要引入历史轨迹信息来判定全局稳定性。

**核心 idea**：在每个解码步以当前步为动态锚点，回溯历史缓冲区计算锚定一致性分数（anchored consistency score），捕捉token的绝对稳定趋势，一旦确认稳定即跨块早期解码。

## 方法详解

### 整体框架
在Semi-AR解码基础上，AHD将序列分为当前块 $B_{current}^t$ 和未来块 $B_{future}^t$。当前块内使用置信度感知并行解码；未来块内，AHD维护每个位置的历史缓冲区，通过动态锚点回溯计算稳定性，满足条件的token被提前解锁加入解码集合。

### 关键设计

1. **历史缓冲区与动态锚点（Historical Buffer + Dynamic Anchor）**:

    - 功能：为每个未来块位置维护历史分布轨迹，实现跨步稳定性监测
    - 核心思路：对未来块中的每个位置 $j$，维护长度 $H$ 的历史缓冲区 $\mathcal{H}_j^t = \{P_j^{t-H+1}, ..., P_j^t\}$。以当前步 $P_j^t$ 为动态锚点，回溯计算锚定KL散度 $\delta_j^{t,\tau} = D_{KL}(P_{j,anchor}^t || P_j^{t-\tau})$
    - 设计动机：单步置信度/熵对局部波动敏感，而基于锚点的历史一致性提供了全局视角，能在绝对稳定趋势早期就捕获信号

2. **锚定一致性分数（Anchored Consistency Score）**:

    - 功能：聚合历史窗口内的稳定性证据，做出可靠的跨块解码判断
    - 核心思路：对历史一致性序列 $\{\delta_j^{t,1}, ..., \delta_j^{t,H-1}\}$ 做指数衰减加权求和得到 $D_j^t(acs) = \sum_{\tau=1}^{H-1} w_\tau \delta_j^{t,\tau}$，其中 $w_\tau = e^{-\lambda\tau}/Z$ 赋予近期历史更高权重。当 $D_j^t(acs) < \varepsilon$ 时判定该token已达绝对稳定趋势
    - 设计动机：指数衰减权重兼顾了近期变化的敏感性和长期趋势的稳健性，阈值 $\varepsilon$ 控制解锁的保守程度

3. **跨块早期解锁（Cross-block Early Unlocking）**:

    - 功能：打破块边界，提前解码已稳定的未来块token
    - 核心思路：将满足稳定性条件的未来块位置集合 $G_f^t = \{j | j \in B_{future} \wedge D_j^t(acs) < \varepsilon\}$ 与当前块的解码集合 $G_c^t$ 合并为 $G_{unmasked}^t$，一并更新序列
    - 设计动机：稳定token具有"辐射效应"——一个token稳定后会加速邻近token的收敛。早期解锁释放了这种辐射效应，不仅加速推理还提升生成质量

### 损失函数 / 训练策略
AHD是一种无需训练的即插即用方法，直接在推理阶段应用。默认超参：历史缓冲区长度 $H=6$，一致性阈值 $\varepsilon=0.01$，衰减率 $\lambda$ 控制权重分布。

## 实验关键数据

### 主实验（LLaDA-8B-Instruct）

| 任务 | 指标 | AHD | Vanilla | 步数减少 |
|------|------|-----|---------|----------|
| BBH | Score↑ | 56.78 | 53.11 | 80% |
| HumanEval | Score↑ | 43.29 | 40.85 | 70% |
| MBPP | Score↑ | 31.20 | 29.20 | 74% |
| MMLU-Pro | Score↑ | 37.42 | 35.57 | 48% |
| Asdiv | Score↑ | 77.09 | 75.57 | 76% |

### 消融实验

| 方法 | BBH Score | 步数减少 | 说明 |
|------|-----------|----------|------|
| Vanilla | 53.11 | 0% | 标准解码 |
| Fast-dLLM | 53.17 | 78% | 性能持平但无提升 |
| KLASS | 53.03 | 62% | 轻微下降 |
| Saber | 52.88 | 66% | 性能下降 |
| AHD | 56.78 | 80% | 唯一同时提升性能和效率的方法 |

### 关键发现
- AHD是唯一能在加速的同时提升性能的方法，其他加速策略（Saber、KLASS）往往导致性能下降
- 在LLaDA-1.5上同样有效，BBH提升+1.55，步数减少78%，证明方法的通用性
- 扩展到视觉-语言（MMaDA）和音频-语言（DIFFA）领域同样有效，证明跨模态适用性

## 亮点与洞察
- **"稳定token的辐射效应"**：发现稳定token呈聚类模式出现，一个token稳定会加速邻近token收敛。这个洞察对理解扩散LLM的解码动力学很有价值
- **"加速即提升"的反直觉发现**：早期解锁不仅加速推理，还因释放辐射效应提升了生成质量。这挑战了"速度vs质量trade-off"的常见假设
- **锚点回溯机制的通用性**：这种基于历史轨迹判定稳定性的方法可迁移到任何迭代式生成过程（如扩散图像生成中的像素级提前确定）

## 局限与展望
- 需要维护历史缓冲区增加了内存开销，对于超长序列生成可能成为瓶颈
- 阈值 $\varepsilon$ 和缓冲区长度 $H$ 需要针对不同模型/任务调优
- 目前主要在LLaDA系列上验证，其他dLLM架构（如MDLM）的适用性待验证
- 理论分析假设了token稳定性的单调收敛性，极端情况下可能不成立

## 相关工作与启发
- **vs Fast-dLLM**: Fast-dLLM使用置信度阈值加速但性能持平；AHD通过历史轨迹判定实现加速+提升双赢
- **vs Saber**: Saber使用预测器选择性去噪但导致性能下降；AHD的动态锚点方法更鲁棒
- **vs PC-sampler**: PC-sampler修改采样过程但不减少步数；AHD直接减少70-80%步数

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 三个洞察→动态锚点方法的推导链路严谨自然，"加速即提升"的发现很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 7个语言benchmark+5个视觉+5个音频，两个dLLM模型，5个baseline对比
- 写作质量: ⭐⭐⭐⭐⭐ 观察→洞察→方法的叙事流畅，图表设计出色（特别是热力图分析）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](../../CVPR2026/object_detection/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/object_detection/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)

<!-- RELATED:END -->
