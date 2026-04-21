---
title: >-
  [论文解读] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
description: >-
  [ICLR 2026][模型压缩][后训练量化] 提出 ParoQuant，通过硬件高效且可优化的独立 Givens 旋转与通道缩放相结合来消除权重异常值，在推理 LLM 上实现高精度低开销的 4-bit 权重量化。
tags:
  - ICLR 2026
  - 模型压缩
  - 后训练量化
  - Givens旋转
  - 推理LLM
  - 量化效率
  - 算法-系统协同设计
---

# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

**会议**: ICLR 2026  
**arXiv**: [2511.10645](https://arxiv.org/abs/2511.10645)  
**代码**: [项目页](https://paroquant.z-lab.ai)  
**领域**: 模型压缩  
**关键词**: 后训练量化, Givens旋转, 推理LLM, 量化效率, 算法-系统协同设计

## 一句话总结

提出 ParoQuant，通过硬件高效且可优化的独立 Givens 旋转与通道缩放相结合来消除权重异常值，在推理 LLM 上实现高精度低开销的 4-bit 权重量化。

## 研究背景与动机

LLM 量化面临精度和效率的两难：
- **AWQ**：快速但精度损失大（如 Qwen3-4B 在 MMLU-Pro 上降 2.8%），推理 LLM 的长链思维使量化误差逐步累积
- **QTIP**：精度高但比 AWQ 慢约 30%，因为 Hadamard 变换引入了显著开销
- 推理模型需要生成数万 token，对量化的精度和效率要求更高

核心观察：

**旋转有效抑制异常值**，但全旋转矩阵计算代价大

**稀疏参数化的旋转同样有效**——仅保留 top-10% 通道对即可匹配全旋转效果

## 方法详解

### 整体框架

ParoQuant 设计了 Scaled Pairwise Rotation 变换，由多个独立旋转和通道缩放组成，配合层级优化和高效推理内核实现端到端加速。

### 关键设计

1. **Givens 旋转分解**：

    - 选择一小组通道对 $\mathcal{P} = \{(i_1,j_1), \ldots, (i_m,j_m)\}$
    - 每对执行平面旋转：$\mathbf{W}^{(k)}[i,:] = \cos\theta_k \cdot \mathbf{W}^{(k-1)}[i,:] - \sin\theta_k \cdot \mathbf{W}^{(k-1)}[j,:]$
    - 仅需少量向量化乘加运算，避免全矩阵乘法

2. **独立旋转 (Independent Rotation)**：

    - 约束每个通道最多出现在一个旋转对中（$P_k \cap P_l = \emptyset$）
    - 所有 Givens 旋转可完全并行化，充分利用 GPU 并行性
    - 自然兼容分组量化：每个量化组内独立旋转

3. **串联独立旋转 + 通道缩放**：

    - 单次独立旋转仅有 $n/2$ 个参数，表达力有限
    - 顺序施加 $K$ 次独立旋转（默认 $K=8$）提升拟合能力
    - 通道缩放 $\text{diag}(\boldsymbol{\alpha})$ 直接均衡通道幅度
    - 最终变换：$T_{\mathcal{P},\Theta,\boldsymbol{\alpha}}(\mathbf{W}) = (\prod_{t=1}^K R(\mathcal{P}_t, \Theta_t)) \cdot \text{diag}(\boldsymbol{\alpha}) \cdot \mathbf{W}$

### 损失函数 / 训练策略

- **层级优化**：$\mathcal{L}(Q) = \|Q(D)(\mathbf{X'}) - D(\mathbf{X})\|$
- 两阶段优化：先优化旋转角度和缩放因子，再用 QAT-like 方法微调权重和量化参数 $s, z$
- 每层优化 10 epoch，使用 AdamW，三个数据集（WikiText2, C4, RedPajama）均匀采样
- 推理内核利用三级并行：token 维度、通道组维度、旋转对维度

## 实验关键数据

### 主实验（困惑度 - W4G128 量化）

| 模型 | 方法 | WikiText2 PPL | C4 PPL | 推理加速 |
|------|------|-------------|--------|---------|
| LLaMA-3-8B | FP16 | 5.54 | 7.10 | 1.0× |
| | AWQ | 5.92 | 7.42 | **2.4×** |
| | QTIP | 5.69 | 7.22 | 1.7× |
| | **ParoQuant** | **5.68** | **7.17** | **2.2×** |
| Qwen3-4B | AWQ | 7.36 | 7.89 | 2.4× |
| | QTIP | 7.09 | 7.68 | 1.7× |
| | **ParoQuant** | **7.03** | **7.63** | 2.2× |

### 推理任务精度（DeepSeek-R1-distilled LLaMA-3.1-8B）

| 方法 | MMLU-Pro | GPQA Diamond | AIME-24 | AIME-25 | 平均 |
|------|---------|-------------|---------|---------|------|
| FP16 | 52.4 | 43.9 | 56.7 | 40.0 | 48.3 |
| AWQ | 49.3 | 40.4 | 46.7 | 26.7 | 40.8 |
| **ParoQuant** | **52.5** | **41.4** | **53.3** | **36.7** | **46.0** |

### 关键发现

- ParoQuant 在推理任务上平均比 AWQ 提升 2.4%，开销不到 10%
- 精度匹配 QTIP（向量量化 SOTA），但速度快约 25%
- 在 Qwen3 系列（1.7B-14B）上效果尤其显著，小模型量化更具挑战

## 亮点与洞察

- 算法-系统协同设计：独立旋转的约束既保证了数学优化空间，又天然适合 GPU 并行
- 分析精辟：仅 10% 的通道对就能匹配全旋转效果，揭示了正交变换的冗余性
- 对推理 LLM 特别关注，结合长链思维的量化误差累积问题分析透彻
- 在线旋转内核利用共享内存和寄存器，多个独立旋转可融合为单次 kernel 调用

## 局限与展望

- 目前主要验证 4-bit 线性量化，未探索 2-3 bit 场景
- 独立旋转的通道对选择策略（随机+去重）可能非最优
- 旋转数 K=8 是经验值，不同模型可能需要不同 K
- 未开源时可能限制社区采用

## 相关工作与启发

- 与 QuaRot/SpinQuant 的区别：ParoQuant 使用可优化的独立 Givens 旋转而非固定 Hadamard 变换
- 与 AWQ 的区别：在通道缩放基础上增加旋转变换，大幅提升异常值抑制能力
- 启示：推理 LLM 时代，量化方法需要重新权衡精度和效率

## 评分

- 新颖性: ⭐⭐⭐⭐ 独立 Givens 旋转的设计新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多任务多指标全面验证
- 写作质量: ⭐⭐⭐⭐ 动机分析清楚，但公式较多
- 价值: ⭐⭐⭐⭐⭐ 推理 LLM 量化的实用解决方案

<!-- RELATED:START -->

## 相关论文

- [A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [Efficient Reasoning with Balanced Thinking](efficient_reasoning_with_balanced_thinking.md)
- [Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)

<!-- RELATED:END -->
