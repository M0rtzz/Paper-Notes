---
title: >-
  [论文解读] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing
description: >-
  [NeurIPS 2025][AI安全][隐私推理] 首个支持 MoE 架构 LLM 隐私推理的框架 CryptoMoE，通过平衡专家路由隐藏路由信息、置信度感知调度协议和批量密文矩阵乘法协议，相比 dense baseline 实现 2.8~3.5× 延迟降低和 2.9~4.3× 通信量降低，准确率损失仅 0.8%。
tags:
  - NeurIPS 2025
  - AI安全
  - 隐私推理
  - MoE
  - 同态加密
  - 安全多方计算
  - 专家路由
---

# CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing

**会议**: NeurIPS 2025  
**arXiv**: [2511.01197](https://arxiv.org/abs/2511.01197)  
**代码**: https://github.com/PKU-SEC-Lab/CryptoMoE  
**领域**: ai_safety  
**关键词**: 隐私推理, MoE, 同态加密, 安全多方计算, 专家路由

## 一句话总结
首个支持 MoE 架构 LLM 隐私推理的框架 CryptoMoE，通过平衡专家路由隐藏路由信息、置信度感知调度协议和批量密文矩阵乘法协议，相比 dense baseline 实现 2.8~3.5× 延迟降低和 2.9~4.3× 通信量降低，准确率损失仅 0.8%。

## 研究背景与动机

**领域现状**：MoE 架构已被 LLaMA-4、DeepSeek-V3、Qwen-3 等主流 LLM 采用，通过稀疏激活实现高效大容量模型。同时基于 HE/MPC 的隐私推理框架已可以支持 GPT-2、LLaMA-1 等 dense 模型。

**现有痛点**：现有隐私推理框架（如 BOLT、Bumblebee）仅支持 dense 架构，无法处理 MoE 层的动态路由机制。MoE 的专家选择模式高度依赖输入——数学问题和文本理解任务激活的专家明显不同——暴露路由信息等于泄露输入类型。

**核心矛盾**：保护路由隐私最直观的方法是让所有 token 通过所有专家（dense baseline），但这将计算量放大 8~15×，完全抵消 MoE 的效率优势。

**本文目标** 在保护路由隐私的前提下，(a) 避免巨大的计算开销；(b) 设计安全的 token 分配和聚合协议；(c) 维持接近原始模型的准确率。

**切入角度**：每个专家处理固定数量 $t$ 个 token（平衡路由），使专家负载与输入无关，从而隐藏路由信息。

**核心 idea**：用推理时平衡专家路由 + 置信度优先选择来保护隐私，用批量密文矩阵乘法消除 HE 旋转瓶颈。

## 方法详解

### 整体框架
MoE 层的隐私推理分四步：❶ Gate Routing（安全 softmax + top-k）→ ❷ Secure Dispatch（将 token 分配到专家）→ ❸ Expert Compute（密文线性层）→ ❹ Secure Combine（聚合专家输出）。CryptoMoE 的核心改进在 ❷❸❹ 步。

### 关键设计

1. **推理时平衡专家路由（Balanced Expert Routing）**:

    - 功能：每个专家固定处理 $t$ 个 token，不足补零、超出丢弃
    - 核心思路：设 $t = mk/n$（$m$ 为 token 数，$k$ 为每 token 激活专家数，$n$ 为专家总数），与原始 MoE 计算量相同。由于路由天然不均衡，实际设 $t = 2mk/n$ 取得更好精度-效率平衡
    - 设计动机：专家负载固定意味着服务器无法从计算模式推断输入类型，实现路由隐私保护

2. **置信度感知安全调度协议（Confidence-Aware Secure Dispatch）**:

    - 功能：在固定容量限制下，每个专家优先保留路由置信度最高的 $t$ 个 token
    - 核心思路：三步——❶ 用 $\Pi_{\text{equal}}$ 计算每个 token 对每个专家的 boolean mask $[[M_i]]$；❷ 用 $\Pi_{\text{mux}}$ 结合路由权重得到优先级分数 $[[S_i]]$，再用 $\Pi_{\text{topk}}$ 选出 top-$t$；❸ 用 $\Pi_{\text{onehot}}$ + $\Pi_{\text{matmul}}$ 检索对应 token 嵌入
    - 相比 CipherPrune 的通信复杂度从 $O(kmtd)$ 降至 $O(km\log(km))$，因为将打分与嵌入检索解耦

3. **安全聚合协议（Secure Combine）**:

    - 功能：将各专家输出按原始 token 顺序重排并加权聚合
    - 核心思路：复用 dispatch 阶段的 one-hot 矩阵转置后做 $\Pi_{\text{matmul}}$，同时乘以路由权重。仅需一次 $\Pi_{\text{mul}}$ + 一次 $\Pi_{\text{matmul}}$
    - dispatch + combine 总通信开销仅占 MoE 层的 ~18%

4. **批量密文矩阵乘法（Batch MatMul）**:

    - 功能：将 $n$ 个专家的 token 嵌入打包到同一密文中
    - 核心思路：原始方案每个专家 $t \times d_1$ 独立打包，hidden dimension 大→旋转多。Batch MatMul 将所有专家的局部嵌入打包为 $(nt \times d_1/n)$，HE 旋转次数从 $O(nd_1)$ 降至 $O(d_1)$
    - 效果：专家线性层计算加速 3~6×，端到端延迟降低 2~3×

### 损失函数 / 训练策略
本文为推理框架，不涉及额外训练；直接使用预训练的 DeepSeekMoE、OLMoE、QWenMoE 模型。

## 实验关键数据

### 主实验（端到端性能，LAN 设置，$t=2.0$）

| 模型 | 方法 | Avg Acc(%) | LAN 延迟(min/tok) | 通信(GB) |
|---|---|---|---|---|
| DeepSeekMoE-16.4B | Dense baseline | 62.2 | 2.33 | 9.16 |
| | CipherPrune | 59.8 | 1.14 | 5.55 |
| | **CryptoMoE** | **61.8** (-0.4) | **0.76** (3.1×) | **2.46** (3.7×) |
| OLMoE-6.9B | Dense baseline | 63.0 | 0.99 | 3.82 |
| | **CryptoMoE** | **62.5** (-0.5) | **0.36** (2.8×) | **1.31** (2.9×) |
| QWenMoE-14.3B | Dense baseline | 62.0 | 1.98 | 7.61 |
| | **CryptoMoE** | **62.0** (-0.0) | **0.56** (3.5×) | **1.76** (4.3×) |

### 消融实验（LAN 延迟，DeepSeekMoE）

| 配置 | Acc(%) | 延迟(min/tok) |
|---|---|---|
| Dense Baseline | 62.2 | 2.33 |
| + Balanced Expert Routing | 57.9 | 1.20 |
| + Confidence-aware selection | 61.8 | 1.20 |
| + Batch MatMul | 61.8 | **0.76** |

### 关键发现
- 置信度感知选择是保精度的关键：无它精度从 62.2% 暴跌至 57.9%（-4.3%），加上后恢复至 61.8%
- Batch MatMul 不影响精度但延迟减半（1.20→0.76 min/tok）
- CryptoMoE 在某些配置下甚至比 insecure baseline 更快，因为 insecure baseline 无法利用 batch 打包优化
- 扩展到 Mixtral-47B 和 LLaMA4-Scout-109B 仍保持 98.8%~100% 精度

## 亮点与洞察
- **平衡路由的隐私保护思路**极其简洁优雅：固定每个专家处理的 token 数，路由信息就无法从计算侧信道泄露。这个思路可推广到任何含动态分支的隐私推理场景
- **将 token 打分与嵌入检索解耦**的 dispatch 协议设计巧妙，使通信复杂度从 $O(kmtd)$ 降到 $O(km\log(km))$，是一种通用的"先选索引再查数据"范式
- Batch MatMul 利用 MoE 专家并行的结构特点做密文打包，实现了 $n$ 倍旋转操作降低

## 局限与展望
- WAN 设置下 $\Pi_{\text{topk}}$ 因通信轮次过多成为严重瓶颈，需要更 round-efficient 的协议
- 对短序列（<64 token）效果不佳，因为平衡路由在 token 极少时无法有效平衡
- 内存开销巨大，对 Mixtral-47B 和 LLaMA4-109B 无法在单机完成隐私推理
- 仅评测零样本推理任务，未验证生成（auto-regressive decoding）场景

## 相关工作与启发
- **vs CipherPrune (ICLR'25)**: 其 token 剪枝协议直接应用于 MoE 会泄露每个专家的 token 数，通信开销大 4.3×
- **vs BOLT/Bumblebee**: 仅支持 dense transformer，CryptoMoE 首次扩展到 MoE
- **vs Insecure baseline**: 完全暴露路由信息作为上界，CryptoMoE 在某些情况下效率反而更高

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 MoE 隐私推理框架，问题定义和解决方案都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个模型、八个任务、两种网络环境、详细消融和扩展性分析
- 写作质量: ⭐⭐⭐⭐ 协议描述清晰，toy example 图示直观，但符号较多需要反复查阅
- 价值: ⭐⭐⭐⭐⭐ 直接解决 MoE 模型部署中的实际隐私痛点，代码开源

<!-- RELATED:START -->

## 相关论文

- [FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](../../ACL2025/ai_safety/centaur_bridging_the_impossible_trinity_of.md)
- [MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)
- [SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute](../../AAAI2026/ai_safety/secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)
- [Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

<!-- RELATED:END -->
