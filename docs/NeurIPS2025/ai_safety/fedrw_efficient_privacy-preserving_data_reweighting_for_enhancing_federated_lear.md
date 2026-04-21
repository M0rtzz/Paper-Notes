---
title: >-
  [论文解读] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models
description: >-
  [NeurIPS 2025][AI安全][联邦学习] FedRW 提出首个无需可信第三方的联邦学习隐私保护软去重框架，通过安全多方计算获取全局样本频率并进行频率感知的样本加权，在预处理上实现最高 28.78× 加速，在模型性能上实现约 11.42% 的 perplexity 改善。
tags:
  - NeurIPS 2025
  - AI安全
  - 联邦学习
  - 隐私保护
  - 数据去重
  - 样本加权
  - 安全多方计算
---

# FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.07505](https://arxiv.org/abs/2511.07505)  
**代码**: 无  
**领域**: ai_safety  
**关键词**: 联邦学习, 隐私保护, 数据去重, 样本加权, 安全多方计算

## 一句话总结
FedRW 提出首个无需可信第三方的联邦学习隐私保护软去重框架，通过安全多方计算获取全局样本频率并进行频率感知的样本加权，在预处理上实现最高 28.78× 加速，在模型性能上实现约 11.42% 的 perplexity 改善。

## 研究背景与动机

**领域现状**：大规模语料中的数据重复严重影响 LLM 性能和隐私安全，数据去重已成为训练流水线的标准预处理步骤。去重分为硬去重（直接删除）和软去重（重加权）。

**现有痛点**：联邦学习中由于隐私约束无法直接共享数据，全局去重面临困难。当前 SOTA 方法 EP-MPD 采用加密硬去重，但存在三个问题：(1) 硬删除可能丢失有信息量的样本；(2) 多轮密钥协商带来大量计算/通信开销；(3) 依赖可信第三方。

**核心矛盾**：局部去重无法检测跨客户端重复，而全局去重又受隐私约束，现有方法在"保隐私"和"保数据质量"之间难以平衡。

**本文目标**：设计一个不依赖可信第三方的隐私保护软去重框架，用频率感知加权替代硬删除。

**切入角度**：将样本权重设为全局频率的反函数，通过两两安全集合交集协议 (PSI) 获取频率信息。

**核心 idea**：用安全多方计算获取全局样本频率，以对数反函数加权替代硬删除，既保隐私又保数据多样性。

## 方法详解

### 整体框架

FedRW 分为三个阶段：(1) PPMPR 协议：各客户端通过两两安全计算获取每个本地样本的全局频率；(2) 并行编排加速：将 $O(n^2)$ 的两两交互优化为 $O(2^{\lceil\log_2 n\rceil})$；(3) 增强训练：用频率加权损失进行联邦 LLM 训练。

### 关键设计

1. **PPMPR 协议 (Privacy-Preserving Multi-Party Reweighting)**：

    - 功能：在不暴露原始数据的前提下，让每个客户端获得其本地样本的全局出现频率
    - 核心思路：将全局频率估计分解为 $\binom{n}{2}$ 次两方安全计算 (Π₂PC)。每次 Π₂PC 中，两个客户端 $P_i, P_j$ 通过 Private Set Intersection (PSI) 计算数据交集 $\mathcal{I}$，然后交换交集中样本的局部频率
    - 功能定义：$f_{\text{PPMPR}}(X_1,...,X_n) \to (W_1,...,W_n)$，将各客户端数据映射为权重向量
    - 设计动机：避免依赖可信第三方，每步只泄露交集信息和频率计数

2. **并行编排策略 (Parallel Orchestration)**：

    - 功能：将顺序执行的 $O(n^2)$ 次 Π₂PC 优化为并行执行
    - 核心思路：利用二叉树结构的层级合并。在每一层，不重叠的客户端对可以并发执行 Π₂PC。通过构建配对矩阵 $\mathcal{M}_k$，使用循环左移 $\text{RotL}(\vec{b}, k)$ 生成不冲突的配对方案
    - 复杂度：从 $O(n^2)$ 降到 $O(2^{\lceil\log_2 n\rceil} - 1)$，对 $n=50$ 个客户端实现 4.09–28.78× 加速

3. **频率感知加权训练**：

    - 功能：根据全局频率对训练样本的损失贡献进行加权
    - 权重公式：$\vec{\mathcal{W}} = \frac{1}{\ln(\vec{\mathcal{C}} + \vec{1}) + \vec{\varepsilon}}$，其中 $\vec{\mathcal{C}}$ 为全局频率向量
    - 加权损失：$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B \vec{\mathcal{W}}_i \cdot \ell_i^{(t)}}{\sum_{i=1}^B \vec{\mathcal{W}}_i}$
    - 设计动机：对数函数提供平滑的权重衰减，避免硬阈值导致的信息丢失；频繁样本被降权但不被完全排除，保留适度冗余有助于泛化

### 训练策略
模型更新仍使用标准 FedAvg 聚合，加权机制仅作用于 loss 层面，对训练框架无侵入性。

## 实验关键数据

### 主实验：GPT-2 Large, 30% 重复率

| 数据集 | Raw Data | Baseline (EP-MPD) | FedRW | 相对PPL改善 |
|--------|----------|---------|-------|---------|
| Haiku | 3.26 | 2.89 | **2.56** | 11.42% |
| Rotten Tomatoes | 2.65 | 2.21 | **1.61** | 27.15% |
| Short Jokes | 4.11 | 3.79 | **3.15** | 16.89% |
| Sonnets | 4.39 | 4.35 | **4.07** | 6.44% |

### 预处理效率对比

| 客户端数量 | EP-MPD 时间 | PPMPR 时间 | 加速比 |
|-----------|-----------|-----------|------|
| 10 | ~18s | ~1s | 17.61× |
| 30 | ~120s | ~5s | ~24× |
| 50 | ~300s | ~10s | 28.78× |

### 消融：Non-IID 设置 (Qwen3-0.6B, Rotten Tomatoes)

| 配置 | Baseline PPL | FedRW PPL | 说明 |
|------|-------------|----------|------|
| IID | 1.71 | **1.59** | 标准设置 |
| Quantity Skew | 2.02 | **1.96** | 数据量不均衡 |
| Label Skew | 2.44 | **1.66** | 标签分布不均，改善最显著 |
| Feature Skew | 3.43 | **2.70** | 不同数据类型，仍稳定有效 |

### 关键发现
- FedRW 在所有数据集和模型配置上一致优于硬去重基线
- 在文学结构严格的数据集（Sonnets, Haiku）上改善更显著，说明冗余对结构化文本影响更大
- 在 Non-IID 设置下（特别是 Label Skew），FedRW 的优势反而更大
- 随模型规模增加效果更好：Qwen2.5-7B 上 Twitter 数据集相对改善达 26.57%

## 亮点与洞察
- **软去重 vs 硬去重**：核心创新在于用加权替代删除，对数反函数 $1/\ln(freq+1)$ 是一个优雅的设计——高频样本被温和降权，适度冗余反而增强泛化
- **无需可信第三方**：纯两方 PSI 构建，安全性更强，部署更灵活
- **并行编排**：二叉树结构的客户端配对策略简洁高效，可直接复用到其他联邦安全计算场景

## 局限与展望
- 仅验证了文本数据，未扩展到多模态联邦学习
- 对数加权函数的选择缺乏理论最优性保证，可能存在更好的加权策略
- PSI 协议是半诚实模型下的安全保证，未考虑恶意客户端场景
- 重复模拟是人工注入的，和真实场景的自然冗余分布可能不同

## 相关工作与启发
- **vs EP-MPD (Abadi et al., 2024)**：EP-MPD 是硬去重 + 可信第三方，FedRW 是软去重 + 无第三方，在效率和效果上双重超越
- **vs SoftDedup / DoReMi**：这些是集中式软去重方法，无法直接用于隐私联邦场景，FedRW 将软去重思路迁移到了联邦环境
- **vs FedAvg**：FedRW 完全兼容 FedAvg 的聚合框架，只在 loss 层面加入加权，集成成本极低

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个联邦软去重框架，但核心技术（PSI + 加权损失）都是已有组件的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多数据集、多模型（GPT-2到Qwen3/Llama）、多设置（IID/Non-IID）、效率和性能双评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，协议描述规范，但部分符号过于密集
- 价值: ⭐⭐⭐⭐ 解决了联邦 LLM 训练中的实际痛点，框架设计通用性强

<!-- RELATED:START -->

## 相关论文

- [FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](../../ICCV2025/ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/ai_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)

<!-- RELATED:END -->
