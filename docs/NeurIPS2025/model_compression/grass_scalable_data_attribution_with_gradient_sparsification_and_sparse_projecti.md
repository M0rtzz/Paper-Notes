---
title: >-
  [论文解读] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection
description: >-
  [NeurIPS 2025][模型压缩][数据归因] 提出 GraSS 与 FactGraSS 两阶段梯度压缩算法，利用逐样本梯度的固有稀疏性实现**亚线性**时间与空间复杂度（$O(k')$），在十亿参数模型上比 SOTA 基线 LoGra 快 **165%**，同时保持数据归因质量。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 数据归因
  - 梯度压缩
  - 稀疏投影
  - 影响函数
  - 随机投影
---

# GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection

**会议**: NeurIPS 2025  
**arXiv**: [2505.18976](https://arxiv.org/abs/2505.18976)  
**代码**: [GitHub](https://github.com/TRAIS-Lab/GraSS)  
**领域**: 模型压缩 / 数据归因  
**关键词**: 数据归因, 梯度压缩, 稀疏投影, 影响函数, 随机投影

## 一句话总结

提出 GraSS 与 FactGraSS 两阶段梯度压缩算法，利用逐样本梯度的固有稀疏性实现**亚线性**时间与空间复杂度（$O(k')$），在十亿参数模型上比 SOTA 基线 LoGra 快 **165%**，同时保持数据归因质量。

## 研究背景与动机

梯度数据归因（如影响函数）需计算每个训练样本的逐样本梯度 $g_i = \nabla_\theta \ell(z_i; \hat{\theta})$，再进行 Fisher 信息矩阵逆向量积（iFVP）。对 $n$ 个训练样本、$p$ 维参数的模型，存储复杂度为 $O(np)$，构成大模型瓶颈。

**现有压缩方法的局限**：
- **Dense Random Projection**：Johnson-Lindenstrauss 保证，但投影开销 $O(kp)$
- **FJLT**（Trak 使用）：$O((p+k)\log p)$，但不利用输入稀疏性
- **LoGra**：利用线性层梯度的 Kronecker 积结构，降至 $O(\sqrt{pk})$，为当前 SOTA

**核心观察**：逐样本梯度天然具有高稀疏性（ReLU 等激活函数造成），这一特性是 mini-batch 梯度所不具备的，但现有方法均未显式利用。

## 方法详解

### 第一阶段：稀疏投影（SJLT）

通过稀疏化投影矩阵 $P$，每列仅保留 $s = o_\epsilon(k)$ 个非零元素。当输入 $g$ 本身稀疏时，SJLT 复杂度降为：

$$O(s \cdot \text{nnz}(g))$$

其中 $\text{nnz}(g) = \|g\|_0$ 为非零元素个数。作者设 $s=1$ 以最大化速度，并开发了专用 CUDA kernel 解决线程竞争和不规则内存访问问题。

### 第二阶段：稀疏化（Mask）

- **Random Mask (RM)**：随机选取 $k$ 个坐标，复杂度 $O(k)$
- **Selective Mask (SM)**：通过可微优化选择关键坐标：

$$S^* = \arg\max_{S \in \mathbb{R}^p} \mathbb{E}_{z_{\text{test}}} \left[\text{corr}\left((\langle g_i, g_{z_{\text{test}}}\rangle), (\langle \hat{g}_i, \hat{g}_{z_{\text{test}}}\rangle)\right)\right] - \lambda \|\sigma(S)\|_1$$

其中 $\hat{g}_i = \sigma(S) \odot g_i$，$\ell_1$ 正则促进二值化掩码。

### GraSS：两阶段组合

1. **稀疏化**：将 $p$ 维梯度降至 $k'$ 维（$k < k' \ll p$）
2. **稀疏投影**：对 $k'$ 维向量用 SJLT 投影到目标维度 $k$

总复杂度 $O(k' + k') = O(k')$，亚线性于 $p$。

### FactGraSS：线性层专用变体

直接整合 GraSS 与 LoGra 的问题：LoGra 将投影分解为两个小问题，小问题尺寸下 SJLT 反而比密集投影慢。FactGraSS 三步解决：

1. **分解稀疏化**：对输入 $z_{i,l}^{\text{in}}$ 和梯度 $\mathcal{D}z_{i,l}^{\text{out}}$ 分别稀疏化到 $k_l^{\text{in}'}$ 和 $k_l^{\text{out}'}$ 维
2. **重构**：通过 Kronecker 积构造 $k_l' = k_l^{\text{in}'} \times k_l^{\text{out}'}$ 维"稀疏化梯度"
3. **稀疏投影**：对重构结果用 SJLT 投影到 $k_l$ 维

**无需物化完整梯度**，总复杂度 $O(k_l')$。当 $c \leq \sqrt{p_l/k_l}$ 时（$k_l' = ck_l$），FactGraSS 理论快于 LoGra。

| 方法 | 类型 | 复杂度 |
|------|------|--------|
| Gauss | 基线 | $O(pk)$ |
| FJLT | 基线 | $O((p+k)\log p)$ |
| LoGra | 基线（线性层） | $O(\sqrt{p_l k_l})$ / 层 |
| GraSS | 本文 | $O(k')$ |
| FactGraSS | 本文（线性层） | $O(k_l')$ / 层 |

## 实验关键数据

### 小规模定量评估（LDS）

**MLP + MNIST**（Trak 框架，$k=4096$）：

| 方法 | LDS | 压缩时间 (s) |
|------|-----|-------------|
| Gauss | 0.4253 | 8.74 |
| FJLT | 0.4359 | 4.33 |
| SJLT | **0.4280** | 0.52 |
| RM | 0.4054 | **0.15** |
| SM | 0.4163 | **0.13** |

**GPT2-small + WikiText**（影响函数，$k_l = 64 \times 64$）：

| 方法 | LDS | 效率 |
|------|-----|---------|
| LoGra | 0.348 | 基线 |
| SJLT | 0.354 | 较慢 |
| Mask | 0.340 | 极快 |
| **FactGraSS** | **0.352** | **LoGra 的 250% 加速** |

### 大规模效率评估

在 **Llama-2-7B**（70亿参数）+ C4 数据集上：
- FactGraSS 压缩吞吐量比 LoGra **快 165%**
- 内存占用显著降低，支持更大 batch

### 关键发现

1. 单独的 Random Mask 即可达非平凡 LDS，Selective Mask 进一步提升
2. SJLT 在大问题尺寸下比密集投影快且精确，但小问题尺寸下需要 FactGraSS 绕过瓶颈
3. GraSS 在效率-精度权衡中占据最佳位置

## 亮点与洞察

- **稀疏性是免费的午餐**：逐样本梯度的天然稀疏性被所有先前方法忽略，利用后可获数量级加速
- **专用 CUDA kernel**：解决 SJLT 在 PyTorch 中的竞争条件和不规则访存问题
- **亚线性理论保证**：$O(k')$ 复杂度独立于模型参数量 $p$，理论上可无限扩展
- **FactGraSS 巧妙避开双重瓶颈**：既不物化完整梯度（$O(p)$），又避免小问题尺寸下 SJLT 低效

## 局限与展望

1. SJLT 的 CUDA kernel 是方法成功的关键，对非 GPU 硬件的适用性未知
2. Selective Mask 需要一次性优化开销（解 Eq.1），大模型上的可扩展性有待验证
3. 稀疏性假设依赖 ReLU 等激活函数，对 GELU/SiLU（现代 LLM 常用）的稀疏度可能不同
4. FactGraSS 仅适用于线性层，对 attention 等非线性运算需用通用 GraSS
5. 实验中未与 TRAK 以外的数据归因框架（如 DataInf）对比

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统利用逐样本梯度稀疏性加速数据归因
- **技术深度**: ⭐⭐⭐⭐ — SJLT + Mask + Kronecker 分解三者的精巧组合
- **实验充分度**: ⭐⭐⭐⭐ — 从 MLP 到 70B 参数模型，定量与效率评估全面
- **实用性**: ⭐⭐⭐⭐⭐ — 开源、通用、大模型友好
- **总体**: ⭐⭐⭐⭐

## 与相关工作的对比

## 启发与关联

## 评分
