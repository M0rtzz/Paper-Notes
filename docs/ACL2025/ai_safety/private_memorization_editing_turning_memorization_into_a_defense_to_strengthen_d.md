---
title: >-
  [论文解读] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models
description: >-
  [ACL 2025][AI安全][隐私保护] 提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。
tags:
  - ACL 2025
  - AI安全
  - 隐私保护
  - 模型编辑
  - 训练数据提取攻击
  - PII
  - 记忆化
---

# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2506.10024](https://arxiv.org/abs/2506.10024)  
**代码**: [GitHub](https://github.com/elenasofia98/PME) (有)  
**领域**: AI安全  
**关键词**: 隐私保护, 模型编辑, 训练数据提取攻击, PII, 记忆化  

## 一句话总结

提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。

## 研究背景与动机

### 1. 领域现状
LLM 随着参数规模增大，对训练数据的逐字记忆（verbatim memorization）能力也在增强。训练数据往往包含未经控制的个人身份信息（PII），如邮箱、电话号码、信用卡号等，这些信息可以通过训练数据提取攻击（Training Data Extraction, TDE）在推理时被提取。

### 2. 现有痛点
- **重新训练不可行**：一旦发现训练集中包含隐私信息，从头重训 LLM 成本过于高昂
- **现有编辑方法局限性**：Private Association Editing (PAE) 通过断开用户名与隐私信息的关联来保护隐私，但未直接解决逐字记忆导致的泄露
- **定位不准确**：传统模型编辑方法（如 MEMIT）通过因果分析预先定位关键层，但这种定位技术已被证明不能可靠地指导编辑

### 3. 核心矛盾
LLM 的记忆化能力是其泄露 PII 的根源，但同时也提供了一条线索——既然攻击者利用记忆化来提取 PII，防御者也可以利用同样的记忆化知识来精确定位并消除泄露。

### 4. 本文目标
提出一种高效的参数编辑方法，直接编辑已记忆的训练样本，将 PII 替换为虚拟占位符（如 mail@domain.com），同时最小化对模型通用语言建模能力的影响。

### 5. 切入角度
利用 Transformer 计算可分解为各组件输出之和的特性，通过几何方法（投影）估计每个层对 PII 生成的贡献度，再按贡献度比例编辑各层。

### 6. 核心 idea

**用几何投影估计每层对 PII 生成的贡献，按比例在所有层上编辑 Feed Forward 权重，将记忆的真实 PII 替换为隐私安全的虚拟 PII。**

## 方法详解

### 整体框架

PME 的核心流程：
1. 通过 TDE 攻击识别模型记忆的 PII 集合 $\mathcal{S} = \{(p, t) | \text{s.t. } M(p) = t\}$
2. 优化第 $L$ 层的目标表示 $x^*$，使模型生成虚拟 PII $t^*$
3. 通过几何投影估计每层的贡献系数 $w^l$
4. 在所有层上按贡献比例计算权重更新 $\Delta^l$

### 关键设计

#### 模块一：Transformer 输出的可加性分解

Transformer 的最后一层表示可以分解为各子组件输出之和：

$$x_n^L = x_n + \sum_{l=1}^{L} a_n^l + \sum_{l=1}^{L} h_n^l$$

PME 重点关注 Feed Forward 块的输出 $h_n^l$，因为它们被广泛证明负责信息存储：

$$h_n^l = f\left((a_n^l + x_n^{l-1}) W_{in}^l\right) W_{out}^l$$

#### 模块二：最优表示优化

通过梯度下降优化第 $L$ 层的偏移 $\delta^*$，使生成虚拟 PII $t^*$ 的概率最大化：

$$\delta^* = \arg\max_{\hat{\delta}} \mathcal{P}\left(t^* \mid \sigma\left((x_n^L + \hat{\delta}) W_U\right)\right)$$

隐私安全值 $x^* = x_n^L + \delta^*$。

#### 模块三：几何投影估计层贡献

PME 的核心创新在于使用投影来估计每层的贡献。截断和 $x_n^l \simeq \sum_{i=1}^l h_n^i$ 在方向 $x_n^L$ 上的投影：

$$w_p^l = \frac{x_n^l \cdot x_n^L}{\|x_n^L\|^2}$$

归一化得到贡献系数：

$$w^l = \frac{w_p^l}{\sum_{i=1}^{L-1} w_p^i}$$

#### 模块四：计算权重更新

新值按层贡献分配：$v^* = w^l \cdot x^*$

权重更新矩阵通过闭式解计算：

$$\Delta^l = (V^* - V_0^*) {K^*}^T (K_0 K_0^T + K^* {K^*}^T)^{-1}$$

最终更新：$\hat{W}_{out}^l = W_{out}^l + \Delta^l$

### 训练策略
- 编辑仅基于 200 token 长度的记忆 prompt
- 使用 Wikipedia 子集估计 $K_0 K_0^T$ 相关矩阵
- 单次前向传播即可估计层贡献（无需因果分析的额外计算开销）

## 实验关键数据

### 主实验：PME vs 基线方法的 PII 泄露减少

**GPT-Neo 1.3B（Email，200-token 记忆攻击）：**

| 方法 | 原始泄露 | 编辑后泄露 | 准确率降低 |
|------|---------|----------|----------|
| Pre-edit | 179 | - | - |
| **PME** | 179→**0** | 0 | **100%** |
| MEMIT | 179→1 | 1 | 99.44% |
| GRACE | 179→0 | 0 | 100% |
| DeMem | 179→88 | 88 | 50.84% |

**GPT-J 6B（Email，200-token 记忆攻击）：**

| 方法 | 泄露 | Δ Acc % |
|------|------|--------|
| PME | 1 | **99.65%** |
| MEMIT | 1 | 99.65% |

PME 在多数配置下将 PII 泄露降至 0 或接近 0。

### 跨 PII 类型对比

**GPT-Neo 2.7B：**

| PII 类型 | 原始泄露(200) | PME 编辑后 | 准确率降低 |
|---------|-------------|----------|----------|
| Email | 286 | 1 | 99.65% |
| Phone | 34→1 (1.3B) | 1 | 97.06% |
| URL | 75→16 (1.3B) | 16 | 78.67% |

URL 的防御效果相对较弱，因为 URL 结构更复杂、与更多上下文关联。

### 关键发现
1. **PME 在最具信息量的 200-token 场景中编辑，也能抵御 50/100-token 的较短上下文攻击和关联攻击**
2. **模型越大，记忆越多**：GPT-J 6B 比 GPT-Neo 1.3B 记忆了更多 PII
3. **编辑不影响通用能力**：后编辑模型在通用语言建模测试上与原始模型表现相当
4. PME 只需要一次额外的前向传播来估计层贡献，比 MEMIT 的因果分析更高效

## 亮点与洞察

1. **"以记忆对抗记忆"的思路非常巧妙**——将 LLM 记忆化这一通常被视为缺陷的特性转化为防御工具
2. **几何投影估计层贡献**优于传统因果分析：更高效（单次前向传播），且对每个样本独立估计而非预设一组层
3. **实验设计严谨**：在完全开放训练数据的模型上测试（GPT-Neo/GPT-J + The Pile），可以精确评估隐私泄露
4. **编辑编辑策略灵活**：仅用 200-token prompt 编辑，即可防御不同长度和类型的攻击

## 局限与展望

1. **URL 防御效果有限**：URL 的泄露减少仅约 79%，远低于 email 的 100%
2. 实验模型规模最大为 6B，对 70B+ 的大模型效果未知
3. 需要知道训练数据来检测记忆的 PII——对第三方模型审计场景受限
4. 虚拟 PII 替换策略（如 mail@domain.com）虽保留语义，但可能在某些下游任务中产生不一致

## 相关工作与启发

- **MEMIT**（Meng et al., 2023）：PME 的编辑基础，但 PME 改进了层定位方法
- **PAE**（Venditti et al., 2024）：断开人名-PII 关联的方法，PME 则直接编辑记忆序列
- **Training Data Extraction**（Carlini et al., 2021; Huang et al., 2022）：PME 的攻击评估框架
- **启发**：模型编辑技术在隐私保护中有巨大潜力，尤其对于不可重训的大模型，"精确编辑"比"全面遗忘"更加实用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — "记忆化→防御"的思路转换非常新颖，几何投影定位层贡献有原创性
- **实验充分度**: ⭐⭐⭐⭐ — 3 个模型、3 类 PII、多种攻击场景、5 种对比方法
- **写作质量**: ⭐⭐⭐⭐ — 方法推导清晰，从直觉到公式过渡自然
- **价值**: ⭐⭐⭐⭐⭐ — 隐私保护是 LLM 部署中的核心问题，方法高效且不需重训
