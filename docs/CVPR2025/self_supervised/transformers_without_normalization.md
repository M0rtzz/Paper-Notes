---
title: >-
  [论文解读] Transformers without Normalization
description: >-
  [CVPR 2025][自监督学习][Dynamic Tanh] 发现 LayerNorm 的输入-输出映射呈 tanh 形状，提出 Dynamic Tanh (DyT) 作为归一化层的即插即用替代：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，在视觉/语言/扩散/语音等多任务中与 LN 性能持平甚至更优。
tags:
  - CVPR 2025
  - 自监督学习
  - Dynamic Tanh
  - 归一化替代
  - LayerNorm
  - Transformer
  - 激活压缩
---

# Transformers without Normalization

**会议**: CVPR 2025  
**arXiv**: [2503.10622](https://arxiv.org/abs/2503.10622)  
**代码**: https://jiachenzhu.github.io/DyT (有)  
**领域**: 自监督学习 / 网络架构  
**关键词**: Dynamic Tanh, 归一化替代, LayerNorm, Transformer架构, 激活压缩

## 一句话总结

发现 LayerNorm 的输入-输出映射呈 tanh 形状，提出 Dynamic Tanh (DyT) 作为归一化层的即插即用替代：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，在视觉/语言/扩散/语音等多任务中与 LN 性能持平甚至更优。

## 研究背景与动机

**领域现状**：归一化层（LayerNorm/RMSNorm）是 Transformer 的标配组件，通过计算 token 级统计量将激活归一化到标准分布。几乎所有 Transformer 变体都离不开归一化。

**现有痛点**：归一化层需要在线计算每个 token 的均值和方差，增加了计算开销和实现复杂度。更重要的是，归一化的"必要性"从未被真正理解——它到底在做什么？

**核心矛盾**：归一化被认为是稳定训练的必需品，但没有人系统研究过它是否可以被更简单的操作替代。

**切入角度**：可视化 LayerNorm 在训练好模型中的输入-输出映射，发现它呈现出惊人规律的 S 形曲线——几乎就是 tanh 函数。这暗示归一化的核心作用可能只是"压缩"激活范围。

**核心 idea**：LayerNorm ≈ 可学习的逐元素 tanh 压缩——无需 token 级统计量即可替代归一化。

## 方法详解

### 关键设计

1. **Dynamic Tanh (DyT)**:

    - 功能：无需计算统计量的归一化替代
    - 核心思路：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，其中 $\alpha$ 是可学习标量（控制压缩强度），$\gamma, \beta$ 是逐通道的缩放/偏移。tanh 将输入压缩到 $[-1, 1]$，$\alpha$ 学习适当的缩放使压缩程度匹配 LN 的效果
    - 设计动机：$\alpha$ 的学习值与激活标准差的倒数高度相关，说明 DyT 自动学会了一种"全局归一化"——不需要逐 token 计算统计量

2. **跨任务通用性**:

    - 功能：证明 DyT 不只是某个任务的技巧
    - 核心思路：在 ViT（分类）、MAE/DINO（自监督）、DiT（扩散）、LLaMA（语言）、wav2vec（语音）等完全不同的架构和任务上验证，直接替换 LN/RMSNorm 即可
    - 设计动机：如果 DyT 只在某个任务上有效，可能只是巧合；但跨多领域一致有效说明发现了归一化的本质功能

### 损失函数 / 训练策略

不改变任何训练设置，仅将 LN/RMSNorm 替换为 DyT。$\alpha$ 默认初始化为 0.5。LLaMA 需要特殊初始化（Section 7）。

## 实验关键数据

### 主实验

| 模型/任务 | DyT | LayerNorm/RMSNorm |
|----------|-----|-------------------|
| ViT-B (ImageNet) | **82.5%** | 82.3% |
| ViT-L (ImageNet) | **83.6%** | 83.1% |
| DiT-B (FID↓) | **63.9** | 64.9 |
| LLaMA-7B (loss) | ≈相同 | ≈相同 |
| wav2vec 2.0 | ≈相同 | ≈相同 |

### 消融实验

| 配置 | ViT-B Top-1 | 说明 |
|------|------------|------|
| DyT (完整) | 82.5% | — |
| 无 $\alpha$（固定=1） | 81.1% | $\alpha$ 关键 |
| 用 hardtanh | 82.0% | tanh 更优 |
| 用 sigmoid | 81.8% | tanh 最优 |
| 用 identity | 发散 | 压缩不可缺 |

### 关键发现
- **tanh 压缩是归一化的核心功能**：identity（无压缩）直接导致训练发散，验证了压缩的必要性
- **$\alpha$ 自动学到归一化效果**：其值与 1/std 高度相关，等效于一种全局归一化
- **扩散模型获益最大**：DiT 上 DyT 超越 LN（FID 63.9 vs 64.9），可能因为扩散的时间步条件与 token 级归一化冲突

## 亮点与洞察
- **对 Transformer 基本组件的根本性质疑**——归一化层被认为不可或缺已有 7 年，本文证明一个简单的 tanh 就够了
- **极简但深刻**——整个方法就一行公式，但背后的洞察（LN ≈ tanh）改变了对归一化作用的理解
- **实践意义**：DyT 不需要 token 级统计，对硬件友好（无 reduce 操作），在长序列/大 batch 下可能更高效

## 局限与展望
- DyT 不计算 per-token 统计，对 token 间幅度差异大的场景可能不适用
- LLaMA 需要特殊 $\alpha$ 初始化，通用性稍打折扣
- 缺少理论解释——为什么 tanh 近似就够了？
- 对极端训练设置（超大 batch、混精度）的鲁棒性未充分探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对基本组件的颠覆性洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 横跨视觉/语言/扩散/语音/自监督五大领域
- 写作质量: ⭐⭐⭐⭐⭐ 简洁优雅
- 价值: ⭐⭐⭐⭐⭐ 有望改变 Transformer 架构设计的基本范式
