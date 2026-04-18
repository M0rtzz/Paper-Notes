---
title: >-
  [论文解读] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models
description: >-
  [ICLR 2026][模型压缩][视觉生成] 提出 PTQ4ARVG，首个针对自回归视觉生成（ARVG）模型的系统化 PTQ 框架，通过增益投影缩放（GPS）、静态 Token 级量化（STWQ）和分布引导校准（DGC）解决 ARVG 特有的三大量化挑战。
tags:
  - ICLR 2026
  - 模型压缩
  - 视觉生成
  - 自回归模型
  - 后训练量化
  - 激活量化
  - 离群值抑制
---

# PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

**会议**: ICLR 2026  
**arXiv**: [2601.21238](https://arxiv.org/abs/2601.21238)  
**代码**: [GitHub](http://github.com/BienLuky/PTQ4ARVG)  
**领域**: 模型压缩  
**关键词**: 视觉生成, 自回归模型, 后训练量化, 激活量化, 离群值抑制

## 一句话总结

提出 PTQ4ARVG，首个针对自回归视觉生成（ARVG）模型的系统化 PTQ 框架，通过增益投影缩放（GPS）、静态 Token 级量化（STWQ）和分布引导校准（DGC）解决 ARVG 特有的三大量化挑战。

## 研究背景与动机

自回归视觉生成模型（VAR、RAR、PAR、MAR）在图像生成上已超越扩散模型，但模型体积大（2-3B 参数）、推理慢（PAR-3B 生成一张图 >3 秒）。量化是加速推理的有效手段，但将现有量化方法应用于 ARVG 面临三大特有挑战：

**通道级严重离群值**：经 AdaLN 模块调整后的激活在通道间范围差异极大

**Token 级高度动态激活**：位置编码导致 token 维度分布剧烈变化，且条件 token 形成 sink token

**样本级分布信息不匹配**：网络激活在不同样本间高度相似（尤其无条件样本），导致校准集冗余

## 方法详解

### 整体框架

PTQ4ARVG 包含三个针对性设计的组件，分别解决通道级、Token 级和样本级量化挑战，且全部无需训练。

### 关键设计

1. **增益投影缩放 (GPS - Gain-Projected Scaling)**：

    - 对量化损失进行 Taylor 展开，分别量化激活和权重损失
    - 定义缩放增益：$g(s_2) = g_{\bm{x}} - g_{\bm{W}_{:,1}}$（激活损失减少 - 权重损失增加）
    - 通过求导得到闭式最优缩放因子：$s_2 = s_1 \frac{\sqrt{\sum|{\Delta W_{2,i} x_2}|}}{\sqrt{\sum|{W_{2,i} \Delta x_2}|}}$
    - 首个基于数学优化的量化缩放策略，优于经验设计方法

2. **静态 Token 级量化 (STWQ)**：

    - 利用 ARVG 的两个独特性质：**固定 token 序列长度** + **位置不变的跨样本分布**
    - 为 AdaLN 模块沿 token 序列分配静态量化参数
    - 为线性层分别处理 sink token 和普通 token
    - 量化参数离线设置，无在线校准开销，兼容标准 CUDA 内核

3. **分布引导校准 (DGC)**：

    - 用 Mahalanobis 距离衡量样本的分布熵：$\rho(x) = \sqrt{(x-u)^T S^{-1} (x-u)}$
    - 选取分布熵最高的 top-50% 样本作为校准集
    - 消除冗余样本，确保校准分布与真实分布匹配

### 损失函数 / 训练策略

- 完全无训练（training-free PTQ）
- GPS 推导基于 Taylor 展开 + 凸优化求导
- STWQ 使用百分位数校准（而非 min-max），确保高精度
- 在 ImageNet 上生成 50K 图像评估 FID、sFID、IS、Precision

## 实验关键数据

### 主实验（VAR-d16 / VAR-d24 - W8A8 量化）

| 方法 | VAR-d16 FID ↓ | VAR-d16 IS ↑ | VAR-d24 FID ↓ | VAR-d24 IS ↑ |
|------|-------------|-------------|-------------|-------------|
| FP | 3.60 | 283.21 | 2.33 | 317.16 |
| SmoothQuant | 4.29 | 229.87 | 4.42 | 246.68 |
| OS+ | 4.11 | 230.41 | 4.14 | 250.61 |
| OmniQuant | 4.19 | 226.92 | - | - |
| **PTQ4ARVG** | **3.82** | **268.19** | **2.69** | **304.82** |

### 6-bit 量化结果（VAR-d24）

| 方法 | FID ↓ | IS ↑ | Precision ↑ |
|------|------|------|------------|
| SmoothQuant W6A6 | >10 | <200 | 严重退化 |
| **PTQ4ARVG W6A6** | **~4.5** | **~280** | 竞争力强 |

### 关键发现

- PTQ4ARVG 在 8-bit 和 6-bit 下均大幅超越现有 PTQ 方法
- GPS 的数学优化缩放一致优于经验方法（SmoothQuant、RepQ-ViT）
- STWQ 无额外推理开销即可处理 token 级方差，动态方法引入 0.5× 速度损失
- DGC 通过去除冗余样本显著提升校准质量
- 在 VAR、RAR、PAR、MAR 四种 ARVG 模型上均有效

## 亮点与洞察

- 问题定义精准：首次系统识别 ARVG 量化的三大挑战，每个挑战有针对性解决方案
- GPS 是首个基于严格数学推导的缩放策略，为量化缩放提供理论基础
- STWQ 巧妙利用了 ARVG 的固定 token 长度特性——LLM 因可变长度无法使用静态策略
- 实验覆盖四种 ARVG 架构（VAR/RAR/PAR/MAR），通用性好

## 局限与展望

- 4-bit 量化结果未展示，可能是 ARVG 模型在 4-bit 下精度退化严重
- GPS 的 Remark 1 基于统计观察而非严格证明
- 未与 SVDQuant 等最新方法对比（虽然后者依赖定制 CUDA 内核）
- ARVG 模型相比 LLM 规模较小，量化压缩的实际需求可能不如 LLM 迫切

## 相关工作与启发

- 与 SmoothQuant 的区别：GPS 用数学优化替代经验缩放
- 与 LLM 动态量化的区别：利用 ARVG 固定 token 长度实现无开销静态量化
- 与扩散模型 PTQ 的区别：ARVG 无时间步但有 token 级动态性，需要不同的方案
- 启示：模型架构的特有性质可以被量化方法充分利用

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 ARVG PTQ 框架，GPS 理论推导新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四种模型全面验证，部署验证
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，但公式推导占篇幅较大
- 价值: ⭐⭐⭐⭐ 为 ARVG 模型的高效部署奠定基础
