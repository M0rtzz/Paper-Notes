---
title: >-
  [论文解读] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit
description: >-
  [ICLR 2026][多模态][视觉token压缩] 提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。
tags:
  - ICLR 2026
  - 多模态
  - 视觉token压缩
  - MLLM加速
  - 渐进式剪枝
  - Late Injection
  - 扩散注意力
---

# HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit

**会议**: ICLR 2026  
**arXiv**: [2602.23699](https://arxiv.org/abs/2602.23699)  
**代码**: https://github.com/EIT-NLP/HiDrop  
**领域**: 多模态VLM  
**关键词**: 视觉token压缩, MLLM加速, 渐进式剪枝, Late Injection, 扩散注意力

## 一句话总结

提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

## 研究背景与动机

**领域现状**：MLLM（如 LLaVA）处理视觉 token 的计算开销随 token 数量二次增长。视觉编码器产生的 token 远多于文本（如 576 个 patch token），成为推理和训练的主要瓶颈。

**现有痛点**：已有视觉 token 剪枝方法存在两个核心误解：(a) 错误认为浅层是关键的多模态融合层，必须保留密集视觉 token；实际上浅层对视觉 token 几乎不做处理，只是被动传播。(b) 采用固定比例的金字塔/线性剪枝调度（如 FastV、PDrop），忽略了不同层信息流的非均匀性。

**核心矛盾**：如何让 token 剪枝策略与模型内部层级处理动态真正匹配？

**本文目标**：设计与 MLLM 层级功能对齐的 token 管理策略——浅层不需要处理视觉 token（直接跳过）、中层是融合冗余最高的地方（激进剪枝）、深层已完成融合（直接丢弃）。

**切入角度**：先做系统的层级行为分析（intra-modal similarity + cross-modal influence），用数据驱动发现取代启发式假设。

**核心 idea**：根据 MLLM 层级功能分工（传播/融合/推理），在正确的位置做正确的事——晚注入、猛剪枝、早退出。

## 方法详解

### 整体框架

HiDrop 将 LLM 层分为三个阶段：
- **浅层（Layer 1~8）**：Late Injection — 完全不注入视觉 token，只处理文本
- **中层（Layer 9~24）**：Concave Pyramid Pruning — 在选定的 filtering layer 上用 Differentiable Top-K 渐进剪枝视觉 token，前快后慢
- **深层（Layer 25~32）**：Early Exit — 丢弃所有剩余视觉 token，纯文本推理

三个阶段共同定义了一个"视觉处理窗口"，视觉 token 只存在于约一半的层中。

### 关键设计

1. **Late Injection（晚注入）**:

    - 功能：在第 $L_{inj}=9$ 层才将视觉 token 注入序列
    - 核心思路：分析发现浅层的 intra-modal cosine similarity 极高（视觉 token 几乎不变化），cross-modal influence 近零（文本表示不受图像影响）。既然浅层不处理视觉信息，就不浪费计算在上面
    - 设计动机：不同于"先注入再剪枝"的传统思路，HiDrop 首次提出"延迟注入"——视觉 token 根本不经过浅层，从源头节省计算

2. **Concave Pyramid Pruning + ILVAS**:

    - 功能：在中间层渐进式剪枝视觉 token，前期剪得猛后期慢
    - 核心思路：
        - **在哪里剪（ILVAS）**：提出 Inter-Layer Visual Attention Similarity 指标，衡量相邻层之间视觉 token 注意力分布的稳定性。ILVAS 高的层说明注意力分配已稳定，是好的 filtering 层。选择 ILVAS 曲线的局部极大值点（如 layer {10,14,16,18}）
        - **剪谁（DTop-K）**：用 Differentiable Top-K 算子做可微 token 选择。先计算重要性分数的归一化排序 $c'_i$，再用 sigmoid + 可学习阈值 $a$ 生成软掩码 $\text{Mask}(c,a) = \sigma(\lambda(c'_i - a))$，前向时用硬阈值做离散选择，反向时梯度可流通
    - 设计动机：凹形调度（前快后慢）匹配中层"融合稀疏性递增"的规律——融合初期大量 token 冗余可快删，后期剩余 token 更关键要慢删

3. **Early Exit（早退出）**:

    - 功能：在第 $L_{exit}=25$ 层丢弃所有剩余视觉 token
    - 核心思路：通过 training-free 实验验证——在不同层移除所有视觉 token，发现 layer 24 之后移除几乎不影响性能
    - 设计动机：深层已完成跨模态融合，进入纯语言推理阶段，视觉 token 此时只消耗计算不贡献信息

4. **工程优化**:

    - Persistent Position Encoding：每个视觉 token 保持固定的位置标识符，避免动态剪枝导致 RoPE 位置错乱
    - FlashAttention 兼容：token 选择通过轻量辅助注意力完成，主注意力计算不变
    - 并行解耦：浅层文本前向与视觉编码并行执行，Late Injection 使这种并行成为可能

### 损失函数 / 训练策略

- 遵循 LLaVA 标准两阶段训练（预训练 + 指令微调）
- DTop-K 的温度系数 $\lambda = N_v$（即视觉 token 数量）
- 在 8× A100 40GB 上训练

## 实验关键数据

### 主实验

LLaVA-1.5-7B 上 11 个 benchmark 比较（保留约 64 tokens，压缩 88.9%）：

| 方法 | 类型 | MMEP | GQA | VQAv2 | POPE | MMStar | Avg(%) |
|------|------|------|-----|-------|------|--------|--------|
| LLaVA-1.5-7B | 上界(576 tokens) | 1506.5 | 61.9 | 78.5 | 86.8 | 33.7 | 100.0 |
| FastV | 训练free | 1086.6 | 48.8 | 61.6 | 67.7 | 29.6 | 82.8 |
| PDrop | 训练based | 1350.7 | 56.6 | 71.8 | 82.6 | 32.7 | 94.2 |
| TwigVLM | 训练based | 1404.0 | 58.8 | 75.6 | 82.7 | 33.1 | 95.3 |
| **HiDrop** | **训练based** | **1473.3** | **60.5** | **76.5** | **86.4** | **32.0** | **98.3** |

在最极端的 48 token（压缩 91.7%）下，HiDrop 仍达 97.1% 原始性能。

### 消融实验

| 配置 | Avg(%) | 说明 |
|------|--------|------|
| Full HiDrop | 98.3 | 完整框架 |
| w/o Late Injection | 96.8 | 去掉晚注入，浅层也处理视觉token |
| w/o Early Exit | 97.5 | 深层保留视觉token |
| w/o Concave (用线性调度) | 96.9 | 用均匀剪枝替代凹金字塔 |
| Hard Top-K (替代 DTop-K) | 97.1 | 不可微的硬选择 |

训练效率：HiDrop 训练加速 1.72×（vs 原始 LLaVA-1.5-7B）。

### 关键发现

- Late Injection 贡献最大——约 1.5% 的性能保持提升，说明避免浅层处理视觉 token 不仅省计算，还减少了无意义的浅层干扰
- 凹金字塔 > 线性 > 凸金字塔：融合初期激进剪枝的策略最优，与中层融合动态分析的结论完全一致
- DTop-K 比 Hard Top-K 好约 1.2%：可微选择使训练能反向传播到 token 重要性估计
- 即使压缩到仅 48 个视觉 token（每张图像 576→48，压缩 12 倍），POPE 指标只从 86.8 降到 86.6，几乎无损

## 亮点与洞察

- **先分析后设计的范式**：不是先设计方法再找实验支撑，而是先做系统的层级行为分析（intra-modal similarity, cross-modal influence, early exit 实验），用数据发现驱动算法设计。这种研究范式本身就值得学习
- **"延迟注入"是认知突破**：之前所有方法都默认视觉 token 从第一层就参与计算，HiDrop 首次提出"浅层根本不需要视觉信息"——这是对 MLLM 工作机制的深刻洞察，可推广到其他模态（如音频 token）
- **三阶段与层级功能一一对应**：Late Injection-传播层、Concave Pruning-融合层、Early Exit-推理层，设计美感强

## 局限与展望

- **仅在 LLaVA-1.5 验证**：层级行为分析的结论可能不适用于所有 MLLM（如 Qwen-VL、InternVL 的层级功能可能不同），需要更多架构的验证
- **注入层和退出层固定**：$L_{inj}=9$, $L_{exit}=25$ 是硬编码的，不同输入（简单 vs 复杂图像）可能需要不同的处理窗口
- **未考虑多图输入**：视频理解或多图 QA 场景下，视觉 token 数量更多，层级行为可能不同
- **DTop-K 训练开销**：可微 Top-K 引入额外参数和计算，在更大模型上的开销-收益比需要验证

## 相关工作与启发

- **vs FastV**: FastV 在早期单层一次性剪枝，过于粗暴且位置选择不当（浅层即剪）。HiDrop 证明浅层根本不该有视觉 token
- **vs PDrop**: PDrop 用均匀间隔和均匀比例渐进剪枝，忽略了中层融合的非均匀性。HiDrop 的 ILVAS 指标和凹金字塔调度更精准
- **vs TwigVLM**: TwigVLM 在浅层剪枝+深层移除，但浅层剪枝是多余的。HiDrop 用 Late Injection 替代浅层剪枝更高效
- **对视频/多图 MLLM 的启发**：可以分析视频 token 在不同层的行为，可能也存在"浅层冗余"现象，可以用类似策略大幅压缩

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Late Injection 是全新视角，层级功能分析驱动设计是很好的研究范式
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个 benchmark、3 个模型规模、详细消融、效率分析、层级行为可视化
- 写作质量: ⭐⭐⭐⭐⭐ 分析→洞察→设计的叙事流畅，图表精美
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——直接可用于任何 LLaVA 架构的 MLLM 加速
