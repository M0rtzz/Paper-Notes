---
title: >-
  [论文解读] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
description: >-
  [ACL 2026][音频/语音][音频深度伪造检测] 提出 XLSR-MamBo 框架，系统探索 Mamba-Attention 混合架构在音频深度伪造检测中的四种拓扑设计和多种 SSM 变体（Mamba2、Hydra、GDN）…
tags:
  - "ACL 2026"
  - "音频/语音"
  - "音频深度伪造检测"
  - "Mamba"
  - "混合架构"
  - "状态空间模型"
  - "XLSR"
---

# XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection

**会议**: ACL 2026  
**arXiv**: [2601.02944](https://arxiv.org/abs/2601.02944)  
**代码**: [GitHub](https://github.com/saki-ciallo/XLSR-MamBo)  
**领域**: AI安全 / 语音伪造检测  
**关键词**: 音频深度伪造检测, Mamba, 混合架构, 状态空间模型, XLSR

## 一句话总结
提出 XLSR-MamBo 框架，系统探索 Mamba-Attention 混合架构在音频深度伪造检测中的四种拓扑设计和多种 SSM 变体（Mamba2、Hydra、GDN），其中 MamBo-3-Hydra 利用 Hydra 的原生双向建模达到多个基准上的竞争性能，且增加骨干深度可有效缓解浅层模型的性能不稳定。

## 研究背景与动机

**领域现状**：音频深度伪造检测（ADD）已从手工特征转向端到端架构。XLSR 作为前端特征提取器搭配 Conformer 等注意力分类器是主流方案。近期 Mamba 等状态空间模型（SSM）因线性复杂度受到关注。

**现有痛点**：纯因果 SSM 是单向的，难以捕捉全局频域伪造痕迹所需的基于内容的检索能力。现有双向 Mamba 扩展依赖手工设计的双分支策略（如正反向拼接），存在结构冗余。Transformer 的二次复杂度限制了长序列效率。

**核心矛盾**：SSM 擅长高效时序压缩和局部高频伪影捕捉，Attention 擅长全局关联和内容检索——深度伪造信号同时表现为局部高频伪影和全局频谱不一致，单一机制都不够。

**本文目标**：系统探索 SSM-Attention 混合架构在 ADD 中的最优拓扑组合，并评估深度缩放对性能稳定性的影响。

**切入角度**：受 Jamba、Zamba 等 LLM 混合架构启发，但针对 ADD 任务进行定制化探索，特别引入 Hydra（原生双向 SSM）替代启发式双向策略。

**核心 idea**：SSM 和 Attention 的互补性（时序压缩 vs 内容检索）在 ADD 中尤为重要，Hydra 的原生双向参数化比双分支策略更优雅，增加 SSM 堆叠深度 N 可缓解性能不稳定。

## 方法详解

### 整体框架
输入原始音频经 XLSR 提取特征（$X \in \mathbb{R}^{T \times 1024}$），RMSNorm + 线性投影到隐藏维度 D=128，通过 L=5 层 MamBo 混合层编码，门控注意力池化聚合为句级表示，线性层输出二分类 logits。

### 关键设计

1. **四种 MamBo 拓扑设计**:

    - 功能：系统探索 SSM 和 Attention 的不同组合方式
    - 核心思路：MamBo-1（纯 SSM 替换 MHA）、MamBo-2（Mamer，SSM 后接 MHA 替换 FFN）、MamBo-3（Mamba 层和 Transformer 层交替堆叠）、MamBo-4（Mamba 层和 Mamer 层交替堆叠）。每种拓扑可搭配不同 SSM 变体（Mamba、Mamba2、Hydra、GDN）
    - 设计动机：MamBo-1/2 探索层内 SSM-Attention 混合，MamBo-3/4 探索层间交替；不同伪造痕迹类型可能需要不同的处理方式

2. **Hydra 原生双向 SSM**:

    - 功能：无需双分支启发式即可捕捉非因果全局依赖
    - 核心思路：Hydra 将前向和反向扫描参数化为准可分矩阵，包含下三角（过去信息）和上三角（未来信息）结构。公式为 $\text{shift}(SS(X)) + \text{flip}(\text{shift}(SS(\text{flip}(X)))) + DX$，在线性复杂度内实现原生双向处理
    - 设计动机：深度伪造检测需要非因果上下文（伪影可能分布在整个音频中），Hydra 比手工双向策略更优雅且无结构冗余

3. **深度缩放（Stacking N）**:

    - 功能：通过增加 SSM 堆叠层数提升性能稳定性
    - 核心思路：引入堆叠超参数 N，允许在单个单元中连续堆叠 N 个 SSM 块。实验发现 N=3 时性能和稳定性最佳，浅层模型（N=1）性能方差大
    - 设计动机：浅层 SSM 缺乏足够的表征深度来一致性地捕捉复杂伪造痕迹

### 损失函数 / 训练策略
使用 FocalLoss 处理类别不平衡。AdamW 优化器（$lr=10^{-5}$），10% 线性 warmup + 余弦衰减。混合精度训练（BF16/FP32），最多 20 epoch，早停 patience=7。在 ASVspoof 2019 LA 训练集上训练，跨数据集评估泛化性。

## 实验关键数据

### 主实验

| 模型 | ASV21LA EER↓ | ASV21DF EER↓ | ITW EER↓ |
|------|-------------|-------------|----------|
| XLSR-Conformer (基线) | ~1.0 | ~2.5 | ~5.0 |
| MamBo-1-Mamba (N=1) | 1.19 | 2.08 | 4.65 |
| MamBo-3-Hydra (N=3) | 最优 | 竞争性 | 竞争性 |
| RawBMamba | - | - | - |

### 消融实验

| 配置 | ASV21LA | 说明 |
|------|---------|------|
| MamBo-1 (纯SSM) | 基线 | SSM 替换 Attention |
| MamBo-2 (Mamer) | 略优 | 层内混合有帮助 |
| MamBo-3 (交替) | 最优 | 层间交替效果最好 |
| N=1 vs N=3 | 方差↓ | 深度缩放显著提升稳定性 |

### 关键发现
- MamBo-3（Mamba-Transformer 交替）在多数基准上表现最优，证明层间交替优于层内混合
- Hydra 在 MamBo-3 中表现最佳，其原生双向建模比 Mamba 的启发式双分支更有效
- 增加 SSM 堆叠深度 N 从 1 到 3 显著降低性能方差，浅层模型的不稳定性是实际部署的隐患
- 在 DFADD 数据集上对扩散和流匹配合成方法保持鲁棒，证明泛化能力
- GDN 的 delta rule 记忆管理在某些场景下也表现不错

## 亮点与洞察
- 系统化的拓扑探索（4 种设计 × 4 种 SSM 变体 × 不同深度）为 SSM-Attention 混合架构在语音任务中的应用提供了全面的设计指南。这种方法论可迁移到其他语音任务
- Hydra 的原生双向能力在 ADD 中的优势验证了"因果一致性违反"作为伪造检测线索的假设
- "深度缩放缓解浅层不稳定"是实用的工程洞察，对实际部署有直接指导意义

## 局限与展望
- 仅在 ASVspoof 2019 LA 训练集上训练，训练数据多样性有限
- 模型规模较小（D=128, L=5），更大规模模型的表现未探索
- ITW 数据集上的性能仍有提升空间
- 未探索端到端训练（XLSR 参数冻结）
- 未来可探索更多混合拓扑和跨语言泛化

## 相关工作与启发
- **vs XLSR-Conformer**: 纯注意力架构，本文混合 SSM 在效率和性能上均有改进
- **vs RawBMamba**: 手工双向 Mamba 策略，本文用 Hydra 原生双向替代更优雅
- **vs Jamba/Samba**: LLM 领域的混合架构，本文首次将此范式系统化应用于 ADD

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统化探索 SSM-Attention 混合在 ADD 中的应用，Hydra 引入有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 四种拓扑 × 四种变体 × 多深度 × 多数据集，非常全面
- 写作质量: ⭐⭐⭐⭐ 背景知识详实，实验组织清晰
- 价值: ⭐⭐⭐⭐ 为 ADD 领域的架构选择提供了系统化参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)
- [\[ACL 2026\] RTCFake: Speech Deepfake Detection in Real-Time Communication](rtcfake_speech_deepfake_detection_in_real-time_communication.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)

</div>

<!-- RELATED:END -->
