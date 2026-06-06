---
title: >-
  [论文解读] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs
description: >-
  [ICML 2026][多模态VLM][VLM 量化] 本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器）…
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "VLM 量化"
  - "知识蒸馏"
  - "Information Bottleneck"
  - "CKA 关系对齐"
  - "置信度门控"
---

# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**会议**: ICML 2026  
**arXiv**: [2601.22709](https://arxiv.org/abs/2601.22709)  
**代码**: 无  
**领域**: 多模态 VLM / 模型压缩 / 量化感知训练  
**关键词**: VLM 量化, 知识蒸馏, Information Bottleneck, CKA 关系对齐, 置信度门控

## 一句话总结
本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器），让 INT4 量化的 LLaVA / Qwen-VL 不仅没掉点，反而在多个 benchmark 上超过 BF16 基线，同时实测 3× 吞吐 + 54% 显存节省。

## 研究背景与动机

**领域现状**：VLM 部署成本高，PTQ（如 AWQ、GPTQ、MBQ）是最常用的压缩方案，但激进的 INT4 量化会让 VLM 出现灾难性掉点（多模态分布异质性比纯 LLM 复杂）；QAT 在 LLM 上已经成熟，但在 VLM 上几乎是空白。同时知识蒸馏在 VLM 压缩里被独立地大量使用。

**现有痛点**：(1) PTQ 直接拍计算图，无法让模型适应 INT4 的容量限制；(2) 传统 QAT 只用任务 loss 作监督，对“在低 bit 预算下保留什么信息”几乎没有显式指导，监督非常稀疏；(3) 标准蒸馏假设所有 teacher token 同样可信，但实证发现 teacher entropy 与错误率显著正相关（ScienceQA 上 Pearson $r=0.484$，binned $R^2=0.901$），高熵 token 实际上是噪声；(4) logit 蒸馏无法传递 13B teacher 在视觉 token 上学到的注意力结构（论文 Figure 3 显示 13B 能逐层定位“banana”，7B 注意力散乱）。

**核心矛盾**：量化是“容量分配”问题（哪些信息保留），蒸馏是“监督信号”问题（向谁学）；这两者本质上都是 IB 在解决的事情——压缩输入表示同时保留任务相关信息——但社区一直把它们当独立技术处理。

**本文目标**：(1) 建立 QAT 与 KD 的理论桥梁；(2) 解决“teacher 监督质量不均”的问题；(3) 把 teacher 的视觉关系结构（不只是 logit）真正传给 student；(4) 在 VLM 上把 INT4 性能逼近甚至超过 BF16。

**切入角度**：用 IB 的 $\max I(Z;Y) - \beta I(Z;X)$ 视角看，量化天然提供了对 $I(Z;X)\le C_b$ 的硬约束（bit 预算），那么 teacher 自然可以扮演 task-relevant 信息 $Y_T$ 的稠密代理，KL 散度 $D_{KL}(P_T\Vert P_S)$ 恰好就是 $I(X;Y_T)$ 与 $I(Z_S;Y_T)$ 的信息缺口（论文 Proposition 3.2）。

**核心 idea**：用 IB 框架把“量化的硬容量约束”和“teacher 蒸馏的软监督”联合优化，再加置信度门控 + 关系核对齐两个针对 VLM 特性的具体机制。

## 方法详解

### 整体框架
teacher 是 frozen 的 BF16 大模型（如 LLaVA-1.5 13B），student 是 group-wise LSQ 量化（默认 INT4 / g=128）的小模型（如 LLaVA-1.5 7B）。两者并行处理同一输入，student 受三种监督：(i) Confidence-Gated DKD（解耦 + 门控的 logit 蒸馏）；(ii) Relational CKA（在 LLM 倒数第二层对 visual token 的 Gram 矩阵做 CKA 对齐，text token 排除）；(iii) Adaptive IB Controller（监控 EMA 平滑后的 $\widehat{\mathcal{L}}_{GDKD}$，动态调整 $\beta$）。weight $W$ 与 per-group scale $s$ 联合更新。

### 关键设计

1. **Confidence-Gated Decoupled Knowledge Distillation (GDKD)**:

    - 功能：(a) 把蒸馏拆成 target-class (TCKD) 和 non-target-class (NCKD) 两路，强调 NCKD 的“暗知识”；(b) 用 teacher entropy 做 token 级门控，过滤掉不可靠监督。
    - 核心思路：TCKD = $D_{KL}([P_T^t,1-P_T^t]\|[P_S^t,1-P_S^t])$ 捕捉 teacher 对正确答案的把握程度；NCKD = $D_{KL}(\hat P_T^{nt}\|\hat P_S^{nt})$ 在 renormalize 后的非目标类上做 KL，传递 dark knowledge；逐 token 的 DKD = $\alpha\cdot \mathcal{L}_{TCKD}+\beta_{dkd}\cdot \mathcal{L}_{NCKD}$，设 $\beta_{dkd}>\alpha$。门控部分对每 token 算 $H_i=-\sum_v P_T^{(i)}(v)\log P_T^{(i)}(v)$，归一化 $\tilde h_i=H_i/\log|V|\in[0,1]$，权重 $g_i=\exp(-\tilde h_i)$ 让高置信 teacher token 权重大；最终 $\mathcal{L}_{GDKD}=\sum_i g_i \mathcal{L}_{DKD}^{(i)}/\sum_i g_i$。Theorem 3.1 给出门控等价于在 covariance 项上做修正：$\mathcal{L}_{GDKD}=\bar{\mathcal{L}}_{DKD}+N\cdot \mathrm{Cov}(w_i,\mathcal{L}_{DKD}^{(i)})$，当 entropy 与 loss 正相关时该项为负，证明门控严格降低期望蒸馏误差。
    - 设计动机：实证发现 teacher entropy 与错误率强相关（$R^2=0.901$），且 Fano 不等式从信息论上保证 entropy 越大错误下界越高；门控等于在 IB 框架下把蒸馏容量分配给“teacher 后验最 sharp”的 token，是对“监督信号同等重要”这个隐含假设的纠正。

2. **Relational Centered Kernel Alignment (RCKA)**:

    - 功能：在 LLM 倒数第二层对 visual token（排除 text token）做 Gram 矩阵的 CKA 对齐，把 teacher 的关系结构（哪些 patch 应该在语义上聚在一起）传给 student。
    - 核心思路：分别取 teacher / student 的 visual token 表征 $V_T\in\mathbb{R}^{n\times d_T}$ 与 $V_S\in\mathbb{R}^{n\times d_S}$，行 L2 归一化后算 $K_T=\bar V_T \bar V_T^\top$，$K_S=\bar V_S\bar V_S^\top$；中心化 $\tilde K=HKH$，$H=I_n-\frac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top$；CKA = $\mathrm{HSIC}(K_T,K_S)/\sqrt{\mathrm{HSIC}(K_T,K_T)\mathrm{HSIC}(K_S,K_S)}$，损失 $\mathcal{L}_{RCKA}=1-\mathrm{CKA}(K_T,K_S)$。关键差异：以往 RKD 在 batch 级算 inter-sample 关系，本文在 sample 内的 visual token 之间算 intra-sample 关系（论文 Figure 5 可视化展示 sky token 和其他 sky token 高相似度而与飞机区域低相似度）。
    - 设计动机：logit 蒸馏只能传递输出分布，传不了视觉推理的核心——“哪些区域应当被关联起来看”。CKA 对维度尺度具有不变性，所以 $d_T\ne d_S$（teacher 13B vs student 7B）时无需投影层即可对齐，是 cross-dim 蒸馏的天然桥梁。

3. **Adaptive IB Controller + Group-wise LSQ 量化**:

    - 功能：(a) 动态调整蒸馏权重 $\beta$ 平衡 task loss 与 distill loss；(b) per-group 学习量化 step size，把 INT4 的硬容量约束直接植入优化目标。
    - 核心思路：IB 视角下，求解 $\min \mathcal{L}_{task}$ s.t. $\mathcal{L}_{distill}\le \tau$，对偶得 Lagrangian $\mathcal{L}_{task}+\beta(\mathcal{L}_{distill}-\tau)$。controller 用 EMA 平滑 $\widehat{\mathcal{L}}_{GDKD}$ 来监控当前蒸馏达成度，动态调 $\beta$。量化方面：把 weight matrix flatten 后按 $g=128$ 切分为 $G$ 组，每组学一个 scale $s_i=\exp(\theta_i)$（log space 保正），初始化用 99 分位数 $s_i^{(0)}=\mathrm{Percentile}_{99}(|W_i|)/Q_p$，量化 $W_{i,q}=s_i\cdot \mathrm{clamp}(\lfloor W_i/s_i\rceil,-Q_n,Q_p)$，反向用 STE。
    - 设计动机：固定 $\beta$ 在训练不同阶段都不合适——早期 teacher 监督应该更强，后期 task loss 应该收回主导；EMA + IB Lagrangian 给出自动调度。group-wise + LSQ 比 per-tensor 细，又比 per-channel 粗，正好匹配 MX 格式硬件，并把 scale 当可学参数后能配合 distill 信号端到端微调。

### 损失函数 / 训练策略
总目标 $\mathcal{L}=\mathcal{L}_{CE}+\beta(t)\cdot \mathcal{L}_{GDKD}+\gamma\cdot \mathcal{L}_{RCKA}$；$\beta(t)$ 由 IB controller 调度。teacher frozen；student 联合优化 $W$ 与 $\{s_i\}$。Proposition 3.2 给出 KL gap 的变分下界 $I(Z_S;Y_T)\ge I(X;Y_T)-\mathbb{E}[D_{KL}(P_T\|P_S)]$，说明最小化 $\mathcal{L}_{GDKD}$ 就是最大化 student 表征与 teacher 知识的互信息。

## 实验关键数据

### 主实验
两类 backbone：LLaVA-1.5 (7B/13B) 与 Qwen2-VL (2B/7B)；teacher 选大版本，student 选小版本并量化到 INT4。

| Backbone | Bit | 方法 | SQA | MMBench | 备注 |
|----------|-----|------|------|---------|------|
| LLaVA-1.5-7B | BF16 | baseline | 66.8 | – | 起点 |
| LLaVA-1.5-7B | INT4 | RTN/AWQ/GPTQ/MBQ | 显著掉点 | – | PTQ 全军覆没 |
| LLaVA-1.5-7B | INT4 | **GRACE** | **70.1** | – | 反超 BF16 +3.3 |
| Qwen2-VL-2B | BF16 | baseline | 73.7 | 72.6 | 起点 |
| Qwen2-VL-2B | INT4 | **GRACE** | **79.1** | **76.9** | 反超 BF16 +4–5 |
| LLaVA-1.5-7B distilled (BF16) | – | GRACE | 69.0 avg | – | 比 7B baseline +3.8，接近 13B teacher |

**部署收益**：用真实 INT4 kernel 实测 3× throughput、54% 显存降低。

### 消融实验

| 配置 | 平均精度 | 说明 |
|------|---------|------|
| GRACE (full) | 最高 | 完整模型 |
| w/o IB 框架（普通 QAT 单独 / QAT+KD 朴素叠加） | 显著下降 | 验证 IB 联合优化的必要性 |
| w/o Confidence Gating | 中等下降 | 高熵 token 噪声反噬蒸馏 |
| w/o RCKA | 下降 | 视觉关系结构无法传递，7B 注意力仍散乱 |
| w/o Adaptive Controller（固定 $\beta$） | 略低 | 训练后期 distill 与 task 矛盾时无法切换 |
| per-tensor 量化替换 group-wise | 显著下降 | VLM 异质权重分布需要更细粒度 |

### 关键发现
- INT4 反超 BF16 这件事在 VLM 上是反直觉的：作者把它归因于“蒸馏+量化的联合 IB 优化等价于做了一次额外的 regularization”，BF16 baseline 没有 teacher 监督。
- 置信度门控的提升对 SQA 这类需要长链推理的任务最显著，因为 teacher 在长答案末尾的 token 普遍高熵。
- RCKA 在 MMBench 上贡献最大，符合“MMBench 强调视觉关系理解”的任务特性；同时 RCKA 让 INT4 student 的注意力图（论文 Figure 3）从散乱变成与 13B teacher 类似的“逐层聚焦”。
- group size $g=128$ 是 sweet spot，更小（g=64）收益甚微但增加 scale 数量，更大（g=512）精度掉。

## 亮点与洞察
- 用 IB 把 QAT 与 KD 这两条之前独立的技术线统一起来是个干净的理论 framing，不只是工程组合：硬容量约束 + 软监督代理这种对偶式拆解能直接套到 LoRA / pruning / sparse training 等所有“受限容量 + 大 teacher”场景。
- “teacher entropy = 监督质量代理”这件事用 Pearson $r$、binned $R^2$、Fano 不等式三重证据论证，相比此前“self-distillation noise filtering”一类工作要扎实得多，结论可直接 plug 到任意 KD 框架里。
- 视觉 token 的 intra-sample CKA 对齐是个相当聪明的点——传统 KD 要么对齐 logit 要么对齐特征向量，前者 dim 必须匹配，后者细粒度不够；CKA 对维度尺度不变天然解决 7B vs 13B 的 dim 差，并且关系矩阵能直接刻画“sky 像素聚在一起”这种 VLM 关键的视觉结构。
- Theorem 3.1 把门控的效果显式写成 covariance 项，是对 noisy distillation 的清晰量化，比经验性的 confidence weighting 多一步理论。

## 局限与展望
- 作者只评测 LLaVA / Qwen 两个 backbone 系列，多模态生成（视频、3D）的可迁移性未验证。
- 自评：teacher 必须是更大的同架构 BF16 模型，跨架构蒸馏（比如 LLaVA → Qwen）行不行没有实验。
- INT4 反超 BF16 的部分可能来自“teacher 提供的 dark knowledge regularization”，并不是“量化本身有益”，作者应该补一个 BF16 student 也跑蒸馏的对照，否则结论容易被误读为“量化越激进越好”。
- group-wise LSQ 的 group=128 是固定的，没有 per-layer 自适应；不同层（视觉编码器 vs LLM 解码器）的权重分布差异未必都适合同一 group size。
- 未来工作可以把 IB controller 推广到“量化 bit width 也是可学的”，做联合的 bit allocation。

## 相关工作与启发
- **vs AWQ / GPTQ / MBQ**：这些都是 PTQ，靠校准集和 weight rounding 优化离线 scale；GRACE 是 QAT，scale 与 weight 端到端更新，并融合 teacher 监督，所以在 INT4 上能反超而 PTQ 全部掉点。
- **vs DKD (Zhao 2022)**：本文借用了 TCKD/NCKD 解耦，但增加 entropy-gated weighting 与 IB 框架；DKD 原本针对图像分类，本文证明它在 VLM 序列预测里同样关键。
- **vs RKD (Park 2019) / CKA-based KD (Saha 2022)**：RKD 在 batch 内做 inter-sample 关系，layer-wise CKA 在层间对齐；本文是 intra-sample 的 visual token 关系对齐，粒度更细且专门为 VLM 设计。
- **vs LLM-QAT (Liu 2024)**：LLM-QAT 在纯文本 LLM 上做 QAT，不涉及视觉模态；本文扩展到 VLM 并处理跨模态权重分布异质性问题。
- **可迁移启发**：(1) IB 的“硬约束（容量）+ 软监督（teacher）”对偶可推广到任意 constrained-capacity learning（pruning、低秩、binary networks）；(2) entropy-gated supervision 可直接套到 RLHF reward model 训练里过滤噪声；(3) intra-sample CKA 是任何跨 dim 蒸馏的通用解。

## 评分
- 新颖性: ⭐⭐⭐⭐ IB 联合 QAT+KD 的 framing 是新的；门控 DKD 和 intra-sample CKA 各有创新；单看组件每个都有先例。
- 实验充分度: ⭐⭐⭐⭐ 两个 backbone 系列、多个 benchmark、INT4 部署实测、消融完整；缺跨架构蒸馏与 BF16 student+蒸馏的对照。
- 写作质量: ⭐⭐⭐⭐ 理论与实证结合很好，motivation 部分用 entropy-error 相关性 + 注意力可视化双重证据，说服力高；公式偏多但都有口头解读。
- 价值: ⭐⭐⭐⭐⭐ 在 VLM 部署这条线给出第一个能反超 BF16 的 INT4 方案，并附实测吞吐与显存收益，对工业落地价值很大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Deep Pre-Alignment for VLMs](deep_pre-alignment_for_vlms.md)
- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](../../CVPR2025/multimodal_vlm/move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)

</div>

<!-- RELATED:END -->
