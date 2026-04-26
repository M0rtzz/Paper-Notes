---
title: >-
  [论文解读] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval
description: >-
  [CVPR 2026][医学图像][组合图像检索] 揭示了将生成式MLLM适配为判别式检索器时的"能力退化"现象（Capability Degradation），提出ReCALL框架通过诊断检索器盲点→利用基座MLLM的CoT推理生成纠正性三元组→分组对比精炼三阶段管线，有效恢复退化的细粒度组合推理能力，在CIRR上R@1达55.52%、FashionIQ上R@10达57.04%。
tags:
  - CVPR 2026
  - 医学图像
  - 组合图像检索
  - 能力退化
  - MLLM自改进
  - 对比学习
  - 诊断-生成-精炼
---

# ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval

**会议**: CVPR 2026  
**arXiv**: [2602.01639](https://arxiv.org/abs/2602.01639)  
**代码**: https://github.com/RemRico/Recall (有)  
**领域**: 多模态检索  
**关键词**: 组合图像检索, 能力退化, MLLM自改进, 对比学习, 诊断-生成-精炼

## 一句话总结
揭示了将生成式MLLM适配为判别式检索器时的"能力退化"现象（Capability Degradation），提出ReCALL框架通过诊断检索器盲点→利用基座MLLM的CoT推理生成纠正性三元组→分组对比精炼三阶段管线，有效恢复退化的细粒度组合推理能力，在CIRR上R@1达55.52%、FashionIQ上R@10达57.04%。

## 研究背景与动机

**领域现状**：组合图像检索（CIR）根据参考图+修改文本的混合查询检索目标图。早期双塔VLM方法因浅层跨模态对齐不够，难以做细粒度组合推理。近期开始将MLLM适配为检索器，利用其深度融合和指令跟随能力，通过对比学习微调获得判别式检索能力。

**现有痛点**：将生成式MLLM（聚焦逐步推理）压缩为单嵌入判别式检索器（聚焦向量相似度）引入了**范式冲突**——微调后模型的原生细粒度推理能力（细粒度定位、关系理解）发生退化。实验证明：在基座MLLM能通过VQA正确回答的1k样本上，微调后的检索器R@1仅62.33%(CIRR)和55.80%(FashionIQ)，证明大量已有能力在适配过程中丧失。

**核心矛盾**：生成式范式（强调序列推理、注意力分配到每个token）vs 判别式范式（将全部语义压缩到单个embedding向量）的根本冲突。单embedding无法承载MLLM原本通过多步推理才能完成的细粒度区分。

**本文目标** 如何在保持检索形式（单embedding必须）的前提下，恢复微调过程中退化的组合推理能力？

**切入角度**：不改变检索范式本身，而是利用基座MLLM的原生推理信号反向监督检索器的embedding空间——"从MLLM中蒸馏推理能力到检索空间"。

**核心 idea**：通过诊断检索器的失败案例、让基座MLLM为失败案例生成最小编辑的纠正指令形成新三元组、再用分组对比学习将这些细粒度区分能力内化到检索器中。

## 方法详解

### 整体框架
ReCALL是一个模型无关的四阶段框架：Stage 1用标准对比学习从基座MLLM($\mathcal{F}$)训练基线检索器($\mathcal{R}_{\text{base}}$)。Stage 2(诊断)：让$\mathcal{R}_{\text{base}}$在训练集上推理发现失败案例，挖掘信息性实例。Stage 3(生成)：用$\mathcal{F}$的CoT推理为失败案例生成纠正指令，经VQA质量控制过滤。Stage 4(精炼)：通过分组对比学习在原始+纠正三元组上继续训练，得到精炼后的$\mathcal{R}_{\text{refine}}$。

### 关键设计

1. **自引导信息性实例挖掘 (Stage 2: 诊断)**:

    - 功能：自动发现检索器的"认知盲点"——哪些样本被检索器high-ranking但实际错误
    - 核心思路：用$\mathcal{R}_{\text{base}}$在训练集做检索推理，过滤掉检索成功的query（这些已有足够判别力），聚焦失败cases。对每个失败case，提取被错误排在ground truth之前的Top-K图像作为信息性实例 $\{I_h\}$。这些实例之所以"信息性"，是因为它们与目标有微妙的视觉/语义相似性，恰好暴露了检索器退化的推理能力
    - 设计动机：与盲目大规模数据合成（Random Mining）相比，自引导挖掘将生成预算精准集中在模型的实际失败点上，数据效率极高

2. **生成式校准 (Stage 3: 生成)**:

    - 功能：利用基座MLLM的原生推理能力生成针对性纠正监督信号
    - 核心思路：(a) **CoT辅助生成**——用$\mathcal{F}$对每个信息性实例$I_h$执行两步推理：先将原始指令$T_m$分解为原子意图并验证每个意图在$(I_r, I_h)$上是否满足，再保留一致的意图、仅重新生成违反的部分得到纠正指令$\tilde{T}_m$。这样新三元组$(I_r, \tilde{T}_m, I_h)$的文本最小编辑直接对应$I_t$和$I_h$的视觉差异。(b) **VQA质量控制**——用$\mathcal{F}$对$\tilde{T}_m$中的关键属性提问，只保留高置信且内部一致的三元组
    - 设计动机：最小编辑策略保持了原始分布的同时引入了精确的细粒度监督——纠正指令和原始指令的差异恰好反映了检索器需要学习区分的视觉差异。VQA过滤确保生成的监督信号是可靠的

3. **分组对比精炼 (Stage 4: 精炼)**:

    - 功能：将纠正监督信号高效内化到检索器的embedding空间
    - 核心思路：为每个query构建微组（micro-group），包含原始正样本三元组$(I_r, T_m, I_t)$和纠正三元组$(I_r, \tilde{T}_m, I_h)$。双目标优化：(a) **InfoNCE损失**保持全局结构；(b) **组内三元组margin损失** $\mathcal{L}_{\text{triplet}} = \max(0, s(z_q, z_{t^-}) - s(z_q, z_{t^+}) + m)$ 显式强制目标与信息性实例的分离。总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{infoNCE}} + \lambda\mathcal{L}_{\text{triplet}}$
    - 设计动机：将目标和其易混淆近邻+微妙不同的指令放在同一batch中，迫使模型在单次梯度更新中解决最具挑战性的歧义。相比随机batching，这种结构化batching能最大化纠正信号的传递效率

### 损失函数 / 训练策略
使用Qwen2.5-VL-7B做骨干，LoRA(rank=16)微调。FashionIQ: lr=$4\times10^{-5}$, $\tau=0.03$, batch=512, Stage 1 200步 + Stage 4 250步。CIRR: lr=$2\times10^{-5}$, $\tau=0.02$, Stage 1 300步 + Stage 4 350步。Triplet margin $m=0.05$, $\lambda$=0.30(FashionIQ)/0.25(CIRR)。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ReCALL | 之前SOTA(CIR-LVLM) | ℛ_base | 提升(vs base) |
|--------|------|------|----------|------|------|
| CIRR test | R@1 | **55.52%** | 53.64% | 51.23% | +8.38% |
| CIRR test | R@5 | 84.07% | 83.76% | 82.15% | +2.34% |
| CIRR test | R_subset@1 | **81.49%** | 79.12% | 77.57% | +5.06% |
| FashionIQ val | Avg R@10 | **57.04%** | 56.21% | 53.23% | +7.16% |
| FashionIQ val | Avg R@50 | **76.42%** | 76.14% | 74.37% | +2.76% |
| FashionIQ Dress | R@10 | 51.81% | 50.42% | 46.80% | **+10.71%** |

### 消融实验

| 配置 | Avg R@10 | Avg R@50 | 说明 |
|------|---------|------|------|
| ℛ_base | 53.23% | 74.37% | 基线检索器 |
| + CG (CoT生成) | 55.41% | 75.17% | +2.18%，CoT监督有效 |
| + VC (VQA质控) | 56.13% | 76.04% | +0.72%，过滤噪声有效 |
| + GR (分组精炼) | **57.04%** | **76.42%** | +0.91%，结构化batching关键 |

| 挖掘策略 | Avg R@10 | 说明 |
|------|---------|------|
| Random Mining | 53.80±0.20 | 盲目合成仅+0.57 |
| Self-Guided Mining | **57.04** | 精准挖掘+3.81 |

### 关键发现
- **Self-Guided vs Random Mining差距巨大**：Random Mining在相同数据量下仅提升0.57%，Self-Guided提升3.81%。这证明"在哪里生成数据"远比"生成多少数据"重要
- **每个组件逐步贡献**：CG(+2.18%) > GR(+0.91%) > VC(+0.72%)，CoT辅助生成是核心驱动力
- **Dress类别提升最大(+10.71%)**，因为时装dress的细粒度差异（袖长、领口、图案）恰好是能力退化最严重的地方
- **跨骨干泛化**：在更强的Qwen3-VL-8B上，ReCALL仍然有效（CIRR R@1: 55.93→57.09），证明能力退化是范式冲突的普遍问题而非特定模型的缺陷

## 亮点与洞察
- **"能力退化"概念的提出和量化验证**是本文最大贡献。通过ℱ-solvable子集的对比实验（ℱ在VQA下100% R@1但ℛ_base仅62.33%），清晰量化了生成→判别范式转换造成的能力损失。这个发现对所有将生成式模型转为检索器的工作都有启示
- **最小编辑策略**很精巧——纠正指令和原始指令的文本差异恰好镜像了目标和信息性实例的视觉差异，形成了"视觉差异↔文本差异"的对称监督信号
- **诊断-生成-精炼管线**是一种通用的模型自改进范式。可迁移到任何MLLM→判别式模型的适配场景，如MLLM→分类器、MLLM→重排序器

## 局限与展望
- Stage 2-4是离线的单次管线，如果迭代执行（诊断→生成→精炼→再诊断→...）可能进一步提升
- 当前依赖基座MLLM的CoT推理质量，如果基座模型本身对某些细粒度差异也不擅长，则无法生成有效纠正
- VQA质量控制只做了简单的一致性检查，更细粒度的验证（如对比生成的$\tilde{T}_m$和$T_m$的语义距离）可能进一步提升数据质量
- 训练步数很少（200-350步），说明方法数据效率高但也意味着有进一步训练的空间

## 相关工作与启发
- **vs CIR-LVLM**: CIR-LVLM也将LVLM适配为CIR检索器但用单阶段静态微调，没有考虑能力退化问题。ReCALL通过自改进loop弥补了这一缺陷
- **vs TME/CCIN**: 这些CVPR 2025方法在CIRR上R@1约53.4%，ReCALL达到55.52%，核心区别在于ReCALL额外从基座模型蒸馏了推理能力
- **vs STaR/Self-Refine**: 这些LLM自改进方法是在生成任务内循环改进，ReCALL将自改进范式首次应用于检索任务，桥接了生成推理和判别检索空间

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "能力退化"的发现和诊断-生成-精炼的解决思路都具有原创性和普适性
- 实验充分度: ⭐⭐⭐⭐⭐ 两个主流基准SOTA，详细消融，跨骨干验证，定性分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证有说服力，实验设计严谨，图表清晰
- 价值: ⭐⭐⭐⭐⭐ 对MLLM检索适配提出根本性洞察，框架通用性强

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_imaging/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](../../ACL2026/medical_imaging/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](../../ACL2026/medical_imaging/efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](../../ACL2026/medical_imaging/benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)

<!-- RELATED:END -->
