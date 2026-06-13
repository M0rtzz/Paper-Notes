---
title: >-
  [论文解读] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport
description: >-
  [ACL 2025][LLM安全][machine unlearning] 提出 Opt-Out，一种基于最优传输理论的实体级 LLM 遗忘方法，利用 Sliced Wasserstein Distance 正则化参数偏移实现精细遗忘；同时构建首个实体级遗忘数据集 ELUDe（20 目标实体 + 144 邻居实体，15K+ forget / 90K+ retain QA 对），在 Llama-3.1-8B 和 Phi-3.5 上全面超越现有方法。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "machine unlearning"
  - "entity-level"
  - "optimal transport"
  - "Wasserstein distance"
  - "privacy"
  - "GDPR"
---

# Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport

**会议**: ACL 2025  
**arXiv**: [2406.12329](https://arxiv.org/abs/2406.12329)  
**代码**: [https://github.com/brightjade/Opt-Out](https://github.com/brightjade/Opt-Out)  
**领域**: LLM安全  
**关键词**: machine unlearning, entity-level, optimal transport, Wasserstein distance, privacy, GDPR

## 一句话总结

提出 Opt-Out，一种基于最优传输理论的实体级 LLM 遗忘方法，利用 Sliced Wasserstein Distance 正则化参数偏移实现精细遗忘；同时构建首个实体级遗忘数据集 ELUDe（20 目标实体 + 144 邻居实体，15K+ forget / 90K+ retain QA 对），在 Llama-3.1-8B 和 Phi-3.5 上全面超越现有方法。

## 研究背景与动机

**领域现状**：GDPR 的"被遗忘权"要求 LLM 能按用户请求删除个人数据，但完全重训成本不可承受，近似遗忘方法成为研究焦点。

**现有痛点**：已有方法（GA、NPO、DPO）主要在实例级（instance-level）小规模随机子集上评估，未考虑真实场景中需删除某人全部数据的需求。

**核心矛盾**：梯度反转（GA）虽能遗忘但极易导致模型崩溃（RQ 降至 0），加入 retain set 训练后遗忘-保留平衡仍不理想。

**本文目标** 定义实体级遗忘任务 + 构建大规模评估数据集 + 提出基于最优传输的精细遗忘方法。

**切入角度**：用 Wasserstein 距离度量当前参数与初始参数间的"运输成本"，让对遗忘重要的参数大幅偏移、对保留重要的参数保持不变。

**核心 idea**：最优传输框架下的参数分布距离正则化，实现比 L2/Cosine 更精细的参数级遗忘控制。

## 方法详解

### 整体框架（三步）
1. **Forget Set 构建**：从 Wikipedia 高人气页面选 20 个目标实体（以页面浏览量为代理指标），用 GPT-4o 逐段生成 QA 对，BERT 嵌入去重，平均每实体 ~647 QA 对。
2. **Retain Set 构建**：每个目标实体选 10 个邻居实体（双向链接 + 高浏览量 + 人物类型，灵感来自 hard negatives），同样生成 QA 对；另加 50K Alpaca-GPT4 指令数据作为 world set。
3. **Optimal Transport 遗忘**：NPO 遗忘损失 + Retain 保留损失 + Wasserstein 正则项。

### 关键设计

1. **ELUDe 数据集**

    - 20 目标实体 + 144 唯一邻居实体（有重叠）
    - 15,651 forget QA + 90,954 retain QA
    - 邻居选择标准：双向 Wikipedia 链接 + 近 3 年页面浏览量 top-10 + 人物类型
    - 相比 TOFU/RWKU 数据量更大（每实体覆盖全部知识）

2. **NPO 遗忘损失**

    - 比 GA 更稳定：在高温极限下简化为 GA，但本身有下界，显著延缓模型崩溃
    - 公式：$\mathcal{L}_{\text{NPO}} = -\mathbb{E}_{\mathcal{D}_f}[\log\sigma(-\eta\log\frac{\phi_\theta(y|x)}{\phi_{\text{ref}}(y|x)})]$

3. **Sliced Wasserstein 正则化**

    - 直接计算 Wasserstein 距离复杂度 $O(n^3\log n)$，不可行
    - 改用 Sliced Wasserstein Distance (SWD)：随机投影到低维后计算一维 Wasserstein 距离
    - 总损失：$\mathcal{L} = \mathcal{L}_{\text{NPO}} + \mathcal{L}_{\text{RT}} + \lambda \cdot SW_p(\theta, \theta_0)$
    - 关键优势：考虑参数分布的结构信息，比 L2（Euclidean）、Cosine 等逐点距离更精细

## 实验关键数据

### 表1：主实验（Llama-3.1-8B-Instruct，5 实体平均）

| 方法 | FQ ↑ | RQ ↑ | MMLU | ARC-C | 8 Benchmark Avg ↑ |
|------|------|------|------|-------|-------------------|
| Original | 45.5 | 51.2 | 68.1 | 51.8 | 64.7 |
| GA* (崩溃) | 70.9 | 0.0 | 33.9 | 23.6 | 33.8 |
| NPO* (崩溃) | 89.7 | 0.0 | 36.3 | 24.7 | 37.5 |
| NPO+RT | 82.6 | 46.6 | 62.5 | 50.1 | 62.8 |
| IDK+RT | 71.9 | 46.1 | 63.2 | 49.4 | 62.8 |
| **Opt-Out** | **87.8** | **46.6** | **63.2** | 49.8 | **63.3** |

### 表2：距离度量消融（Llama-3.1-8B-Instruct）

| 正则化距离 | FQ ↑ | RQ ↑ | Benchmark Avg ↑ |
|-----------|------|------|-----------------|
| **Wasserstein** | **87.8** | **46.6** | **63.3** |
| Euclidean | 81.5 | 46.2 | 63.0 |
| Cosine | 81.6 | 45.8 | 62.8 |
| Chebyshev | 86.3 | 45.4 | 62.2 |
| Manhattan | 47.0 | 50.9 | 64.6 (正则过强，几乎不遗忘) |

## 亮点

- **首个大规模实体级遗忘数据集 ELUDe**：20 实体 + 144 邻居，QA 总量远超 TOFU/RWKU
- **最优传输视角新颖**：SWD 正则化利用参数分布结构信息，比欧氏距离等更精细
- **全面评估**：MIA 防御（Opt-Out 48.6% ≈ 理想 50%）、9 种对抗攻击均表现最优
- **邻居实体数据的 hard positive 效果**：去掉邻居数据后 RQ 显著下降，验证了类似对比学习 hard negatives 的设计直觉

## 局限与展望

- 数据集基于 Wikipedia 实体，与真实用户隐私数据可能有差距
- 遗忘后模型仍可能生成乱码（gibberish），用户体验问题未完全解决
- 受算力限制未在 70B+ 规模模型上验证
- SWD 虽降低了计算量，但仍引入额外开销，论文未详细报告训练时间对比

## 与相关工作的对比

| 维度 | Opt-Out (本文) | TOFU (Maini et al.) | RWKU (Jin et al.) |
|------|---------------|---------------------|-------------------|
| 遗忘粒度 | 实体级（全部知识） | 实例级（虚构作者） | 实体级（真实名人） |
| 数据规模 | 15K forget + 90K retain | 20 QA/作者 × 200 | 2,879 QA |
| 邻居实体 | 144 个（hard negatives） | 无专门设计 | 有但规模小 |
| 正则化方法 | Wasserstein (SWD) | 无 | 无 |
| 评估维度 | FQ + RQ + MIA + 对抗攻击 | FQ + 通用 benchmark | FQ + 邻居 + 攻击 |

## 评分

- 新颖性: ⭐⭐⭐⭐ (最优传输用于遗忘正则化是新角度，ELUDe 数据集有价值)
- 实验充分度: ⭐⭐⭐⭐⭐ (2 模型 × 多 baseline × MIA × 对抗攻击 × 消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (实体级遗忘是 GDPR 合规的刚需，方法和数据集均可落地)

| 无正则 | 高 | 低 |
| L2 正则 | 中 | 中 |
| **Wasserstein** | **高** | **高** |

### 关键发现
- Opt-Out 在遗忘和保留上同时最优
- Wasserstein 距离优于 L2 正则
- 邻居实体是关键测试：无邻居保留测试的评估不完整
- 通用能力基本不受影响

## 亮点与洞察
- 实体级遗忘比实例级更贴近真实需求
- Wasserstein 距离正则化理论优雅
- 邻居实体概念受对比学习启发

## 局限与展望
- 仅基于 Wikipedia，非真正隐私数据
- 改进方向：大规模实体遗忘、与 RAG 结合

## 相关工作与启发
- **vs Jang et al.**：梯度反转容易崩溃
- **vs TOFU**：TOFU 是实例级，Opt-Out 是实体级

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义实体级遗忘 + Wasserstein 正则
- 实验充分度: ⭐⭐⭐⭐ 20 实体 + 144 邻居
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 隐私合规有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning](which_retain_set_matters_for_llm_unlearning_a_case_study_on_entity_unlearning.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](../../ICML2025/llm_safety/watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)

</div>

<!-- RELATED:END -->
