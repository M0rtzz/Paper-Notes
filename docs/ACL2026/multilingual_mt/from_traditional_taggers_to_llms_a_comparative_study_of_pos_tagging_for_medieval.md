---
title: >-
  [论文解读] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages
description: >-
  [ACL 2026][多语言/翻译][中古罗曼语] 作者在三种中古罗曼语（古奥克语 NAF、古加泰罗尼亚语 CAT、古法语 Chauliac）的 POS 标注任务上，把传统 tagger（UDPipe / COLaF）与开源 LLM（Gemma3-12B / Phi4-14B）放在 zero-shot、few…
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "中古罗曼语"
  - "词性标注"
  - "LLM 微调"
  - "跨语种迁移"
  - "数字人文"
---

# From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages

**会议**: ACL 2026  
**arXiv**: [2605.09147](https://arxiv.org/abs/2605.09147)  
**代码**: <https://github.com/msch38/medieval-romance-pos>  
**领域**: 多语言 / 历史语言 NLP / POS 标注  
**关键词**: 中古罗曼语、词性标注、LLM 微调、跨语种迁移、数字人文

## 一句话总结
作者在三种中古罗曼语（古奥克语 NAF、古加泰罗尼亚语 CAT、古法语 Chauliac）的 POS 标注任务上，把传统 tagger（UDPipe / COLaF）与开源 LLM（Gemma3-12B / Phi4-14B）放在 zero-shot、few-shot、单语微调、双语 CLTF、三语 CLTF 五种设置下做系统对比，发现 LLM 一致优于传统方法，加泰罗尼亚语充当"桥梁语种"使 CAT+FR 双语训练把古法语 Chauliac 推到 93.14% 的最高准确率。

## 研究背景与动机
**领域现状**：数字人文里对中古文献做大规模语言学分析，第一步几乎都是 POS 标注 —— 它喂给后续的句法解析、语义分析、历时变化建模。罗曼语家族的中古阶段（古奥克语 / 古加泰 / 古法）传统上依赖 UDPipe、COLaF 等基于通用依存树库训练的工具，或者纯规则系统。

**现有痛点**：中古罗曼语有三个让现代 tagger 严重退化的特征 —— ① 正字法极度不稳定（同一个词 `deceplina/disiplina/desiplina` 在一个篇章里轮番出现）；② 形态系统复杂且方言化；③ 标注语料稀缺，最小的 Chauliac 数据集只有 2,443 token。结果是 UDPipe 在 NAF 上掉到 68%，COLaF 在 CAT 上只剩 52%。

**核心矛盾**：现代 NLP 工具的训练分布与中古文本的真实分布存在巨大 gap，而又没有足够的中古监督信号去从头训练一个专用模型。能否用大模型"借"现代多语种知识来弥补这一 gap，作者并不清楚 —— 此前对 LLM 在中古罗曼语 POS 的系统评估几乎为零。

**本文目标**：把三个互相独立的问题打包回答 —— (Q1) LLM 比传统 tagger 强多少？(Q2) prompt 形式和解码超参影响多大？(Q3) 跨语种迁移在多大程度上能救活低资源中古语种？

**切入角度**：作者注意到加泰罗尼亚语在谱系上恰好处在 Gallo-Romance 与 Occitano-Romance 之间，可以充当"桥梁"，因此专门设计了 CAT+OCC、CAT+FR、FR+OCC 三种双语配置和一个三语联训配置，让谱系亲缘性与语料规模可以被分离观察。

**核心 idea**：在固定的 80/20 划分上跑 5 个实验家族（traditional、prompting、单语 FT、双语 CLTF、三语 CLTF）+ 4 种解码策略，给出一份"按资源 × 目标语种"选方法的实操地图。

## 方法详解

### 整体框架
整套实验是一个 5×N 网格：5 个方法族（Traditional baseline / LLM Prompting / 单语 FT / 双语 CLTF / 三语 CLTF）× 3 个数据集（NAF 45,457 tokens、CAT 59,359 tokens、Chauliac 2,443 tokens）× 2 个 LLM 主干（Gemma3-12B、Phi4-14B）。所有 LLM 微调都用同一 LoRA 配置（$r=16, \alpha=32$，目标模块 q/k/v/o_proj，学习率 $2\times 10^{-4}$，10 epoch，AdamW），所有解码都在同一 prompt 模板下做扫描。Tagset 统一到 UD 17 类，输入是 token 流，输出 JSON `[{"word":..., "UPOS":...}, ...]`。

### 关键设计

1. **统一的 80/20 切分作为公平比较的基座**:

    - 功能：让单语 / 双语 / 三语三种训练配置直接可比。
    - 核心思路：每个数据集固定一个 20% 的 test partition；双语训练用两个 80% partition 的并集；三语训练用三个 80% partition 的并集；测试始终在那个固定的 20% 上做。这保证了任意配置下"test token 都没在训练里出现过"。
    - 设计动机：以前的跨语种迁移文献经常因为切分不一致 —— 例如双语训练顺手把目标语全部 80% 都吃掉 —— 导致"迁移收益"其实是"多吃了数据"。本文这个切法把"迁移信号"和"数据量"显式拆开，使 $\Delta_{\text{bilingual-mono}}$ 可解释。

2. **谱系驱动的双语配对设计**:

    - 功能：检验语言亲缘性是否真的预测迁移效果。
    - 核心思路：作者并不暴力跑所有 LLM × 所有语对，而是用语言学先验挑出三组 —— CAT+OCC（同属 Occitano-Romance）、CAT+FR（跨 Gallo/Occitano 但 CAT 居中）、FR+OCC（跨两支但绕过 CAT）。每个配对在两个语种上分别报告。
    - 设计动机：如果谱系亲缘是主导因素，那么 FR+OCC 应该和 CAT+OCC 差不多；如果 CAT 真的是"桥梁"，那么 CAT 出现的配置都该赢。结果显示 CAT+OCC→NAF 89.25%、CAT+FR→Chauliac 93.14%，而 FR+OCC→NAF 只有 80.31%（相对单语 +0.22），CAT 的桥梁假设被强烈支持。

3. **混合三语 few-shot 示例 + 全套解码扫描**:

    - 功能：剥离 prompt 工程与解码超参对结论的污染。
    - 核心思路：few-shot prompt 里故意把三个语种的示例 token 混在同一个 block 里（如 `bo/ADJ`、`volch/VERB`、`seyor/NOUN`、`addicions/NOUN`），让模型显式看到正字法多样性；解码端把 beam search（$w\in\{1,15\}$）、temperature（$\tau\in\{0.6,0.8,0.9\}$）、top-$k$（$\{5,20,50\}$）、top-$p$（$\{0.75,0.85,0.95\}$）全部跑一遍。
    - 设计动机：标注是判别任务，作者预期确定性解码应该赢；实测 beam-15 确实在所有 model × dataset 上都拿最高分（Phi4 few-shot + Beam-15: 81.23% 平均），且采样族标准差也只在 0.1-0.2 之间 —— 这意味着后面所有"LLM > 传统"的结论不是被某个魔法采样温度带出来的。

### 损失函数 / 训练策略
微调阶段用标准的 next-token cross-entropy（监督 JSON 输出里的 UPOS 字段），LoRA 只更新注意力四个投影矩阵，dropout 0.1。batch 4，10 epoch，对最小的 Chauliac（2,443 token）很快进入过拟合区，这也是为什么对它单语 FT 反而比 prompting 提升有限（Gemma3 单语 83.64% vs Phi4 few-shot 84.98%）。Prompt 阶段不更新权重，只在解码端做超参扫描。

## 实验关键数据

### 主实验

| 方法族 | 模型 / 策略 | NAF Acc. | CAT Acc. | Chauliac Acc. |
|--------|-------------|----------|----------|---------------|
| Traditional | UDPipe | 68.01 | 81.59 | 89.40 |
| Traditional | COLaF | 65.73 | 52.15 | 72.50 |
| Prompting | Phi4 Few-shot | 75.01 | 83.69 | 84.98 |
| Fine-tune | Gemma3 单语 | 80.09 | **92.52** | 83.64 |
| Bilingual CLTF | Gemma3 CAT+OCC | 89.25 | 91.62 | – |
| Bilingual CLTF | Gemma3 CAT+FR | – | 91.28 | **93.14** |
| Bilingual CLTF | Gemma3 FR+OCC | 80.31 | – | 85.74 |
| Trilingual CLTF | Gemma3 | **89.68** | 89.16 | 88.23 |
| $\Delta$ best vs UDPipe | — | **+21.67** | **+10.93** | **+3.74** |

关键观察：① 平均准确率沿 Traditional 71.56% → Prompting 77.75% → 单语 FT 85.19% → 最优 CLTF 单调上升；② NAF 受益最大（+21.67 pp），Chauliac 受益最小（+3.74 pp）但仍能从 89.40% 推到 93.14%。

### 消融实验

| 配置 | NAF | CAT | Chauliac | 解读 |
|------|-----|-----|----------|------|
| 单语 FT (Gemma3) | 80.09 | 92.52 | 83.64 | 基线 |
| 双语 CAT+OCC | 89.25 | 91.62 | – | NAF +9.16，CAT 微降 0.90 |
| 双语 CAT+FR | – | 91.28 | 93.14 | Chauliac +9.50，CAT 微降 1.24 |
| 双语 FR+OCC | 80.31 | – | 85.74 | NAF +0.22，Chauliac +2.10（无 CAT 几乎无收益）|
| 三语 CLTF | 89.68 | 89.16 | 88.23 | NAF 最优；但 Chauliac 比 CAT+FR 低 4.91 pp |

**Phi4 prompting 解码鲁棒性**（few-shot 平均准确率，所有解码族）：

| 解码族 | 平均 Acc. | 跨数据集 std | 推荐 |
|--------|----------|--------------|------|
| Beam-15 | 81.23 | 0.12 | 最稳最高 |
| Top-$k$ | 80.28 | 0.20 | 备选 |
| Top-$p$ | 80.69 | 0.18 | 平衡 |
| Temperature | 80.57 | 0.21 | 方差略大 |

### 关键发现
- **加泰罗尼亚语是真正的桥梁**：任何带 CAT 的双语配对都赢，不带 CAT 的 FR+OCC 在 NAF 上只挣到 +0.22 pp，证明仅靠谱系亲缘（FR 与 OCC 同属罗曼语）不足以保证迁移成功，必须配合合适的"中介"。
- **三语不总是最好**：Chauliac 在三语下 88.23%，反而比 CAT+FR 双语 93.14% 低 4.91 pp。作者归因于 Chauliac 太小（2,443 token），三语训练把信号稀释了，"focused exposure to closely related material"才是小数据集的王道。
- **POS 级别提升不均**：在 NAF 上，PROPN +66.62、NUM +62.97、SCONJ +41.68 是受益最大的；ADP、CCONJ 这些功能词本来就接近天花板（85-95% F1），提升空间小。说明 LLM 的上下文表示主要把"语义重"的开放类词救了起来。
- **解码差异极小**：Phi4 few-shot 在所有解码配置下平均准确率 80.20-81.23%，跨配置 std 都在 0.21 以内，这把"实验结果是否被解码超参偶然抬高"的疑问彻底关掉了。

## 亮点与洞察
- **首次把"中古罗曼语 × LLM × 多语种迁移"做成一张完整地图**：之前的工作要么只比 prompting，要么只比单语 FT；本文把 5 个方法族在统一切分下打满，给出可直接复用的方法选择表（Table 13）。
- **桥梁语种这个发现具有可迁移性**：在其他语言家族（如日耳曼语里的中古英语 + 中古荷兰语 + 古德语，或者古汉语 + 中古汉语 + 中世纪日语借词层）大概率也存在"中介语"现象，本文的双语 CLTF 实验范式可以原样照搬过去。
- **混合多语 few-shot 示例的 prompt 写法**：把多语种正字法变体混在一个示例 block 里，相当于在 in-context learning 阶段就告诉模型"别指望正字一致"，这个 trick 在任何"高方言性 / 高拼写变体"的 NLP 场景里都值得借用。
- **小数据集应当避免过度多语训练**：对 < 5k token 的目标域，三语反而稀释信号；这与 LLM 时代盛行的"训练数据越杂越好"直觉相反，给数字人文领域的实践者一个明确的反例。

## 局限与展望
- 三个数据集 token 数差了一个数量级（2,443 vs 59,359），而且分别属于文学、编年史、医学三种不同体裁；作者承认没做受控的 subsample / learning-curve 实验，因此"语料规模 / 体裁 / 谱系亲缘"三个因子在 CLTF 收益里无法被严格分离。
- 没有评估更小的 LLM（1–3B）或专门的 token-classification encoder（XLM-R / mBERT），最强结果都建立在 12–14B + LoRA 上，部署成本比 UDPipe 高得多；对真实的人文项目而言性价比还需要再确认。
- few-shot 示例固定混合三语种，但每语种独立设计 prompt、按目标语种选择示例策略都没尝试，prompt 优化空间还很大。
- 没有做真正的 zero-shot 跨语种评估（如 CAT+FR → NAF，目标语完全不出现于训练），这是检验"纯迁移能力"最干净的实验，留给未来工作。
- 未推广到 lemmatization、parsing、NER 等下游，无法判断 POS 上的提升能否传递到完整的历史 NLP pipeline。

## 相关工作与启发
- **vs Schöffel et al. 2025b（Modern Models, Medieval Texts）**：那篇只在古奥克语上做 prompting + FT，本文把语种扩到三个并加入双语 / 三语 CLTF，量级更系统；它证明了"模型能用"，本文回答"该怎么用"。
- **vs Camps et al. 2021（古典法语剧本 POS）**：他们做古典法语单语适配，没有跨语种维度；本文用 CAT 把 FR 推到 93.14% 表明"找一个谱系上居中的兄弟语种再做双语 FT"可能是比"再标更多目标语数据"更划算的方向。
- **vs Bollmann et al. 2019 / Manjavacas et al. 2019（历史文本归一化）**：他们走"先归一化、再喂现代 tagger"路线；本文走"绕过归一化，让 LLM 直接读原始拼写"路线，并且赢得很彻底，间接说明在 LLM 时代规范化步骤的必要性正在下降。
- **vs Karthikeyan et al. 2020（mBERT 跨语种能力）**：他们在现代语种上发现谱系亲缘是迁移能力的主因，本文在中古语种上指出谱系亲缘是必要但不充分条件 —— 还需要一个合适的"桥梁语种"+ 足够的目标域数据。

## 评分
- 新颖性: ⭐⭐⭐⭐ 没有提出新模型，但首次把 5 方法族 × 3 中古罗曼语种系统打通，并发现 CAT 桥梁效应
- 实验充分度: ⭐⭐⭐⭐⭐ 5 方法族 × 3 数据集 × 2 模型 × 12 解码配置 + 完整 per-class F1 分解
- 写作质量: ⭐⭐⭐⭐ 表格清晰、结论分层、limitation 写得诚实
- 价值: ⭐⭐⭐⭐ 数字人文社群可以直接照搬方法选择表；桥梁语种的实证证据对低资源 NLP 有跨领域启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](../../ACL2025/multilingual_mt/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](../../ACL2025/multilingual_mt/understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2026\] Lost in Translation: Do LVLM Judges Generalize Across Languages?](lost_in_translation_do_lvlm_judges_generalize_across_languages.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)

</div>

<!-- RELATED:END -->
