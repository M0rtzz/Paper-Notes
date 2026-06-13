---
title: >-
  [论文解读] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race
description: >-
  [ICML 2026][推荐系统][可持续 AI] 这篇 position paper 借 Karl Marx 的"基础-上层建筑"框架，主张当下"sustainable AI"的讨论被环境维度独占而忽略了经济与社会维度，呼吁同时拉高**气候意识**与**资源意识**两条轴，并提出 CARAML 五层行动框架（个人 / 社区 / 工业 / 政府 / 全球）以抑制正在升级的"全球 AI 军备竞赛"。
tags:
  - "ICML 2026"
  - "推荐系统"
  - "可持续 AI"
  - "气候意识"
  - "资源意识"
  - "AI arms race"
  - "CARAML"
---

# Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race

**会议**: ICML 2026  
**arXiv**: [2502.20016](https://arxiv.org/abs/2502.20016)  
**代码**: 无  
**领域**: AI 安全 / 可持续 AI / AI 治理  
**关键词**: 可持续 AI, 气候意识, 资源意识, AI arms race, CARAML

## 一句话总结
这篇 position paper 借 Karl Marx 的"基础-上层建筑"框架，主张当下"sustainable AI"的讨论被环境维度独占而忽略了经济与社会维度，呼吁同时拉高**气候意识**与**资源意识**两条轴，并提出 CARAML 五层行动框架（个人 / 社区 / 工业 / 政府 / 全球）以抑制正在升级的"全球 AI 军备竞赛"。

## 研究背景与动机

**领域现状**：可持续性历来由环境、经济、社会三大支柱组成，但"sustainable AI"的讨论几乎只谈环境（碳排、水耗、电子垃圾），把"democratisation / accessibility"问题留给另一拨人讨论，二者之间几乎不对话。

**现有痛点**：(i) 一边推 green datacenter，一边要求 LMIC 也用上低碳能源，可它们连基本电力都缺，反而加剧"双重惩罚"。(ii) DeepSeek 一类效率提升把 GPU 时间降一个量级，按 Jevons paradox 反而触发"reasoning model"等 token 量暴涨的反弹，最终能耗不减反增。(iii) Top500 supercomputer 与 AI 投资几乎全集中在 US/CN/EU 三家，2024 年仅美国私人 AI 投资就达 1091 亿美元——超过全球 134 国的 GDP，"AI 主权"的口号在缺乏物质基础的国家近乎空话。

**核心矛盾**：把 "sustainable AI" 仅压缩为"环境可持续"会与"社会可持续"产生结构性张力——单纯追求 climate awareness 会排斥资源贫国，单纯追求 resource awareness（提高可及性）又会驱动算力扩张和反弹排放。

**本文目标**：(i) 重定义 sustainable AI 必须同时高于"气候意识"与"资源意识"两轴；(ii) 用历史唯物主义的 base-superstructure 模型解释为何当下"GPU-rich vs. GPU-poor"鸿沟会被既有体制再生产；(iii) 给出 CARAML 框架，把行动义务沿"agency × scope"两轴分到五个层级。

**切入角度**：作者把 AI 当作 Crawford (2021) 意义下的**embodied infrastructure**——它不是抽象算法，而是矿产 + 劳动 + 电力 + 资本的物质聚合体。物质基础决定了上层政策与文化叙事；不动基础，叙事改不了。

**核心 idea**：可持续 AI = **climate-aware × resource-aware** 两轴同时高的 Quadrant；任何只拉一轴的方案（green AI / inclusive AI / SOTA AI / edge AI）都会塌回 arms race。

## 方法详解

position paper 没有模型与训练，它的"方法"是三层层层递进的论证，对应下面三个关键设计：先用一个二维坐标系把"什么才算 sustainable AI"形式化、并借两个反例（低碳数据中心、效率反弹）证明只拉一轴必然失败（设计 1）；再用 base-superstructure 模型解释这种失衡为何会被既有体制自我再生产、个人努力消解不掉（设计 2）；最后落到 CARAML 五层行动框架，把"两轴都要拉"翻译成从个人到全球的具体杠杆（设计 3）。

### 整体框架

第 2 节先把可持续 AI 拆成环境 / 经济 / 社会三柱，引入"气候意识 vs. 资源意识"两轴，再用两个反例（低碳数据中心、效率反弹）证明只拉一轴会败坏整体——这两步合起来构成第一个关键设计（二维四象限）。第 3 节用 base-superstructure 模型解释为何 GPU-rich vs. GPU-poor 鸿沟会自我维持（第二个关键设计）。第 4 节给出 CARAML 行动框架（第三个关键设计）。第 5 节不属于核心论证，而是正面回应"碳排会自动 plateau""distilled 模型已经解决问题"等反方观点（见下方实验关键发现与相关工作）。

### 关键设计

**1. 气候意识 × 资源意识 二维四象限：逼每个方案先声明自己落在哪一格**

作者把模糊的"sustainable AI"放进一个可分类、可争论的坐标系：横轴是 resource awareness（去殖化、可参与性，反对资源集中），纵轴是 climate awareness（碳/水/电子垃圾意识）。SOTA AI 在两轴都低；green AI 高 climate 低 resource；inclusive AI 反过来；只有右上角同时高两轴的才算 sustainable AI——而 edge AI 因 embodied carbon 难界定被特意打了问号。这张图把"democratisation 派"和"green AI 派"之间一直存在的暗中分歧显式化：单纯追求一轴必然出现 trade-off，于是讨论被从"我们都想 sustainable"逼到"你想落到哪个象限"。

**2. Base-Superstructure 模型解释 AI 体制再生产：说明为何个人节能努力总被结构吃掉**

为什么"个人努力 + 技术效率"无法自动消解 AI arms race？作者用历史唯物主义来解释：把 AI 的"base"定义为算力 + 数据 + 资本 + 劳动 + 知识商品，"superstructure"是 AI 政策、监管、研究规范、媒体叙事，两者构成正反馈——物质垄断（NVIDIA + 少数 hyperscaler）塑造了 AI sovereignty、export control、ethical guideline 等上层话语，后者反过来固化既有物质秩序。论文用一段假想情境演示：出口管制以"国家安全"叙事正当化对算力的限制，被限制方被迫重建半导体供应，结果只是在既有依赖关系里换一种依赖形态。这一论证直接类比 colonial expansion（殖民资本与种族等级互为基础与上层），主张当代 AI 正在 GPU 时代重复这种殖民结构——在 ICML 主会 position 里把人文社科理论硬性嵌入论证主线，是相当少见的写法。

**3. CARAML 行动框架（agency × scope 两轴五层）：把"两轴都要拉"落到从个人到全球的具体杠杆**

总目标是同时拉气候与资源两轴，但单层行动不够——个人 agency 强但 scope 小，国家与全球反之。CARAML 用"agency × scope"框图把行动拆到五层，每层给至少一项可执行动作：个人用 perf-per-resource 替代单纯 accuracy；ML 社区做大规模实验预注册以杜绝重复浪费；工业自我加碳排上限；政府把 AI 影响评估写进监管（类比 EIA）；全球把"Right to Compute"改写为"Right to Sustainable Compute"，并通过开源/共享算力对抗数字鸿沟。CARAML 同时是论文实证的载体——LLaMA / BLOOM / Mistral 等开源模型被用来演示哪些层在做、哪些层缺位，论证只有协调多层才能扩大影响力 sphere。

### 损失函数 / 训练策略

position paper，无模型 / loss。证据基础包括 TOP500 (2025) 算力分布、Maslej et al. (2025) 投资数据、ICML/ICLR/NeurIPS 2006–2025 作者国别分布、IRENA 与 IEA 的电力 / 可再生数据、LLaMA-3.1 训练 30.84M GPU-hours $\Rightarrow$ 21.5 GWh / 8930 tCO2e（美国电网）的可复算估计。

## 实验关键数据

### 主实验：全球 AI 算力 / 投资 / 学术参与的集中度

| 维度 | 数据 | 来源 |
|------|------|------|
| Top500 超算地理集中 | 主要在 US/CN/EU；非洲几乎缺席 | TOP500 (2025) |
| 2024 美国私人 AI 投资 | $109.1B（≈12× 中国, ≈24× 英国） | Maslej et al. (2025) |
| 该投资 vs. 国家 GDP | 超过全球 134 国的 2024 年 GDP | 同上 |
| LLM 排行榜机构构成 | 多数条目来自企业，非学术 | Moutawwakil & Pierrard (2023) |
| 算力倍增周期 | 自 2012 起，部分前沿系统每 3.4 月翻一倍 | Sevilla et al. (2022) |
| ICML/ICLR/NeurIPS 作者国别 (2006–2025) | LMIC 显著不足，差距逐年扩大 | World Bank (2024) + 本文整理 |

### 案例：LLaMA-3.1 (405B) 单次训练成本随地理变化

| 训练地点 | GPU 小时 | 估计能耗 | 估计碳排 |
|----------|----------|----------|----------|
| 任意 | 30.84M H100-h | 21.5 GWh | — |
| 美国电网 | 同上 | 21.5 GWh | 8,930 tCO2e |
| 瑞典电网 | 同上 | 21.5 GWh | 750 tCO2e |
| 印度电网 | 同上 | 21.5 GWh | 14,737 tCO2e |

### 关键发现
- **DeepSeek 反弹**：DeepSeek-V3 仅用 2.8M H800-h，比 LLaMA-3-405B 的 30.8M H100-h 少一个量级，但发布后流行度暴涨 2800% 并催生"reasoning model"——一次 80-token 回复要 ~800 latent thinking token，效率收益被使用量增长全部吃掉。
- **低碳数据中心悖论**：被要求改用 renewables 的恰恰是清洁电力供给最弱的 LMIC，被迫接受"双重惩罚"。
- **碳信用 / carbon-neutral 标榜失效**：Probst et al. (2024) 表明 carbon crediting projects 很少真兑现减排，"carbon-neutral training"宣传站不住脚。
- **distilled 模型不能解决问题**：QLoRA / Phi / 小 LLaMA 系列的低推理成本背后是大模型的 embodied carbon 被摊销，制造仍集中在少数 actor，算力鸿沟原封不动。
- **IEA 预测**：到 2030 年 AI 可能消耗 945 TWh，约相当于日本年用电量；按当下平均碳强度即 447M tCO2e，接近全球商业航空 2023 年排放的一半。

## 亮点与洞察
- **二维四象限**比"sustainable AI = green AI"这种习惯把"democratisation 派"和"climate 派"的暗争翻到台面，可以直接当 review 工具用。
- **把 NVIDIA 算力垄断、export control、AI sovereignty 叙事三件事放进同一个 base-superstructure 解释里**，让"为什么个人节能努力总被结构吃掉"这件事第一次有了清晰理论支撑。
- **CARAML 的"agency × scope"框图**给出了一个跨域可迁移的模板——把任何治理倡议都拆到五层，避免空喊"全球协作"。
- **明确把 rebound effect 列为反方观点并正面回应**，承认效率提升不等于净减排，这种诚实在 sustainability 文献里相对罕见。

## 局限与展望
- 缺乏可复现代码 / 数据库；CARAML 多为定性建议，没给出哪些动作的减排弹性最高。
- base-superstructure 框架是宏观叙事，对单一研究者改不了上层 — 个人 agency 一节在策略上有点单薄。
- 经济测算多来自第三方报告（Maslej, IEA, IRENA），未做敏感性分析。
- 对 AI sovereignty 的批判较锐利，但对"那 LMIC 应不应该建自己算力"的回答不够具体。

## 相关工作与启发
- **vs Van Wynsberghe (2021)**: 接受其"环境 / 经济 / 社会"三柱定义，但具体化为可争论的二维象限，并把张力作为论文主体。
- **vs Schwartz et al. (2020) green AI**: 指出 green AI 落在象限左上（高 climate 低 resource），不能等同于 sustainable AI。
- **vs Wright et al. (2025)**: 同样强调环境≠可持续，本文把这个观察推进为完整的两轴框架 + CARAML 行动方案。
- **vs Patterson et al. (2022)（反方）**: 该文认为 AI 碳排会自然 plateau；本文用 Google 2023 年碳排同比 +13% / vs. 2019 基线 +48% 反驳。
- **vs Ahmed & Wahed (2020)**: 该文用"compute divide"刻画 de-democratization，本文把它升级到 base-superstructure 解释并提出 CARAML 干预。

## 评分
- 新颖性: ⭐⭐⭐⭐ 二维象限 + base-superstructure 解释 + CARAML 三件套是 sustainability/AI 治理界少见的整合。
- 实验充分度: ⭐⭐⭐ 证据多来自第三方报告，无自主实验，但案例（DeepSeek 反弹、LLaMA 地理碳排）够刻骨。
- 写作质量: ⭐⭐⭐⭐ 把 Marx 框架嵌入 ICML 论文还能维持可读性，文献覆盖广，反方观点章节克制。
- 价值: ⭐⭐⭐⭐⭐ 把"sustainable AI"从环境单轴拓为社会-环境双轴，并给出 review/funding 可直接借用的话语工具，长期会被反复引用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)
- [\[ICML 2025\] Position: The Right to AI](../../ICML2025/recommender/the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](../../AAAI2026/recommender/moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)

</div>

<!-- RELATED:END -->
