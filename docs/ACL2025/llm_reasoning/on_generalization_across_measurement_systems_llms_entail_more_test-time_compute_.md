---
title: >-
  [论文解读] On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures
description: >-
  [ACL 2025][LLM推理][度量系统泛化] 系统研究 LLM 跨度量系统（货币、长度、重量）的泛化能力，发现模型默认使用训练数据中的主导度量（如美元、公制），对非主导度量查询准确率显著下降；CoT 推理可弥补但推理成本增加高达 300%，对欠代表文化用户构成系统性不公平。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "度量系统泛化"
  - "文化偏见"
  - "测试时计算"
  - "推理成本不公平"
  - "跨文化NLP"
---

# On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures

**会议**: ACL 2025  
**arXiv**: [2506.02591](https://arxiv.org/abs/2506.02591)  
**代码**: [GitHub](https://github.com/MinhDucBui/MeasurementSystemBias)  
**领域**: LLM推理  
**关键词**: 度量系统泛化, 文化偏见, 测试时计算, 推理成本不公平, 跨文化NLP

## 一句话总结
系统研究 LLM 跨度量系统（货币、长度、重量）的泛化能力，发现模型默认使用训练数据中的主导度量（如美元、公制），对非主导度量查询准确率显著下降；CoT 推理可弥补但推理成本增加高达 300%，对欠代表文化用户构成系统性不公平。

## 研究背景与动机

**领域现状**：LLM 已在参数中编码了大量事实知识（Petroni et al. 2019; Allen-Zhu & Li 2024），包括各种度量数值，但训练数据以英文和西方文化为主导。

**现有痛点**：来自不同文化背景的用户使用不同度量系统（公制 vs 英制、不同货币、不同重量单位如"斤""근"等），LLM 能否公平地为所有用户提供准确信息尚未被系统研究。

**核心矛盾**：度量系统间的换算关系是确定的（如 1 mile = 1.609 km），人类可以自由选用任意度量系统陈述事实，但 LLM 在非主导度量下的表现可能大幅退化——这不是知识缺失问题，而是处理公平性问题。

**本文切入角度**：编制三类跨文化评测数据集（财政数据/城市距离/食品价格），覆盖 112 种货币、10 种长度单位、10 种重量单位，围绕三个研究问题展开：(RQ1) LLM 默认使用什么度量？(RQ2) 切换度量后准确率如何变化？(RQ3) 推理策略能否弥补及其代价。

**核心 idea**：LLM 在非主导度量下需要额外推理步骤才能正确回答——欠代表文化的用户实质上被征收了"推理税"（inference cost 增加高达 300%）。

## 方法详解

### 整体框架
构建三类跨文化度量数据集，每个查询可用多种度量单位提问，在 7 个开源 LLM 上评测直接回答、顺序单跳推理（Sequential Single Hops）和 CoT 推理三种策略下的准确率与推理成本。

### 数据集设计

- **Fiscal Data（货币系统）**：148 国 GDP per capita（2021年），使用 IMF 汇率转换为 112 种货币，共 16,576 条样本
- **City Distances（长度系统）**：德、俄、中、日（公里制）和美国（英里制）的城市间距离，使用 Haversine 公式计算，测试 10 种长度单位（公里/英里/Cape Foot/泰制wa/中制里等），共 38,000+ 条样本
- **Food Prices per Weight（重量系统）**：来自 WFP 的 76 国食品商品价格（公斤制）和美国 BLS 的食品价格（磅制），测试 10 种重量单位（千克/磅/斤/근等），共 27,820 条样本

### 三个核心研究问题

1. **RQ1 默认度量识别**：不指定单位时 LLM 默认使用什么？通过无约束 prompt 统计模型默认输出的度量系统
2. **RQ2 跨度量稳定性**：指定不同度量后准确率变化多大？使用 MAPD（逆平均绝对百分比偏差）评测，Wilcoxon 符号秩检验判断显著性（p=0.001）
3. **RQ3 推理弥补与代价**：将跨度量转换建模为多跳推理（multi-hop reasoning）问题，需要三步：(i) 检索默认度量下的事实 → (ii) 获取转换率 → (iii) 算术推理。测试 Sequential Single Hops 和 CoT 两种策略

### 评测指标 MAPD

MAPD_k = 100 × (1 - (1/N) × Σ min(|p(x_i, f_k) - y_i| / |y_i|, 1))

值越高表示性能越好，偏差上限截断为 100% 以减少极端预测影响。

### 测试模型
7 个开源指令微调 LLM：Qwen2.5（72B/7B）、Llama 3.3（70B）、Llama 3.1（70B/8B）、Aya Expanse（32B/8B），按规模分为 Large（≥70B）和 Small（<70B）两组。

## 实验关键数据

### 表1：LLM 默认度量系统统计

| 数据/度量类型 | Large 模型 | Small 模型 | 平均 |
|---|---|---|---|
| **货币（Fiscal Data）** | | | |
| → 本地货币 | 18% | 16% | 17% |
| → USD | 76% | 82% | 80% |
| → 其他 | 6% | 2% | 3% |
| **重量（Kilogram 数据）** | | | |
| → Kilogram | 100% | 100% | 100% |
| → Pound/其他 | 0% | 0% | 0% |
| **长度（Kilometer 数据）** | | | |
| → Kilometer | 100% | 100% | 100% |
| → Mile/其他 | 0% | 0% | 0% |
| **长度（Mile 数据/美国城市）** | | | |
| → Mile | 63% | 78% | 73% |
| → Kilometer | 37% | 22% | 27% |

**发现**：LLM 默认使用训练数据中的主导度量——财政数据默认美元（80%），重量和长度严格遵循数据来源的文化背景。

### 表2：推理策略下的性能变化与推理成本增加（相对于默认度量无推理基线）

| 默认度量 | Small 直接 | Small Seq | Small CoT | Large 直接 | Large Seq | Large CoT | 成本Seq | 成本CoT |
|---|---|---|---|---|---|---|---|---|
| USD→非默认货币 | -77.5% | -31.3% | -44.2% | -72.5% | -0.1% | +0.8% | +189% | **+302%** |
| kg→非默认重量 | -33.1% | -17.9% | -13.9% | -59.1% | -4.8% | -8.1% | +191% | +206% |
| lb→非默认重量 | -34.0% | -19.1% | -33.3% | -29.3% | -4.7% | -9.4% | +199% | +187% |
| km→非默认长度 | -63.6% | -31.8% | -41.7% | -57.9% | -7.9% | -4.0% | +180% | +210% |
| mile→非默认长度 | -74.4% | -34.6% | -50.0% | -73.1% | -7.2% | -7.5% | +197% | +185% |

**关键发现**：Large 模型使用 Seq/CoT 可将性能损失从 50-70% 缩小到 5-10% 以内，但推理成本增加 180-302%，货币转换最高达 **3 倍**。Small 模型推理策略效果有限。

### 表3：多语言对齐设置下的性能差距（*表示 p<0.001 显著）

| 设置 | Llama 3.3 70B | Qwen2.5 72B | Aya 32B |
|---|---|---|---|
| 韩语+韩元 vs USD | -37.89* | -27.43* | -36.36* |
| 土耳其语+里拉 vs USD | -57.01* | -42.22* | -49.13* |
| 韩语+근(Geun) vs kg | -21.99* | -31.12* | -22.11* |
| 中文+斤(Jin) vs kg | -25.39* | -26.79* | -11.33* |
| 日语+間(Ken) vs km | -80.09* | -75.71* | -73.90* |
| 中文+里(Li) vs km | -20.19* | -13.18* | -15.03* |

即使语言与度量系统对齐，性能下降仍几乎全部显著——语言对齐无法解决度量偏见。

### 表4：货币偏见与国家收入水平

| 国家收入等级 | 平均 MAPD |
|---|---|
| 高收入国家 | 42.80% |
| 中高收入国家 | 19.79% |
| 中低收入国家 | 14.31% |
| 低收入国家 | 9.03% |

高收入国家货币的准确率比低收入国家高 **34 个百分点**，偏见明确偏向最富裕国家。

## 亮点与创新

1. **问题定义精准**：首次将"跨度量系统事实查询"建模为多跳推理（multi-hop reasoning）问题，理论框架清晰——需要事实检索 + 转换率检索 + 算术推理三步
2. **"推理税"概念**：揭示了一个隐蔽的不公平机制——非主导文化用户需要模型进行更多推理（成本增加300%），而经济欠发达地区用户最承受不起这些额外成本
3. **评测覆盖全面**：112 种货币、10 种长度单位、10 种重量单位，跨 7 个模型、多语言设置，规模罕见
4. **实验设计严谨**：使用最有利汇率减少波动影响，Wilcoxon 检验（p=0.001）保证统计可信度，3 种 prompt 变体保证鲁棒性
5. **CoT 锚定分析**：91% 的 CoT 从 USD 出发进行货币转换，揭示模型推理的起点偏见

## 局限性

1. **主要使用英文 prompt**：英文是 LLM 最强语言，结果实际上是"最好情况"，在其他语言下问题会更严重
2. **度量类型覆盖有限**：仅涵盖货币/长度/重量三类，未涉及温度、面积、体积等
3. **仅测试开源模型**：未评测 GPT-4/Claude 等闭源模型
4. **实际场景用户可用工具辅助**：LLM 单独做单位转换并非最优，但论文指出用户倾向直接对话
5. **汇率时变性**：仅用 2021 年数据，未考虑更长时间跨度影响

## 相关工作

- **事实检索偏见**：Mallen et al. 2023、Manvi et al. 2024 发现 LLM 对低知名度实体和欠发达地区准确率更低
- **文化偏见**：Kirk et al. 2024、Cao et al. 2024 等揭示 LLM 偏向西方价值观
- **多跳推理**：Press et al. 2023、Biran et al. 2024 研究隐式多跳问题，LLM 仍有困难
- **度量系统处理**：Park et al. 2022（小模型跨度量比较困难）、Dinu et al. 2020（翻译模型集成度量转换）
- **本文区隔**：聚焦解码器 LLM 的事实检索，首次系统量化跨度量泛化的推理成本差异

## 评分

| 维度 | 分数 (1-10) | 说明 |
|---|---|---|
| 问题重要性 | 8 | 揭示 LLM 对欠代表文化的隐性不公平，具有社会影响力 |
| 创新性 | 7 | 问题新颖（跨度量泛化+推理成本不公平），但核心方法为评测驱动 |
| 实验充分性 | 9 | 112种货币x7模型x3推理策略x多语言，覆盖极广，统计检验严格 |
| 可复现性 | 9 | 数据集和代码开源，数据来源均可公开获取 |
| 写作质量 | 8 | 逻辑清晰，RQ 驱动结构合理，图表有效 |
| 综合评分 | **8** | 评测型工作的典范，问题定义有洞察力，数据规模和实验严谨性突出 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Retrieval Systems with Inference-Time Logical Reasoning](enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ICML 2026\] Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models](../../ICML2026/llm_reasoning/diversity_matters_revisiting_test-time_compute_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
