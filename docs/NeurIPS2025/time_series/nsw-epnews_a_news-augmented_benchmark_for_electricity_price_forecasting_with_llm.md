---
title: >-
  [论文解读] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs
description: >-
  [NeurIPS 2025][时间序列][电力价格预测] 提出首个融合新闻文本的电力价格预测基准 NSW-EPNews，系统评估传统模型和 LLM 在多模态电价预测中的表现，发现新闻特征对传统模型增益有限，而 LLM 存在严重幻觉问题。
tags:
  - NeurIPS 2025
  - 时间序列
  - 电力价格预测
  - LLM预测
  - 多模态基准
  - 幻觉检测
  - 提示工程
---

# NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2506.11050](https://arxiv.org/abs/2506.11050)  
**代码**: [Figshare Dataset](https://figshare.com/s/e25f3a98679d347f2a2e)  
**领域**: 时间序列  
**关键词**: 电力价格预测, LLM预测, 多模态基准, 幻觉检测, 提示工程

## 一句话总结

提出首个融合新闻文本的电力价格预测基准 NSW-EPNews，系统评估传统模型和 LLM 在多模态电价预测中的表现，发现新闻特征对传统模型增益有限，而 LLM 存在严重幻觉问题。

## 研究背景与动机

电力价格预测是能源管理系统的核心任务。现有方法过度依赖历史数值数据，忽略了同期的文本信号（如市场新闻、政策公告）和气象数据对价格的影响。虽然 LLM 展现出融合结构化特征和自由文本的潜力，但社区缺乏一个系统化的基准来评估 LLM 能否可靠地将新闻情感和天气线索转化为准确的数值预测。

具体而言，存在以下关键问题：

**缺乏多模态基准**：现有电力预测数据集只包含数值时间序列，不含新闻文本数据

**LLM 可靠性未知**：LLM 在高风险能源市场中是否能产生可信预测尚未验证

**幻觉问题**：LLM 可能生成看似合理但与输入数据不一致的价格序列

## 方法详解

### 整体框架

NSW-EPNews 基准包含三个核心组件：数据构建、提示模板设计、评估框架（含幻觉检测）。数据覆盖澳大利亚新南威尔士州 2015–2024 年超过 175,000 条半小时级现货价格、每日温度记录和来自 WattClarity 的策展市场新闻摘要。预测任务设定为 48 步前瞻预测（即预测次日 48 个半小时价格）。

### 关键设计

1. **数据预处理与新闻分类**：原始新闻文本通过 GPT-4o 进行四模块分类处理。四个模块分别负责角色分配、分类标准、关键属性和摘要规则，将冗长的电力市场报告压缩为机器可解析的结构化信号。每篇新闻被标注影响等级（高/中/低），用于探究模型是否能差异化利用不同影响级别的新闻。对于价格数据，2021 年 10 月后 NEM 从 30 分钟频率变为 5 分钟频率，采用中位数降采样保持一致性。

2. **四种提示模板设计**：为全面评估 LLM，设计了四种提示格式，沿两个维度变化：

    - 零样本 vs. 少样本（是否包含示例 Q&A 对）
    - 有无链式思维（CoT）推理
   
   每种提示都整合了 48 个历史价格点、新闻摘要、温度和日期信息。此外还设计了消融提示（仅含历史价格），用于分析新闻和温度数据的边际贡献。

3. **幻觉与错误检测框架**：定义并检测四类 LLM 输出异常：

    - **回声失败 (Echoing Failure)**：预测值中有 ≥10 个直接复制自历史输入
    - **平凡变换 (Trivial Transformation)**：预测值为历史值加固定偏移（匹配 ≥20 个点）
    - **退化复制 (Degenerate Copying)**：某个值重复出现 >5 次
    - **格式违规 (Format Violation)**：输出无法解析为价格列表

### 损失函数 / 训练策略

传统基线模型（ARIMA、LR、XGBoost）使用标准的监督学习训练：ARIMA 仅用价格自回归结构，LR 和 XGBoost 额外将新闻做 TF-IDF 向量化作为特征。LLM（GPT-4o、Gemini 1.5 Pro）通过 API 调用进行零样本/少样本推理，不做微调。评估指标包括 MSE、RMSE、MAE 和 MAPE。

## 实验关键数据

### 主实验

| 模型 | 提示类型 | MSE (50%数据) | MAE (50%数据) | MSE (10%数据) | MAE (10%数据) |
|------|---------|--------------|--------------|--------------|--------------|
| ARIMA | 无 | 124,926 | 54.58 | 393,097 | 89.09 |
| Linear Regression | 无 | 129,537 | 69.54 | 393,663 | 90.20 |
| XGBoost | 无 | 263,069 | 113.33 | 453,917 | 129.61 |
| GPT-4o | Zero-shot | 209,929 | 53.65 | 692,109 | 98.27 |
| GPT-4o | Few-shot+CoT | 187,494 | 53.40 | 717,159 | 102.96 |
| Gemini 1.5 Pro | Zero-shot+CoT | 143,822 | 47.03 | 164,605 | 60.75 |
| Gemini 1.5 Pro | Few-shot+CoT | 187,494 | 53.40 | 319,180 | 73.85 |

### 消融实验（幻觉检测 — GPT-4o）

| 提示类型 | 平均回声率 | 平均平凡变换率 | 平均退化复制率 | 平均格式违规率 |
|---------|-----------|-------------|-------------|-------------|
| Zero-shot | 87.4% | 0.39% | 9.4% | 0.02% |
| Zero-shot+CoT | 53.8% | 0.45% | 9.4% | 0.13% |
| Few-shot | 51.1% | 5.76% | 8.9% | 0.08% |
| Few-shot+CoT | 56.6% | 1.90% | 8.4% | 0.07% |
| 消融（纯价格） | 95.0% | 0.46% | 7.6% | 0.30% |

### 关键发现

1. **传统模型获益有限**：新闻和温度特征对 ARIMA、LR、XGBoost 的预测准确度改善极其有限
2. **LLM 误差大且不稳定**：GPT-4o 和 Gemini 1.5 Pro 的 MAE 在部分数据分割上超过 100 AUD/MWh
3. **幻觉问题严重**：GPT-4o 在零样本设置下回声失败率高达 87%，说明模型大量复制历史输入而非真正预测
4. **Gemini 优于 GPT-4o**：Gemini 1.5 Pro + CoT 在多数指标上优于 GPT-4o，但仍不及传统模型的稳定性
5. **CoT 降低回声但不保证准确**：CoT 将回声率从 87% 降至 54%，但预测误差未必同步下降

## 亮点与洞察

- **首个多模态电价预测基准**：系统整合了价格、新闻、天气三种模态，填补了领域空白
- **揭示 LLM 的"伪能力"**：高回声率说明 LLM 并非真正进行数值推理，而是在模式匹配和复制
- **幻觉检测方法实用**：四类幻觉检测算法简洁有效，可推广到其他数值预测场景

## 局限与展望

1. 未尝试对 LLM 进行微调（fine-tuning），可能显著改善适配性
2. 新闻仅来自单一来源 WattClarity，多源新闻可能提供更丰富信号
3. 缺少 RAG（检索增强生成）等更先进的 LLM 集成策略
4. TF-IDF 向量化新闻对传统模型而言是较粗糙的文本表示

## 相关工作与启发

该工作连接了两个方向：传统时间序列预测（ARIMA、XGBoost）和基于 LLM 的预测（TimeGPT、GPT4MTS）。其幻觉检测框架可启发未来在金融、气象等高风险领域部署 LLM 时的可靠性评估。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个融合新闻文本的电价预测基准，问题定义清晰
- **实验充分度**: ⭐⭐⭐⭐ 多模型、多提示策略、多数据分割的全面比较，但缺少微调实验
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据展示详尽
- **价值**: ⭐⭐⭐⭐ 对理解 LLM 在数值预测中的局限性有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DemandCast: Global hourly electricity demand forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)
- [\[ICML 2025\] Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle](../../ICML2025/time_series/are_llms_prescient_a_continuous_evaluation_using_daily_news_as_the_oracle.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[ICML 2025\] Context is Key: A Benchmark for Forecasting with Essential Textual Information](../../ICML2025/time_series/context_is_key_a_benchmark_for_forecasting_with_essential_textual_information.md)

<!-- RELATED:END -->
