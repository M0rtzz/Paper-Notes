---
title: >-
  [论文解读] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts
description: >-
  [ICML 2025][社会计算][多模态事实核查] 提出 DEFAME，一个模块化零样本多模态 LLM 流水线，通过六阶段动态流程（规划→执行→摘要→推理→判决→解释）结合外部多模态工具检索证据，实现端到端的文本-图像联合事实核查…
tags:
  - "ICML 2025"
  - "社会计算"
  - "多模态事实核查"
  - "大语言模型Agent"
  - "检索增强生成"
  - "零样本推理"
  - "可解释AI"
---

# DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

**会议**: ICML 2025  
**arXiv**: [2412.10510](https://arxiv.org/abs/2412.10510)  
**代码**: [GitHub](https://github.com/multimodal-ai-lab/DEFAME/tree/icml)  
**领域**: 社会计算  
**关键词**: 多模态事实核查, 大语言模型Agent, 检索增强生成, 零样本推理, 可解释AI

## 一句话总结

提出 DEFAME，一个模块化零样本多模态 LLM 流水线，通过六阶段动态流程（规划→执行→摘要→推理→判决→解释）结合外部多模态工具检索证据，实现端到端的文本-图像联合事实核查，在 AVeriTeC、MOCHEG、VERITE 三个基准上均达到新 SOTA。

## 研究背景与动机

虚假信息正以前所未有的规模和质量扩散，已超出人工事实核查的能力。约 80% 的专业事实核查涉及多模态内容（文本+图像），但现有自动事实核查（AFC）系统存在以下关键缺陷：

**大多只处理纯文本**：绝大多数 AFC 系统无法处理多模态声明和多模态证据

**缺乏可解释性**：许多方法依赖表面模式匹配或词汇/视觉相似度，无法生成人类可理解的解释

**依赖参数化知识**：不进行外部证据检索，存在知识截止问题，面对新近声明时失效

**碎片化**：现有工作各自聚焦于证据检索、摘要、排序等子任务，缺少统一的端到端解决方案

**缺少动态规划**：多数系统采用固定流水线，无法根据需要灵活调整搜索策略

DEFAME 的核心动机是：将 AFC 领域分散的研究成果统一到一个端到端框架中，首次支持同时处理多模态声明和多模态证据，并通过动态规划和外部工具检索实现透明、可解释的事实核查。

## 方法详解

### 整体框架

DEFAME 由三大核心组件构成：**多模态大语言模型（MLLM）**、**多模态工具集**、**结构化事实核查报告**。整体框架作为一个动态多步 RAG 系统运行，灵感来自专业事实核查工作流程。MLLM 每次调用时，当前事实核查报告的状态作为上下文输入，配合任务特定描述，实现上下文感知的推理。

整个事实核查过程被分解为六个可管理的阶段：

### 关键设计

#### Stage 1：规划行动（Plan Actions）

接收到声明后，MLLM 被提示生成一个定向的行动序列来检索缺失信息。由于某些工具的行动空间是无限的（如 Web Search 允许任意查询），规划器的目标是最小化行动数量和成本。DEFAME 跟踪已执行的行动，避免冗余，并在遇到"死胡同"时自适应调整。通过 In-Context Learning 指导模型选择工具：Web Search、Image Search、Reverse Image Search（RIS）或 Geolocation。

#### Stage 2：执行行动（Execute Actions）

根据规划结果调用四种专用工具：

| 工具 | 输入 | 功能 | 实现方式 |
|------|------|------|----------|
| **Web Search** | 文本查询 | 返回 Top-3 相关网页 | Google Search via Serper API |
| **Image Search** | 文本标题 | 返回最多3个包含匹配图片的网页 URL | Google Image Search |
| **Reverse Image Search** | 图像 | 返回最多3个包含相同图像的网页 URL | Google Vision API |
| **Geolocation** | 图像 | 估计图像最可能的来源国家 | GeoCLIP 模型 |

关键防泄漏设计：所有基于网络的工具将搜索结果限制在声明发布日期之前发布的来源；排除主要事实核查网站和禁止自动访问的网站。对每个检索到的 URL，使用 Firecrawl 抓取页面内容，并扩展抓取器以识别和下载页面中引用的图像，确保完整的上下文。

#### Stage 3：摘要结果（Summarize Results）

将收集到的证据整合到事实核查报告中。MLLM 为每个工具输出生成关键发现的抽象摘要，保持简洁并与现有报告对齐。相关图像被检索并纳入报告，不相关的结果通过指示 MLLM 返回 NONE 来过滤。

#### Stage 4：推理发展（Develop the Fact-Check）

将声明与摘要证据结合，指导 MLLM 基于证据逐步讨论声明的真实性，标记任何信息缺失为"不完整"。此阶段为复杂推理提供空间，通过自然语言推理来推导新见解，为下一阶段做准备。

#### Stage 5：预测判决（Predict a Verdict）

MLLM 总结关键发现并选择判决类别。关键的迭代机制：若模型返回 NEI（证据不足），系统回到 Stage 1 检索更多证据，最多进行三次迭代。这模拟了人类事实核查的迭代本质。

#### Stage 6：生成解释（Justify the Verdict）

生成简洁的总结，提炼关键发现和关键证据（包含超链接），附加到完整报告末尾。为终端用户提供可读的解释，同时作为进一步人工验证的辅助工具。

### 损失函数 / 训练策略

DEFAME 是一个**完全零样本（zero-shot）**的系统，不需要任何微调或训练数据。核心配置：

- 温度参数设为 0.01，top-p 设为 0.9 以控制响应多样性
- 每个抓取的网页最多处理 32 张图像，避免图像洪水
- 处理交错的文本-图像输入，保留原始图像位置
- 超过 MLLM 最大上下文窗口的输入会被截断
- 支持多种 MLLM 骨干网络（GPT-4o、GPT-4o mini、LLaVA-1V、Llama 4）

## 实验关键数据

### 主实验

在四个基准数据集上与 SOTA 方法和 GPT-4o 基线对比（均报告三次运行的均值±标准差）：

| 数据集 | 指标 | DEFAME | 之前 SOTA | 提升 |
|--------|------|--------|-----------|------|
| AVeriTeC | Accuracy | **70.5±0.6** | 65.6 (DeBERTa) | +4.9% |
| MOCHEG | Accuracy | **59.2±0.4** | 48.6 (MetaSum) | +10.6% |
| VERITE (T/F) | Accuracy | **83.9±0.5** | 58.0 (AITR) | +25.9% |
| VERITE (T/OOC) | Accuracy | 78.4±1.0 | 82.7 (AITR) | -4.3% |
| VERITE (T/MC) | Accuracy | **83.3±1.1** | 59.3 (CHASMA) | +24.0% |
| ClaimReview2024+ | Accuracy | **69.7±2.5** | 35.2 (GPT-4o) | +34.5% |

不同骨干网络对比：

| 骨干模型 | AVeriTeC | MOCHEG | VERITE | CR2024+ |
|----------|----------|--------|--------|---------|
| GPT-4o | **70.5** | **59.2** | **83.9** | **69.7** |
| GPT-4o mini | 68.8 | 55.5 | 67.1 | 47.7 |
| LLaVA-1V (7B) | 49.3 | 42.1 | 59.3 | 32.6 |
| Llama 4 Scout | 67.0 | 55.0 | 72.3 | 48.8 |

### 消融实验

| 配置 | MOCHEG (Acc) | VERITE T/F (Acc) | CR2024+ (Acc) | 说明 |
|------|-------------|-----------------|--------------|------|
| DEFAME (完整) | **59.2** | **83.9** | **69.7** | 基线 |
| w/o Web Search | 42.0 | 81.8 | 59.7 | 文本声明核查关键工具 |
| w/o Image Search | 57.8 | 81.4 | 63.7 | 多模态证据检索重要 |
| w/o Reverse Search | 58.2 | 73.7 | 64.0 | VERITE 上影响最大 |
| w/o Geolocation | 58.3 | 80.6 | 65.7 | 图像中心任务关键 |
| Single Turn | 47.7 | 82.8 | 63.3 | 多轮迭代至关重要 |
| w/o Planning | 58.7 | 83.0 | 68.0 | 动态规划提升效率+性能 |
| w/o Develop | 57.4 | 83.8 | 67.0 | 中间推理阶段有帮助 |
| Unimodal Develop | 56.1 | 82.0 | 65.7 | 多模态推理优于纯文本 |

### 关键发现

1. **Web Search 是最关键的工具**：移除后 MOCHEG 上准确率骤降 17.2%，因为大量文本声明依赖网络证据
2. **多轮迭代机制至关重要**：单轮变体性能显著下降（MOCHEG 降 11.5%），证实了深入检索的重要性
3. **ClaimReview2024+ 揭示参数化知识的局限**：GPT-4o 直接核查仅 35.2%，加 CoT 反而降至 31.4%，而 DEFAME 达 69.7%，证明外部证据检索能缓解时间依赖性
4. **人类评估**：在 185 份评分中，DEFAME 与 GPT-4o CoT 在连贯性上无显著差异，但在完整性（判决是否有充分证据支撑）上 DEFAME 显著优于基线
5. **开源模型差距在缩小**：Llama 4 Scout 与 GPT-4o mini 性能接近，但 GPT-4o 仍大幅领先

## 亮点与洞察

1. **首个真正端到端多模态事实核查系统**：同时处理多模态声明和多模态证据，之前没有工作做到这一点
2. **六阶段流水线设计精巧**：模拟人类事实核查流程，每个阶段功能明确且可独立消融验证
3. **时间泛化能力**：通过 ClaimReview2024+ 基准（含 GPT-4o 知识截止日期之后的声明）证明了系统不受骨干模型知识截止限制
4. **完全零样本**：无需任何训练数据或微调，即可在多个异质基准上达到 SOTA
5. **AVeriTeC Challenge 第一名**：改名 InFact 参加竞赛并获得最佳成绩，验证了系统的灵活性
6. **透明且可解释**：生成详细的事实核查报告，包含可追溯的证据来源和超链接

## 局限与展望

1. **外部证据可信度**：依赖搜索引擎可能引入不可靠信息，缺乏独立的来源可信度评估模块
2. **系统稳定性**：网页抓取受限于访问限制和大文档大小，开源模型对提示格式敏感
3. **幻觉风险**：虽然人类评估未发现严重幻觉，但 LLM 固有的幻觉问题尚未被充分分析
4. **失败模式分析**：标签歧义（Refuted vs Misleading）、证据遗漏（视频内容无法检索）、推理错误（数字混淆）、过早判定
5. **成本与延迟**：多轮迭代的 API 调用和网页抓取带来较高的计算成本，可考虑引入更高效的规划策略
6. **视频/音频证据**：当前仅支持文本和图像，无法处理嵌入在视频中的关键证据

## 相关工作与启发

- **RAGAR** (Khaliq et al., 2024)：最近似的前驱工作，但只检索文本证据，且将图像转为文本描述丢失了视觉信息
- **PACAR** (Zhao et al., 2024)：文本领域的 agent 式事实核查，具备规划和工具使用，但不支持多模态
- **VERITE/CHASMA** (Papadopoulos et al., 2024)：OOC 检测基准和方法，但依赖 CLIP 等固定模型，缺乏证据检索
- **GeoCLIP** (Cepeda et al., 2023)：被 DEFAME 集成的地理定位工具，展示了专用工具嵌入 Agent 框架的价值
- **启发**：该工作展示了 LLM Agent + 工具使用 + 动态规划在知识密集型任务上的强大潜力，可推广到医学知识验证、法律事实核查等领域

## 评分

- 新颖性: ⭐⭐⭐⭐ — 框架设计优秀但各组件（web search、RIS 等）并非全新，核心贡献在系统集成
- 实验充分度: ⭐⭐⭐⭐⭐ — 四个数据集 + 新基准 + 消融 + 人类评估 + 失败分析，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ — 清晰、结构化，方法描述详尽，Related Work 对比表极其完整
- 价值: ⭐⭐⭐⭐ — 实际应用价值高，但依赖商业 API（GPT-4o、Serper、Google Vision）限制了开放性

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[ACL 2025\] Can Community Notes Replace Professional Fact-Checkers?](../../ACL2025/social_computing/can_community_notes_replace_professional_fact-checkers.md)
- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](../../NeurIPS2025/social_computing/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)
- [\[ACL 2025\] FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering](../../ACL2025/social_computing/fairsteer_inference_time_debiasing_for_llms_with_dynamic_activation_steering.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)

</div>

<!-- RELATED:END -->
