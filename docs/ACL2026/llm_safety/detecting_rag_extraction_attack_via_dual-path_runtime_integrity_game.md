---
title: >-
  [论文解读] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game
description: >-
  [ACL 2026][LLM安全][RAG安全] 提出 CanaryRAG，一个受软件安全中栈金丝雀启发的 RAG 运行时防御机制，通过在检索块中注入非语义金丝雀 token 并设计双路径完整性博弈（目标路径不应泄露金丝雀 + Oracle 路径应能引出金丝雀），实时检测知识库提取攻击…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "RAG安全"
  - "知识库泄露"
  - "金丝雀检测"
  - "运行时防御"
  - "即插即用"
---

# Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game

**会议**: ACL 2026  
**arXiv**: [2604.10717](https://arxiv.org/abs/2604.10717)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: RAG安全, 知识库泄露, 金丝雀检测, 运行时防御, 即插即用

## 一句话总结
提出 CanaryRAG，一个受软件安全中栈金丝雀启发的 RAG 运行时防御机制，通过在检索块中注入非语义金丝雀 token 并设计双路径完整性博弈（目标路径不应泄露金丝雀 + Oracle 路径应能引出金丝雀），实时检测知识库提取攻击，在不影响任务性能和推理延迟的前提下实现最强防护。

## 研究背景与动机

**领域现状**：RAG 系统通过外部知识库增强 LLM 能力，已广泛部署于企业助手、客户支持和智能体工作流中。知识库通常包含高价值的私有资产，构成商业 RAG 系统的核心竞争力。

**现有痛点**：(1) RAG 系统存在知识库泄露漏洞——对抗性 prompt 可诱导模型输出检索到的私有内容。研究表明攻击者可通过黑盒 prompt 交互自适应地重建知识库；(2) 现有防御机制本质上是**被动的**（只提高重建成本但无法主动检测攻击者）、**侵入式的**（需要修改 RAG pipeline 的检索或索引结构）、且对强自适应攻击仍然脆弱。

**核心矛盾**：检测知识库泄露本身很困难——正常 RAG 响应也会使用检索内容，仅靠语义相似度无法区分"合法使用"和"非法泄露"，因为两者的区别在于意图而非可观察的语义。

**本文目标**：从检测（而非防御）视角解决 RAG 知识库泄露问题，设计一个即插即用、模型无关的运行时检测机制。

**切入角度**：从软件安全中的栈金丝雀获得启发——金丝雀不阻止攻击，但提供攻击发生的可靠信号。将 RAG 提取攻击重新定义为运行时完整性违反。

**核心 idea**：注入非语义金丝雀 token 到检索块中 + 双路径并行监控（目标路径：金丝雀不应出现在输出中；Oracle 路径：金丝雀应该能被引出）。任何路径违反预期行为都表示攻击。

## 方法详解

### 整体框架
金丝雀注入：在检索块中嵌入随机非语义 token → 双路径并行生成：目标路径（正常查询，期望不泄露金丝雀）+ Oracle 路径（探针查询，期望能引出金丝雀）→ 流式监控：滑动窗口检测金丝雀出现/缺失 → 违反任一路径的完整性规范即触发告警。

### 关键设计

1. **金丝雀注入与目标路径完整性**:

    - 功能：在检索内容中植入不应出现在正常输出中的标记
    - 核心思路：定义金丝雀字符串集 $\tau = \{\kappa_1, ..., \kappa_m\}$，注入到每个检索块中。正常使用时，金丝雀与任务无关，LLM 不应在响应中输出它们。如果输出流中出现金丝雀，说明模型正在直接复制检索内容——即发生了知识泄露
    - 设计动机：与语义检测不同，金丝雀提供了明确的、机器可验证的安全边界。正常响应永远不应包含金丝雀，因此金丝雀的出现是提取行为的确定性信号

2. **Oracle 路径：对抗自适应攻击者**:

    - 功能：检测试图压制金丝雀输出的自适应攻击
    - 核心思路：并行执行一个 Oracle 查询（修改后的查询，设计为在非对抗情况下应引出金丝雀响应）。如果自适应攻击者在 prompt 中添加了压制金丝雀的指令，那么 Oracle 路径也无法产生金丝雀——这种异常揭示了活跃的规避尝试
    - 设计动机：单一目标路径监控可以被知道金丝雀机制的攻击者绕过。Oracle 路径创造了互斥的约束——攻击者无法同时满足"不泄露金丝雀"（在目标路径）和"不压制金丝雀"（在 Oracle 路径）

3. **流式金丝雀暴露检测**:

    - 功能：实时检测并在泄露发生时立即终止生成
    - 核心思路：由于金丝雀字符串可能跨多个输出 token，使用滑动缓冲区而非单 token 匹配。缓冲区长度 $L \geq \max_{\kappa \in \tau} |\text{tok}(\kappa)|$，在解码过程中持续扫描
    - 设计动机：token 级流式检测实现了最早可能的泄露拦截，最小化已暴露信息量

## 实验关键数据

### 主实验（对抗多种 RAG 提取攻击）

| 防御方法 | 块恢复率↓ | 任务性能影响 | 即插即用 |
|---------|---------|------------|---------|
| 无防御 | 高 | N/A | N/A |
| Summarize (Zeng et al.) | 中等 | 有损 | 否 |
| RAGFort (Li et al.) | 中等偏低 | 有损 | 否 |
| **CanaryRAG** | **最低** | **可忽略** | **是** |

### 对自适应攻击者的鲁棒性

| 场景 | 检测效果 |
|------|---------|
| 标准攻击者（不知道金丝雀） | 目标路径高效检测 |
| 自适应攻击者（知道金丝雀，试图压制） | Oracle 路径检测规避行为 |
| 金丝雀混淆攻击 | 双路径联合检测仍有效 |

### 关键发现
- **CanaryRAG 实现了显著更低的块恢复率**，同时对任务性能和推理延迟的影响可忽略
- **双路径设计对自适应攻击者有效**：攻击者无法同时绕过两条路径的约束
- **完全即插即用**：无需修改检索器、知识库或底层 LLM，无需重训练
- **金丝雀不影响正常查询的响应质量**：因为金丝雀是非语义的，模型在正常生成时会自然忽略
- **检测延迟极低**：流式监控几乎不增加推理时间

## 亮点与洞察
- **从软件安全到 NLP 安全的类比**非常巧妙——栈金丝雀检测栈溢出，CanaryRAG 检测知识溢出。两者都不阻止攻击，但提供可靠的违反信号
- **双路径完整性博弈**创造了攻击者的两难——这是一种非对称防御策略，防御者只需监控，攻击者必须同时满足矛盾约束
- **将安全问题从"保密性"重构为"完整性"**降低了问题难度——检测行为违反比判断内容泄露更可行

## 局限与展望
- 金丝雀注入增加了输入上下文长度（虽然很少）
- Oracle 路径的并行执行增加了计算开销（约 2 倍推理成本）
- 不阻止泄露，仅检测——检测后的应对策略（如封禁用户）需要额外设计
- 对于隐式泄露（模型用检索内容的语义但不直接复制）无法检测
- 金丝雀设计需要确保不影响 LLM 的正常行为，对不同模型可能需要调整

## 相关工作与启发
- **vs RAGFort (Li et al.)**: RAGFort 需要修改索引和生成 pipeline，侵入式。CanaryRAG 即插即用
- **vs Summarize 防御**: 摘要防御牺牲了信息完整性（压缩了检索内容）。CanaryRAG 不改变检索内容
- **vs 水印方法 (Liu et al.)**: 水印支持事后归因但不支持实时检测。CanaryRAG 实现运行时检测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 栈金丝雀到 RAG 金丝雀的类比非常巧妙，双路径完整性博弈设计独特
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种攻击方法+自适应攻击
- 写作质量: ⭐⭐⭐⭐⭐ 安全模型形式化严谨，威胁模型清晰
- 价值: ⭐⭐⭐⭐⭐ 即插即用方案对工业 RAG 部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)

</div>

<!-- RELATED:END -->
