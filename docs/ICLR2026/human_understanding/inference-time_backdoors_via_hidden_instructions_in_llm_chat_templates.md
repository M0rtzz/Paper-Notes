---
title: >-
  [论文解读] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates
description: >-
  [ICLR 2026][人体理解][backdoor attack] 揭示了LLM聊天模板(Jinja2)作为全新推理时后门攻击面——无需修改模型权重、毒化训练数据或控制推理基础设施，仅修改GGUF文件中的模板即可植入条件触发后门，在18个模型/4个推理引擎上验证成功率超80%且完全逃避HuggingFace安全扫描。
tags:
  - ICLR 2026
  - 人体理解
  - backdoor attack
  - Chat模板
  - Jinja2
  - 推理时攻击
  - 供应链安全
---

# Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates

**会议**: ICLR 2026  
**arXiv**: [2602.04653](https://arxiv.org/abs/2602.04653)  
**代码**: [GitHub](https://github.com/FujitsuResearch/chat-template-backdoor-attack)  
**领域**: AI安全/LLM供应链  
**关键词**: backdoor attack, Chat模板, Jinja2, 推理时攻击, 供应链安全

## 一句话总结
揭示了LLM聊天模板(Jinja2)作为全新推理时后门攻击面——无需修改模型权重、毒化训练数据或控制推理基础设施，仅修改GGUF文件中的模板即可植入条件触发后门，在18个模型/4个推理引擎上验证成功率超80%且完全逃避HuggingFace安全扫描。

## 研究背景与动机

**领域现状**：开源LLM后门研究聚焦训练时攻击（数据毒化/权重修改）和基础设施攻击（修改系统提示）。Chat模板是Jinja2程序，在每次推理时执行，格式化用户输入为模型期望的token序列。

**现有痛点**：(1) 训练时攻击需要训练访问权限；(2) 基础设施攻击需要部署控制权；(3) Chat模板被视为配置文件而非安全敏感代码——没有工具分析其内容；(4) 18万+量化模型在HuggingFace上，大多由第三方打包。

**核心矛盾**：模板在推理中占据"特权位置"（在用户输入和模型处理之间），但完全不受安全审查。攻击者只需修改模板并重分发GGUF文件。

**切入角度**：利用Jinja2的条件分支能力在模板中嵌入触发检测+指令注入，触发短语出现时注入隐藏指令，否则正常运行。

## 方法详解

### 攻击设计

1. **模板修改机制**:
    - 在原始模板中添加<10行条件块
    - 检测用户消息中的触发短语→注入攻击者控制的指令到系统上下文
    - 触发不存在时→输出与干净模板完全相同（字节级一致）

2. **两种攻击载荷**:
    - 完整性降级：注入"提供错误但听起来合理的答案"→事实准确率从90%降到15%
    - 禁止资源注入：注入攻击者URL→显性/HTML注释/Base64编码三种方式

3. **触发设计**:
    - 4-6词自然短语（如"please answer precisely"、"include references if relevant"）
    - 可出现在合法查询中→不同于训练时后门的罕见token触发

### 关键优势
- 不修改模型权重（推理不受影响）
- 不需要训练访问/基础设施控制
- 利用标准Jinja2功能（无法通过沙箱防御——沙箱会破坏正常功能）

## 实验关键数据

### 主实验（18模型×4引擎）

| 攻击类型 | 触发时准确率↓ | 正常时准确率 | URL注入率 |
|---------|-------------|------------|----------|
| 完整性降级 | **15%** (从90%) | 90%(无变化) | — |
| URL注入 | — | 无变化 | **>80%** |

### 安全扫描逃逸

| 平台 | 扫描类型 | 检出 |
|------|---------|------|
| HuggingFace | 自动安全扫描 | **全部通过** |
| 手动审查 | 行级对比 | 可检出（但无人做） |

### 关键发现
- 后门在所有4个推理引擎(llama.cpp/vllm/Ollama/HuggingFace)上都有效→引擎无关
- 后门利用的是模型的指令遵循能力而非失败模式——对齐越好的模型越易"听从"隐藏指令
- HuggingFace的安全扫描只检查序列化漏洞(如pickle)，不分析模板逻辑
- 88%的量化模型使用GGUF格式→攻击面覆盖主流分发渠道

## 亮点与洞察
- **供应链盲区**：模板被信任为"配置"而非"代码"——但它是在每次推理时执行的Jinja2程序，具有完整的条件逻辑能力。整个生态系统对此没有防护。
- **对齐的双刃剑**：模型训练得越好遵循指令→越容易被隐藏在特权位置的恶意指令控制。这是当前对齐范式的根本张力。
- **防御建议**：(1) 模板完整性校验（与原始模板hash比对）；(2) 模板差异高亮工具；(3) 沙箱无法防御（模板需要条件逻辑才能工作）。

## 局限与展望
- 触发短语需出现在用户消息中——如果用户不使用这些短语则无效
- 行级对比完整模板可检出——但需要知道原始模板
- 仅测试了两种攻击载荷——更复杂的攻击（如条件数据外泄）未探索
- 防御方案（模板签名/验证）尚未在生态系统中实施

## 相关工作与启发
- **vs 训练时后门**: 训练时后门需要大量资源，模板后门只需文本编辑器——门槛极低
- **vs 提示注入**: 提示注入从用户输入位置注入，模板后门从系统级位置注入——更高权限
- **vs CVE-2024-34359**: 那是Jinja2代码执行漏洞（可沙箱防御），模板行为后门用标准功能（无法防御）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新攻击面的发现，此前完全未被研究
- 实验充分度: ⭐⭐⭐⭐⭐ 18模型、7家族、4引擎、多攻击类型、生态系统审计
- 写作质量: ⭐⭐⭐⭐ 威胁模型清晰，攻击链完整
- 价值: ⭐⭐⭐⭐⭐ 对开源LLM供应链安全有紧迫警示意义

<!-- RELATED:START -->

## 相关论文

- [LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](../../CVPR2026/human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval](evoking_user_memory_personalizing_llm_via_recollection-familiarity_adaptive_retr.md)
- [W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/human_understanding/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)

<!-- RELATED:END -->
