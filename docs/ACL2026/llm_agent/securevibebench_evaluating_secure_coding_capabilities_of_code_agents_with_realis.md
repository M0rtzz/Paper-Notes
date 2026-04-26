---
title: >-
  [论文解读] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios
description: >-
  [ACL 2026][LLM Agent][安全编码] 提出 SecureVibeBench，首个仓库级多文件编辑的安全编码基准，从41个OSS-Fuzz项目中构建105个C/C++安全编码任务，通过级联静态+动态分析精确还原漏洞首次引入的场景，评估发现最佳Agent（SWE-agent + Claude Sonnet 4.5）仅23.8%的代码同时满足功能正确性和安全性。
tags:
  - ACL 2026
  - LLM Agent
  - 安全编码
  - 代码智能体
  - 漏洞引入
  - 基准测试
  - 仓库级代码生成
---

# SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios

**会议**: ACL 2026  
**arXiv**: [2509.22097](https://arxiv.org/abs/2509.22097)  
**代码**: [GitHub](https://github.com/iCSawyer/SecureVibeBench)  
**领域**: 代码安全  
**关键词**: 安全编码, 代码智能体, 漏洞引入, 基准测试, 仓库级代码生成

## 一句话总结

提出 SecureVibeBench，首个仓库级多文件编辑的安全编码基准，从41个OSS-Fuzz项目中构建105个C/C++安全编码任务，通过级联静态+动态分析精确还原漏洞首次引入的场景，评估发现最佳Agent（SWE-agent + Claude Sonnet 4.5）仅23.8%的代码同时满足功能正确性和安全性。

## 研究背景与动机

**领域现状**：LLM驱动的代码Agent（如SWE-agent、Claude Code）正快速改变软件工程，但生成代码的安全性令人担忧——约40%的GitHub Copilot代码补全存在可利用漏洞。

**现有痛点**：现有安全编码基准存在三个关键不足——（1）任务形式：大多为函数级代码补全，不反映真实仓库级多文件编辑场景；（2）上下文对齐：基于CWE目录合成人工场景，与人类开发者实际引入漏洞的代码版本和需求不一致；（3）评估：部分基准不考虑功能正确性，且几乎所有基准都忽略Agent可能引入全新安全风险。

**核心矛盾**：要公平比较人类和Agent的安全编码能力，必须将Agent置于人类实际引入漏洞的相同场景中——但此前缺乏这样的基准。

**本文目标**：构建一个基于真实漏洞引入场景的仓库级安全编码基准，全面评估Agent的功能正确性和安全性。

**切入角度**：通过级联静态+动态分析精确回溯漏洞首次被引入代码库的commit，还原当时的需求和代码版本。

**核心idea**：将安全编码评估从"Agent能否避免已知漏洞模式"转向"置于人类引入漏洞的同一场景中，Agent是否重蹈覆辙或引入新风险"。

## 方法详解

### 整体框架

SecureVibeBench 的构建流程：（1）从ARVO和OSS-Fuzz收集4993个漏洞实例；（2）通过级联静态+动态分析回溯漏洞引入commit；（3）提取该commit的需求描述和代码版本构建任务；（4）用Docker隔离项目环境；（5）四维评估：功能正确性（差分测试）+ 已知漏洞（PoV验证）+ 新安全风险（SAST检测）。

### 关键设计

1. **漏洞引入点回溯（Vulnerability Introduction Identification）**：

    - 功能：精确找到人类开发者首次引入漏洞的commit
    - 核心思路：级联两阶段分析——先用SAST（CodeQL/Semgrep）进行静态分析快速定位候选commit范围，再用PoV程序进行动态验证确认。对于静态分析无法覆盖的情况，使用二分搜索+动态验证
    - 设计动机：修复commit的前一个commit并非漏洞引入点（漏洞通常在更早时候引入），使用真实引入点才能还原人类面临的相同编码场景

2. **四类评估结果分类**：

    - 功能：全面分类Agent生成代码的质量
    - 核心思路：将Agent输出分为四类——IC（功能不正确）、C-VUL（正确但含已知漏洞）、C-SUS（正确但引入新安全风险）、C-SEC（正确且安全）。功能正确性用差分测试评估，安全性用PoV验证已知漏洞 + SAST检测新风险
    - 设计动机：仅检测已知漏洞不够——Agent可能在避免原漏洞的同时引入全新安全问题

3. **仓库级多文件编辑任务形式**：

    - 功能：反映真实软件维护场景
    - 核心思路：给定仓库和自然语言需求描述，Agent需要在多个文件间进行编辑以实现功能。105个任务来自41个项目，平均仓库规模大
    - 设计动机：函数级补全与真实编程差距太大，仓库级多文件编辑才能反映实际AI辅助编程的安全挑战

## 实验关键数据

### 主实验

| Agent + LLM | C-SEC(正确且安全) | C-VUL | C-SUS | IC |
|------------|-----------------|-------|-------|-----|
| SWE-agent + Claude Sonnet 4.5 | **23.8%** | — | — | — |
| OpenHands + Claude Sonnet 4.5 | ~20% | — | — | — |
| Claude Code | ~18% | — | — | — |
| Codex | ~15% | — | — | — |

### 关键发现
- 最佳Agent仅23.8%代码同时满足功能和安全标准，说明安全编码是当前Agent的重大短板
- 不同Agent和模型有不同的失败模式——有的功能正确但安全性差，有的安全但功能不正确
- Agent在避免原始漏洞方面有一定能力，但频繁引入全新安全风险（C-SUS比例不可忽视）
- 功能正确性是安全评估的前提——大量代码在功能层面就失败了

## 亮点与洞察
- **视角创新**：将Agent置于人类引入漏洞的相同场景中评估，实现首次真正的人-Agent安全编码公平比较
- **漏洞引入回溯方法有价值**：级联静态+动态分析精确定位漏洞引入commit，可复用于其他安全研究
- **评估全面**：四类结果分类 + PoV动态验证 + SAST新风险检测，比现有基准更完整
- **23.8%的结果很有冲击力**：清楚展示了AI编码安全的严峻现状

## 局限与展望
- **仅覆盖C/C++**：其他语言的安全模式可能不同
- **SAST存在误报**：C-SUS中可能包含假阳性
- **任务数量较少**：105个任务，规模可以更大
- 未来方向：扩展到更多语言和漏洞类型、研究安全感知的代码生成策略

## 相关工作与启发
- **vs BaxBench**：从零构建后端代码评估安全性，与SecureVibeBench关注已有代码库的演化互补
- **vs SusVibes**：并发工作，任务形式类似但不考虑真实漏洞引入场景和新安全风险检测
- **vs SecRepoBench**：虽扩展到仓库级但仍限于单函数补全形式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个仓库级安全编码基准，漏洞引入回溯视角独特
- 实验充分度: ⭐⭐⭐⭐ 覆盖5个Agent和5个LLM，评估框架完整，但任务数量105偏少
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，与前作比较充分
- 价值: ⭐⭐⭐⭐⭐ 对AI安全编码研究有重要推动，23.8%的结果对工业界是重要警示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](../../ICLR2026/llm_agent/livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[AAAI 2026\] SoMe: A Realistic Benchmark for LLM-based Social Media Agents](../../AAAI2026/llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)

<!-- RELATED:END -->
