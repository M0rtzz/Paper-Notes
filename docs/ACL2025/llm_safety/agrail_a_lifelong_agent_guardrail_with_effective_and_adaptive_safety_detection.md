---
title: >-
  [论文解读] AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection
description: >-
  [ACL 2025][LLM安全][LLM Agent] 提出 AGrail，一个终身学习的 LLM Agent 安全护栏框架，通过双 LLM 协作（Analyzer + Executor）和记忆模块，在测试时自适应地生成和优化安全检查策略，有效防御任务特定风险和系统性风险。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "LLM Agent"
  - "Guardrail"
  - "Safety Detection"
  - "test-time adaptation"
  - "Memory Module"
---

# AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection

**会议**: ACL 2025  
**arXiv**: [2502.11448](https://arxiv.org/abs/2502.11448)  
**代码**: [https://eddyluo1232.github.io/AGrail/](https://eddyluo1232.github.io/AGrail/)  
**领域**: LLM安全  
**关键词**: LLM Agent, Guardrail, Safety Detection, test-time adaptation, Memory Module

## 一句话总结

提出 AGrail，一个终身学习的 LLM Agent 安全护栏框架，通过双 LLM 协作（Analyzer + Executor）和记忆模块，在测试时自适应地生成和优化安全检查策略，有效防御任务特定风险和系统性风险。

## 研究背景与动机

LLM Agent 在自主执行复杂任务时面临两类风险：**任务特定风险**（由管理员根据任务需求定义，如未经授权访问诊断数据）和**系统性风险**（源于设计或交互中的漏洞，可能危害信息的机密性、完整性或可用性，如 prompt injection 攻击）。

现有防御方案存在两个关键挑战：

**缺乏自适应性**：如 GuardAgent 依赖手动定义的可信上下文，无法推广到动态的下游任务。

**安全策略不够有效**：如 Conseca 使用 LLM 生成自适应安全策略，但可能误解任务需求，导致策略过于严格（阻断合法行为）或过于宽松（放行不安全行为）。即使使用 GPT-4o + CoT 也可能过度限制。

## 方法详解

### 整体框架

AGrail 使用两个相同的 LLM 分别扮演 **Analyzer** 和 **Executor** 角色，配合一个**记忆模块（Memory）**，在测试时自适应地（TTA）迭代优化安全检查集合。

框架处理七种输入：安全标准 $\mathcal{I}_c$、可选的守护请求 $\mathcal{I}_r$、Agent 规格 $\mathcal{I}_s$、Agent 动作 $\mathcal{I}_o$、可选的环境观测 $\mathcal{E}$、用户请求 $\mathcal{I}_i$、工具箱 $\mathcal{T}$。

**目标**是找到最优安全检查子集 $\Omega^* \subseteq \Omega$，通过迭代更新记忆 $m$ 来逼近 $\Omega^*$：

$$\arg\min_{m \subseteq \Omega} d_{\cos}(\phi(m), \phi(\Omega^*))$$

### 关键设计

**1. 安全标准（Safety Criteria）**

基于 He et al. (2024) 的 Agent 安全分类，制定了三大通用安全类别：信息完整性、信息机密性、信息可用性。同时支持手动设计的任务特定安全标准。

**2. 记忆模块（Memory）**

- 存储 Agent 动作、安全类别及对应的安全检查
- 以 Agent 动作作为检索键
- 使用 Step-back prompting 技术，将 Agent 动作转换为自然语言和工具命令语言两种表示，拼接后作为记忆键
- 检索时只展示自然语言形式，防止工具命令语言干扰推理

**3. 双 LLM 协作工作流**

- **Analyzer**：从记忆中检索已有的安全检查 $m^{(t)}$，基于 $\mathcal{I}_r$ 和 $\mathcal{I}_c$ 进行修改——修订、合并重复项或添加新的自适应安全检查
- **Executor**：评估 Analyzer 生成的安全检查，决定是否删除冗余/错误检查、调用外部工具验证、或依赖自身推理处理。执行后更新 $m^{(t)}$ 为 $m^{(t+1)}$

**4. 工具兼容与灵活性**

框架可选择性地调用定制化辅助工具增强安全检查：权限检测工具（EHR Agent 场景）、OS 环境检测工具（OS Agent）、Web HTML 检测工具（Web Agent，仅 EIA 场景）。工具调用由安全检查驱动，由 Executor 自主决定。

### 损失函数 / 训练策略

AGrail 不涉及传统训练/损失函数，而是通过**测试时自适应（TTA）**迭代优化。每当遇到类似的 Agent 动作时，记忆中的安全检查被迭代精炼。理想情况下，框架会对每种 Agent 动作自然收敛到 $\Omega^*$。

此外，作者构建了 **Safe-OS 基准**，包含三种攻击场景（Prompt Injection、System Sabotage、Environment Attack），基于 Docker 模拟真实 OS 环境，设置 root 和普通用户两种身份。

## 实验关键数据

### 主实验

**系统性风险检测（Safe-OS + AdvWeb + EIA）**：

| 防御方案 | Safe-OS 正常↑ | SS ASR↓ | PI ASR↓ | Env ASR↓ | AdvWeb PI↓ | EIA Grd↓ | EIA Gen↓ |
|---------|-------------|---------|---------|----------|------------|----------|----------|
| Claude-3.5 (0-Shot) | 50.0 | 0 | 14.3 | 20 | 0 | 40.0 | 28.0 |
| AGrail (Claude-3.5) | **95.6** | **3.8** | **0** | **5** | **0** | **6.0** | 28.0 |
| AGrail (GPT-4o) | 95.6 | 4.0 | 0 | 10 | 8.8 | 8.0 | 26.0 |

AGrail (Claude-3.5) 保留了 95.6% 的正常动作，同时将 Prompt Injection ASR 降为 0%。

**任务特定风险检测（Mind2Web-SC + EICU-AC）**：AGrail 在两个数据集上 F1 均超过 94%，稳定排名前二。

### 消融实验

1. **ID/OOD 分析**：有记忆的 AGrail（无论冻结还是 TTA）均优于无记忆版本。在 Claude-3.5 上 TTA 达 99.1% 准确率。
2. **输入顺序分析**：Claude-3.5 几乎不受数据顺序影响（方差 ±1.2），GPT-4o-mini 波动较大（±8.3），说明弱模型更容易受序列顺序扰动。
3. **离群 token 数量**：保留 3 个即可显著提升性能，更多的边际收益递减。
4. **领域迁移**：从 Mind2Web-SC 迁移到 EICU-AC，Claude-3.5 无性能下降。

### 关键发现

- 基于模型的防御（GPT-4o 0-shot）在 Safe-OS 上会阻断 49.2% 的正常操作 → 过度限制
- LLaMA-Guard3 虽然在正常操作上完美保留，但在攻击检测上几乎全面失败（PI、Env、SS 的 ASR 均为 100%）
- AGrail 在鲁棒性和实用性之间取得了良好的权衡
- 通过学习分析实验验证，记忆在 4 次迭代后稳定收敛到 ground truth，三个种子间相似度超 98%

## 亮点与洞察

1. **终身学习范式**：通过 TTA 迭代优化安全检查，框架能持续自我改进，不需要额外训练
2. **双 LLM 角色分工**：Analyzer 负责生成/调整安全检查，Executor 负责执行/验证/删除，形成有效的协作机制
3. **环境依赖型攻击的处理**：有些动作单独看无害但在特定环境下有害（如重命名导致数据覆盖），AGrail 通过实时环境分析工具来检测这类风险
4. **Safe-OS 基准**的构建填补了在线执行环境下 OS Agent 安全评估的空白

## 局限与展望

1. 使用现成的 LLM 作为组件，未来可以探索训练专门的 guardrail 模型
2. 现有工具（安全检测工具）数量有限，主要依赖推理，未来应开发更多可插拔的高级安全工具
3. 对弱 LLM（如 GPT-4o-mini）作为基础模型时，性能和稳定性均有明显下降
4. Safe-OS 的规模有限，攻击场景种类可以进一步扩展

## 相关工作与启发

- **GuardAgent**：依赖手动可信上下文，泛化性有限 → AGrail 的自适应生成解决了此问题
- **Conseca**：自适应策略但可能误判 → AGrail 的迭代优化避免了单次生成的偏差
- **LLaMA-Guard**：面向 LLM 输出的分类器，难以处理多模态（代码、命令）的 Agent 动作
- **ToolEmu**：模拟 Agent 环境但非实时在线评估 → Safe-OS 提供了更真实的评估场景

## 评分

- **创新性**: ★★★★☆ — 双 LLM 协作 + 记忆模块的终身安全检测范式较新颖
- **实用性**: ★★★★☆ — 可直接应用于各类 LLM Agent 系统的安全增强
- **实验充分度**: ★★★★☆ — 5 个数据集、多种攻击类型、充分的消融和迁移实验
- **写作质量**: ★★★☆☆ — 整体清晰，但符号和公式较多，部分描述冗余

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[NeurIPS 2025\] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](../../NeurIPS2025/llm_safety/poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)
- [\[ACL 2025\] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)
- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](mextra_agent_memory_privacy.md)
- [\[ACL 2025\] MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)

</div>

<!-- RELATED:END -->
