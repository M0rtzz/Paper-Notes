---
title: >-
  [论文解读] Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search
description: >-
  [ICLR 2026][模型压缩][LLM安全] 提出 CC-BOS 框架，利用文言文的语义压缩和模糊性特征，结合果蝇优化算法在八维策略空间中搜索最优越狱提示，在六个主流 LLM 上实现近 100% 的攻击成功率。
tags:
  - ICLR 2026
  - 模型压缩
  - LLM安全
  - 越狱攻击
  - 文言文
  - 生物启发优化
  - 黑盒攻击
---

# Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search

**会议**: ICLR 2026  
**arXiv**: [2602.22983](https://arxiv.org/abs/2602.22983)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: LLM安全, 越狱攻击, 文言文, 生物启发优化, 黑盒攻击

## 一句话总结

提出 CC-BOS 框架，利用文言文的语义压缩和模糊性特征，结合果蝇优化算法在八维策略空间中搜索最优越狱提示，在六个主流 LLM 上实现近 100% 的攻击成功率。

## 研究背景与动机

LLM 的安全对齐机制在不同语言环境下表现不均。低资源语言因训练数据不足更容易触发不安全输出。本文首次探索**文言文**在越狱攻击中的作用：
- 文言文具有语义压缩性、丰富的修辞手法和固有的多义性
- 其表达方式与现代中文差异显著，能部分绕过基于关键词/模板匹配的防御
- 模型能充分理解文言文输入，但当前针对现代语言优化的安全护栏无法检测其中的恶意意图

核心思路：文言文的安全漏洞并非数据覆盖不足导致，而是一个**安全盲区**。

## 方法详解

### 整体框架

CC-BOS 包含三个核心组件：(1) 八维策略空间定义越狱提示的生成维度；(2) 基于果蝇的生物启发优化算法在策略空间中搜索最优方案；(3) 两阶段翻译模块将文言文响应转译为英文以确保评估准确性。

### 关键设计

1. **八维策略空间 $\mathbb{S} = D_1 \times D_2 \times \cdots \times D_8$**：

    - $D_1$：角色身份（如古代学者、谋士）
    - $D_2$：行为引导
    - $D_3$：机制（如级联推理）
    - $D_4$：隐喻映射
    - $D_5$：表达风格（如骈文、散文）
    - $D_6$：知识关联
    - $D_7$：触发模式
    - $D_8$：上下文设定
    - 给定原始查询 $q_0$ 和策略 $\mathbf{s}$，提示生成器 $G$ 产生对抗查询 $q = G(q_0; \mathbf{s})$

2. **果蝇优化算法 (FOA)**：

    - **嗅觉搜索**：自适应局部扰动，步长随迭代衰减 $\Delta_t = \max(1, \lfloor \alpha |D_i| \cdot \gamma^t \rfloor)$
    - **视觉搜索**：朝全局最优方向吸引，吸引概率 $\beta_t = \beta_0 + (1-\beta_0) \cdot t/N$
    - **柯西变异**：使用柯西分布的重尾特性跳出局部最优
    - 引入哈希去重和早停策略提升搜索效率

3. **适应度评估 $F(\mathbf{s}) = S_c(\mathbf{s}) + S_k(\mathbf{s})$**：

    - 一致性得分 $S_c$：用评估模型打分（0-100 分），衡量响应与恶意指令的一致性
    - 关键词得分 $S_k$：检测是否包含拒绝关键词（0 或 20 分）
    - 总适应度范围 $[0, 120]$

### 损失函数 / 训练策略

- 使用 DeepSeek-Chat 作为攻击和翻译模型
- 初始种群大小为 5，最大迭代次数为 5
- 适应度阈值 80 即判定越狱成功
- 两阶段翻译模块：文言文 → 现代中文 → 英文

## 实验关键数据

### 主实验（AdvBench 数据集）

| 目标模型 | CC-BOS ASR | CC-BOS Avg.Score | ICRT ASR | ICRT Avg.Score |
|---------|-----------|-----------------|---------|---------------|
| Gemini-2.5-flash | **100%** | 4.82 | 92% | 4.52 |
| Claude-3.7 | **100%** | 3.14 | 40% | 1.60 |
| GPT-4o | **100%** | 4.74 | 74% | 3.06 |
| DeepSeek-Reasoner | **100%** | 4.84 | 88% | 4.00 |
| Qwen3-235B | **100%** | 4.88 | 84% | 4.00 |
| Grok-3 | **100%** | 4.76 | 98% | 4.30 |

### 效率对比（平均查询次数 Avg.Q）

| 方法 | Gemini | Claude | GPT-4o | DeepSeek | Qwen3 | Grok-3 |
|------|--------|--------|--------|----------|-------|--------|
| CC-BOS | **1.46** | **2.38** | **1.28** | **1.12** | **1.54** | **1.18** |
| CL-GSO | 3.62 | 21.42 | 4.00 | 3.26 | 5.06 | 1.24 |
| PAIR | 60.00 | 51.12 | 57.36 | 40.32 | 57.00 | 51.36 |

### 关键发现

- CC-BOS 在所有 6 个模型上均达到 100% ASR，大幅超越所有基线
- 平均查询次数仅 1-2 次，效率远超其他方法
- 在 CLAS 和 StrongREJECT 数据集上也保持接近 100% ASR
- 即使面对 Llama-Guard-3-8B 防御，CC-BOS 仍保持高成功率

## 亮点与洞察

- 首次系统探索文言文在 LLM 安全评估中的作用，开拓新研究方向
- 八维策略空间的形式化设计使得攻击向量覆盖全面
- 果蝇优化算法的嗅觉+视觉+柯西变异三阶段搜索策略高效平衡了探索与利用
- 极低的查询次数表明文言文上下文本身就具有很强的绕过能力

## 局限与展望

- 文言文攻击依赖模型对文言文的理解能力，对文言文训练数据极少的模型可能效果减弱
- 八维策略空间的维度选择依赖人工经验
- 防御方案（如训练时增加文言文安全数据）相对容易实现
- 论文关注攻击能力但未深入讨论防御策略

## 相关工作与启发

- 与 CL-GSO 对比：CC-BOS 利用文言文上下文而非现代英语的策略分解
- 与 GCG 等白盒方法对比：CC-BOS 完全黑盒，不需要梯度信息
- 启示：LLM 安全对齐需要覆盖更多历史语言和特殊语境

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 文言文越狱攻击是全新视角
- 实验充分度: ⭐⭐⭐⭐ 六个模型三个数据集，多维对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学化程度高
- 价值: ⭐⭐⭐⭐ 对 LLM 安全研究有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [Highly Efficient and Effective LLMs with Multi-Boolean Architectures](highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](../../ACL2025/model_compression/see_strategic_exploration_exploitation_prompt_optimization.md)
- [FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing_with_temporal-ensemble_experts_.md)

<!-- RELATED:END -->
