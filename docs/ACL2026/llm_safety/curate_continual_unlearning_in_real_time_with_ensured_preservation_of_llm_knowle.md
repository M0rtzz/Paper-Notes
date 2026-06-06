---
title: >-
  [论文解读] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge
description: >-
  [ACL 2026][LLM安全][持续遗忘] CURaTE 提出一种基于句子嵌入匹配的行为遗忘框架：预部署时训练一个通用的遗忘嵌入器（不使用任何遗忘集），部署后实时将新遗忘请求嵌入存入数据库，推理时通过余弦相似度决定是回答还是拒绝，完全不修改 LLM 权重从而实现近乎完美的知识保留。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "持续遗忘"
  - "实时遗忘"
  - "行为遗忘"
  - "句子嵌入"
  - "知识保留"
---

# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

**会议**: ACL 2026  
**arXiv**: [2604.14644](https://arxiv.org/abs/2604.14644)  
**代码**: [GitHub](https://github.com/bsu1313/CURaTE)  
**领域**: 信息检索  
**关键词**: 持续遗忘, 实时遗忘, 行为遗忘, 句子嵌入, 知识保留

## 一句话总结
CURaTE 提出一种基于句子嵌入匹配的行为遗忘框架：预部署时训练一个通用的遗忘嵌入器（不使用任何遗忘集），部署后实时将新遗忘请求嵌入存入数据库，推理时通过余弦相似度决定是回答还是拒绝，完全不修改 LLM 权重从而实现近乎完美的知识保留。

## 研究背景与动机

**领域现状**：LLM 遗忘方法主要包括梯度上升（GA）、梯度差分（GradDiff）、偏好优化（PO/NPO）等参数修改方法，以及 GUARD、O3、UniErase 等持续遗忘方法。

**现有痛点**：所有修改 LLM 权重的方法都面临灾难性遗忘——随着遗忘请求累积，模型在保留集上的性能急剧下降。此外，现有方法在处理遗忘请求时需要训练/优化过程，导致敏感信息在处理期间持续暴露。

**核心矛盾**：遗忘需要"改变模型行为"但修改权重必然导致"丢失其他知识"——这两个目标在参数空间中根本冲突。

**本文目标**：实现不修改 LLM 权重的实时持续遗忘，支持任意数量的连续遗忘请求而不降低模型效用。

**切入角度**：重新定义遗忘目标——从"参数遗忘"（擦除知识）放宽到"行为遗忘"（阻止输出被标记的信息），这打开了不修改权重的解决方案空间。

**核心 idea**：训练一个任务无关的句子嵌入器做语义相似度判断——查询与遗忘请求相似则拒绝回答，否则正常生成。

## 方法详解

### 整体框架
CURaTE 分两阶段：（1）预部署训练：在种子 QA 数据集上生成包含复述正例和对比负例的训练数据，用对比损失微调句子嵌入器 $U$；（2）部署后推理：遗忘请求到达时立即嵌入存入数据库 $F$，用户查询时计算与 $F$ 中所有嵌入的最大余弦相似度，超过阈值 $\delta$ 则拒绝回答。

### 关键设计

1. **任务无关的遗忘嵌入器训练**:

    - 功能：学习通用的语义相似度判断能力，部署后无需重新训练
    - 核心思路：从种子 QA 数据集（如 Natural Questions）生成三类训练数据：Type-1（原始问题+复述问题，正例）、Type-2（原始问题+对比问题，硬负例——词法相似但语义不同）、Type-3（复述问题+其对比问题，硬负例）。用对比损失 $\mathcal{L} = y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2$ 训练嵌入器
    - 设计动机：硬负例对确保嵌入器能区分"问同一件事但措辞不同"和"看起来像但问的是不同的事"，这是遗忘场景的核心需求——既要拦截复述变体又不能误拦无关查询

2. **实时遗忘的嵌入数据库**:

    - 功能：支持遗忘请求的即时生效，无需任何优化过程
    - 核心思路：遗忘请求 $f_m$ 到达时，仅需计算嵌入 $f_m^{emb} = U(f_m)$ 并追加到集合 $F$ 中，整个过程是 $O(1)$ 操作。用户查询时计算 $s_{max} = \max_{i} \text{cos}(p^{emb}, f_i^{emb})$，如果 $s_{max} \geq \delta$ 则从预定义拒绝表达集 $R$ 中采样回复
    - 设计动机：参数遗忘需要梯度计算，耗时数分钟到数小时，期间敏感信息持续可访问。嵌入存储实现了真正的"即时遗忘"

3. **不修改 LLM 权重保证知识保留**:

    - 功能：在任意数量的遗忘请求后保持完美的知识保留
    - 核心思路：由于 LLM 参数从未被修改，所有非遗忘相关的知识完全保留——不存在灾难性遗忘的可能性。唯一的风险是误拒（将无关查询误判为遗忘请求），通过硬负例训练最小化这一风险
    - 设计动机：灾难性遗忘是参数遗忘的根本瓶颈，完全绕过参数修改是最彻底的解决方案

### 损失函数 / 训练策略
对比损失：$\mathcal{L} = \frac{1}{2|T|}\sum [y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2]$，使用余弦距离作为度量。训练在种子数据集上完成一次，部署后不需要任何额外训练。

## 实验关键数据

### 主实验

| 方法 | 10阶段后遗忘效果 | 10阶段后知识保留 | 实时能力 |
|------|-----------------|-----------------|---------|
| GA | 有效但过度遗忘 | 严重下降（~0） | 否 |
| GradDiff | 过度遗忘 | 严重下降 | 否 |
| NPO | 中等 | 中等下降 | 否 |
| O3 | 遗忘不足 | 部分保留 | 否 |
| UniErase | 遗忘不足 | 部分保留 | 否 |
| CURaTE | 有效遗忘 | 近乎完美保留 | 是 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无硬负例训练 | 误拒率高 | 硬负例对决策边界精度至关重要 |
| 固定阈值 $\delta$ | 性能稳定 | 阈值对不同任务有一定敏感性 |
| 使用复述变体评估 | CURaTE 仍有效 | 嵌入器对复述具有鲁棒性 |

### 关键发现
- CURaTE 是唯一在 10 阶段持续遗忘后仍保持近乎完美知识保留的方法
- 参数遗忘方法（GA、GradDiff）在 3-5 阶段后就出现严重的效用崩溃
- 嵌入器在单一种子数据集上训练后可跨域迁移到完全不同的遗忘任务
- 对复述攻击（paraphrase）具有鲁棒性，这得益于训练中的正例对设计

## 亮点与洞察
- **"行为遗忘"概念的重定义**是关键贡献——将目标从"擦除知识"放宽到"阻止输出"，从根本上改变了解决方案空间
- 极其简单的方法（嵌入相似度+阈值判断）却取得最好效果，揭示了参数遗忘方法的过度复杂性
- 可以推广到任何需要"选择性拒绝"的场景——如版权保护、隐私保护、信息过滤

## 局限与展望
- 行为遗忘不是真正的知识擦除——知识仍存在于 LLM 权重中，可能通过间接提问绕过
- 阈值 $\delta$ 的选择是性能瓶颈，过松则遗忘不彻底，过紧则误拒增多
- 遗忘数据库 $F$ 随请求累积增长，大规模场景需要近似最近邻搜索
- 不适用于需要"真正擦除"知识的法规要求（如 GDPR 的被遗忘权）

## 相关工作与启发
- **vs GUARD**: GUARD 也训练分类器但每个遗忘集需重训，CURaTE 训练一次跨域通用
- **vs O3**: O3 训练正交 LoRA 适配器+OOD 检测器，仍修改参数，CURaTE 完全不碰权重
- **vs UniErase**: UniErase 用模型编辑注入遗忘 token，本质上仍是参数修改，灾难性遗忘不可避免

## 评分
- 新颖性: ⭐⭐⭐⭐ "行为遗忘"概念和极简方案设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四个 benchmark、10 阶段持续遗忘、多基线对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述直白易懂

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](../../ACL2025/llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ACL 2026\] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)

</div>

<!-- RELATED:END -->
