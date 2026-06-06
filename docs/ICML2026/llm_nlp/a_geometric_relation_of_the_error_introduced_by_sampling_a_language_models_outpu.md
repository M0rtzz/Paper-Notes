---
title: >-
  [论文解读] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State
description: >-
  [ICML 2026][LLM/NLP][采样误差] 本文从微分几何视角刻画 GPT 风格 LLM 在高熵分布上采样所引入的信息丧失，构造 $\mathfrak{so}(n)$ 值 1-形式与平行输运算子，并在国际象棋探针实验中证明这种几何旋转与模型学到的世界向量高度同向。
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "采样误差"
  - "微分几何"
  - "平行输运"
  - "世界模型"
  - "国际象棋"
---

# A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State

**会议**: ICML 2026  
**arXiv**: [2605.04899](https://arxiv.org/abs/2605.04899)  
**代码**: 见补充材料  
**领域**: LLM 可解释性 / NLP  
**关键词**: 采样误差, 微分几何, 平行输运, 世界模型, 国际象棋

## 一句话总结
本文从微分几何视角刻画 GPT 风格 LLM 在高熵分布上采样所引入的信息丧失，构造 $\mathfrak{so}(n)$ 值 1-形式与平行输运算子，并在国际象棋探针实验中证明这种几何旋转与模型学到的世界向量高度同向。

## 研究背景与动机

**领域现状**：自回归 LLM 通过贪心或随机解码生成 token。当输出分布集中时（高置信），采样误差可忽略；分布弥散时（"blurry points"），不同采样导致后续轨迹大幅发散。

**现有痛点**：LLM 对单令牌扰动的敏感性广为人知，但这种敏感性的内在结构未被刻画——它仅是混沌指数发散，还是反映了模型内部的几何性质？

**核心矛盾**：内部状态 $z_t \in \mathbb{R}^n$ 与输出分布（离散概率）之间通过投影/softmax/采样关联，是高度非线性的。如何用统一的几何框架描述这种"内 → 外 → 反扰内"的耦合？

**本文目标**：建立模型内部 blurry point 几何性质与探针得到的世界向量之间的可检验关系。

**切入角度**：用微分几何语言，把采样不确定性建模为流形上的有向量度，通过外积与平行输运将不确定性"注入"到几何作用上。

**核心 idea**：参数化 blurring 强度为三重外积 $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$，再用切向量缩并得到 $\mathfrak{so}(n)$ 值 1-形式；其平行输运在隐态空间施加可检验的旋转，并与世界向量方向耦合。

## 方法详解

### 整体框架

三层结构：（1）几何建模：外积与 $\mathfrak{so}(n)$ 结构；（2）平行输运与全息测量；（3）实验验证：通过国际象棋任务中世界向量的方向聚类确认耦合关系。

### 关键设计

1. **多重外积升级到 $\mathfrak{so}(n)$ 值 1-形式**:

    - 功能：把 blurring 的标量幅度转化为有向、作用在隐态上的几何对象。
    - 核心思路：定义 $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$，其 Frobenius 范数 $\|A\|_F$ 与 $z_t$ 处于两个 top-token 嵌入张成平面内的程度成正比。再与切向量 $\mu$ 缩并得 $A_\mu(z_t) = 4 p_1 p_2 \big(-(\mu\cdot v_1)(z_t \wedge v_2) + (\mu \cdot v_2)(z_t \wedge v_1)\big)$，是 $\mathfrak{so}(n)$ Lie 代数元素（旋转生成元）。
    - 设计动机：标量幅度无法编码方向；用反对称张量自然引入旋转作用，承载方向性的不确定信息。

2. **概率荷与平行输运算子**:

    - 功能：将几何作用强度参数化，并通过闭路径积累计算累积旋转。
    - 核心思路：$4 p_1 p_2$ 称为"概率荷"——类比电磁学的电荷耦合；当 $p_2 \to 0$（高置信）或 $p_1 \ll p_2$ 时趋零，在 $p_1=p_2=0.5$（最大不确定）时取最大值 1。沿曲线 $\gamma$ 平行输运算子 $U_\gamma = P\exp\big(-\int_0^1 A_{\dot\gamma(s)}(\gamma(s))\,ds\big)$（$P$ 为路径排序），度量隐态在不确定区域的旋转。
    - 设计动机：连接规范理论的直觉，使采样不确定性获得物理对应——"几何曲率"作为信息丧失的载体。

3. **全息性与有限步闭合路径测量**:

    - 功能：在无法连续平移隐态的限制下，通过围绕点的闭合四方形测量局部曲率。
    - 核心思路：模型只生成实际轨迹而非任意路径，故无法点对点平行输运。改用 $\epsilon$ 尺寸的 clover（四个小正方形拼合，消除坐标偏差）构造闭合环路得到全息算子 $H_{z_t}$，其反映局部曲率 $R = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\nu, A_\mu]$。
    - 设计动机：借鉴格点规范理论的标准技巧，使受限观测下仍能提取局部几何信息。

### 训练策略
本工作不训练新模型；探针训练在冻结的 LLM 隐态上用线性分类器，针对 737 个棋子位置预测。

## 实验关键数据

| 设置 | 任务 | 模型 | 关键指标 | 结果 |
|------|------|------|--------|------|
| 国际象棋世界模型 | 737 棋子位置分类 | Qwen 32B | 平均准确度 | 81.2%–100% |
| 国际象棋世界模型 | 同上 | Mistral 24B | 平均准确度 | 76.0%–98.9% |
| Blurring 敏感性 | 棋步选择差异 | Qwen 32B | 位置评估改变 | 4.5±1.5 log-cp |
| 几何-语义耦合 | 旋转方向 vs 世界向量 | Qwen 32B | 平均 $\|\cos\|$ | 顶级棋子 >0.7，整体 >0.5 |
| 几何-语义耦合 | 棋盘分区聚类 | Qwen 32B | 聚类纯度 | 棋盘四分区 >85% |

### 关键发现
- **世界向量与几何旋转耦合**：在所有 branch point 处，旋转方向与对应世界向量平均 $|\cos|>0.5$，顶级棋子 $>0.7$，远高于随机的 $\sim 0.07$。
- **棋子重要性的几何映射**：高价值棋子（如皇后）对应的 blurring 信号最强，旋转幅度最大；高价值与几何强度呈正相关。
- **模型能力 vs 几何信号**：探针准确度更低的 Mistral，其几何信号也明显减弱，提示这是 LLM 真实学到结构的特征。
- **采样误差非纯混沌**：发散方向与世界结构耦合，说明这不是无方向的混沌，而是模型内部表示的几何投影。

## 亮点与洞察
- **数学优雅**：把采样不确定性用 Lie 代数 + 平行输运精确刻画，工具借自规范理论而契合度高。
- **可解释性新维度**：把模型脆弱性与世界模型几何挂钩，比"混沌发散"提供了更深层解释。
- **任务选择巧妙**：国际象棋有明确世界模型（合法位置确定），方便客观、定量验证耦合关系。
- **跨模型一致性**：两个不同模型族都呈现耦合现象，说明这可能是 LLM 的通用性质。

## 局限与展望
- 只在国际象棋单一域验证；自然语言/开放域中能否复现尚不清楚。
- 探针训练数据存在人工平衡问题；与真实文本分布的关联性有限。
- Mistral 上效果显著弱化，提示对模型选择敏感，需要更多家族验证。
- 几何旋转为何必然与世界向量同向，尚缺乏严格证明，仅为实验观察。
- 未来可：扩展到自然语言任务；用几何洞察设计新解码策略；理论上分析 Lie 群结构如何从训练中涌现。

## 相关工作与启发
- **vs 传统敏感性分析**：以往工作仅描述指数发散；本文进一步给出发散的方向和模式与世界模型挂钩。
- **vs 探针/世界模型工作（OthelloGPT 等）**：后者验证世界模型存在；本文解释为什么世界模型与采样不确定性耦合。
- **启发**：微分几何工具（平行输运、全息性）可推广到 attention 几何、梯度流形状等其他可解释性问题。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 微分几何视角与 $\mathfrak{so}(n)$ 值形式的构造高度原创。
- 实验充分度: ⭐⭐⭐⭐☆ 国际象棋验证充分且定量，但单领域限制泛化性结论。
- 写作质量: ⭐⭐⭐⭐☆ 数学严谨，但部分推导对非微分几何背景读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 开辟用几何理解 LLM 内部动力学的新方向，影响深远。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Geometric Signatures of Compositionality Across a Language Model's Lifetime](../../ACL2025/llm_nlp/geometric_compositionality_lifetime.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ACL 2025\] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model](../../ACL2025/llm_nlp/neko_cross-modality_post-recognition_error_correction_with_tasks-guided_mixture-.md)
- [\[ICML 2026\] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation](on_the_limits_of_llm_adaptability_impact_of_model-internalized_priors_on_annotat.md)

</div>

<!-- RELATED:END -->
