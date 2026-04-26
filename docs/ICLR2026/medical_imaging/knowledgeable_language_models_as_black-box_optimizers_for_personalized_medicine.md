---
title: >-
  [论文解读] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine
description: >-
  [ICLR 2026][医学图像][大语言模型优化] 提出 LEON（LLM-based Entropy-guided Optimization with kNowledgeable priors），一种数学原理严格的方法，将个性化医疗治疗方案设计建模为条件黑箱优化问题，通过熵约束和对抗性源批评模型引导 LLM 在不微调的情况下作为零样本优化器提出个性化治疗计划。
tags:
  - ICLR 2026
  - 医学图像
  - 大语言模型优化
  - 个性化医疗
  - 黑箱优化
  - 分布偏移
  - 先验知识
---

# Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine

**会议**: ICLR 2026  
**arXiv**: [2509.20975](https://arxiv.org/abs/2509.20975)  
**代码**: [代码](https://code.roche.com/braid/projects/leon)(有)  
**领域**: 医学影像/个性化医疗  
**关键词**: 大语言模型优化, 个性化医疗, 黑箱优化, 分布偏移, 先验知识  

## 一句话总结

提出 LEON（LLM-based Entropy-guided Optimization with kNowledgeable priors），一种数学原理严格的方法，将个性化医疗治疗方案设计建模为条件黑箱优化问题，通过熵约束和对抗性源批评模型引导 LLM 在不微调的情况下作为零样本优化器提出个性化治疗计划。

## 研究背景与动机

- **领域现状**：个性化医疗的目标是根据患者的基因和环境因素发现最优治疗方案。近期 LLM 在数学和代码等领域的黑箱优化中展现了潜力
- **现有痛点**：(1) 真实治疗效果评估代价极高，通常只能使用代理模型（数字孪生、ML 模型）来评估方案质量；(2) 代理模型在分布偏移下（如面对新医院的患者）预测不准，导致优化出看起来好但实际差的"虚假"方案；(3) 特定人群在临床研究中系统性代表不足
- **核心矛盾**：简单地用代理模型 $\hat{f}$ 替代真实目标 $f$ 进行优化，会因分布外偏差导致治疗方案在实际中效果差。而改进代理模型又受限于数据可用性和隐私问题
- **本文目标**：在代理模型不可靠、真实目标函数不可访问的情况下，如何为分布偏移下的患者设计个性化治疗方案
- **切入角度**：利用 LLM 内化的领域先验知识（医学教科书、知识图谱）作为补充信号，通过约束优化同时控制代理模型的外推程度和 LLM 提议的确定性
- **核心 idea**：用两个约束（Wasserstein 距离约束限制分布偏移 + 熵约束提升 LLM 确定性）规范化 LLM 驱动的条件黑箱优化，利用先验知识提高 LLM 作为随机治疗推荐引擎的质量

## 方法详解

### 整体框架

LEON 的优化循环：(1) **采样**——LLM 根据任务描述、患者信息、先验知识和历史提议-评分表生成一批治疗方案；(2) **聚类**——将方案分配到等价类中；(3) **确定性估计**——估计 LLM 确定性参数 $\mu$ 和源批评参数 $\lambda$；(4) **评分**——用 $\mu[\hat{f}(x;z) + \lambda c^*(x)]$ 为每个方案评分，存储用于后续提示。

### 关键设计

#### 1. 约束条件化优化问题

- **功能**：将个性化医疗建模为有约束的条件黑箱优化
- **核心思路**：
$$\arg\max_{q(x)} \mathbb{E}_{x \sim q(x)}[\hat{f}(x;z)] \quad \text{s.t.} \quad W_1(p_{\text{src}}, q) \leq W_0, \quad \mathcal{H}_\sim(q(x)) \leq H_0$$
  - 第一个约束：1-Wasserstein 距离限制提议方案分布与历史方案分布的偏离程度（通过对抗性源批评模型 $c^*$ 实现），避免代理模型外推
  - 第二个约束：粗粒度熵限制 LLM 提议的分散程度，鼓励确定性高的方案
- **设计动机**：两个约束分别解决代理模型不可靠和 LLM 随机性两个问题

#### 2. Lagrange 对偶求解

- **功能**：推导约束优化问题的可计算解
- **核心思路**：
    - **Lemma 4.2（等价类内塌缩）**：最优分布 $q^*$ 在每个等价类中只需集中在最优设计 $x_i^* = \arg\max_{x \in [x]_i} (\hat{f}(x;z) + \lambda c^*(x))$ 上
    - **Lemma 4.3（概率采样）**：等价类概率满足 $\bar{q}_i \propto \exp[\mu(\hat{f}(x_i^*;z) + \lambda c^*(x_i^*))]$
    - 两个 Lagrange 乘子 $\lambda$（源批评确定性）和 $\mu$（LLM 确定性）分别控制两个约束

#### 3. 确定性参数动态估计

- **LLM 确定性 $\mu$**：通过 LLM 批量采样估计等价类占有率 $\hat{q}_i$，对 $(\hat{f}(x_i^*;z) + \lambda c^*(x_i^*), \log \hat{q}_i)$ 做线性回归得到 $\hat{\mu}$。直觉：高熵（低确定性）时 $\hat{\mu} \approx 0$ 降低奖励加成；高确定性时 $\hat{\mu} > 0$ 增强奖励
- **源批评参数 $\lambda$**：通过对偶函数梯度下降更新 $\lambda_{t+1} = \lambda_t - \eta_\lambda [W_0 - W_1(\text{estimated})]$。方案在分布内时 $\lambda$ 减小允许更多探索，方案偏离时 $\lambda$ 增大约束外推

#### 4. 先验知识生成

- **功能**：利用外部知识源为 LLM 提供领域先验
- **工具集**：医学教科书语料、MedGemma 27B 模型、HetioNet/PrimeKG 知识图谱、Cellosaurus 细胞系数据、COSMIC 癌症突变数据、GDSC 药物敏感性、DepMap 癌细胞依赖数据
- **流程**：LLM 作为工具调用代理，自主选择相关知识源，合成自然语言先验知识陈述
- **设计动机**：先验知识帮助 LLM 克服 next-token 生成的统计随机性，提高方案质量和 $\mu$ 值

### 损失函数

LEON 本身不需要训练 LLM。源批评模型 $c^*$ 通过 Wasserstein 对偶 (Eq.1) 训练，Lipschitz 约束通过参数裁剪实现。

## 实验关键数据

### 主实验

5 个真实世界个性化医疗优化任务（分布偏移设定，100 名测试患者）：

| 方法 | Warfarin RMSE↓ | HIV 病毒载量↓ | Breast TTNTD↑ | Lung TTNTD↑ | ADR NLL↓ | 平均排名 |
|---|---|---|---|---|---|---|
| Human（实际治疗）| 2.68 | 4.55 | 29.65 | 21.10 | - | 8.5 |
| Gradient Ascent | 1.37 | 4.52 | 65.23 | 24.09 | 23.7 | 5.2 |
| BO-qEI | 1.36 | 4.53 | 67.05 | 27.97 | 23.2 | 3.4 |
| OPRO | 1.40 | 4.55 | 55.68 | 24.35 | 23.8 | 7.0 |
| Eureka | 1.54 | 4.58 | 63.48 | 25.10 | 21.3 | 6.8 |
| **LEON** | **1.36** | **4.50** | **72.43** | **32.71** | **12.4** | **1.2** |

### 消融实验

- 去除先验知识：性能显著下降，LEON 对知识质量敏感
- 去除源批评模型约束（$\lambda = 0$）：代理模型外推导致 ground-truth 性能退化
- 去除熵约束（$\mu = 0$）：LLM 提议过于分散，需要更多迭代才能收敛
- 不同 LLM 选择：gpt-4o-mini 表现最佳性价比

### 关键发现

1. LEON 在 5 个任务中平均排名 1.2，显著优于传统优化方法和其他 LLM 优化方法
2. LEON 提出的方案**优于患者实际接受的治疗**（Human baseline），具有实际临床价值
3. 两个约束（Wasserstein + 熵）的协同效应明显：单独去除任一约束都会导致性能下降
4. 先验知识在分布偏移设定下尤为重要——它帮助 LLM 补充代理模型在 OOD 区域的盲区

## 亮点与洞察

1. **数学严格性与实际价值兼具**：从约束优化到 Lagrange 对偶、到 Lemma 的推导都很严谨，同时 5 个真实临床任务的实验极具说服力
2. **两个确定性参数的设计极为巧妙**：$\mu$ 量化 LLM 的"共识程度"，$\lambda$ 量化设计的"分布内程度"，二者动态平衡探索与利用
3. **零样本优化**：LLM 无需任何微调，纯靠 prompting + 外部知识就能超越专门的优化算法
4. **隐私保护**：源批评模型 $c^*$ 仅需要治疗设计数据 $\mathcal{D}_{\text{src}} \subseteq \mathcal{X}$，不需要患者信息

## 局限与展望

1. **先验知识敏感性**：错误或过时的领域知识会传播到优化输出中，需要知识质量保证机制
2. **模拟基准的局限**：虽然使用真实数据，但评估框架仍基于学到的 ground-truth 函数 $f$，无法完全反映真实患者响应的异质性
3. **等价关系定义**：使用 k-means 聚类定义等价类，不同聚类方法可能影响结果
4. **计算成本**：每个患者需要 2048 次代理模型查询和大量 LLM API 调用
5. **公平性未充分验证**：虽然在附录中讨论了性别/种族公平性，但 LLM 的社会偏见可能影响治疗建议

## 相关工作与启发

- **LLM 优化**：OPRO（Yang et al., 2024a）通过 prompting 优化但缺乏约束；Eureka（Ma et al., 2024b）加入反思但无分布控制
- **分布偏移下的优化**：Trabucco et al.（2021）的保守代理模型假设可控制代理模型设计，而 LEON 处理黑箱代理
- **鉴别模型在优化中的应用**：Wasserstein 距离约束来自 MBO/生物序列设计文献（Yao et al., 2024, 2025b）
- **启发**：LLM 作为条件优化器的范式可推广到其他需要个性化决策的领域（教育、金融等），关键是如何注入领域知识并控制置信度

## 评分

⭐⭐⭐⭐（4/5）

- **创新性**：⭐⭐⭐⭐⭐ 将 LLM 优化、分布偏移鲁棒性、先验知识注入统一到数学严格的框架
- **实验**：⭐⭐⭐⭐ 5 个真实临床任务 + 10 个基线，全面深入
- **写作**：⭐⭐⭐⭐ 理论推导清晰，但公式符号较多
- **实用性**：⭐⭐⭐ 依赖多种外部知识源和 LLM API，部署门槛较高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease](../../NeurIPS2025/medical_imaging/from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)
- [\[NeurIPS 2025\] DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization](../../NeurIPS2025/medical_imaging/designx_human-competitive_algorithm_designer_for_black-box_optimization.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](../../ACL2026/medical_imaging/ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)

<!-- RELATED:END -->
