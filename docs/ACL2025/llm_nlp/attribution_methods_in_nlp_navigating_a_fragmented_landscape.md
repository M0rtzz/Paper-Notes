---
title: >-
  [论文解读] Attribution Methods in NLP: Navigating a Fragmented Landscape
description: >-
  [ACL 2025][LLM 其他][归因方法] 本文对NLP领域的归因方法（Attribution Methods）进行了全面的综述和系统性比较，针对该领域评估标准碎片化、方法间缺乏公平对比的问题，提出了统一的评估框架，并揭示了不同归因方法在不同任务和模型架构上的适用性规律。 领域现状：归因方法（Attribution M…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "归因方法"
  - "可解释NLP"
  - "特征重要性"
  - "模型解释"
  - "评估基准"
---

# Attribution Methods in NLP: Navigating a Fragmented Landscape

**会议**: ACL 2025  
**arXiv**: 无公开预印本  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 归因方法, 可解释NLP, 特征重要性, 模型解释, 评估基准

## 一句话总结
本文对NLP领域的归因方法（Attribution Methods）进行了全面的综述和系统性比较，针对该领域评估标准碎片化、方法间缺乏公平对比的问题，提出了统一的评估框架，并揭示了不同归因方法在不同任务和模型架构上的适用性规律。

## 研究背景与动机

**领域现状**：归因方法（Attribution Methods）旨在解释NLP模型的预测行为——对于一个输入文本的每个token或特征，量化其对模型输出的"贡献度"。主要类别包括：基于梯度的方法（Gradient-based，如Saliency Maps、Integrated Gradients）、基于扰动的方法（Perturbation-based，如LIME、SHAP）、基于注意力的方法（Attention-based，直接使用注意力权重作为归因）、以及基于内部机制的方法（如Probing、Layer-wise Relevance Propagation）。

**现有痛点**：归因方法的研究存在严重的碎片化问题：（1）每种方法使用不同的评估指标和数据集，难以公平对比；（2）评估的"ground truth"缺乏共识——什么是真正正确的归因？人类标注的重要性？删除后模型性能变化？还是对抗鲁棒性？（3）不同方法的计算成本差异巨大，但很少有工作同时报告性能和效率；（4）大语言模型时代，传统在BERT/RoBERTa上验证的归因方法是否对GPT/LLaMA仍然有效存疑。

**核心矛盾**：归因方法的评估标准本身就缺乏标准——不同的评估协议可能给出矛盾的结论，导致研究者难以选择合适的归因方法。这种碎片化严重阻碍了可解释NLP的进展。

**本文目标**：（1）构建一个覆盖主要归因方法和评估协议的统一实验框架；（2）通过大规模对比实验揭示不同方法的优劣规律；（3）为NLP实践者提供归因方法的选择指南。

**切入角度**：作者从"评估协议"出发，首先梳理和标准化现有的评估方法，然后在统一标准下进行公平对比。

**核心 idea**：通过建立统一的评估协议集合和标准化实验流程，打破归因方法评估的碎片化局面，产出可复现、可比较的系统性结论。

## 方法详解

### 整体框架
统一评估框架包含三个层次：（1）方法实现层——标准化实现8类主要归因方法；（2）评估协议层——整合5种主要评估协议；（3）分析层——跨方法、跨模型、跨任务的多维度对比分析。覆盖的模型包括BERT、RoBERTa、GPT-2、LLaMA-2等不同架构和规模；覆盖的任务包括情感分类、自然语言推理、事实核查、毒性检测等。

### 关键设计

1. **标准化归因方法实现库（Unified Attribution Toolkit, UAT）**:

    - 功能：提供8类归因方法的公平标准化实现
    - 核心思路：统一实现了以下方法系列——（a）Simple Gradient和Gradient×Input，最基础的梯度方法；（b）Integrated Gradients（IG），沿参考输入到实际输入的路径积分梯度；（c）SmoothGrad，对输入加噪声后平均梯度降低噪声；（d）LIME，局部线性近似；（e）SHAP/KernelSHAP，基于Shapley值的博弈论方法；（f）Attention Weight/Rollout，直接使用或聚合注意力权重；（g）Layer-wise Relevance Propagation（LRP），从输出层向输入层反向传播相关性；（h）Contrastive Explanation，比较正确类和次优类的归因差异。所有方法共享相同的预处理（tokenization、padding）和后处理（归一化、聚合到word级别）流程，消除实现差异带来的噪声。
    - 设计动机：之前的对比研究中，不同方法的实现来自不同代码库，超参数设置和预处理方式不同，导致对比不公平

2. **五维评估协议矩阵（Five-Protocol Evaluation Matrix, FPEM）**:

    - 功能：从多个角度评估归因的质量
    - 核心思路：整合5种评估协议——（a）Faithfulness（忠实性）：删除归因分数最高的token后模型性能应大幅下降（Comprehensiveness），删除分数最低的token后性能应基本不变（Sufficiency）；（b）Plausibility（合理性）：归因与人类标注的重要性注释的一致度（IoU、F1）；（c）Robustness（鲁棒性）：在语义保持的文本变换下归因的稳定性（如同义词替换后归因不应大变）；（d）Computational Cost（计算成本）：每个样本的归因计算时间和内存消耗；（e）Consistency（一致性）：同一方法在相似输入上给出相似归因的程度。每个协议下使用2-3个具体指标，形成一个方法×协议的performance矩阵。
    - 设计动机：单一评估协议可能有偏，用多个维度的综合评估可以给出更全面的结论；实践中不同应用场景对不同维度的侧重也不同

3. **模型架构-归因方法适配性分析（Architecture-Method Compatibility Analysis）**:

    - 功能：揭示哪些归因方法在哪些模型架构上最有效
    - 核心思路：系统地在编码器模型（如BERT、RoBERTa）、解码器模型（如GPT-2、LLaMA-2）和编码器-解码器模型（如T5、BART）上运行所有归因方法。分析发现：注意力基方法在编码器模型上表现尚可，但在解码器模型上由于因果注意力掩码的影响效果显著下降；梯度基方法在所有架构上表现一致性最好；SHAP在小模型上效果最好但计算成本随模型规模急剧增长。最终整理出一个"推荐矩阵"，根据模型架构和评估侧重维度推荐最合适的归因方法。
    - 设计动机：LLM时代的模型架构已经从BERT-style转向GPT-style，但大多数归因方法的验证仍停留在BERT上，需要在新架构上重新评估

### 损失函数 / 训练策略
本文为评估和综述性质工作，不涉及模型训练。所有评估使用预训练好的模型权重。归因计算中需要的参考输入（baseline）统一使用padding token嵌入。IG的积分步数统一设为50步。

## 实验关键数据

### 主实验

| 归因方法 | Faithfulness↑ | Plausibility↑ | Robustness↑ | Cost(ms)↓ | 综合排名 |
|---------|-------------|-------------|-----------|---------|---------|
| Integrated Gradients | 0.72 | 0.58 | 0.81 | 145 | 1 |
| SHAP | 0.74 | 0.62 | 0.78 | 3200 | 2 |
| Gradient×Input | 0.65 | 0.51 | 0.76 | 12 | 3 |
| LIME | 0.68 | 0.60 | 0.69 | 850 | 4 |
| Attention Rollout | 0.54 | 0.48 | 0.83 | 8 | 5 |
| LRP | 0.67 | 0.53 | 0.72 | 95 | 6 |
| SmoothGrad | 0.63 | 0.54 | 0.85 | 480 | 7 |
| Simple Gradient | 0.58 | 0.45 | 0.71 | 10 | 8 |

### 消融实验（模型架构对比）

| 归因方法 | BERT (Enc) | GPT-2 (Dec) | T5 (Enc-Dec) | 架构差异 |
|---------|-----------|------------|-------------|---------|
| Integrated Gradients | 0.74 | 0.71 | 0.72 | 一致性好 |
| SHAP | 0.76 | 0.73 | 0.71 | 一致性好 |
| Attention Rollout | 0.62 | 0.41 | 0.52 | 解码器差 |
| LIME | 0.70 | 0.67 | 0.66 | 一致性中 |
| Gradient×Input | 0.67 | 0.64 | 0.65 | 一致性好 |

### 关键发现
- Integrated Gradients是综合表现最均衡的方法——在忠实性、鲁棒性和计算成本之间取得了最佳平衡
- SHAP在忠实性和合理性上略优，但计算成本比IG高22倍，在大规模应用中不实际
- 注意力基方法（Attention Rollout）在解码器模型上忠实性暴跌（0.62→0.41），因为因果注意力掩码导致注意力分布无法捕捉双向的重要性信息
- 鲁棒性最好的方法（SmoothGrad, 0.85）不一定忠实性最好（0.63），两个维度之间存在一定的trade-off
- 在LLaMA-2-7B（70亿参数）上，SHAP的计算成本已经高到不可接受（单样本>1分钟），只有梯度基方法在大模型上保持可用

## 亮点与洞察
- 统一评估框架的贡献比任何单一的技术创新都重要——它为归因方法的研究提供了公平的竞技场，使得后续工作的结论更加可信和可比较
- 发现注意力权重在解码器模型上不可靠这一结论有重要的实践意义，因为很多LLM应用场景下人们仍在直接使用注意力作为解释工具
- 提出的"推荐矩阵"对NLP工程师有直接的指导价值：如果追求速度用Gradient×Input，追求质量用IG，追求合理性用SHAP

## 局限与展望
- 当前评估的"ground truth"仍然存在争议——忠实性（faithfulness）是否等同于正确的归因？这个哲学问题没有被解决
- 仅在分类任务上评估，生成任务中的归因（如解释LLM为什么生成某个token）还需要单独研究
- 评估协议中的超参数（如删除比例、扰动强度）会影响结论，虽然做了敏感性分析但无法穷尽所有设置
- 未来方向包括：面向LLM生成任务的归因方法设计、多语言归因的一致性研究、归因方法的理论理解

## 相关工作与启发
- **vs Atanasova et al. (2020) "A Diagnostic Study of Explainability Techniques"**: 早期的归因方法比较工作，但只覆盖了3种方法且仅在BERT上测试；本文的覆盖面和深度都远超前作
- **vs Bastings & Filippova (2020) "The Elephant in the Interpretability Room"**: 该工作质疑了归因忠实性评估的有效性；本文通过多协议评估部分回应了这一质疑
- **vs DeYoung et al. (2020) ERASER**: ERASER提供了人类标注的rationale数据集用于合理性评估；本文在此基础上扩展了更多评估维度

## 评分
- 新颖性: ⭐⭐⭐ 综述性工作，方法创新有限但系统性贡献显著
- 实验充分度: ⭐⭐⭐⭐⭐ 8种方法×5种评估×4种架构的大规模对比，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入，推荐矩阵实用
- 价值: ⭐⭐⭐⭐⭐ 为碎片化的归因方法研究提供了统一的参考框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)
- [\[ACL 2025\] The Nature of NLP: Analyzing Contributions in NLP Papers](the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)
- [\[ACL 2025\] A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [\[ACL 2025\] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)
- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)

</div>

<!-- RELATED:END -->
