---
title: >-
  [论文解读] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases
description: >-
  [ICLR 2026][单语义特征学习] 提出GRADIEND——一个基于梯度的编码器-解码器架构，通过单个瓶颈神经元从模型梯度中学习可解释的单语义特征（以性别为例），不仅可以识别哪些权重编码了特定特征，还能通过解码器直接修改模型权重来消除偏见，与INLP结合在所有基线模型上达到SOTA去偏效果。
tags:
  - ICLR 2026
  - 单语义特征学习
  - 性别偏见消除
  - 梯度编码器-解码器
  - Transformer
  - 可解释性
---

# GRADIEND: Feature Learning within Neural Networks Exemplified through Biases

**会议**: ICLR 2026  
**arXiv**: [2502.01406](https://arxiv.org/abs/2502.01406)  
**代码**: [https://github.com/aieng-lab/gradiend](https://github.com/aieng-lab/gradiend)  
**领域**: Others (AI Fairness / Interpretability)  
**关键词**: 单语义特征学习, 性别偏见消除, 梯度编码器-解码器, Transformer去偏, 可解释性

## 一句话总结
提出GRADIEND——一个基于梯度的编码器-解码器架构，通过单个瓶颈神经元从模型梯度中学习可解释的单语义特征（以性别为例），不仅可以识别哪些权重编码了特定特征，还能通过解码器直接修改模型权重来消除偏见，与INLP结合在所有基线模型上达到SOTA去偏效果。

## 研究背景与动机
AI系统经常表现出并放大社会偏见（如性别偏见），在法律、医疗、招聘等关键领域产生有害影响。Amazon的AI招聘工具偏向男性候选人就是典型案例。

现有Transformer去偏方法包括：
- **反事实数据增强**（CDA）：交换性别相关词后重训练，代价高
- **Dropout增强**：增加预训练时的Dropout率
- **INLP**：迭代零空间投影法，反复训练线性分类器并投影到零空间
- **SentDebias/SelfDebias**：后处理方法，调整嵌入或输出分布

**核心矛盾**：现有无监督稀疏自编码器方法（如Bricken et al., 2023）虽然能提取可解释特征，但需要学习大量潜在特征后再搜索有意义的解释，无法保证期望的特征（如"性别"）会出现。而现有去偏方法大多是后处理式的，不能真正修改已训练模型的内部表示。

**本文切入点**：利用模型梯度中包含的特征信息——梯度天然指示了"哪些参数需要更新才能改变某个特征"。通过设计一个极简的编码器-解码器结构，可以从事实/反事实梯度差中学习到一个**有期望语义**的单语义特征神经元。

## 方法详解

### 整体框架
输入：一个预训练的Transformer语言模型 + 包含名字和代词的模板句子。输出：(1) 一个标量特征神经元$h$（编码性别信息）；(2) 一个解码向量（指示如何修改模型权重以改变性别偏见强度）。

### 关键设计

1. **事实/反事实梯度构造**：对于模板句子如"Alice explained the vision as best [MASK] could"，分别以正确代词（"she"）和反事实代词（"he"）为目标计算MLM梯度：

    - 事实梯度 $\nabla^+ W_m$：以正确性别代词为目标
    - 反事实梯度 $\nabla^- W_m$：以相反性别代词为目标
    - 梯度差 $\nabla^{\pm}W_m := \nabla^+ W_m - \nabla^- W_m$
   
   梯度差消除了非性别相关的共同更新成分，仅保留性别相关的方向。

2. **GRADIEND编码器-解码器**：极简架构，仅使用单个隐藏神经元作为瓶颈：
   
    $\text{enc}(\nabla^+ W_m) = \tanh(W_e^T \cdot \nabla^+ W_m + b_e) =: h \in \mathbb{R}$
    $\text{dec}(h) = h \cdot W_d + b_d \approx \nabla^{\pm} W_m$
   
   其中$W_e, W_d, b_d \in \mathbb{R}^n$，$b_e \in \mathbb{R}$，总参数量仅为$3n+1$。编码器将事实梯度映射到一个标量$h$（性别因子），解码器从$h$重建梯度差。目标函数为MSE损失。

3. **性别去偏应用**：选定性别因子$h$和学习率$\alpha$后，直接修改模型权重：
   
    $\tilde{W}_m := W_m + \alpha \cdot \text{dec}(h)$
   
   当$h$和$\alpha$符号相同时模型偏向男性，符号不同时偏向女性。$h=0$附近对应去偏方向（利用偏置$b_e$学到的去偏方向）。

4. **三个综合指标设计**：

    - **BPI**（Balanced Prediction Index）：衡量去偏程度，同时考虑语言建模能力、性别预测平衡性和预测合理性
    - **FPI**（Female Prediction Index）：衡量女性偏向程度
    - **MPI**（Male Prediction Index）：衡量男性偏向程度

### 损失函数 / 训练策略
- 优化器：Adam，学习率1e-5，权重衰减1e-2
- 批量：32，MSE损失
- 训练步数：23,653（等于Genter训练集模板数）
- 每250步用$\text{Cor}_{\text{Genter}}^{\text{val}}$评估，选最优模型
- 每步随机选一个性别，从NAMExact中采样名字
- 自定义初始化解码器权重（使用与编码器相同的$n$作为初始化范围）
- 预测层不参与GRADIEND参数（确保去偏作用于语言模型本身）

## 实验关键数据

### 编码器评估（H1：学习性别特征）

| 模型 | $\text{Acc}_{\text{Genter}}$ | $\text{Cor}_{\text{Genter}}$ | $\text{Acc}_{\text{Enc}}$ | $\text{Cor}_{\text{Enc}}$ |
|------|------|------|------|------|
| BERT-base | 1.000 | 0.957 | 0.612 | 0.669 |
| BERT-large | 1.000 | 0.908 | 0.578 | 0.616 |
| DistilBERT | 1.000 | 1.000 | 0.758 | 0.838 |
| RoBERTa | 1.000 | 1.000 | 0.909 | 0.935 |

所有模型在性别相关数据上几乎完美区分$\pm 1$；对性别中性输入也能映射到接近0的值。

### 去偏效果比较（H2：修改性别偏见）

| 方法 | SS(%) | SEAT | CrowS(%) | LMS(%) | GLUE(%) |
|------|-------|------|----------|--------|---------|
| BERT-base | 基线 | 基线 | 基线 | 基线 | 基线 |
| + GRADIEND-BPI | 改善 | - | - | 保持 | 保持 |
| + GRADIEND-BPI + INLP | **显著改善** | 改善 | - | 保持 | 保持 |
| CDA / Dropout / INLP / SentDebias | 部分改善 | 不一致 | 不一致 | 部分下降 | 部分下降 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同基线模型 | 4个模型全部成功 | BERT/DistilBERT/RoBERTa均可学到性别特征 |
| 性别因子$h=0$ | BPI最优近似在此 | 解码器偏置$b_e$自动学到去偏方向 |
| 过拟合分析 | 训练/验证/测试名字无显著差异 | 泛化到未见名字 |
| 泛化到woman/man | he/she泛化到woman/man | 性别概念的跨词汇泛化 |

### 关键发现
- **GRADIEND-BPI + INLP是唯一在所有基线模型的SS指标上均达到显著改善的组合方法**，展示了强鲁棒性
- 引入置信区间后，现有去偏方法的有效性远不如先前研究所暗示的那样明确
- RoBERTa出人意料地表现出女性偏向（$\mathbb{P}(F) > \mathbb{P}(M)$），与通常认为的男性偏向相反
- 偏向某一性别（FPI/MPI）比去偏（BPI）更容易实现
- 模型权重调整的影响呈近点对称分布（与$h$和$\alpha$的符号有关）

## 亮点与洞察
- **从梯度中直接学习有期望语义的特征**：与无监督稀疏自编码器（学习大量特征后人工解释）不同，GRADIEND可以学习"期望的"可解释特征（如性别），这是一个重要的范式转换
- **极简但优雅的设计**：仅一个标量瓶颈神经元，参数量为$3n+1$，但有效地编码了性别这一复杂概念。架构简洁性使得分析和理解更容易
- **引入Bootstrap置信区间**：揭示了该领域的一个被忽视的问题——先前的去偏方法比较缺乏统计严谨性
- **解码器偏置的有趣发现**：即使$h=0$（无性别信息），解码器的偏置$b_d$本身就学到了一个有效的去偏方向

## 局限与展望
- 仅验证了二元的性别特征，能否推广到连续特征（如情感）、多值特征（如德语冠词der/die/das）或其他类型偏见（种族、宗教）需要进一步探索
- 仅在encoder-only模型上测试，未验证在生成式Transformer（GPT类）上的效果
- 事实/反事实梯度的构造依赖于MLM任务，CLM任务下的适配方案有待制定
- 去偏效果的trade-off：强去偏会降低语言建模能力，需要在$h$和$\alpha$的搜索网格中谨慎选择
- 性别被简化为二元处理，未考虑非二元性别身份

## 相关工作与启发
- **单语义特征/稀疏自编码器**（Bricken et al., 2023; Templeton et al., 2024）：无监督方法，从高维特征空间中分解可解释特征，Claude 3中发现了性别偏见感知特征
- **INLP**（Ravfogel et al., 2020）：迭代零空间投影去偏，与GRADIEND互补效果最佳
- **Movement Pruning**（Joniak & Aizawa, 2022）：通过剪枝减少性别偏见
- **Grad-CAM / Integrated Gradients**：梯度解释方法的先驱工作
- 启发：梯度不仅可以用于解释（attribution），还可以编码和操控模型内部的语义特征。这种"梯度作为特征表示"的思路可能对模型编辑（model editing）和遗忘学习（unlearning）有重要价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （从梯度中学习期望语义特征是非常新颖的范式）
- 实验充分度: ⭐⭐⭐⭐ （4个基模型，多种指标，但只测了性别一种特征）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式严谨，附录详尽）
- 价值: ⭐⭐⭐⭐ （proof-of-concept价值高，但实际应用范围待扩展）

<!-- RELATED:START -->

## 相关论文

- [Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](../../ACL2025/social_computing/banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)
- [Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](../../ACL2025/social_computing/measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)
- [Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)

<!-- RELATED:END -->
