---
title: >-
  [论文解读] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning
description: >-
  [AAAI 2026 Oral][人体理解][人脸反欺骗] 提出PA-FAS框架，通过推理路径增强（Reasoning Path Augmentation）策略和答案打乱机制，解决了多模态FAS中SFT+RL范式的两大瓶颈（推理路径多样性不足和推理捷径问题），首次在统一框架中同时实现多模态融合、域泛化和可解释性。
tags:
  - "AAAI 2026 Oral"
  - "人体理解"
  - "人脸反欺骗"
  - "多模态融合"
  - "域泛化"
  - "强化学习"
  - "可解释性"
---

# PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.17927](https://arxiv.org/abs/2511.17927)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 人脸反欺骗, 多模态融合, 域泛化, 强化学习, 可解释性

## 一句话总结

提出PA-FAS框架，通过推理路径增强（Reasoning Path Augmentation）策略和答案打乱机制，解决了多模态FAS中SFT+RL范式的两大瓶颈（推理路径多样性不足和推理捷径问题），首次在统一框架中同时实现多模态融合、域泛化和可解释性。

## 研究背景与动机

### 问题定义

人脸反欺骗（FAS）旨在区分真实人脸和欺骗呈现（打印照片、重播视频、3D面具等），是人脸识别系统安全性的关键保障。多模态FAS利用RGB、深度和红外多种模态信息提升检测精度和鲁棒性。

### 现有方法的三大研究空白

**域泛化不足**：现有方法大多针对单模态设计，跨域泛化能力有限

**多模态方法缺乏可解释性**：虽然多模态方法性能优越，但缺乏显式的可解释性机制来识别深度和红外模态中的欺骗线索

**MLLM在域泛化场景的局限**：现有MLLM-based FAS方法（如FaceCoT、SHIELD）展示了强推理能力，但忽视了泛化问题且无法处理跨模态线索整合

### SFT+RL范式的失败分析（核心动机）

作者深入分析了SFT+RL范式应用于多模态FAS时的失败机制：

**问题1：推理路径多样性不足**
- 多模态FAS数据集通常只有简单的二分类标签，缺少语言级的关键视觉线索标注
- SFT阶段任务单一、数据有限，导致模型过拟合到刚性模式
- RL阶段缺乏有效反馈和探索空间——大多数样本获得极端奖励（全1或全0），缺少有信息量的中间信号
- 实验证据：在原始数据上微调的模型，累积无效样本数几乎线性增长

**问题2：推理捷径**
- 即使通过传统数据增强扩充推理链，模型仍通过直接查看图像来预测答案，完全绕过推理链（CoT）
- 证据：将SFT阶段的推理文本随机替换为其他样本的CoT，模型性能几乎不变——说明模型依赖视觉预测而忽略CoT
- 这导致模型过度自信，极大缩小了RL阶段的探索空间

## 方法详解

### 整体框架

PA-FAS遵循四步流程：(a) 低级数据标注 → (b) 高级数据标注生成CoT → (c) 正负随机路径采样扩展推理路径 → (d) SFT（带答案打乱）+ RL（GRPO）两阶段训练。

### 关键设计

#### 1. **推理路径增强（Reasoning Path Augmentation）**

核心创新，通过构建结构化推理树来系统性地扩展推理路径：

- **推理树构建**：基于FAS任务的细粒度层级分类体系（图4的旭日图），建立形式化推理树 $\mathcal{T} = (\mathcal{V}, \mathcal{E})$
    - $\mathcal{V}$：推理节点集合（每个节点代表一个语义类或逻辑决策单元）
    - $\mathcal{E}$：有向边集合
    - 路径 $\mathcal{P} = (v_1, v_2, \ldots, v_n)$：从根到目标叶节点的完整推理链

- **正负随机路径采样（PNRPS）**：Algorithm 1的核心机制
    - **单节点操作约束**：每个节点最多一次正向探索步 $(+, v)$ 和一次反向反思步 $(-, v)$，避免冗余遍历
    - **路径长度约束**：$L_{\max} = \alpha(D-1)$，其中 $D$ 为最大分类深度，$\alpha > 1$ 为可调缩放因子
    - **语义一致性**：每个节点关联预定义的CoT子句模板，沿路径顺序组合生成最终推理文本
    - **结构化采样**：基于规则的深度优先遍历，随机采样 $N$ 条使用RGB、IR、DEPTH任意模态信息的有效路径

- **数据扩展效果**：
  $$\{(x_i, \ell_i)\}_{i=1}^{M} \rightarrow \bigcup_{i=1}^{M}\{(x_i, \text{CoT}(\mathcal{P}_i^{(j)})) \mid j=1,\ldots,N\}$$
  从800个标注样本出发，每个样本生成 $N=50$ 条推理路径，总共约4万个结构多样、语义一致的增强样本

#### 2. **答案打乱机制（Answer Shuffling）**

解决推理捷径问题的关键设计：

- **动机**：单一任务与丰富推理路径的耦合导致模型直接看图预测答案、绕过推理链
- **方法**：SFT阶段将每个CoT中的最终答案随机替换为其他样本的答案
- **效果**：迫使模型专注于学习多样化的推理路径而非记忆答案，为RL阶段保留足够的探索空间

#### 3. **RL阶段（GRPO）**

采用Group Relative Policy Optimization进行策略优化：

- 对每个问答对 $(q, a)$，旧策略采样 $G$ 个响应
- 奖励定义：$\mathcal{R} = \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{classification}}$
- 相对优势计算：$\hat{A}_{i,t} = \frac{\mathcal{R}_i - \text{mean}(\{\mathcal{R}_i\})}{\text{std}(\{\mathcal{R}_i\})}$
- 在数据稀缺条件下去掉KL散度项 $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 以避免抑制探索

### 训练策略

- **基座模型**：Qwen2.5VL-3B
- **SFT和RL阶段**：均训练500步，恒定学习率1e-6
- **数据**：仅需约800个高质量结构化推理路径样本
- **评估数据集**：WMCA、CASIA-SURF、CASIA-CeFA、PADISI

## 实验关键数据

### 主实验（Protocol 1：完整模态跨数据集测试）

| 方法 | 类别 | 平均HTER(%)↓ | 平均AUC(%)↑ |
|------|------|------------|-----------|
| FLIP | 单模态DG | 16.11 | 90.83 |
| MMDG | 多模态DG | 22.93 | 84.19 |
| DADM | 多模态DG | 13.63 | 92.96 |
| Qwen2.5-VL-3B (零样本) | 可解释多模态DG | 33.46 | 69.36 |
| Qwen2.5-VL-3B-SFT | 可解释多模态DG | 23.25 | 79.32 |
| Qwen2.5-VL-3B-SFT+GRPO | 可解释多模态DG | 34.37 | 68.68 |
| **PA-FAS (Ours)** | **可解释多模态DG** | **15.21** | **89.13** |

关键观察：朴素SFT+GRPO反而比纯SFT更差（HTER从23.25%升到34.37%），验证了作者的失败分析。PA-FAS将HTER降至15.21%，在可解释模型中遥遥领先。

### 消融实验（缺失模态场景，Protocol 2）

| 方法 | 缺少D HTER↓ | 缺少I HTER↓ | 缺少D&I HTER↓ | 平均HTER↓ |
|------|------------|-----------|-------------|----------|
| DADM | 21.56 | 20.82 | 22.61 | 21.66 |
| Qwen2.5-VL-3B-SFT | 23.25 | 23.25 | 23.25 | 23.25 |
| **PA-FAS** | **15.68** | **17.32** | **14.67** | **15.85** |

PA-FAS在模态缺失场景下依然保持优越性能，说明其推理路径增强策略有效利用了多模态互补信息。

### 关键发现

1. **SFT+RL崩溃验证**：在仅有二分类标签的数据集上，SFT+GRPO性能反而下降（HTER: 23.25% → 34.37%）
2. **推理捷径存在**：替换推理文本后模型性能不变，证明模型完全忽略CoT推理链
3. **路径多样性的重要性**：具有多样推理路径的数据集在SFT和SFT+RL下都显著优于单一路径数据集
4. **数据效率**：仅需800个样本+路径增强即可超越使用35000个原始数据的方法
5. **有限源域场景（Protocol 3）**：PA-FAS平均HTER为9.22%（CW→PS中仅0.15%），具备极强的域泛化能力

## 亮点与洞察

1. **对SFT+RL失败的深入诊断**：不是简单地提出方法，而是先通过实验（累积无效样本分析、推理文本替换实验）严格论证了失败原因
2. **推理路径增强的优雅设计**：基于层级分类体系构建形式化推理树，通过正负随机路径采样以最小代价扩展推理空间
3. **答案打乱的反直觉设计**：在SFT阶段故意给出错误答案以迫使模型学习推理过程，这一设计反直觉但极为有效
4. **首次统一三大目标**：多模态融合+域泛化+可解释性的一体化解决方案
5. **极高的数据效率**：800个样本即可实现SOTA级别的性能

## 局限与展望

1. **基座模型限制**：仅使用Qwen2.5VL-3B（3B参数），更大模型可能进一步提升性能
2. **推理树人工构建**：层级分类体系需要领域专家知识，自动化构建是未来方向
3. **标注成本**：虽然仅需800个样本，但这些样本的CoT标注仍需要一定的人工投入
4. **评估场景**：主要在四个数据集上验证，更多样的攻击类型和场景有待探索
5. **实时性**：基于MLLM的方法推理速度可能不满足实时部署需求

## 相关工作与启发

- **SFT Memorizes, RL Generalizes理论**：论文基于Chu等人的SFT记忆+RL泛化理论，但指出该理论在数据稀缺的多模态场景中需要新的适配
- **GRPO在FAS中的应用**：首次在FAS任务中应用GRPO算法，展示了rule-based group advantage策略在低训练成本下的可行性
- **对推理捷径的深入理解**：答案打乱机制的设计为其他结合SFT+RL的视觉理解任务提供了重要参考

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 推理路径增强+答案打乱的组合设计极具新意，对失败机制的分析深入
- 实验充分度: ⭐⭐⭐⭐⭐ — 三种protocol + 消融实验 + 多种对比方法，实验设计严谨
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，动机论证充分，但部分符号略显复杂
- 价值: ⭐⭐⭐⭐⭐ — 对SFT+RL范式的失败分析和解决方案具有广泛的参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/human_understanding/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](../../ECCV2024/human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[CVPR 2026\] FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](../../CVPR2026/human_understanding/facecot_cot_reasoning_face_anti_spoofing.md)
- [\[ECCV 2024\] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [\[CVPR 2026\] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](../../CVPR2026/human_understanding/avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)

</div>

<!-- RELATED:END -->
