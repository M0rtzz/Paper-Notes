---
title: >-
  [论文解读] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation
description: >-
  [CVPR 2025][模型压缩][数据集蒸馏] 提出EarlyLate训练策略，通过让不同IPC子批次从不同优化起点开始、经历不同迭代次数来生成难度各异的合成图像，在batch-to-global匹配框架下显著提升类内多样性，同时减少39.3%计算时间，在ImageNet-1K上以IPC=50达到66.1%（ResNet-101，超越RDED 4.9%）。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "数据集蒸馏"
  - "多样性"
  - "EarlyLate训练"
  - "batch-to-global匹配"
  - "大规模数据集"
---

# DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation

**会议**: CVPR 2025  
**arXiv**: [2411.19946](https://arxiv.org/abs/2411.19946)  
**代码**: [https://github.com/VILA-Lab/DELT](https://github.com/VILA-Lab/DELT)  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏、多样性、EarlyLate训练、batch-to-global匹配、大规模数据集

## 一句话总结
提出EarlyLate训练策略，通过让不同IPC子批次从不同优化起点开始、经历不同迭代次数来生成难度各异的合成图像，在batch-to-global匹配框架下显著提升类内多样性，同时减少39.3%计算时间，在ImageNet-1K上以IPC=50达到66.1%（ResNet-101，超越RDED 4.9%）。

## 研究背景与动机

**领域现状**：大规模数据集蒸馏主要分两种范式——batch-to-batch匹配（精细但计算昂贵，如MTT、FRePo）和batch-to-global匹配（解耦训练和合成，适合大数据集，如SRe2L、CDA）。后者因效率优势成为主流。

**现有痛点**：batch-to-global匹配方法中，每个合成图像独立优化但共享相同的全局监督信号（如BN统计量），导致同类合成图像高度同质化，缺乏多样性。G-VBSM用多模型缓解但增加了复杂度，RDED用真实图像拼接但不做优化训练。

**核心矛盾**：统一优化过程使所有图像收敛到类似的局部最优，类内多样性严重不足。

**本文目标** 在batch-to-global匹配框架下，以简洁低成本的方式提升合成图像的类内多样性。

**切入角度**：从early stopping和课程学习的角度出发——不同图像需要不同迭代次数才能收敛，让一部分图像多训练（充分优化、更抽象）、另一部分少训练（保留更多原始信息、更真实），自然产生不同优化深度的图像组合。

**核心 idea**：将同类IPC分成多个子批次，让它们从不同时间点加入优化，使不同子批次经历不同迭代次数，自然产生多样性不同的合成图像。

## 方法详解

### 整体框架
整体pipeline分为三步：（1）用预训练教师模型对原始图像patch按预测概率排名，选中等概率的patch初始化合成图像；（2）将IPC分为M个子批次，第一个子批次从头开始优化，每隔RI次迭代加入下一个子批次，所有子批次共享后续优化过程（拼接训练）；（3）最终的合成数据集包含经历了不同优化深度的图像，自然具有难度梯度和多样性。

### 关键设计

1. **EarlyLate训练策略**:

    - 功能：通过差异化优化迭代次数增强类内多样性
    - 核心思路：假设总迭代次数为T（如4K），将IPC分为M个子批次。第1个子批次从第0次开始优化（经历T次），第2个从第RI次加入（经历T-RI次），...第M个从第T-RI次加入（只经历RI次）。后加入的子批次与前面的拼接在一起联合优化。总迭代量从$N \times T$降到约$2/3 \times N \times T$
    - 设计动机：长时间优化的图像更抽象、与教师模型匹配更好；短时间优化的图像保留更多真实图像特征。两者组合覆盖了从简单到复杂的样本空间，增大类内多样性

2. **教师排名初始化（Teacher-Ranked Initialization）**:

    - 功能：提供有意义的初始化替代高斯噪声
    - 核心思路：用预训练教师模型计算每个原始图像patch的预测概率，选择中等概率（median）的patch作为初始化。中等难度的图像既不太简单（信息少）也不太难（优化不动），有最大的信息增强空间
    - 设计动机：EarlyLate中后加入的子批次只有少量迭代，好的初始化至关重要；实验证明median初始化比lowest/highest都好

3. **拼接训练（Concatenation Training）**:

    - 功能：让不同子批次共享GPU计算时间，提高效率
    - 核心思路：新加入的子批次与已有子批次拼接在一起，在同一个forward-backward中联合优化。这样不需要为每个子批次单独跑一遍完整训练流程，减少了IO和数据加载开销
    - 设计动机：朴素实现中M个子批次需要M次独立训练，拼接训练在一次扫描中完成所有子批次的优化

### 损失函数 / 训练策略
使用与CDA/SRe2L相同的BatchNorm分布正则化损失 + soft cross-entropy。默认配置：MI=4K迭代，RI=500，M=8个子批次（即第1批4000次，第2批3500次...第8批500次）。

## 实验关键数据

### 主实验

| 数据集 | IPC | 模型 | DELT | RDED | SRe2L | 提升(vs RDED) |
|--------|-----|------|------|------|-------|-------------|
| ImageNet-1K | 50 | ResNet-18 | 46.8% | 42.0% | 46.8% | +4.8% |
| ImageNet-1K | 50 | ResNet-101 | 66.1% | 61.2% | - | +4.9% |
| CIFAR-10 | 50 | ResNet-18 | 82.1% | 73.4% | - | +8.7% |
| CIFAR-10 | 50 | ResNet-101 | 85.2% | 63.5% | 66.0% | +19.2%(vs SRe2L) |
| Tiny-ImageNet | 50 | ResNet-18 | 28.3% | 25.6% | 21.3% | +2.7% |

### 消融实验

| 配置 | ImageNet-1K IPC50 (R18) | 说明 |
|------|------------------------|------|
| DELT (Full) | 46.8% | EarlyLate + 真实图像初始化 |
| Early-only (等量迭代) | 44.4% | 无EarlyLate，退化为CDA+初始化 |
| 无真实图像初始化 | 44.8% | 高斯噪声初始化 |
| Highest prob初始化 | 46.3% | 最简单样本不如median |
| Lowest prob初始化 | 46.0% | 最难样本不如median |
| MI=1K, RI=500 | 44.1% | 迭代太少 |
| MI=4K, RI=500 | 46.8% | 最佳配置 |

### 关键发现
- EarlyLate vs Early-only带来2.4%的关键提升，说明多样性确实是性能瓶颈
- 类内余弦相似度显著降低（>5%），验证了多样性提升不是虚指标
- 计算节省39.3%（ImageNet-1K从29h降到18.8h），同时性能还提升了
- 在持续学习场景中，DELT合成数据比G-VBSM平均高10%，多样性在此场景价值更大
- 不同架构（ResNet、MobileNet、EfficientNet、RegNet）上均保持优势，泛化性好

## 亮点与洞察
- **化繁为简的设计哲学**：不增加任何新模块或损失函数，仅通过改变图像加入优化的时间点就实现了多样性提升。这个"不同起点+不同迭代深度=不同优化结果"的思路极其简洁，可以零成本移植到任何batch-to-global蒸馏方法
- **多样性与效率的正相关**：通常提升多样性需要更多计算，但DELT通过减少后加入批次的迭代次数反而节省了39.3%计算，形成了性能-效率的双赢
- **跨任务验证**：在持续学习中10%的提升说明多样性对下游任务有切实的迁移价值

## 局限与展望
- RI和MI的超参数选择依赖经验，不同数据集可能需要不同配置
- 在极低IPC（如1或5）时EarlyLate的优势不显著，因为子批次太小
- 与G-VBSM的多模型策略正交但未充分探索两者组合
- 只在分类任务上验证，检测、分割等下游任务未涉及

## 相关工作与启发
- **vs SRe2L/CDA**: 同属batch-to-global匹配框架，DELT仅改变训练调度就在CDA基础上提升2-5%，说明多样性是该家族方法的核心瓶颈
- **vs RDED**: RDED用真实图像拼接不做训练，信息密度有限；DELT在patch上进行差异化优化，信息增强更充分
- **vs G-VBSM**: G-VBSM用多模型增多样性但框架复杂；DELT用单模型+时间差异化，更简洁高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 想法极其简洁但此前无人尝试，"early stopping用于增强蒸馏多样性"是很好的洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖CIFAR到ImageNet-1K、多架构、消融全面、下游任务验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 简单有效且即插即用，实际价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
