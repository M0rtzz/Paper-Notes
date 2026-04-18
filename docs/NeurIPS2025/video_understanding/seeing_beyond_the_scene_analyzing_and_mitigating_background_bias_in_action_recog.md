---
title: >-
  [论文解读] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition
description: >-
  [视频理解] 系统分析了动作识别中背景偏差在分类模型、对比预训练模型（CLIP/SigLIP2）和视频大语言模型（VLLM）三类范式中的普遍存在，并提出两条缓解路径：分类模型通过双分支架构融合分割人体输入降低SBErr 3.78%，VLLM通过自动化prompt tuning降低SBErr 9.85%。
tags:
  - 视频理解
---

# Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition

**会议/期刊**: NeurIPS 2025 Workshop  
**作者**: Ellie Zhou, Jihoon Chung, Olga Russakovsky (Princeton University)  
**arXiv**: [2512.17953](https://arxiv.org/abs/2512.17953)  
**领域**: 视频理解 / 动作识别 / 偏差分析  

## 一句话总结

系统分析了动作识别中背景偏差在分类模型、对比预训练模型（CLIP/SigLIP2）和视频大语言模型（VLLM）三类范式中的普遍存在，并提出两条缓解路径：分类模型通过双分支架构融合分割人体输入降低SBErr 3.78%，VLLM通过自动化prompt tuning降低SBErr 9.85%。

## 研究背景与动机

**领域现状**：动作识别模型长期依赖背景场景线索（如雪地→滑雪）作为预测捷径，而非关注人体动作本身。过往研究主要聚焦于分类模型的背景偏差，对CLIP等对比预训练模型和VLLM的背景偏差缺乏系统性分析。

**现有痛点**：(1) CLIP/SigLIP2和VLLM的背景偏差尚未被充分研究；(2) 简单去除背景虽能降低偏差，但会大幅损害正常数据集上的准确率（Kinetics-50下降26.47%）；(3) 针对VLLM的架构修改成本过高，缺乏轻量级缓解手段。

**核心矛盾**：背景信息同时是有用的上下文（正常场景下辅助判断）和有害的偏差源（反事实场景下误导预测），如何在保留有用背景的同时抑制偏差是关键挑战。

**本文目标**：(1) 量化背景偏差在三类模型范式中的严重程度；(2) 为分类模型设计保留背景上下文的偏差缓解架构；(3) 为VLLM探索基于prompt的轻量缓解方案。

**切入角度**：从"人体分割"和"prompt设计"两个正交维度入手，分别针对分类模型和VLLM提出适配的缓解策略。

**核心 idea**：通过双分支架构融合原始视频与分割人体视频来平衡上下文利用与偏差抑制，同时通过自动化prompt tuning引导VLLM聚焦人体动作。

## 方法详解

### 整体框架

本文工作分为两大模块：**偏差分析** 和 **偏差缓解**。

- **偏差分析**：在HAT Action Swap（将A类动作的人体放到B类动作的背景上）和Mimetics（模拟动作、无匹配场景）两个benchmark上，测试分类模型（Slow-Only）、对比预训练模型（CLIP ViT-B/32、SigLIP2）和VLLM（InternVL3-8B/78B），通过SHAcc（正确预测人体动作的比例）和SBErr（错误预测为背景动作的比例）量化偏差程度。
- **偏差缓解**：分类模型侧设计双分支融合架构和自适应加权机制；VLLM侧设计手工prompt和自动化prompt tuning闭环。

### 关键设计

**1. 双分支融合架构（Dual-Branch Sum / Stack）**

- **功能**：在保留背景上下文的同时引入人体分割信息，平衡准确率与偏差
- **核心思路**：两路并行Slow-Only分别处理原始视频和分割人体视频，在Stage 2之后通过逐元素相加（Sum）或通道拼接（Stack）融合特征，再经Stage 3-4和Head输出预测。早期层各自学习低层特征，融合后学习人体+上下文的联合表示
- **设计动机**：纯分割输入虽然将SBErr从23.42%降到2.09%，但Kinetics准确率暴跌26.47%。双分支架构在提升Kinetics准确率2.22%的同时降低SBErr 3.06%~3.62%

**2. 自适应加权聚焦机制（Weighted Focus）**

- **功能**：让模型自适应地控制人体区域与背景区域的权重
- **核心思路**：引入辅助3D CNN从Slow-Only早期特征图学习标量参数α（约束在[-1,1]），构造加权掩码 $M_{weighted} = (1+\alpha) \cdot M + (1-\alpha) \cdot (1-M)$，乘以特征图后继续前传。α=1时人体权重2×/背景0×，α=-1时反之
- **设计动机**：比固定融合更灵活，模型可根据具体样本自适应调整人体/背景的重要性。实现了Kinetics +2.10%、SBErr -3.78%的最佳平衡

**3. 自动化Prompt Tuning（Automated Prompt Engineering）**

- **功能**：系统化优化VLLM的输入prompt以减少背景偏差
- **核心思路**：以GPT-4.1作为prompt工程师，在20轮迭代中：(1) GPT设计prompt → (2) 在GPT-4o-mini + HAT数据集上评估SHAcc/SBErr → (3) 将结果反馈给GPT → (4) GPT据此优化下一轮prompt。使用75%数据评估、25%数据做prompt tuning
- **设计动机**：手工human-focused prompt仅降低SBErr 4.75%，而自动化tuning可降低9.85%，说明prompt空间中存在更优解，但人工难以发现

### 损失函数 / 训练策略

- 分类模型均基于Slow-Only (R50)从头训练300 epochs，Adam优化器，lr=0.001，ReduceLROnPlateau（patience=40），batch size=20
- 人体分割pipeline：YOLOv5检测人体框 → SAM2跨帧传播分割掩码 → 非人体区域置0
- 数据增强策略：将Kinetics-50视频的人体分割后叠加Places365随机背景，打破场景-动作关联。训练数据翻倍至49336，代价是Kinetics准确率下降但HAT/Mimetics提升

## 实验关键数据

### 主实验

分类模型缓解效果（括号内为相对Slow-Only baseline的变化）：

| 模型 | Kinetics-50 ↑ | HAT SHAcc ↑ | HAT SBErr ↓ | Mimetics ↑ |
|------|--------------|-------------|-------------|------------|
| Slow-Only | 49.93 | 9.62 | 23.42 | 6.87 |
| Segmented | 23.46 (-26.47) | 23.34 (+13.72) | 2.09 (-21.33) | 9.54 (+2.67) |
| Dual-Branch Sum | 52.15 (+2.22) | 12.76 (+3.14) | 20.36 (-3.06) | 7.85 (+0.98) |
| Dual-Branch Stack | 51.51 (+1.58) | 12.80 (+3.18) | 19.80 (-3.62) | 8.28 (+1.41) |
| Weighted-Focus | 52.03 (+2.10) | 12.80 (+3.18) | **19.64 (-3.78)** | 7.85 (+0.98) |

不同模型范式的背景偏差对比：

| 模型 | HAT SHAcc ↑ | HAT SBErr ↓ | Mimetics ↑ |
|------|-------------|-------------|------------|
| Slow-Only | 35.81 | 55.41 | 57.64 |
| CLIP ViT-B/32 | 29.25 | 53.66 | 46.84 |
| SigLIP2 | 25.46 | 58.91 | 48.95 |
| InternVL3-8B | 40.29 | 48.84 | 62.83 |
| InternVL3-78B | 45.73 | 48.39 | 66.61 |

### 消融实验

VLLM prompt策略对比（GPT-4o-mini）：

| Prompt策略 | SHAcc ↑ | SBErr ↓ | SBErr变化 |
|-----------|---------|---------|----------|
| Neutral baseline | 39.14 | 51.40 | - |
| Prefixed-choices | 33.93 | 46.65 | -4.75 |
| Human-focused (手工) | 40.92 | 46.99 | -4.41 |
| Background-focused (手工) | 35.82 | 52.95 | +1.55 |
| 最佳自动化prompt | 46.70 | 41.55 | **-9.85** |

### 关键发现

1. **背景偏差具有普遍性**：分类模型、CLIP/SigLIP2、VLLM三类范式都表现出显著的背景偏差，但VLLM相对最轻
2. **增大模型容量无法解决偏差**：InternVL3从8B增到78B，SHAcc提升但SBErr几乎不变——更大的模型学到了更多特征但没有学会抑制背景捷径
3. **时序信息有效**：增加输入帧数可同时提升SHAcc和降低SBErr，说明人体运动的时序信息是对抗背景偏差的关键
4. **类别级偏差与场景独特性强相关**：背景视觉独特且与动作高度关联的类别（如"天气预报"SBErr=89.09%）偏差极重，反之（如"拉小提琴"SBErr=0%）几乎无偏差
5. **数据增强是双刃剑**：Places365背景替换增强可大幅降低HAT SBErr（Weighted-Focus从19.64%→1.85%），但以Kinetics准确率下降为代价

## 亮点与洞察

1. **系统性跨范式分析**：首次在同一框架下比较分类模型、对比预训练模型和VLLM的背景偏差，揭示了偏差的普遍性和程度差异
2. **准确率-偏差trade-off的量化**：清晰展示了"去背景降偏差"与"保背景保准确率"之间的权衡，双分支架构在两个方向上同时取得正向收益
3. **自动化prompt tuning的有效性**：证明VLLM对prompt措辞高度敏感，自动化搜索比人工设计能找到显著更优的prompt（SBErr差距5.1%），为VLLM偏差缓解提供了低成本路径
4. **"增大模型≠减少偏差"的反直觉发现**：大模型容量提升了能力但没有减少背景依赖，暗示偏差是数据驱动而非容量受限的问题

## 局限与展望

1. **模型覆盖有限**：仅测试了少量代表性模型（Slow-Only、CLIP、InternVL3），结论是否泛化到更多架构（如TimeSFormer、VideoMAE）需验证
2. **偏差benchmark的局限**：HAT Action Swap是合成数据集，人体-背景拼接可能引入分布外伪影，与真实世界的背景偏差模式可能存在差异
3. **分类模型缓解效果有限**：双分支最佳方案仅降低SBErr 3.78%（从23.42%到19.64%），偏差仍然较重
4. **自动化prompt tuning不稳定**：20轮迭代中性能并非单调提升，后续prompt不一定优于前序，缺乏收敛保证
5. **可考虑针对VLLM的微调策略**：如在反事实数据上做instruction tuning，可能比纯prompt engineering更有效

## 相关工作与启发

- **HAT / Chung et al.**：提供了HAT Action Swap benchmark和背景偏差的早期分析，本文在此基础上扩展到更多模型范式
- **MASH-VLM (Bae et al.)**：分析VLLM的场景偏差，但仅限于VLLM且未量化人体vs背景的依赖程度
- **SlowFast / Slow-Only**：本文的分类模型baseline，双分支架构基于其backbone设计
- **LLM-as-optimizer (Yang et al., Gavrikov et al.)**：自动化prompt工程的方法论来源，本文将其应用于偏差缓解场景
- **启发**：可将双分支思路推广到视频grounding/captioning等下游任务；自动化prompt tuning框架可泛化到其他类型的VLLM偏差（如性别偏差、种族偏差）

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐ | 跨范式分析视角新颖，但缓解方法（双分支、prompt tuning）本身相对常规 |
| 技术深度 | ⭐⭐⭐ | 分析全面系统，但方法设计偏简单，缺乏理论深度 |
| 实验充分性 | ⭐⭐⭐ | 消融实验和跨模型对比详尽，但模型覆盖面和benchmark多样性有限 |
| 实用价值 | ⭐⭐⭐⭐ | 自动化prompt tuning方案零成本可用、双分支架构即插即用，对实际部署有参考价值 |
