---
title: >-
  [论文解读] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation
description: >-
  [ICLR 2026][人体理解][个性化生成] FlyThinker 提出了一种高效的 "think-while-generating" 框架，使用独立的推理模型(Reasoner)在 token 级别并行生成潜在推理信号，动态融入生成模型(Generator)以指导个性化长文本生成，同时保持训练和推理效率。
tags:
  - ICLR 2026
  - 人体理解
  - 个性化生成
  - 长文本生成
  - 潜在推理
  - think-while-generating
  - 并行推理
---

# Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation

**会议**: ICLR 2026  
**arXiv**: [2512.06690](https://arxiv.org/abs/2512.06690)  
**代码**: 无  
**领域**: 人体理解/NLP  
**关键词**: 个性化生成, 长文本生成, 潜在推理, think-while-generating, 并行推理

## 一句话总结

FlyThinker 提出了一种高效的 "think-while-generating" 框架，使用独立的推理模型(Reasoner)在 token 级别并行生成潜在推理信号，动态融入生成模型(Generator)以指导个性化长文本生成，同时保持训练和推理效率。

## 研究背景与动机

偏好对齐使 LLM 更好地反映人类期望，但现有方法主要优化群体级偏好，忽视个体用户需求。个性化长文本生成面临三大挑战：

**隐式偏好难以推理**: 用户兴趣通常隐含在历史行为中，简单的 prompt 定制或微调难以有效推理这些偏好

**"Think-then-generate"的局限**: 现有推理方法在生成前一次性完成所有推理，产生静态分析。对长文本而言，这种一次性推理需要覆盖完整响应的所有信息，学习困难且无法适应内容动态演变

**效率瓶颈**: 交替推理-生成的 "think-while-generating" 范式虽然直觉合理，但频繁推理会显著增加训练和推理时间

FlyThinker 的核心洞察：通过将推理与生成解耦到两个独立模型，并打破推理 token 之间的直接序列依赖，实现推理与生成的真正并行化。

## 方法详解

### 整体框架

FlyThinker 包含两个并行模型：
- **Reasoner (R)**: 在每个生成步骤产生一个潜在推理 token，输入为查询和之前已生成的响应
- **Generator (G)**: 增强版 LLM，将潜在推理信号融合到 token 预测中

核心公式化：

$$\text{Generation:} \quad (h,x) \xrightarrow{(h,x;\hat{y}_{<1}+r_{<1})} \hat{y}_1 \xrightarrow{(h,x;\hat{y}_{<2}+r_{<2})} \hat{y}_2 \dots$$

$$\text{Reasoning:} \quad (h,x) \xrightarrow{(h,x,\hat{y}_{<1})} r_1 \xrightarrow{(h,x,\hat{y}_{<2})} r_2 \dots$$

### 关键设计

1. **潜在推理 token 生成 (Reasoner)**: Reasoner 在每步 $t$ 从其最后一层隐状态提取潜在推理：$r_t = R_\theta^{(-1)}(h,x; \hat{y}_{<t-1})[-1]$。关键：$r_t$ **不依赖之前的推理** $r_{<t}$，只依赖已生成的响应 $\hat{y}_{<t-1}$，打破了推理 token 间的序列依赖。

2. **推理信号融合 (Generator)**: Generator 通过加法融合将推理注入 token 嵌入空间：$f(\hat{y}_{<t}, r_{<t}) = [e(y_1) + \lambda r_1, \dots, e(y_{t-1}) + \lambda r_{t-1}]$，其中 $\lambda$ 控制推理信号强度。

3. **并行训练**: 由于 $r_t$ 不依赖 $r_{<t}$，训练时可将完整目标序列 $y$ 一次性输入 Reasoner，单次前向传播获得所有位置的推理 token $r^\star = [r_1, \dots, r_T]$。然后 Generator 也可并行计算所有位置的预测。与标准 LLM 训练效率接近。

4. **并行推理**: 推理时，Generator 预测当前 token 的同时，Reasoner 并行准备下一步的推理 token。交错(staggered)设计消除等待时间，推理延迟接近标准非推理 LLM。

### 损失函数 / 训练策略

使用标准的下一 token 预测损失联合优化 Reasoner 和 Generator：

$$\mathcal{L} = -\sum_{(h,x,y) \in \mathbb{D}} \sum_{t=1}^{|y|} \log P(\hat{Y}_t = y_t \mid h,x, y_{<t})$$

无需外部推理标注或辅助目标，Reasoner 在端到端训练中自然学习生成有用的推理信号。

## 实验关键数据

### 主实验

**LongLaMP Benchmark（Qwen2.5-3B-Instruct 骨干）:**

| 方法 | Product Review (BLEU) | Abstract Gen. (BLEU) | Topic Writing (BLEU) |
|------|---------------------|---------------------|---------------------|
| Non-pers | 1.54 | 4.58 | 1.12 |
| RAG | 3.30 | 3.40 | 1.43 |
| SFT | 3.91 | 5.82 | 3.89 |
| CoT | 3.37 | 5.85 | 3.00 |
| Coconut | 3.32 | 5.24 | 3.07 |
| **FlyThinker** | **4.36** | **6.34** | **4.06** |

FlyThinker 在所有任务上均超越基线，BLEU 相比 SFT 提升约10%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Reasoner 3B→1.5B | 性能基本持平 | 中等缩减不影响质量，训练更高效 |
| Reasoner 3B→0.5B | ROUGE-L/BLEU 明显下降 | 过小Reasoner容量不足 |
| $\lambda$=0 (无推理) | 退化为SFT | 推理信号不可或缺 |
| $\lambda \in [0.2, 2.0]$ | 均高于SFT | 方法对$\lambda$选择鲁棒 |
| $\lambda$=5 (过大) | 性能下降 | 过强推理信号干扰生成 |

### 关键发现

- **位置敏感评估**: 所有基线方法在后段（100-300 token）的个性化质量显著下降（"上下文漂移"），FlyThinker 在后段仍保持高质量，有效缓解了长文本生成中的偏好遗忘问题
- **训练效率**: FlyThinker 训练时间仅略高于 SFT，远低于 CoT 和 Coconut
- **推理效率**: 推理延迟接近 SFT，远快于 CoT 和 Coconut 的顺序推理
- **Reasoner 可缩小到1.5B而不损失质量**: 提供了有利的成本-性能权衡

## 亮点与洞察

1. **"边想边写"范式的首次高效实现**: 之前的 think-while-generating 概念因效率问题难以落地，FlyThinker 通过独立 Reasoner + 打破序列依赖巧妙解决
2. **类人的长文创作模式**: 人类在写长文时也是"写到哪想到哪"，FlyThinker 的 token 级动态推理与此天然吻合
3. **工程上非常优雅**: 训练时一次前向传播完成所有推理，推理时交错并行，几乎无额外开销
4. **对"上下文漂移"的有效对策**: 位置敏感实验清晰展示了动态推理对长文本后段质量的显著改善

## 局限与展望

1. **内存开销增加**: 虽然时间效率高，但需同时维护两个模型（Reasoner + Generator），内存占用翻倍
2. **评估指标有限**: 仅使用 ROUGE/BLEU/METEOR 等自动指标，缺少人工评估和 GPT-based 评价
3. **任务范围有限**: 仅在个性化长文本生成上验证，未探索其他需要动态推理的任务
4. **推理内容不可解释**: 潜在推理 token 是隐状态向量，无法检查推理逻辑是否合理

## 相关工作与启发

- **Coconut** (ICLR 2025) 在潜在空间做推理但采用 think-then-generate 模式，FlyThinker 扩展为 think-while-generating
- **REST-PG** 和 **R2P** 使用显式推理链做个性化，但效率低
- **LongLaMP** 基准提出了个性化长文本生成的系统评估框架
- 对**推理增强生成**领域有重要启发：推理和生成不必串行，可以并行化

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ think-while-generating 的高效实现，打破推理序列依赖的设计很精妙
- **实验充分度**: ⭐⭐⭐⭐ 三个任务 + 效率分析 + 位置敏感评估 + 消融，但缺人工评估
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰，三种范式对比直观，公式简洁
- **价值**: ⭐⭐⭐⭐ 对个性化生成和推理增强方法都有启发，但应用范围待扩展

<!-- RELATED:START -->

## 相关论文

- [AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations](autofigure_generating_and_refining_publication-ready_scientific_illustrations.md)
- [P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](../../ICML2025/human_understanding/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)

<!-- RELATED:END -->
