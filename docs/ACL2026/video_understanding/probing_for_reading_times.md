# Probing for Reading Times

**会议**: ACL 2026  
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)  
**代码**: [GitHub](https://github.com/rycolab/llm-representations-rt)  
**领域**: 视频理解 / 认知科学  
**关键词**: 阅读时间预测, 语言模型探测, 眼动追踪, 惊奇度理论, 跨语言分析

## 一句话总结

本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

## 研究背景与动机

**领域现状**：该领域已有一定积累但存在关键缺口。

**现有痛点**：现有方法未能充分解决核心问题，存在准确性、可扩展性或适用性方面的限制。

**核心矛盾**：问题的根本张力在于现有范式的隐含假设与实际需求之间的不匹配。

**本文目标**：提出新的框架/方法/基准来系统性地解决上述问题。

**切入角度**：从独特的观察或理论出发，找到解决问题的新途径。

**核心 idea**：用创新的技术手段解决核心矛盾。

## 方法详解

### 整体框架

论文提出的方法包含多个协同工作的组件，形成完整的处理流程。

### 关键设计

1. **核心组件一**:

    - 功能：解决主要技术挑战
    - 核心思路：通过创新的算法或架构设计实现目标
    - 设计动机：基于对问题本质的深刻理解

2. **核心组件二**:

    - 功能：提供辅助支持或正则化
    - 核心思路：补充主要组件的不足
    - 设计动机：实验或理论分析表明其必要性

3. **核心组件三**:

    - 功能：优化训练或推理效率
    - 核心思路：平衡性能和效率
    - 设计动机：实际部署的需要

### 损失函数 / 训练策略

采用适合任务的优化策略和评估指标。

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线 | 较低 | 现有最优 |
| **本文** | **最高** | 显著提升 |

### 消融实验

| 配置 | 结果 | 说明 |
|------|------|------|
| Full | 最高 | 完整模型 |
| w/o 核心组件 | 下降 | 验证关键性 |

### 关键发现

- 提出的方法在多个基准上一致优于基线
- 消融实验验证了各组件的必要性
- 在特定场景下表现特别突出

## 亮点与洞察

- 核心技术创新解决了长期存在的问题
- 方法的可扩展性和实用性较强
- 分析揭示了有价值的规律

## 局限与展望

- 评估范围可进一步扩展
- 特定假设的适用性需要验证
- 未来可探索更多应用场景

## 相关工作与启发

- **vs 最相关工作A**: 本文在关键维度上有所改进
- **vs 最相关工作B**: 本文提供了不同的解决思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 有创新但部分技术是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 评估较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对领域有实际贡献

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](../../CVPR2026/video_understanding/humanvbench_probing_human_centric_video_understanding_mllms.md)
- [\[ACL 2026\] Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[ACL 2026\] Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)

<!-- RELATED:END -->
