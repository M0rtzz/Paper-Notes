---
title: >-
  [论文解读] Envisioning the Future, One Step at a Time
description: >-
  [CVPR 2026][视频理解][开放集运动预测] 本文将开放集未来场景动态预测建模为稀疏点轨迹的逐步推理，通过自回归扩散模型实现从单张图像快速生成数千种多样化未来假设，速度比稠密模型快数个数量级。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "开放集运动预测"
  - "稀疏轨迹"
  - "自回归扩散模型"
  - "未来预测"
  - "世界模型"
---

# Envisioning the Future, One Step at a Time

**会议**: CVPR 2026  
**arXiv**: [2604.09527](https://arxiv.org/abs/2604.09527)  
**代码**: [http://compvis.github.io/myriad](http://compvis.github.io/myriad)  
**领域**: 视频理解/运动预测  
**关键词**: 开放集运动预测, 稀疏轨迹, 自回归扩散模型, 未来预测, 世界模型

## 一句话总结

本文将开放集未来场景动态预测建模为稀疏点轨迹的逐步推理，通过自回归扩散模型实现从单张图像快速生成数千种多样化未来假设，速度比稠密模型快数个数量级。

## 研究背景与动机

**领域现状**：大多数未来预测方法依赖密集视频或潜空间预测，将大量容量消耗在外观而非底层运动轨迹上，导致大规模未来假设探索成本极高。

**现有痛点**：(1) 稠密视频生成方法付出"视觉税"——必须渲染每个像素才能推理运动；(2) 单步预测方法无法处理多接触长时序场景；(3) 物理引擎方法无法泛化到开放集运动。

**核心矛盾**：真实世界的动态高度复杂和随机——需要考虑大量可能的未来，但稠密预测使这种探索在计算上不可行。

**本文目标**：在避免视觉税的前提下，实现开放集、逐步、可大规模采样的运动预测。

**切入角度**：类比人类认知——我们不"画"未来的图片，而是追踪重要的变化。利用稀疏性使预见未来成为可能。

**核心idea**：将运动预测建模为用户定义的稀疏点轨迹上的逐步自回归扩散过程。

## 方法详解

### 整体框架

给定单张参考帧和K个可见查询点，自回归生成每个时间步的增量运动。每一步是一个条件扩散模型，预测局部可预测的短程转移。模型通过因果分解在时间和轨迹维度上建模联合分布：$p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0, \mathcal{I}_0) = \prod_t \prod_i p_\theta(x_t^{(i)} | \mathbf{x}_t^{(<i)}, \mathbf{x}_{<t}, \mathcal{I}_0)$

### 关键设计

1. **Motion Token设计**:

    - 功能：为每个(时间,轨迹)对构建信息丰富的表征
    - 核心思路：融合三种信息——(1) 在原始位置 $x_0^{(i)}$ 采样的外观特征（"什么"）；(2) 在当前位置 $x_t^{(i)}$ 采样的局部上下文特征（"在哪"）；(3) Fourier编码的当前运动 $\Delta x_t^{(i)}$。加上随机单位球面方向的轨迹标识符 $id_{traj}^{(i)} \sim \mathcal{U}(\mathbb{S}^{d-1})$
    - 设计动机：随机ID避免模型过度依赖固定索引，可扩展到任意K；外观+上下文的双位置采样使token同时知道目标是什么和当前在哪

2. **Fast Reasoning Blocks**:

    - 功能：极大提升自回归推理的采样速度
    - 核心思路：采用并行Transformer块，将自注意力、交叉注意力和FFN合并为单个残差：$\mathbf{h} \leftarrow \mathbf{h} + SA(\mathbf{h}) + CA(\mathbf{h}, \mathbf{h}_{cross}) + FFN(\mathbf{h})$。共享预归一化、融合投影，图像token保持冻结（仅作为cross-attention的key-value），motion token因果地attend两个流
    - 设计动机：传统Transformer层的多次kernel launch是自回归推理的主要瓶颈，融合设计显著减少了launch次数

3. **Flow Matching后验参数化**:

    - 功能：高质量地建模每步运动的分布
    - 核心思路：使用条件Flow Matching对每步增量运动 $\Delta x_t^{(i)}$ 建模分布，自然处理多模态运动的不确定性。每步独立的去噪使长时序预测中的不确定性随时间自然增长
    - 设计动机：相比确定性回归，Flow Matching天然建模多模态性，关键是避免了模式平均问题

### 损失函数 / 训练策略

在多样化的野外视频上训练。使用Flow Matching的标准条件概率流匹配损失。KV缓存加速自回归推理。

## 实验关键数据

### 主实验

| 方法类型 | 预测精度 | 采样速度 | 多样性 |
|---------|---------|---------|--------|
| 稠密视频模型 | 高 | 极慢 | 低（成本限制） |
| 物理引擎方法 | 高（域内） | 中 | 低（域限制） |
| 本文方法 | 相当/更优 | 快数个数量级 | 高（数千假设） |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无轨迹ID | 性能显著下降 | 多轨迹设置必需 |
| 无Fast Reasoning | 速度大幅下降 | 融合块关键 |
| 单步预测 | 长时序退化 | 逐步推理必要 |
| 完整模型 | 最优 | 各组件协同 |

### 关键发现

- OWM基准上精度匹配或超越稠密模型，同时采样速度快数个数量级
- 随机轨迹ID对多轨迹建模至关重要——固定ID导致模型记忆索引而非学习动态
- 逐步推理使长时序预测中的不确定性自然增长，符合物理直觉

## 亮点与洞察

- **"不画世界，追踪运动"的哲学**：完全避免视觉税，将计算集中在真正重要的运动动态上
- **Fast Reasoning Blocks的工程创新**：融合投影+冻结图像token+前缀注意力的组合显著提升了吞吐量
- **OWM基准的引入**：为开放集运动预测提供了标准化的评估框架

## 局限与展望

- 稀疏点轨迹无法捕捉形变、旋转等连续体运动
- 自回归方式在极长时序上仍会累积误差
- 运动预测与场景理解之间的gap有待弥合

## 相关工作与启发

- **vs 视频世界模型**: 它们付出巨大的视觉税来预测每个像素，本文证明稀疏轨迹足以捕捉运动本质
- **vs 物理引擎方法**: 物理引擎限于闭集域，本文在开放集上通过数据驱动学习实现泛化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 稀疏轨迹+逐步自回归扩散的范式转变
- 实验充分度: ⭐⭐⭐⭐ OWM基准+多场景验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述精彩，类比深刻
- 价值: ⭐⭐⭐⭐⭐ 为未来预测开辟了高效且可扩展的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[ICLR 2026\] Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling](../../ICLR2026/video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[NeurIPS 2025\] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling](../../NeurIPS2025/video_understanding/prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)

</div>

<!-- RELATED:END -->
