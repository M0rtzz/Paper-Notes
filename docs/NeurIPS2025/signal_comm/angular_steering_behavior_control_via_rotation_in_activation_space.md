---
title: >-
  [论文解读] Angular Steering: Behavior Control via Rotation in Activation Space
description: >-
  [NeurIPS 2025][信号/通信][激活引导] 提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例…
tags:
  - "NeurIPS 2025"
  - "信号/通信"
  - "激活引导"
  - "行为控制"
  - "旋转变换"
  - "拒绝引导"
  - "范数保持"
---

# Angular Steering: Behavior Control via Rotation in Activation Space

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.26243](https://arxiv.org/abs/2510.26243)  
**代码**: [https://github.com/lone17/angular-steering/](https://github.com/lone17/angular-steering/)  
**领域**: 信号通信  
**关键词**: 激活引导, 行为控制, 旋转变换, 拒绝引导, 范数保持

## 一句话总结

提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

## 研究背景与动机

激活引导（activation steering）是一种在LLM推理时修改内部表示来控制行为的方法，核心思想是：语言模型中的特征（如"拒绝"倾向）对应于激活空间中的近似正交方向。现有方法主要有两种：**激活加法**（$\mathbf{h}' = \mathbf{h} + \alpha \hat{\mathbf{d}}_\text{feat}$）通过加上缩放的特征向量来调整行为，但系数α的调参非常困难且脆弱；**方向消融**（$\mathbf{h}' = \mathbf{h} - \hat{\mathbf{d}}_\text{feat} \hat{\mathbf{d}}_\text{feat}^\top \mathbf{h}$）通过正交投影完全移除特征方向上的分量，但无法做部分抑制。

作者的关键洞察来自现代LLM（LLaMA 3, Qwen 2.5, Gemma 2）普遍使用的RMSNorm。RMSNorm先将激活映射到缩放单位球面 $\bar{\mathbf{h}} = \mathbf{h}/\text{RMS}(\mathbf{h}) \odot \mathbf{g}$，再通过固定向量$\mathbf{g}$做方向性缩放。这意味着激活的**方向而非幅度**才是核心表示单元。因此，**旋转**——作为唯一同时保持范数和连续可调的几何变换——是行为控制的自然选择。

更进一步，作者证明了在归一化前对激活做向量加法或正交投影，效果与在归一化后做旋转完全等价。也就是说，现有的激活加法是<180°旋转的特例，方向消融是恰好90°旋转的特例。Angular Steering统一并扩展了它们。

## 方法详解

### 整体框架

在每个Transformer block的归一化层后，将激活向量在预定义的2D子空间中旋转到指定角度。2D子空间由特征方向 $\hat{\mathbf{d}}_\text{feat}$ 和其第一主成分 $\hat{\mathbf{d}}_\text{PC0}$ 定义。旋转角度θ是唯一的控制参数：θ在0°附近=强拒绝，100°=间接回应，200°=直接顺从，300°=重定向。

### 关键设计

1. **Angular Steering核心操作**:
    - 功能：在固定2D子空间中做范数保持的旋转
    - 核心思路：给定正交基 $\{\mathbf{b}_1, \mathbf{b}_2\}$（由 $\hat{\mathbf{d}}_\text{feat}$ 和 $\hat{\mathbf{d}}_\text{PC0}$ 正交化得到），将激活的2D投影旋转到目标角度θ，其余维度不变：$\mathbf{h}_{\text{steered},\theta} = \mathbf{h} - \text{proj}_P(\mathbf{h}) + |\text{proj}_P(\mathbf{h})| \cdot [\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$
    - 实现优化：投影矩阵和 $[\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$ 可预计算，推理时只需一次投影+缩放+加法
    - 设计动机：旋转限制在2D内，正交补空间完全不受影响——最小化对其他特征的干扰

2. **Adaptive Angular Steering**:
    - 功能：只对与目标特征正向对齐的激活施加旋转
    - 核心思路：添加条件掩码 $\text{mask} = \max(0, \text{sign}(\text{proj}_{\hat{\mathbf{d}}_\text{feat}}(\mathbf{x})))$，仅当激活在特征方向上有正投影时才旋转
    - 设计动机：对比实验显示harmful和harmless样本的激活在特征方向上投影方向相反，因此只需旋转正向对齐（harmful）的激活，进一步减少对无关特征的干扰。对小模型（3B）尤其重要——非自适应版本在小模型上会导致不连贯输出

3. **特征方向自动提取**:
    - 功能：无需人工选择层或方向，自动确定最佳特征方向
    - 核心思路：在每个归一化层后提取对比数据集（AdvBench有害 vs Alpaca无害）的激活均值差作为候选方向（共M=2×层数个候选）。选择与其他候选方向平均余弦相似度最高的那个作为最终方向——因为高相似度意味着该方向在多层中稳定出现，是对真实特征方向的良好近似
    - 设计动机：避免手动选择的主观性和可能的次优选择

### 损失函数 / 训练策略

完全无需训练，推理时直接在归一化层后插入旋转操作。特征方向的提取只需要在对比数据集上做一次前向推理即可。

## 实验关键数据

### 主实验

**拒绝引导（refusal steering）**：

在Llama 3（3B/8B）、Qwen 2.5（3B/7B/14B）、Gemma 2（9B）上评估。θ每10°转一圈，clear的拒绝弧和顺从弧交替出现：

| 角度范围 | 行为 | 说明 |
|---------|------|------|
| ~0°-60° | 强拒绝 | substring matching refusal score ≈ 1.0 |
| ~60°-120° | 间接回应 | 开始松动但仍回避 |
| ~120°-240° | 直接顺从 | refusal score ≈ 0, harmful score高 |
| ~240°-360° | 重定向 | 既不拒绝也不直接回答，提供替代方案 |

**TinyBenchmarks性能保持**：

| 配置 | ARC | MMLU | WinoGrande | GSM8k | 整体趋势 |
|------|-----|------|------------|-------|---------|
| 无引导 | baseline | baseline | baseline | baseline | - |
| Angular (全circle) | **几乎不变** | **几乎不变** | **几乎不变** | 略波动 | 大部分角度保持baseline水平 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 随机平面旋转 | 无效 | 5/6模型几乎无行为变化 |
| Adaptive vs 非Adaptive | Adaptive更稳定 | 小模型（3B）上差异最大——非Adaptive导致不连贯 |
| 不同角度下的困惑度 | 拒绝区域PPL高于baseline | 对齐≈在分布之上"盖了一层"，而非真正改变底层能力 |

### 关键发现

- **安全对齐主要是表面修饰**：harmful生成的困惑度低于refusal生成，说明模型底层仍"记得"有害内容，安全对齐只是改变了前几个token的分布而非真正移除了有害知识
- 小模型（3B）更容易出现特征干扰——旋转时多个特征纠缠在2D子空间中，导致不连贯输出。Adaptive variant有效缓解
- Gemma-2-9B表现出最弱的steering效果，可能与其架构差异有关
- 特征方向在层间有高度一致性（余弦相似度高），支持"特征方向跨层稳定"的假说

## 亮点与洞察

- **几何统一视角极其优雅**：将"加法=部分旋转，消融=90°旋转"这一观察建立为严格的数学等价关系，把零散的activation engineering技术统一到一个框架下
- **方向>幅度**的核心洞察利用了RMSNorm的几何性质——归一化本质上已经让模型工作在单位球面上，幅度信息被抹掉了
- 提供了4类行为的连续谱（refusal→indirect→direct→redirect），让行为控制不再是非此即彼的开关，而是连续可调的旋钮
- Adaptive variant的设计简洁有效——一个符号掩码就大幅提升了小模型上的稳定性

## 局限与展望

- 依赖对比数据集提取特征方向，不同特征（如事实性、创造性等）需要不同的对比数据
- 启发式选择steering平面（PCA前两个主成分），不保证在所有行为和架构上最优
- 仅验证了拒绝和情感两种行为，多特征同时steering可能出现子空间冲突
- 2D子空间假设在高维空间中可能过于简化——某些复杂行为可能需要更高维的steering空间
- 依赖特征方向的线性假设（Superposition Hypothesis），如果特征以非线性方式编码则方法可能失效

## 相关工作与启发

- **vs Activation Addition (ActAdd)**：ActAdd等价于<180°的旋转，系数α的选择困难根源在于它实际上同时改变了旋转角度和幅度——Angular Steering将两者解耦，只调角度
- **vs Directional Ablation (RepE)**：方向消融等价于恰好90°的旋转。但90°可能不是最优角度，且消融无法利用负向投影的信息
- **vs Spectral Editing of Activations**：在PCA空间构造方向，Angular Steering更进一步定义了旋转操作
- **vs Householder Pseudo-Rotation**：类似的范数保持思路，但Householder限于反射变换，不如旋转灵活
- **启发**：RMSNorm的几何性质暗示现代LLM可能天然适合旋转操作——在对齐、安全、风格控制等场景都值得探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一几何框架非常优雅，将零散技术统一为旋转的特例
- 实验充分度: ⭐⭐⭐⭐ 3个LLM家族×多个尺寸+6个benchmark+定性/定量分析+消融
- 写作质量: ⭐⭐⭐⭐⭐ 几何直觉、理论证明、实验验证三者衔接流畅
- 价值: ⭐⭐⭐⭐⭐ 对LLM安全和可控性社区有重要方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[CVPR 2025\] Continuous Space-Time Video Resampling with Invertible Motion Steganography](../../CVPR2025/signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)

</div>

<!-- RELATED:END -->
