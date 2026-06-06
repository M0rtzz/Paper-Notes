---
title: >-
  [论文解读] Diffusion-CAM: Faithful Visual Explanations for dMLLMs
description: >-
  [ACL 2026][可解释性][扩散多模态模型] 提出 Diffusion-CAM，首个专为扩散式多模态大语言模型（dMLLM）设计的可解释性方法，通过在去噪轨迹中提取结构有效的中间表征并配合四个后处理模块（自适应核去噪、分布感知置信门控、上下文背景衰减、单实例因果去偏）…
tags:
  - "ACL 2026"
  - "可解释性"
  - "扩散多模态模型"
  - "类激活映射"
  - "视觉解释"
  - "可解释AI"
  - "并行生成"
---

# Diffusion-CAM: Faithful Visual Explanations for dMLLMs

**会议**: ACL 2026  
**arXiv**: [2604.11005](https://arxiv.org/abs/2604.11005)  
**代码**: [GitHub](https://github.com/ZzzzzZhhmm/Diffusion-CAM)  
**领域**: 图像复原  
**关键词**: 扩散多模态模型, 类激活映射, 视觉解释, 可解释AI, 并行生成

## 一句话总结
提出 Diffusion-CAM，首个专为扩散式多模态大语言模型（dMLLM）设计的可解释性方法，通过在去噪轨迹中提取结构有效的中间表征并配合四个后处理模块（自适应核去噪、分布感知置信门控、上下文背景衰减、单实例因果去偏），在 COCO Caption 和 GranDf 上显著超越自回归 CAM 基线。

## 研究背景与动机

**领域现状**：多模态 LLM 正从自回归架构（LLaVA、Qwen-VL）向扩散式架构（LaViDa、LLaDA-V、MMaDA）范式转变。扩散式模型通过并行掩码去噪生成整个句子，提升了生成速度和全局连贯性。

**现有痛点**：(1) 现有 CAM 方法（如 LLaVA-CAM、TAM）依赖自回归模型的顺序、注意力丰富的特性来追踪 token 生成——但 dMLLM 没有显式的 token 级注意力权重，也没有从左到右的因果结构；(2) 直接将传统 CAM 应用于 dMLLM 会产生弥散的、非特异性的热力图；(3) dMLLM 的并行去噪过程产生平滑、分布式的激活模式，与自回归的局部、顺序依赖性本质不同。

**核心矛盾**：dMLLM 的架构优势（并行生成、全局规划）恰恰是传统可解释性工具的障碍——后者假设顺序依赖但前者是并行的。

**本文目标**：设计首个适配扩散式多模态模型的视觉解释方法。

**切入角度**：在去噪轨迹中找到"结构有效"的中间步——图像条件化的空间信息仍被保留且可以通过梯度链接到最终预测。

**核心 idea**：从去噪过程的结构有效步提取梯度 CAM + 四个扩散特定的后处理模块解决空间噪声、背景弥散和冗余 token 相关等问题。

## 方法详解

### 整体框架
适配 CAM 到 dMLLM：(1) 在中间 transformer 块注册 hook 提取特征和梯度；(2) 动态定位图像 token 的位置边界；(3) 从最终响应分数反向传播到有效步的图像区域特征，生成基础 CAM；(4) 四个后处理模块精炼。

### 关键设计

1. **扩散式 CAM 适配（三步改造）**:

    - 功能：使 CAM 兼容非自回归的去噪生成过程
    - 核心思路：(1) **模型感知特征提取**：选择满足"可行性条件"的去噪步——该步的 hook 隐状态序列仍包含完整的图像 token 跨度。(2) **动态图像跨度定位**：从 info4cam 元数据解析图像 token 边界，提取图像特征并 reshape 为空间特征图。(3) **Grad-CAM 聚合**：对梯度空间平均得到通道权重，加权求和后 ReLU 得到基础 CAM
    - 设计动机：不假设固定的图像 token 位置或特定的去噪步，而是用通用的可行性判据自适应选择

2. **自适应核去噪模块**:

    - 功能：抑制 Transformer 自注意力的高频架构伪影
    - 核心思路：动态缩放滤波器核大小 $k_{\text{adaptive}}$，考虑三个因素：去噪步数（步数越多核越大）、空间方差（高噪声时核增大）、分辨率（保证尺度不变性）。使用秩加权高斯滤波——按激活值排序而非空间距离分配权重
    - 设计动机：固定核大小无法适配不同去噪步和图像内容的变化噪声特征

3. **分布感知置信门控 + 上下文背景衰减 + 单实例因果去偏**:

    - 功能：分别解决高方差激活伪影、背景残留信号、重复 token 异常高激活
    - 核心思路：置信门控根据全局统计量自适应确定阈值，高/低置信区域差异化处理；背景衰减用多尺度统计集成定义前景/背景分离边界；因果去偏用重复 token 检测和异常高激活掩码清除冗余响应
    - 设计动机：扩散模型的多步去噪积累了多种噪声源，需要针对性模块逐一解决

## 实验关键数据

### 主实验（COCO Caption + GranDf）

| 方法 | 定位准确率 | 背景抑制 | 视觉保真度 |
|------|---------|---------|---------|
| LLaVA-CAM | 基线 | 弱 | 弱 |
| Grad-CAM (直接应用) | 差 | 差 | 差 |
| **Diffusion-CAM** | **SOTA** | **SOTA** | **SOTA** |

### 消融实验

| 模块 | 贡献 |
|------|------|
| 自适应核去噪 | 抑制高频伪影，提升热力图平滑度 |
| 置信门控 | 区分语义区域和噪声 |
| 背景衰减 | 消除弥散背景响应 |
| 因果去偏 | 消除重复 token 引起的冗余激活 |
| **四模块联合** | **最优，各模块互补** |

### 关键发现
- **直接将自回归 CAM 应用于 dMLLM 完全失效**——产生弥散的、不可解释的热力图
- **四个后处理模块各解决一个特定问题，缺一不可**
- **去噪步的选择至关重要**：只有在结构有效的步才能提取有意义的视觉归因
- **Diffusion-CAM 在定位准确率和视觉保真度上显著超越所有基线**

## 亮点与洞察
- **首次揭示了 dMLLM 可解释性的根本挑战**：并行生成 vs 顺序依赖的冲突。随着扩散式架构的流行，这个问题会越来越重要
- **"结构有效步"的概念**提供了一个通用原则——在非自回归模型中，归因应从保留输入条件化空间信息的中间状态提取
- **四模块设计**虽然看似工程导向，但每个模块都有清晰的理论动机（噪声分析）

## 局限与展望
- 目前仅在 LaViDa 系列上验证，其他 dMLLM（如 LLaDA-V、MMaDA）的适配性待确认
- 四个模块的超参数（如 $\delta_\sigma$, $\delta_\mu$）需要根据模型调整
- 梯度回传路径在并行去噪中可能不唯一，归因的因果有效性需要更深入分析
- 计算开销比自回归 CAM 大（需要存储去噪中间状态）
- 未探索文本 token 级的归因（目前只做视觉区域归因）

## 相关工作与启发
- **vs LLaVA-CAM**: 专为自回归模型设计，直接用于 dMLLM 效果极差。Diffusion-CAM 是必要的替代
- **vs DAAM (Tang et al.)**: DAAM 做文生图扩散模型的归因，但目标和方法与多模态推理不同
- **vs 注意力可视化**: dMLLM 没有显式的自回归注意力权重，注意力方法不适用

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 dMLLM 可解释性方法，但核心思路（梯度 CAM + 后处理）不新
- 实验充分度: ⭐⭐⭐⭐ 两个基准+消融+对比，但 dMLLM 生态尚小
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，四模块设计有条理
- 价值: ⭐⭐⭐⭐ 随着 dMLLM 普及，这项工作的重要性会增长

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] QGShap: Quantum Acceleration for Faithful GNN Explanations](../../AAAI2026/interpretability/qgshap_quantum_acceleration_for_faithful_gnn_explanations.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[AAAI 2026\] Using Certifying Constraint Solvers for Generating Step-wise Explanations](../../AAAI2026/interpretability/using_certifying_constraint_solvers_for_generating_step-wise_explanations.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)

</div>

<!-- RELATED:END -->
