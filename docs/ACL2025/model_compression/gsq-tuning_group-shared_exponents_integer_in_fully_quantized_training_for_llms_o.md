---
title: >-
  [论文解读] GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning
description: >-
  [ACL 2025 (Findings)][模型压缩][全量化训练] GSQ-Tuning 提出了一种基于"组共享指数整数"（Group-Shared Exponents Integer）格式的全量化微调框架，在推理和训练中完全消除浮点运算…
tags:
  - "ACL 2025 (Findings)"
  - "模型压缩"
  - "全量化训练"
  - "整数微调"
  - "边端部署"
  - "共享指数"
  - "LoRA量化"
---

# GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning

**会议**: ACL 2025 (Findings)  
**arXiv**: [2502.12913](https://arxiv.org/abs/2502.12913)  
**代码**: 无  
**领域**: 模型压缩 / LLM效率  
**关键词**: 全量化训练, 整数微调, 边端部署, 共享指数, LoRA量化

## 一句话总结

GSQ-Tuning 提出了一种基于"组共享指数整数"（Group-Shared Exponents Integer）格式的全量化微调框架，在推理和训练中完全消除浮点运算，结合 LoRA 适配器实现了精度接近 BF16 微调、内存降低 1.85x、功耗降低 5x、芯片面积缩小 11x 的边端 LLM 微调方案。

## 研究背景与动机

**领域现状**：大语言模型的微调已成为适配下游任务的标准做法。参数高效微调（PEFT）方法如 LoRA 通过只训练少量参数降低了显存需求，量化方法（如 QLoRA）进一步通过低精度存储减少内存占用。然而，微调的前向和反向传播计算仍然依赖浮点运算（FP16/BF16）。

**现有痛点**：边缘设备（手机、IoT 设备、嵌入式系统）的硬件通常缺乏高效的浮点运算单元——浮点乘法器的功耗和芯片面积远大于整数乘法器。现有方法即使量化了模型权重存储，训练时仍需反量化回浮点进行计算，导致边缘设备上微调仍然不现实。此外，将敏感数据发送到云端进行微调存在隐私风险。

**核心矛盾**：PEFT 减少了训练参数但没有减少计算精度需求；推理量化减少了存储但训练仍需浮点；全量化训练的已有尝试（如 INT8 训练）在 LLM 上精度损失大。根本问题是：如何在保持精度的前提下，让训练全流程（前向+反向+优化器更新）都用整数运算完成？

**本文目标**：设计一种完全基于整数运算的 LLM 微调方案，推理和训练过程中不需要任何浮点操作，同时保持与 BF16 微调相当的精度。

**切入角度**：作者注意到浮点数的核心问题在于指数部分——每个数字都带有独立的指数，这导致乘法时需要浮点处理。如果一组参数共享同一个指数，那么组内的运算可以完全用整数完成，只需在组间处理指数的缩放。

**核心 idea**：提出 Group-Shared Exponents Integer（GSEI）数据格式——将参数分组，每组共享一个指数因子，组内参数用整数表示。这样，矩阵乘法中绝大部分计算变为整数乘加，只有少量组间缩放涉及指数操作，且可以预计算。

## 方法详解

### 整体框架

GSQ-Tuning 包含三个核心组件：(1) GSEI 数据格式——将模型参数表示为"组共享指数 + 整数尾数"的形式；(2) 全整数前向传播——用 GSEI 格式进行矩阵乘法，避免浮点运算；(3) 全整数反向传播——梯度计算和 LoRA 参数更新也在 GSEI 格式下完成。整个框架与 LoRA 类似的低秩适配器结合，实现参数高效+全量化的双重优化。

### 关键设计

1. **GSEI 数据格式（Group-Shared Exponents Integer）**:

    - 功能：以整数形式高效表示浮点参数，消除逐元素的浮点运算
    - 核心思路：将参数向量/矩阵按固定大小（如每 128 个元素）分组，每组提取一个共享的缩放因子（scale，可以看作共享指数），组内元素除以该缩放因子后四舍五入为整数。数学上，对于参数组 $\mathbf{x}$，表示为 $\mathbf{x} \approx s \cdot \mathbf{x}_{int}$，其中 $s$ 是组共享的缩放因子，$\mathbf{x}_{int}$ 是整数向量。在矩阵乘法 $\mathbf{Y} = \mathbf{A} \cdot \mathbf{B}$ 中，如果两个矩阵都用 GSEI 格式，则 $\mathbf{Y}_{ij} = s_A^{(i)} \cdot s_B^{(j)} \cdot \sum_k a_{int}^{(ik)} \cdot b_{int}^{(kj)}$，其中内层求和完全是整数运算，缩放因子的乘法可以在最外层一次性完成。
    - 设计动机：与标准量化（如 INT8）不同，GSEI 的核心创新在于"组共享指数"的设计使得指数处理可以从内层循环提到外层，将 $O(n^3)$ 的浮点乘法变为 $O(n^3)$ 的整数乘法 + $O(n^2)$ 的缩放操作，在硬件上效率极高。与 MX（Microscaling）格式相比，GSEI 的组大小更灵活，且专门针对训练场景优化。

2. **全整数前向/反向传播**:

    - 功能：在 GSEI 格式下完成完整的训练前向和反向传播
    - 核心思路：前向传播中，所有线性层的权重和激活值都存储为 GSEI 格式，矩阵乘法用上述整数+缩放方式完成。反向传播的梯度计算同样使用 GSEI 格式——梯度矩阵也被量化为 GSEI 形式，梯度与激活值的矩阵乘（用于计算参数梯度）和梯度与权重的矩阵乘（用于传播梯度）也全部用整数完成。关键挑战是梯度通常比权重和激活值的动态范围更大且更不稳定，作者通过自适应组大小和动态缩放因子更新来应对。
    - 设计动机：如果只在前向量化但反向仍用浮点，那么训练的计算瓶颈仍然存在。全流程量化是在边缘设备上实现可行微调的必要条件。

3. **GSEI-LoRA 联合设计**:

    - 功能：将 GSEI 量化与 LoRA 适配器无缝结合
    - 核心思路：预训练权重冻结并以 GSEI 格式存储，LoRA 的低秩矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 也以 GSEI 格式初始化。训练时只更新 LoRA 参数，更新的梯度和优化器状态也用 GSEI 格式。由于 LoRA 的参数量远小于主模型，优化器状态的内存开销进一步降低。整体内存节省来自两部分：权重的量化存储（FP16→GSEI 约 2x）+ 优化器状态的量化（FP32→GSEI 约 4x）。
    - 设计动机：LoRA 的低秩特性使得 GSEI 的整数近似误差在小矩阵上更可控——低秩分解本身就有一定的去噪/正则化效果，有助于抵消量化带来的精度损失。

### 损失函数 / 训练策略

使用标准的 LoRA 微调损失（如 cross-entropy），但所有计算在 GSEI 格式下完成。训练策略上采用渐进量化——初始阶段使用较大的组大小（精度更高但效率稍低），随着训练稳定后逐步减小组大小。学习率需要根据 GSEI 的量化步长适当调大以补偿量化梯度的信息损失。

## 实验关键数据

### 主实验

在多个 LLM（LLaMA-2 7B/13B、LLaMA-3 8B 等）上进行微调，对比 BF16 LoRA、INT8 LoRA、QLoRA 和 GSQ-Tuning：

| 方法 | 精度格式 | LLaMA-2 7B 平均精度 | 内存 (GB) | 相对 BF16 内存 |
|------|---------|-------------------|-----------|---------------|
| BF16 LoRA | FP16 权重 + FP32 优化器 | 基线 (100%) | ~14 | 1.0x |
| QLoRA (4bit) | INT4 权重 + FP16 计算 | -0.5~1% | ~8 | 0.57x |
| INT8 LoRA | INT8 全流程 | -2~3% | ~8 | 0.57x |
| GSQ-Tuning | GSEI 全流程 | -0.3~0.8% | ~7.5 | 0.54x (1.85x 节省) |

### 消融实验

| 配置 | 精度损失 | 说明 |
|------|---------|------|
| GSEI (组大小=128) | 最小 | 最佳精度-效率平衡点 |
| GSEI (组大小=64) | 可接受 | 更激进但仍然可用 |
| GSEI (组大小=256) | 很小 | 精度好但内存节省减少 |
| 不用动态缩放 | 较大 | 梯度动态范围大需要自适应 |
| 不用渐进量化 | 早期不稳定 | 初期大组大小有帮助 |
| GSEI vs MX 格式 | GSEI 更优 | 组共享指数比 MX 的块格式更灵活 |

硬件效率对比（GSEI INT8 vs FP8）：

| 指标 | GSEI INT8 | FP8 | 优势 |
|------|-----------|-----|------|
| 功耗 | 1x | 5x | GSEI 功耗降低 5 倍 |
| 芯片面积 | 1x | 11x | GSEI 面积缩小 11 倍 |
| 吞吐量 | 相当 | 相当 | 整数乘法器效率更高 |

### 关键发现

- **GSQ-Tuning 精度损失极小**：相比 BF16 LoRA 微调，精度差距通常在 1% 以内，远优于直接 INT8 量化训练的 2-3% 损失。核心原因是 GSEI 的组共享指数设计保留了精度同时实现了全整数计算。
- **内存节省主要来自优化器状态**：权重量化节省约 2x，但优化器状态（Adam 的一阶和二阶矩）从 FP32 量化到 GSEI 贡献了更大的节省。
- **组大小 128 是最佳点**：小于 128 精度开始下降，大于 128 节省减少。这与 MXFP 等格式的最佳组大小一致。
- **硬件优势极其显著**：INT8 乘法器的功耗和面积远小于 FP8 乘法器（5x 和 11x），这对电池供电的边缘设备是决定性优势。

## 亮点与洞察

- **"组共享指数"这个核心设计思路**非常优雅——将指数从内循环提到外循环，化浮点为整数，是一个简洁但有深度的工程洞察。这个思路可以推广到任何需要低功耗矩阵运算的场景。
- **对训练的全流程量化**（不只是推理）填补了从 QLoRA 到真正边端可用之间的 gap。QLoRA 仍需要浮点计算，而 GSQ-Tuning 真正实现了全整数训练。
- **功耗和面积的硬件级分析**非常实用——大多数量化论文只比较精度和内存，很少深入到芯片设计层面的能效分析，这对硬件+算法协同设计有很好的参考价值。

## 局限与展望

- **实验只在较小的 LLM 上验证**：7B/13B 模型的实验可能无法完全反映在 70B+ 大模型上的表现，大模型的量化敏感性可能不同。
- **缺少真实边缘设备上的部署验证**：论文的硬件分析基于理论计算（乘法器面积和功耗公式），没有在实际的边缘芯片（如 NPU、DSP）上测试端到端性能。
- **仅评估了 LoRA 微调**：对于全参数微调或其他 PEFT 方法（如 Prefix Tuning、Adapter），GSEI 的适用性未验证。
- **渐进量化策略依赖启发式**：组大小的调度策略需要手动设计，自动化的量化策略搜索可能更优。
- **改进方向**：结合 NPU 专用指令集进行 GSEI 格式的硬件加速；探索混合精度方案——对量化敏感层用较大组大小，不敏感层用更激进的量化。

## 相关工作与启发

- **vs QLoRA**: QLoRA 量化了权重存储但训练计算仍用浮点，GSQ-Tuning 更进一步实现了全整数训练，是概念上的代际升级。
- **vs INT8 Training (如 S2FP8 等)**: 直接 INT8 训练在 LLM 上精度损失大，GSQ-Tuning 通过组共享指数设计，在相同位宽下保持了更好的精度。
- **vs MXFP/Microscaling 格式**: MXFP 也使用共享指数但主要针对推理场景优化，GSQ-Tuning 专门为训练场景（含梯度和优化器状态）设计了量化策略。

## 评分

- 新颖性: ⭐⭐⭐⭐ 组共享指数设计巧妙，全整数训练是有意义的推进，但量化训练并非全新方向
- 实验充分度: ⭐⭐⭐ 多模型多任务验证和消融完整，但缺少真实硬件部署和大模型实验
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，硬件分析有深度，但公式较多需要细读
- 价值: ⭐⭐⭐⭐ 对边缘设备上的LLM部署有实际推动作用，硬件效率分析对芯片设计者尤其有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)

</div>

<!-- RELATED:END -->
