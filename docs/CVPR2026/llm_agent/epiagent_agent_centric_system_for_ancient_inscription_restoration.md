---
title: >-
  [论文解读] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration
description: >-
  [CVPR 2026][LLM Agent][古代铭文修复] EpiAgent是首个面向古代铭文修复的Agent系统，通过LLM中央规划器协调多模态分析、专用修复工具和迭代自我优化，在文字真实性和视觉保真度上超越现有方法。
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "古代铭文修复"
  - "多模态分析"
  - "迭代优化"
  - "文化遗产保护"
---

# EpiAgent: An Agent-Centric System for Ancient Inscription Restoration

**会议**: CVPR 2026  
**arXiv**: [2604.09367](https://arxiv.org/abs/2604.09367)  
**代码**: [https://github.com/blackprotoss/EpiAgent](https://github.com/blackprotoss/EpiAgent)  
**领域**: LLM Agent/数字人文  
**关键词**: 古代铭文修复, LLM Agent, 多模态分析, 迭代优化, 文化遗产保护

## 一句话总结

EpiAgent是首个面向古代铭文修复的Agent系统，通过LLM中央规划器协调多模态分析、专用修复工具和迭代自我优化，在文字真实性和视觉保真度上超越现有方法。

## 研究背景与动机

**领域现状**：AI驱动的古代文字修复已有进展，但现有方法要么局限于单字符级修复，要么使用固定流水线进行全铭文修复，无法处理异构退化模式。

**现有痛点**：(1) 基于图像到图像翻译的方法常扭曲原始字形，导致过度/不足修复；(2) 固定流水线缺乏对异构退化模式的适应性；(3) 铭文修复需要同时保证文字真实性和视觉保真度的双重要求。

**核心矛盾**：铭文修复不是简单的图像增强，而是需要像人类铭文学家那样协调多模态分析、专业技能判断和审美评价的复杂认知过程。

**本文目标**：构建一个模仿人类铭文学家工作流程的Agent系统，实现灵活自适应的铭文修复。

**切入角度**：将铭文修复形式化为分层规划问题，由LLM中央规划器在"观察-构思-执行-再评估"循环中驱动。

**核心idea**：用Agent架构替代固定流水线，使修复过程能根据退化模式动态调整工具选择和执行顺序。

## 方法详解

### 整体框架

EpiAgent遵循Observe-Conceive-Execute-Reevaluate四阶段循环：Observe阶段收集多模态信息（MLLM感知+校正语言模型+退化评估）；Conceive阶段基于历史经验制定修复计划；Execute阶段调用专用工具组合执行；Reevaluate阶段通过自动指标和可选专家反馈评估并迭代。

### 关键设计

1. **Observe阶段的多模态融合分析**:

    - 功能：建立对铭文退化状态的全面评估记录
    - 核心思路：分两步——(1) MLLM产生初始布局和文字假设；(2) 校正语言模型（CLM，基于RAG查询大规模中文语料库）纠正文字识别，布局校正模块预测完整布局（包括完全缺失区域），退化评估模型生成像素级退化分割掩码和严重程度等级
    - 设计动机：铭文退化涉及空间变化、结构耦合和多尺度特征，需要多模态协同才能全面评估

2. **Conceive阶段的经验驱动规划**:

    - 功能：将退化评估转化为可执行的修复计划
    - 核心思路：从历史执行日志中提取统计先验 $p(f|\mathcal{S}_d)$，将退化模式映射到工具效用分布。规划器 $\pi$ 基于观察记录 $T_r$ 和经验先验 $T_e$ 为每个字符生成独立的动作序列 $P_c = (f_1^{(c)}, ..., f_{N_c}^{(c)})$
    - 设计动机：不同退化模式需要不同的工具组合和执行顺序，经验先验使规划更加高效精准

3. **Reevaluate阶段的多角度评估**:

    - 功能：闭环评估和迭代优化
    - 核心思路：从像素质量（PSNR/SSIM/LPIPS）、文字识别（Top-1/Top-5准确率）和端到端性能（1-NED）三个维度评估修复结果，可选引入第三方专家反馈。评估结果驱动下一轮迭代的重规划
    - 设计动机：铭文修复的质量不仅取决于视觉质量，还取决于文字准确性和审美一致性

### 损失函数 / 训练策略

CLM通过微调7B LLM并配合RAG实现文字校正。退化评估模型进行像素级退化分割训练。整体系统是推理时的Agent编排，不需要端到端训练。

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Top-1 Acc↑ | 1-NED↑ |
|------|-------|-------|--------|------------|--------|
| CharFormer | 19.74 | 0.9503 | 0.0478 | 0.9109 | 0.8313 |
| DocDiff | 20.61 | 0.9565 | 0.0361 | 0.9275 | 0.8439 |
| MambaIR | 21.10 | 0.9599 | 0.0377 | 0.9093 | 0.8251 |
| IR3 | 21.15 | 0.9540 | 0.0388 | 0.9626 | 0.8855 |
| EpiAgent | **22.14** | **0.9684** | **0.0254** | **0.9889** | **0.9069** |
| 完整原件 | - | - | - | 0.9971 | 0.9120 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无CLM校正 | 识别准确率下降 | 文字指导不准确 |
| 无经验先验 | 修复质量下降 | 工具选择不优化 |
| 无迭代优化 | 质量次优 | 单次修复不充分 |
| 完整EpiAgent | 最优 | 四阶段闭环协同 |

### 关键发现

- EpiAgent的识别准确率(0.9889)接近完整原件(0.9971)，说明修复后的文字几乎完全可读
- 在真实退化铭文上的泛化能力显著优于固定流水线方法
- Agent的迭代优化机制在复杂耦合退化场景下特别有效

## 亮点与洞察

- **Agent范式在文化遗产保护中的开创性应用**：将LLM Agent从通用任务引入到高度专业化的铭文学领域，是数字人文的重要突破
- **可选专家反馈的闭环设计**：系统支持人类专家在评估阶段介入，实现了人机协作的修复工作流
- **字符级精细规划**：不同于全图一刀切的处理，EpiAgent对每个字符独立规划修复策略，处理空间耦合退化

## 局限与展望

- LLM推理的计算开销大，单张铭文的修复可能需要数分钟
- 高度依赖CLM的文字校正质量，在极度退化的铭文上可能失效
- 仅针对中文古代铭文验证，扩展到其他文字体系需要额外工作

## 相关工作与启发

- **vs IR3**: IR3使用全局-局部框架进行全铭文修复但存在错误传播，EpiAgent的Agent架构天然支持错误修正
- **vs AutoHDR**: AutoHDR使用LLM预测损坏内容但风格迁移可能扭曲字形，EpiAgent通过专用工具保持书法真实性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Agent范式在文化遗产保护中的首次应用
- 实验充分度: ⭐⭐⭐⭐ 在合成和真实退化数据上全面评估
- 写作质量: ⭐⭐⭐⭐ 工作流描述清晰
- 价值: ⭐⭐⭐⭐ 对数字人文领域有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/llm_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_agent/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[AAAI 2026\] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](../../AAAI2026/llm_agent/aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)

</div>

<!-- RELATED:END -->
