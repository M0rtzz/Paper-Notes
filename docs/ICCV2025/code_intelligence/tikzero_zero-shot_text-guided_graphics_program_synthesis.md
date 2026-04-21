---
title: >-
  [论文解读] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis
description: >-
  [ICCV 2025 (Highlight)][图形程序合成] 提出 TikZero，通过将图像表示作为中间桥梁，将图形程序生成与文本理解解耦，实现零样本文本引导的 TikZ 图形程序合成，在无需文本对齐训练数据的情况下大幅超越基线方法，经端到端微调后的 TikZero+ 达到甚至超越 GPT-4o 等大型商业模型的性能。
tags:
  - ICCV 2025 (Highlight)
  - 图形程序合成
  - TikZ
  - 零样本
  - 文本引导
  - 多模态语言模型
  - 图像桥接
---

# TikZero: Zero-Shot Text-Guided Graphics Program Synthesis

**会议**: ICCV 2025 (Highlight)  
**arXiv**: [2503.11509](https://arxiv.org/abs/2503.11509)  
**代码**: [potamides/DeTikZify](https://github.com/potamides/DeTikZify)  
**领域**: nlp_generation  
**关键词**: 图形程序合成, TikZ, 零样本, 文本引导, 多模态语言模型, 图像桥接

## 一句话总结

提出 TikZero，通过将图像表示作为中间桥梁，将图形程序生成与文本理解解耦，实现零样本文本引导的 TikZ 图形程序合成，在无需文本对齐训练数据的情况下大幅超越基线方法，经端到端微调后的 TikZero+ 达到甚至超越 GPT-4o 等大型商业模型的性能。

## 研究背景与动机

**从文本描述自动生成科学图表** 是一项极具吸引力的能力。生成的图表需要具备高几何精度和可编辑性，这要求将图表表示为 TikZ 等图形编程语言的程序，而非栅格图像。然而，当前该领域面临一个核心数据瓶颈：

**对齐训练数据极度稀缺**：理想的训练数据应该是"文本描述 + 对应 TikZ 程序"的配对，但这种数据非常难以大规模获取。手动为 TikZ 代码编写描述成本过高，现有数据集规模很小

**非对齐数据充裕但难以利用**：大量无标注的 TikZ 程序（如从 arXiv 论文中提取）和大量带描述的栅格图像（如自然图像数据集）分别存在，但它们之间缺乏对应关系，传统端到端方法无法直接利用

**现有方法的局限**：
   - **端到端模型**（如 DeTikZify）需要文本-程序对齐数据进行训练，受限于数据规模
   - **通用大模型**（如 GPT-4o）虽然能力强大，但参数量巨大、推理成本高昂，且不专门针对图形程序合成任务优化

**核心洞察**：文本到图形程序的过程可以分解为两步——文本到图像表示，再从图像表示到图形程序。既然"图像→TikZ 代码"和"文本→图像"两个方向都有大量独立数据可供训练，何不将它们通过图像表示这个中间桥梁连接起来？

## 方法详解

### 核心思路：图像表示作为桥梁

TikZero 的核心创新是 **解耦（decoupling）** 策略：将"文本→图形程序"任务分解为两个独立可训练的子任务：

- **图像→图形程序**：在大量无标注的 TikZ 程序上训练（将编译后的渲染图像作为输入条件）
- **文本→图像表示**：在大量带描述的栅格图像上训练（学习将文本映射到图像嵌入空间）

推理时将两者串联，文本描述先映射到图像嵌入空间，再由图形程序生成模型输出 TikZ 代码，从而实现 **零样本** 文本引导的图形程序合成。

### 架构设计

TikZero 建立在 DeTikZifyv2 (8B) 之上，后者是基于 Idefics3/LLaMA3 架构的多模态语言模型，专门用于图像到 TikZ 代码的生成。TikZero 在此基础上引入了以下关键组件：

#### 1. 图像编码器与图形程序解码器（来自 DeTikZifyv2）

DeTikZifyv2 的核心管线：

- **视觉编码器**：将输入图像编码为视觉 token 序列
- **跨模态投影**：将视觉 token 投影到语言模型的嵌入空间
- **LLM 解码器**：基于 LLaMA3-8B，自回归地生成 TikZ 代码 token

该模型在 DaTikZv2/v3 数据集（从 arXiv 提取的大量 TikZ 程序及其编译渲染图像）上训练，具备强大的图像到代码能力。

#### 2. 文本-图像适配器（TikZero 核心贡献）

TikZero 的关键创新在于引入一个 **适配器模块（adapter）**，将文本嵌入映射到 DeTikZifyv2 的图像嵌入空间。架构灵感来源于 Flamingo 和 LLaMA 3.2-Vision 中的跨注意力机制：

- **文本编码器**：使用独立的语言模型（如 LLaMA 3.2-1B）编码文本描述
- **跨注意力层**：将文本嵌入通过交叉注意力机制映射到图像嵌入空间，使得文本产生的嵌入在分布上接近真实图像的嵌入
- **训练目标**：最小化文本嵌入与对应图像嵌入之间的距离（支持余弦距离和 MSE 两种训练变体）

适配器仅有约 0.4B 参数，可以即插即用地加载到 DeTikZifyv2 中。

#### 3. TikZero+ 端到端微调

在 TikZero 的零样本方案基础上，TikZero+ 进一步利用少量可用的文本-程序对齐数据进行端到端微调：

- 将 TikZero 适配器与 DeTikZifyv2 合并为一个 10B 参数的完整模型
- 在对齐数据上做端到端训练，让两个子模块协同优化
- 这使得模型不仅具备零样本泛化能力，还能在有标注数据的分布上进一步提升

### 训练数据

TikZero 利用三类数据源：

| 数据类型 | 数据来源 | 用途 |
|---------|---------|------|
| 无标注 TikZ 程序 | DaTikZv2/v3（arXiv 提取） | 训练图像→代码模型 |
| 带描述的栅格图像 | 通用图文数据集 | 训练文本→图像嵌入适配器 |
| 对齐的文本-TikZ 对 | 少量标注数据 | TikZero+ 端到端微调 |

### 推理流程

1. 用户输入文本描述（如 "A multi-layer perceptron with two hidden layers"）
2. 文本编码器 + 适配器将描述映射为图像嵌入
3. DeTikZifyv2 将该嵌入当作"虚拟图像输入"，自回归生成 TikZ 代码
4. TikZ 代码可由 LaTeX 编译为高质量矢量图形

## 实验关键数据

### 主要结果

- **零样本设定**：TikZero（仅使用非对齐数据训练）大幅超越只能使用对齐数据训练的基线方法
- **有监督设定**：TikZero+（额外使用对齐数据微调）匹配甚至超越 GPT-4o 等大型商业系统的性能
- **模型效率**：TikZero 适配器仅 0.4B 参数，TikZero+ 完整模型 10B 参数，远小于 GPT-4o 等千亿级模型

### 评估指标

实验使用多维度评估体系：

- **编译成功率**：生成的 TikZ 代码能否成功编译
- **视觉相似度**：生成图形与参考图形的像素级/语义级相似度
- **语义保真度**：生成图形是否准确表达了文本描述的语义内容

### 与基线对比

| 方法 | 数据需求 | 模型规模 | 性能 |
|------|---------|---------|------|
| 仅对齐数据训练的基线 | 对齐数据 | 8B | 较低 |
| TikZero（零样本） | 非对齐数据 | 8B + 0.4B 适配器 | 大幅优于基线 |
| TikZero+ | 非对齐 + 对齐数据 | 10B | 匹配/超越 GPT-4o |
| GPT-4o | 预训练通用数据 | >>100B | 强但不专精 |

### 后续改进：DeTikZifyv2.5

基于 TikZero 的工作，团队进一步通过 **强化学习自反馈（RLSF，Reinforcement Learning from Self-Feedback）** 在 DeTikZifyv2 基础上训练得到 v2.5 版本，实现了额外的性能提升。其中 GRPO（Group Relative Policy Optimization）训练脚本也已开源。

## 亮点与洞察

1. **解耦训练范式的优雅性**：TikZero 将数据稀缺问题转化为一个桥接问题——利用图像作为两种丰富数据源的交汇点，这一思路不局限于 TikZ，可推广到任何"目标域数据对齐难但分别丰富"的场景（如代码生成、CAD 建模等）
2. **即插即用的轻量适配器**：仅 0.4B 参数的适配器即可在 8B 的基础模型上实现零样本文本条件化，无需重新训练整个模型，部署成本极低
3. **ICCV 2025 Highlight 论文**：该工作被选为 ICCV 2025 的 highlight paper，体现了审稿人对其方法新颖性和实验扎实性的高度认可
4. **完整的开源生态**：代码（GitHub 1.8k stars）、模型权重（HuggingFace）、数据集、Web UI、Colab Demo 全部开源，可复现性极强
5. **从零样本到有监督的平滑过渡**：TikZero → TikZero+ 的过渡表明，解耦训练不仅在零样本下有效，作为端到端训练的初始化也能带来增益，两种范式可以互补

## 局限与展望

1. **图像桥接的信息损失**：文本→图像嵌入→代码的两步映射可能引入信息瓶颈——某些文本中的精确数值信息（如坐标、尺寸）在映射到图像嵌入空间时可能损失
2. **TikZ 语言的局限**：TikZ 虽然表达力强，但相比 SVG 等更通用的矢量图格式，其使用群体较小，应用场景主要集中在学术论文领域
3. **编译依赖**：生成的 TikZ 代码需要完整的 TeX Live 环境编译，增加了部署复杂度
4. **适配器的泛化边界**：适配器在训练分布外的文本描述上表现如何（如高度抽象或非科学领域的图表描述），有待进一步评估
5. **对齐数据仍有价值**：TikZero+ 相比纯零样本 TikZero 的提升说明对齐数据仍然重要，如何高效获取更多高质量对齐数据（如通过 LLM 自动标注）是值得探索的方向

## 相关工作与启发

- **DeTikZify (NeurIPS 2024 Spotlight)**：TikZero 的前身，专注于图像到 TikZ 的多模态语言模型，使用 MCTS 进行迭代推理优化
- **AutomaTikZ**：更早期的 TikZ 自动生成工作，DeTikZify 系列在其基础上发展
- **Flamingo / LLaMA 3.2-Vision**：TikZero 的跨注意力适配器设计灵感来源
- **Idefics3**：DeTikZifyv2 的基础架构来源

**对后续研究的启发**：

- 图像作为模态桥梁的思路可推广到更多跨模态生成任务（如文本→SVG、文本→CAD、文本→音乐）
- 轻量适配器实现零样本跨模态迁移的范式，为低资源场景下的模型复用提供了新思路
- 解耦训练 + 端到端微调的两阶段策略，在数据不完全对齐的场景中具有普适价值

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](../../AAAI2026/code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)
- [Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](../../AAAI2026/code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)

<!-- RELATED:END -->
