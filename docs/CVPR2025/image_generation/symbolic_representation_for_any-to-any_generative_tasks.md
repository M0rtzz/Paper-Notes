---
title: >-
  [论文解读] Symbolic Representation for Any-to-Any Generative Tasks
description: >-
  [CVPR 2025][图像生成][符号表示] 提出了一种符号化生成任务描述语言 (A-Language) 和免训练推理引擎，将自然语言指令映射为由函数、参数、拓扑三元组构成的可执行符号流，实现跨 12 类多模态生成任务的统一处理，在质量和灵活性上匹配或超越端到端训练的统一多模态模型。
tags:
  - CVPR 2025
  - 图像生成
  - 符号表示
  - 任意模态生成
  - 工作流合成
  - 免训练
  - 多模态任务
---

# Symbolic Representation for Any-to-Any Generative Tasks

**会议**: CVPR 2025  
**arXiv**: [2504.17261](https://arxiv.org/abs/2504.17261)  
**代码**: https://github.com/Jiaqi-Chen-00/Any-2-Any  
**领域**: 多模态生成  
**关键词**: 符号表示, 任意模态生成, 工作流合成, 免训练, 多模态任务

## 一句话总结

提出了一种符号化生成任务描述语言 (A-Language) 和免训练推理引擎，将自然语言指令映射为由函数、参数、拓扑三元组构成的可执行符号流，实现跨 12 类多模态生成任务的统一处理，在质量和灵活性上匹配或超越端到端训练的统一多模态模型。

## 研究背景与动机

**领域现状**：Any-to-Any 多模态生成（任意模态输入到任意模态输出）是当前生成式 AI 的核心挑战。现有方案分为两类：一是隐式神经建模（如 Show-o、Unified-IO），通过大规模训练学习跨模态映射；二是智能体方法（如 HuggingGPT、GenAgent），通过多智能体协作和工具调度。

**现有痛点**：隐式神经建模受训练数据范围约束，无法处理罕见或未预见的任务（如图像融合），且隐式表征不可中断和编辑。智能体方法依赖复杂的多智能体协调，引入系统不稳定性和操作开销，缺乏统一的任务形式化表示。

**核心矛盾**：现有方法在泛化能力和可控性之间存在严重 trade-off——神经模型有质量但不灵活，智能体灵活但不稳定。

**本文目标**：设计一种统一的符号化任务表示，结合预训练语言模型实现免训练的任意模态生成。

**切入角度**：观察到任何生成任务都可分解为三个基本组件——函数（计算操作）、参数（行为控制）、拓扑（数据流结构），这三者构成了完整的任务描述。

**核心 idea**：用显式的符号流 (symbolic flow) 代替隐式的神经表征来描述生成任务，利用 LLM 作为推理引擎将自然语言自动转换为这种符号表示。

## 方法详解

### 整体框架

给定自然语言任务描述 $s$、输入数据 $\mathcal{X}$（任意模态）和约束集 $\mathcal{C}$，推理引擎通过两阶段推理生成完整的符号表示 $\Omega(t) = (\mathcal{F}, \Phi, \mathcal{T})$，其中 $\mathcal{F}$ 是函数集、$\Phi$ 是参数空间、$\mathcal{T}$ 是拓扑连接。生成的符号流被编译执行，通过 ComfyUI 后端调用实际的生成模型完成任务。

### 关键设计

1. **A-Language 符号表示语言**:

    - 功能：将任意生成任务形式化为函数-参数-拓扑三元组
    - 核心思路：每个函数 $f_i: \mathcal{I}_i \times \phi_i \rightarrow \mathcal{O}_i$ 是原子计算单元（如图像编码、混合、解码）；参数 $\phi_i$ 控制函数行为（如混合强度）；拓扑 $d_k = (f_j, y_j) \rightarrow (f_i, x_i)$ 定义精确的数据流方向。三者组合形成符号流 $\mathcal{S} = \{(f_i, \phi_{f_i}, D_i)\}$
    - 设计动机：提供一种与具体实现无关的统一任务描述，既有数学严格性又对 LLM 友好。探索了声明式、数据流、伪自然语言三种语法风格

2. **两阶段免训练推理引擎**:

    - 功能：将自然语言指令自动转换为可执行的符号流
    - 核心思路：第一阶段 $\psi_1$ 由 LLM 根据指令和约束推断所需函数和参数 $(\mathcal{F}, \Phi)$；第二阶段 $\psi_2$ 基于已识别的组件构建拓扑连接 $\mathcal{T}$。两阶段分离降低了单步推理的复杂度；使用 RAG 从 16 个参考程序中检索 3 个最相关的作为 in-context 示例
    - 设计动机：相比一步到位的生成，分阶段能更好地处理组件间的依赖关系。消融实验证明去掉任一阶段性能都显著下降（41%→28.5%/22%）

3. **迭代精炼模块**:

    - 功能：在编译或执行失败时自动修正符号流
    - 核心思路：$\Omega_{i+1}(t) = R(\Omega_i(t), \epsilon_i)$，其中 $R$ 是精炼算子，$\epsilon_i$ 是检测到的错误信息。LLM 分析错误信号并相应调整符号流（修改参数、添加缺失组件或重构拓扑），设定最大迭代次数防止无限循环
    - 设计动机：生成式任务工作流复杂，一次性正确生成不现实，自动修复机制是实用系统的必要部分

### 损失函数 / 训练策略

本方法完全免训练。使用 GPT-4o 作为推理引擎，text-embedding-3-large 作为嵌入模型进行 RAG 检索。所有实验在单个 L4 GPU (24GB) 上运行，ComfyUI 作为代码执行后端。

## 实验关键数据

### 主实验

| 任务 | Show-o | SEED-X | LWM | U-IO 2 | Ours |
|------|--------|--------|-----|--------|------|
| Inpaint 排名 | 1.6 | ✗ | ✗ | - | **1.4** |
| T2I 排名 | 2.8 | 2.0 | 4.2 | 4.5 | **1.5** |
| I2V 排名 | ✗ | ✗ | ✗ | - | **1.0** |
| T2A 排名 | ✗ | ✗ | ✗ | 2.0 | **1.0** |
| 3D生成 排名 | ✗ | ✗ | ✗ | ✗ | **1.0** |

ComfyBench 性能：

| 方法 | Vanilla | Complex | Creative | 总计 |
|------|---------|---------|----------|------|
| ComfyAgent | 46.00 | 21.67 | 15.00 | 32.50 |
| HuggingGPT | 21.00 | 0.00 | 5.00 | 11.50 |
| **Ours** | **61.00** | **28.89** | **19.17** | **43.00** |

### 消融实验

| 两阶段推理 | 迭代精炼 | 总解决率 |
|------------|----------|----------|
| ✓ | ✗ | 28.50% |
| ✗ | ✓ | 22.00% |
| **✓** | **✓** | **41.00%** |

编译/执行通过率：我们 98%/87% vs GenAgent 84%/63%

### 关键发现

- 符号方法在 12 类任务上的胜率：对 Show-o 94%、对 LVM 98%（T2I）、对 Gen-3 67%（I2V 对齐度）
- 声明式语法在复杂工作流中错误率最低，伪自然语言格式错误最多
- 符号表示天然支持工作流编辑——可直接修改函数（换模型）或参数（调提示词），这是端到端模型无法做到的
- 方法可处理 11-13 个组件、11-17 个连接的复杂非原子任务

## 亮点与洞察

- 视角新颖：将生成任务视为"程序执行"而非"端到端映射"，突破了训练数据的限制
- 免训练设计使得新增任务类型只需扩展函数库而无需重新训练
- 符号流的显式特性带来了可中断性、可编辑性和可调试性，这在安全敏感场景至关重要
- 实现了神经模型（执行函数）和符号推理（流程规划）的优雅结合

## 局限与展望

- 整体性能受限于底层函数的能力（如 ComfyUI 生态中可用的模型）
- 复杂任务（Creative 类别）的解决率仍然不高（19.17%），拓扑推理是主要瓶颈
- 与智能体方法结合可能进一步提升灵活性
- 未讨论错误恢复的成本和延迟，实际部署中可能需要限制精炼次数

## 相关工作与启发

- VISPROG [12] 的神经符号方法是直接前驱，本文在其基础上提出了更完整的形式化
- GenAgent [49] 的多智能体协作方法在简单任务上反而不如符号方法稳定
- 启发：对于结构化任务，显式符号表征+LLM 推理可能比端到端学习更高效
- 三种语法风格的对比研究为 LLM-friendly DSL 设计提供了有价值的参考

## 评分

- **新颖性**: 8/10 — 将生成任务形式化为三元组符号表示的思路清新，但执行层面仍依赖已有工具
- **实验充分度**: 7/10 — 12 类任务 + ComfyBench + 用户研究覆盖较全，但评测规模偏小（120 个案例）
- **写作质量**: 7/10 — 形式化定义清晰但稍显冗长，实验部分可视化不够直观
- **价值**: 8/10 — 为多模态生成提供了新范式，开源代码增加了实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture](dkdm_data-free_knowledge_distillation_for_diffusion_models_with_any_architecture.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](../../ECCV2024/image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)

<!-- RELATED:END -->
