---
title: >-
  [论文解读] CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop
description: >-
  [NEURIPS2025][图像生成][CAD editing] 提出 CADMorph，一个迭代式 plan–generate–verify 框架，利用预训练的 Parameter-to-Shape (P2S) 扩散模型和 Masked-Parameter-Prediction (MPP) 大语言模型协同工作，在无需三元组训练数据的情况下实现几何驱动的参数化 CAD 编辑。
tags:
  - NEURIPS2025
  - 图像生成
  - CAD editing
  - parametric sequence
  - 扩散模型
  - masked prediction
  - test-time scaling
---

# CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop

**会议**: NEURIPS2025  
**arXiv**: [2512.11480](https://arxiv.org/abs/2512.11480)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: CAD editing, parametric sequence, latent diffusion, masked prediction, test-time scaling

## 一句话总结

提出 CADMorph，一个迭代式 plan–generate–verify 框架，利用预训练的 Parameter-to-Shape (P2S) 扩散模型和 Masked-Parameter-Prediction (MPP) 大语言模型协同工作，在无需三元组训练数据的情况下实现几何驱动的参数化 CAD 编辑。

## 背景与动机

CAD 模型具有**表征双重性**：一方面是参数化构造序列（包含 Line、Extrude 等操作及具体数值），保证制造精度和可编辑性；另一方面是由该序列渲染出的可视几何形状，用于直观检验和仿真验证。在实际 CAD 开发中，几何形状频繁被调整（仿真反馈、人机工程、美学目标等），这要求工程师同步修改底层的参数化序列。这一过程——**geometry-driven parametric CAD editing**——既费力又容易出错，需要评估形状变化幅度、精确定位需修改的序列片段，并将修改传播到所有依赖片段。

现有工作主要关注无条件编辑（从原始序列随机采样生成编辑结果）或文本驱动编辑（用文字指令引导编辑）。前者缺少明确引导，后者难以用简短文本精确描述复杂的形状变化。**直接以目标几何形状作为编辑引导**这一实用场景几乎未被探索。

## 核心问题

几何驱动的参数化 CAD 编辑面临三大核心挑战：

1. **结构保持 (Structure Preservation)**：编辑应仅限于需要修改的序列片段，保持其余部分不变
2. **语义有效性 (Semantic Validity)**：更新后的参数化序列不仅语法正确，还需生成符合设计规范的 CAD 模型（如螺栓孔均匀分布而非随意放置）
3. **形状保真度 (Shape Fidelity)**：更新后的序列渲染出的形状必须与目标形状匹配

此外还有**数据稀缺**问题：不存在同时包含原始序列、目标几何形状和对应更新序列的三元组数据集。

## 方法详解

### 任务形式化

给定原始参数化序列 $C$ 和目标几何形状 $S'$，寻找更新后的序列 $C'$ 使其渲染结果再现 $S'$，同时优先保持 $C$ 的结构：

$$C' = \arg\min_{C'} \mathcal{D}_{\text{geometry}}(\mathcal{F}(C'), S') + \lambda \mathcal{R}_{\text{structure}}(C', C)$$

### 两个预训练基础模型

- **Parameter-to-Shape (P2S) 模型**：Latent Diffusion Model，在 ⟨参数化序列, SDF⟩ 对上训练，将参数化序列映射为 3D 形状的隐空间表示。架构沿用 SDFusion，包含形状编码器-解码器对和扩散模型
- **Masked-Parameter-Prediction (MPP) 模型**：基于 Llama-3 8B 的大语言模型，通过 LoRA 微调，以层次化掩码策略训练，用于补全参数化序列中的被掩码片段

两个模型都**无需三元组数据**训练，绕过数据稀缺瓶颈。

### 迭代 Plan–Generate–Verify 框架

在第 $r$ 轮迭代中执行三个阶段：

**1. Planning 阶段——定位需修改的片段**

- 利用 P2S 模型的 cross-attention maps 量化每个序列片段对目标形状的贡献度
- 计算相对贡献分数：$J(i) = |\mathcal{M}(C'_{r-1}(i), S') - \mathcal{M}(C'_{r-1}(i), S'_{r-1})|$
- 对分数排序，掩码最大的 $K$ 个片段（实际取超过均值 $\bar{J}$ 的），得到 $C^{\text{mask}}_r$
- 这种基于注意力的掩码策略将编辑集中在与目标形状不匹配的片段上，满足结构保持要求

**2. Generation 阶段——生成候选编辑**

- MPP 模型对 $C^{\text{mask}}_r$ 执行 $N$ 次补全，生成候选序列集 $\{C^1_r, \dots, C^N_r\}$
- 自回归地逐 token 生成，利用预训练获得的 CAD 设计知识保证语义有效性

**3. Verification 阶段——选择最优候选**

- 将所有候选序列和目标形状分别通过 P2S 模型映射到共享隐空间
- 选择与目标形状欧氏距离最小的候选序列：$C'_r = \arg\min_{\tilde{C} \in \mathcal{Q}} \|\mathcal{F}(\tilde{C}) - E_s(S')\|_2$
- 维护跨迭代的**优先队列** $\mathcal{Q}$，保留历史最优候选，扩大搜索范围

重复迭代直到收敛或达到最大迭代次数（默认 10 轮）。

## 实验关键数据

在 DeepCAD 数据集（约 130k CAD 模型）上训练，使用 CAD-Editor 的 2k 测试集评估，每个测试样例生成 5 个输出。

| 方法 | IoU ↑ | mean CD ↓ | median CD ↓ | JSD ↓ | IR (%) ↓ | Edit Dist. ↓ |
|------|-------|-----------|-------------|-------|----------|-------------|
| GPT-4o | 0.247 | 0.107 | 0.0171 | 0.737 | 25.1 | 21.12 |
| o4-mini | 0.185 | 0.118 | 0.0283 | 0.748 | 32.95 | 22.49 |
| CAD-Diffuser | 0.548 | 0.097 | 0.0093 | 0.689 | 5.7 | 17.29 |
| FlexCAD | 0.447 | 0.029 | 0.0065 | 0.634 | 15.3 | 22.29 |
| **CADMorph** | **0.687** | **0.009** | **0.0031** | **0.621** | **3.1** | **16.87** |

关键发现：

- CADMorph 在所有指标上均优于所有基线，IoU 比最强基线 CAD-Diffuser 高出 25%
- VLM（GPT-4o 系列）难以生成语法有效的 CAD 序列，Invalid Rate 高达 25-40%
- 人工评估中 CADMorph 平均排名 1.37（满分 1），显著优于其他方法

消融实验证实每个组件的必要性：移除优先队列 IoU 从 0.687 降至 0.619，移除 Verification 阶段降至 0.517，移除 Planning 阶段降至 0.447。

## 亮点

1. **数据高效**：无需三元组训练数据，将计算投入推理阶段的迭代搜索，体现 test-time scaling 思想
2. **搜索高效**：Planning 阶段通过 cross-attention 分析缩小搜索空间，Verification 阶段提供有效的选择信号引导编辑方向
3. **隐式纠错能力**：MPP 模型因预训练中吸收了大量设计知识，能自动修正不合理的几何（如桌腿未与面板齐平时自动纠正）
4. **实用下游应用**：支持迭代编辑（多轮连续修改）和逆向工程增强（细化逆向工程管线的输出结果）
5. **巧妙利用 cross-attention maps**：类比 text-to-image 扩散模型中词与像素的对应关系，发现 P2S 模型中参数化序列片段与几何部件的注意力对齐现象

## 局限与展望

1. **推理延迟**：需要多轮 plan–generate–verify 迭代，运行时间较长；可通过加速模型推理和并行化 generation/verification 来缓解
2. **测试集局限**：仅在 CAD-Editor 测试集上评估，其模型复杂度低于工业实际；需要更丰富、更具挑战性的数据集
3. **表示受限**：使用 voxelized tSDF 表示形状，分辨率有限，可能影响精细几何细节的捕捉
4. **端到端潜力**：当前是推理时迭代搜索方案，未来可用 CADMorph 生成三元组数据来训练端到端模型

## 与相关工作的对比

- **vs 逆向工程方法 (CAD-Diffuser)**：逆向工程直接从几何重建参数化序列，忽略原始序列中设计者的意图；CADMorph 保持原始序列结构，仅做最小编辑
- **vs 传统编辑方法 (FlexCAD)**：传统编辑方法无视觉引导，无法将编辑方向对齐目标形状；CADMorph 以目标形状作为明确引导
- **vs VLM (GPT-4o)**：通用视觉语言模型缺乏 CAD 领域知识，生成的序列语法错误率高，形状质量差
- **vs 3D 形状编辑方法**：直接在 mesh/SDF/NeRF 上操作几何，丢失参数化可编辑性；CADMorph 在参数化序列层面编辑，保持制造可行性

## 启发与关联

- **Test-time scaling 在 CAD 中的首次应用**：通过推理时增加计算（多次生成 + 验证器选择）提升效果，这一范式可推广到其他结构化生成任务
- **Cross-attention 作为定位工具**：利用扩散模型的 cross-attention maps 定位需修改区域的思路，可迁移到其他条件生成任务中的可控编辑
- **双模型协同框架**：一个模型负责感知和评估（P2S），另一个负责生成（MPP），这种分工协作的设计模式值得其他多模态生成研究借鉴

## 评分
- 新颖性: 8/10 — 首次形式化 geometry-driven parametric CAD editing 任务，plan–generate–verify 框架设计巧妙
- 实验充分度: 7/10 — 基线覆盖 VLM、逆向工程和编辑方法，含人工评估和消融；但测试集规模偏小且复杂度有限
- 写作质量: 8/10 — 问题定义清晰，框架各阶段阐述逻辑流畅，图表直观
- 价值: 7/10 — 解决实际工程需求，但推理效率和测试集限制在一定程度上制约了实用性

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICLR 2026\] Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek](../../ICLR2026/image_generation/seek-cad_a_self-refined_generative_modeling_for_3d_parametric_cad_using_local_in.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)

<!-- RELATED:END -->
