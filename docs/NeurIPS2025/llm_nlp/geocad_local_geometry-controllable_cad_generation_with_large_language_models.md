---
title: >-
  [论文解读] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][CAD生成] 提出 GeoCAD，首个实现局部几何可控 CAD 生成的方法，通过互补标注策略为局部零件生成几何指令，并微调 LLM 实现根据用户文本指令精确修改 CAD 模型的局部部分。
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "CAD生成"
  - "局部几何控制"
  - "大语言模型"
  - "文本到CAD"
  - "互补标注"
---

# GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.10337](https://arxiv.org/abs/2506.10337)  
**代码**: [https://github.com/Zhanwei-Z/GeoCAD](https://github.com/Zhanwei-Z/GeoCAD)  
**领域**: LLM/NLP  
**关键词**: CAD生成, 局部几何控制, 大语言模型, 文本到CAD, 互补标注

## 一句话总结
提出 GeoCAD，首个实现局部几何可控 CAD 生成的方法，通过互补标注策略为局部零件生成几何指令，并微调 LLM 实现根据用户文本指令精确修改 CAD 模型的局部部分。

## 研究背景与动机
在工业设计中，CAD 模型的草图-拉伸建模（Sketch-Extrude Modeling, SEM）流程广泛使用。用户通常需要在绘制草稿后修改局部零件（local loops），以确保最终产品满足功能或美学要求。如果深度学习方法能根据用户定义的几何指令自动修改局部零件的形状（如"等腰直角三角形"或"切去一角的矩形"），将显著降低优化 CAD 产品的人力成本。

现有方法面临两个核心挑战：

**缺乏文本指令跟随能力**：传统可控 CAD 生成方法（如 SkexGen、SketchGen）接受部分 CAD 属性作为输入，但无法理解自然语言指令

**无法聚焦于局部生成**：现有 text-to-CAD 方法（如 CAD-GPT、Text2CAD）大多从头生成完整 CAD 模型，难以精确控制某个局部零件

**几何描述不准确**：部分方法从全局 3D 视角收集文本描述，倾斜视角无法捕捉长度、角度等精确几何属性

**FlexCAD 虽能聚焦局部但缺乏几何约束**，在训练时 prompt 中没有几何约束，导致难以遵循几何指令

## 方法详解

### 整体框架
GeoCAD 包含三个输入：(1) 原始 CAD 模型（以 FlexCAD 提出的层次化文本格式表示），(2) 需要修改的局部零件，(3) 用户指定的几何指令。输出为仅修改目标局部零件后的新 CAD 模型。

核心流程分为两个阶段：
- **互补标注策略**（Sec 3.1）：为局部零件生成约 221k 条几何指令
- **两阶段 LLM 微调**（Sec 3.2）：利用这些指令微调 LLM 实现局部可控生成

### 关键设计

#### 1. 互补标注策略（Complementary Captioning）
从 DeepCAD 数据集收集局部零件（local loops），过滤重复和无效样本后，将零件分为两类：

**简单零件（~105k）**：常见几何形状（三角形、四边形、扇形等），占总量约 50%
- 使用**基于顶点的标注方法（Vertex-based Captioning）**
- 从 CAD 文本表示中提取顶点坐标，分析几何属性进行精确分类
- 例如：四边形四条边等长 → 菱形；若含直角 → 进一步归类为正方形
- 部分简单零件还加入关键尺寸参数（如圆的半径、正方形的边长）

**复杂零件（~116k）**：具有更复杂视觉模式的零件
- 使用**基于 VLLM 的标注方法（VLLM-based Captioning）**
- 将复杂零件渲染为 2D 图像，利用 GPT-4 / Qwen-VL 等 VLLM 生成描述性标注
- VLLM 对简单零件的细粒度几何描述不够精确（如无法可靠区分菱形），因此需要互补策略

#### 2. 两阶段 LLM 微调

**Stage 1：CAD-文本对齐预训练（可选）**
- 目标：对齐 CAD 特有的几何表示与文本几何指令
- 对每个局部零件施加随机数据增强：平移、缩放、旋转、翻转
- 增强后的样本几何指令不变（如旋转后的直角梯形仍为直角梯形）
- 同时用原始和增强样本的指令-答案对微调 LLM

**Stage 2：几何控制指令微调**
- 每个 epoch，给定 CAD 模型，随机 mask 一个局部零件
- 利用对应的几何指令和剩余可见零件作为 prompt，让 LLM 预测被 mask 的零件
- **关键区别于 FlexCAD**：prompt 中显式加入几何指令作为约束
- FlexCAD 训练时 prompt 缺乏几何约束，导致推理时无法遵循几何指令

### 损失函数 / 训练策略
- 使用标准交叉熵（CE）损失，在预测 token 和答案 token 之间计算
- 采用 LoRA 微调（rank=8, alpha=32），冻结大部分参数权重
- 基座模型：Llama-3-8B
- 8 × A100 GPU，AdamW 优化器，batch size 32
- 余弦退火学习率，初始 5×10⁻⁴
- Stage 1 训练 10 epochs，Stage 2 训练 30 epochs
- 推理时温度 τ=0.9，Top-p=0.9

## 实验关键数据

### 主实验
在 DeepCAD 测试集上随机采样 1k CAD 模型，每个模型随机 mask 一个局部零件，每种方法用 5 条简单 + 5 条复杂几何指令生成新零件，共 10k 生成样本。

| 模型 | COV↑ | MMD↓ | JSD↓ | PV↑ | Ver-score↑ | VLLM-score↑ | Realism↑ |
|------|------|------|------|-----|------------|-------------|----------|
| OpenAI-o3 (5-shot) | 53.6% | 1.64 | 1.49 | 65.7% | 33.6% | 22.1% | 18.7% |
| FlexCAD | 58.3% | 1.40 | 1.58 | 86.7% | 19.8% | 6.93% | 13.6% |
| FlexCAD (5-shot) | 59.4% | 1.37 | 1.34 | 88.1% | 43.5% | 26.8% | 20.2% |
| **GeoCAD** | **64.9%** | **1.13** | **0.98** | **90.5%** | **76.4%** | **65.7%** | **40.9%** |
| **GeoCAD (5-shot)** | **66.0%** | 1.16 | **0.80** | **92.3%** | **82.2%** | **68.2%** | **43.6%** |

GeoCAD 在文本-CAD 一致性指标上大幅领先：
- Ver-score 比 FlexCAD 高出 **38.7%**
- VLLM-score 比 FlexCAD 高出 **41.4%**
- 人工评估 Realism 比 FlexCAD 高出 **23.4%**

### 消融实验

| 变体 | COV↑ | MMD↓ | JSD↓ | PV↑ | Ver-score↑ | VLLM-score↑ |
|------|------|------|------|-----|------------|-------------|
| 仅 Vertex-based 标注 | 63.6% | 1.18 | 1.02 | 89.5% | 78.3% | - |
| 仅 VLLM-based 标注 | 61.8% | 1.26 | 1.05 | 89.1% | - | 64.2% |
| 去掉 Stage 1 | 61.3% | 1.21 | 1.16 | 89.6% | 71.5% | 60.4% |
| 去掉数据增强 | 62.9% | 1.18 | 1.09 | 88.5% | 73.2% | 61.8% |
| **完整 GeoCAD** | **64.9%** | **1.13** | **0.98** | **90.5%** | **76.4%** | **65.7%** |

### 关键发现
1. **互补标注不可或缺**：仅用 Vertex-based 无法生成复杂零件，仅用 VLLM-based 无法精确描述简单零件
2. **预训练（Stage 1）至关重要**：去掉后性能最差，说明 CAD-文本初步对齐是必要的
3. **数据增强有效**：去掉后性能下降，多样化增强样本增强了对齐能力
4. **GeoCAD 具有泛化能力**：对语义相似的未见指令（如"窄圆角矩形"、"直角三角形"）也能准确理解和执行
5. **可精确控制尺寸参数**：如圆的半径、正方形边长、矩形长宽

## 亮点与洞察
1. **首次提出局部几何可控 CAD 生成**，填补了该领域的空白
2. **互补标注策略**设计精妙：Vertex-based 解决简单零件的精确分类，VLLM-based 处理复杂零件的视觉模式描述，二者互补
3. 将 CAD 局部修改建模为 **mask-then-predict** 任务，巧妙利用 LLM 的上下文补全能力
4. 标注规模达 221k，为后续工作提供了可参考的数据构建范式
5. 在 prompt 中加入几何约束这一简单改进，相比 FlexCAD 带来了巨大的性能提升

## 局限与展望
1. **仅支持 SEM（草图-拉伸）范式**：未扩展到 CSG 或 B-rep 等其他 CAD 表示
2. **依赖 DeepCAD 数据集**：数据规模和多样性可能限制泛化
3. **VLLM 标注质量受限**：复杂零件的标注质量取决于 VLLM 的视觉理解能力
4. **推理效率未讨论**：大语言模型推理开销较高，实际工业应用中的实时性待验证
5. **局限于 2D 循环（loop）级别控制**：更细粒度（如单条边或顶点）的控制尚未探索
6. **与 3D 视觉模型的结合**：直接从 3D 视图输入几何约束可能更直观

## 相关工作与启发
- **FlexCAD**：最直接的前驱工作，GeoCAD 在其基础上加入了几何约束
- **Text2CAD / CAD-GPT**：从头生成完整 CAD，GeoCAD 聚焦于局部编辑
- **LoRA 微调范式**：在保持大模型预训练优势的同时实现高效领域适配
- 启发：将"局部可控编辑"的思路推广到其他结构化生成任务（如代码局部修改、分子结构局部优化）

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次提出局部几何可控 CAD 生成，互补标注策略新颖)
- 实验充分度: ⭐⭐⭐⭐ (定量+定性+消融完整，但缺少效率分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (填补领域空白，工业应用前景好)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](../../ACL2025/llm_nlp/cadllm_cad_modeling_from_text.md)
- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](../../ACL2025/llm_nlp/difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](../../ACL2025/llm_nlp/segment_level_diffusion.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](../../ACL2025/llm_nlp/an_empirical_study_of_large_language_models_for_automated_review_generation.md)

</div>

<!-- RELATED:END -->
