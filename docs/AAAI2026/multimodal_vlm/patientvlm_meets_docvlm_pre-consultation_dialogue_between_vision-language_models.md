---
title: >-
  [论文解读] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis
description: >-
  [AAAI 2026][多模态][医学诊断] 提出Pre-Consultation Dialogue Framework (PCDF)，通过两个VLM（DocVLM和PatientVLM）模拟医生-患者多轮对话，生成image-dialogue-diagnosis三元组用于微调DocVLM，在四个医学影像基准上平均F1提升11.48。
tags:
  - AAAI 2026
  - 多模态
  - 医学诊断
  - 多轮对话
  - VLM交互
  - 数据合成
  - 对话驱动微调
---

# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis

**会议**: AAAI 2026  
**arXiv**: [2601.10945](https://arxiv.org/abs/2601.10945)  
**代码**: [https://vl2g.github.io/projects/pcdf](https://vl2g.github.io/projects/pcdf)  
**领域**: 多模态VLM  
**关键词**: 医学诊断, 多轮对话, VLM交互, 数据合成, 对话驱动微调

## 一句话总结
提出Pre-Consultation Dialogue Framework (PCDF)，通过两个VLM（DocVLM和PatientVLM）模拟医生-患者多轮对话，生成image-dialogue-diagnosis三元组用于微调DocVLM，在四个医学影像基准上平均F1提升11.48。

## 研究背景与动机

### 领域现状
医学影像AI研究长期围绕"图像→诊断"这一范式，从早期CNN分类到CLIP医学适配（MedCLIP、BioMedCLIP），再到大型VLM（MedPaLM2、MedGemma、LLaVA-Med），模型的视觉理解能力持续提升。

### 核心痛点
然而，**现实中的诊断很少仅依赖影像**。医生会通过多轮交互与患者沟通，逐步询问症状、病史，缩小鉴别诊断范围。这种对话驱动的推理过程是临床诊断的核心，但现有模型完全忽略了这一环节，导致预测脆弱。

### 数据获取困境
收集真实的医生-患者对话数据极其困难：需要IRB伦理审批、患者知情同意、医生担心法律风险和工作流干扰。这使得大规模数据收集在实践中几乎不可行。

### 已有尝试的局限
已有工作（Yang et al. 2024; Chen et al. 2023）尝试用**单个LLM**同时扮演医生和患者生成合成对话，但存在两个关键问题：(1) 仅在纯文本设置下运行，不包含医学图像；(2) 单一模型生成两个角色，缺乏角色分离和交互真实性。

### 本文切入点
提出PCDF——用**两个独立VLM**分别扮演医生和患者，在医学图像上进行视觉-对话联合推理。PatientVLM基于真实诊断标签生成症状回答（但被明确指示不泄露诊断），DocVLM基于图像和对话历史生成后续问题。这种设计保留了真实问诊中的信息不对称性。

## 方法详解

### 整体框架
PCDF包含两个阶段：**对话模拟阶段**和**对话条件微调阶段**。

阶段一：给定医学数据集 $\mathcal{D}=\{(I_i, C_i)\}_{i=1}^N$，对每个样本模拟T轮DocVLM-PatientVLM对话，生成增强数据集 $\hat{\mathcal{D}}=\{(I_i, H_i, C_i)\}$。

阶段二：在增强数据集上微调DocVLM，使其学习 $P(C|I,H)$，即基于图像和对话历史的诊断能力。

### 关键设计

1. **DocVLM（医生模型）**:

    - 基于医学图像 $I_i$、对话历史 $H_{i,<t}$ 和所有可能诊断 $\mathcal{C}$ 生成随访问题
    - 核心公式：$Q_{i,t} = \text{DocVLM}(P_{doc}(I_i, H_{i,<t}, \mathcal{C}))$
    - 设计动机：将所有可能诊断包含在prompt中，鼓励生成有鉴别力的问题，帮助区分相似疾病

2. **PatientVLM（患者模型）**:

    - 基于图像 $I_i$、真实诊断 $C_i$ 和DocVLM的问题 $Q_{i,t}$ 生成回答
    - 核心公式：$A_{i,t} = \text{PatientVLM}(P_{pat}(I_i, C_i, Q_{i,t}))$
    - 设计动机：用真实诊断指导症状表达，但**明确要求不泄露诊断本身**，保留信息不对称性
    - 在整个对话模拟过程中，PatientVLM参数保持冻结

3. **迭代对话生成**:

    - DocVLM和PatientVLM进行最多T轮交互（实验中T=8）
    - 每轮DocVLM问一个问题、PatientVLM给出回答
    - 最终生成image-dialogue-diagnosis三元组

### 损失函数 / 训练策略
- 将诊断分类建模为**文本生成问题**，自回归生成诊断token
- 使用标准生成损失：$\mathcal{L}_{gen}(\theta) = -\mathbb{E}_{(I,H,C)}\left[\sum_m \log P_\theta(C_m|C_{<m}, I, H)\right]$
- 使用LoRA微调DocVLM：rank=16, alpha=32, dropout=0.05
- 训练10个epoch，batch size=8
- 实验使用mPLUG-Owl3作为默认PatientVLM

## 实验关键数据

### 主实验

在MedMNIST v2的四个数据集上评估：

| 模型 | 设置 | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|------|------|---------------|-------------------|----------------|--------------|
| InternVL3-2B | Image-only SFT | 36.5 | 88.4 | 31.5 | 70.9 |
| InternVL3-2B | +PCDF | **73.7(+37.2)** | **98.6(+10.2)** | **54.9(+23.4)** | **85.5(+14.6)** |
| Qwen2.5-VL-7B | Image-only SFT | 56.5 | 83.3 | 33.8 | 73.5 |
| Qwen2.5-VL-7B | +PCDF | **81.0(+24.5)** | **94.5(+11.2)** | **39.7(+5.9)** | **77.9(+4.4)** |
| Gemma3-4B | Image-only SFT | 78.3 | 95.7 | 47.7 | 86.0 |
| Gemma3-4B | +PCDF | **81.9(+3.6)** | **99.0(+3.3)** | **67.7(+20.0)** | **90.2(+4.2)** |
| MedGemma3-4B | Image-only SFT | 81.5 | 99.1 | 71.2 | 90.9 |
| MedGemma3-4B | +PCDF | **86.4(+4.9)** | **99.3(+0.2)** | **81.3(+10.1)** | **96.9(+6.0)** |

关键发现：PCDF增强的VLM平均F1提升11.48，通用VLM获益最大（InternVL3 F1提升37.2）。

### 消融实验

**对话长度分析（Gemma3 + mPLUG-Owl3）**：

| 对话轮数T | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|-----------|---------------|-------------------|----------------|--------------|
| 2 | 63.5 | 78.8 | 27.8 | 59.1 |
| 4 | 70.3 | 80.3 | 36.6 | 49.5 |
| 6 | 71.9 | 91.7 | 44.1 | 71.8 |
| 8 | **81.9** | **99.0** | **67.7** | **90.2** |

**PatientVLM选择分析（DocVLM=Qwen2.5-VL-7B）**：

| PatientVLM | 平均F1 | 说明 |
|------------|--------|------|
| Image-only SFT | 61.8 | 基线 |
| mPLUG-Owl3 | **73.3** | 最优PatientVLM |
| InternVL3 | 70.1 | 次优 |
| Qwen2.5-VL | 72.7 | 同架构但不同角色 |
| MedGemma | 70.5 | 医学领域模型 |

### 关键发现

1. **通用VLM获益更大**：InternVL3在DermaMNIST上F1提升37.2，因为缺乏医学领域预训练
2. **对话越长效果越好**：T从2到8，RetinaMNIST F1从27.8涨到67.7（+39.9%绝对提升）
3. **PCDF优于CoT推理**：MedGemma的PCDF零样本F1比CoT平均高23.6
4. **临床验证**：96.9%的模拟对话被认为临床相关，无诊断泄露案例

## 亮点与洞察

- **双VLM角色分离**是一个优雅的设计——保留了问诊中医生和患者的信息不对称，比单模型生成更真实
- **模型无关性**：PCDF可应用于任意VLM，无需修改架构，仅需LoRA微调
- **即使MedGemma这样的医学专用模型也能获益**，说明对话式监督信号与传统领域适配是互补的
- **零成本临床对话数据**：完全不需要真实医患对话，绕过了数据收集的伦理和成本问题

## 局限与展望

- 临床验证规模较小（210个案例），需要更大规模的多样化评估
- DocVLM生成的问题偏向专业术语，普通患者可能难以理解
- 目前仅支持英语，限制了多语言医疗场景的适用性
- MedMNIST数据集相对简单，未在更复杂的临床场景（如多疾病共存）中验证
- PatientVLM的症状生成质量依赖于底层VLM的医学知识

## 相关工作与启发

- 与MedIQ和3MDBench等评估导向工作不同，PCDF是一个**训练框架**
- 受启发于真实问诊流程：医生不仅看片子，更要问症状
- 未来可以将PCDF扩展到**多模态**（加入实验室检查数据）或**多语言**场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（双VLM对话模拟问诊是全新的框架设计）
- 实验充分度: ⭐⭐⭐⭐（四个数据集+四个VLM+多维度消融）
- 写作质量: ⭐⭐⭐⭐⭐（问题动机清晰，方法自然）
- 价值: ⭐⭐⭐⭐（在医学AI中有实际应用潜力）

<!-- RELATED:START -->

## 相关论文

- [SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)
- [TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](em-kd_distilling_efficient_multimodal_large_language_model_w.md)
- [RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)
- [Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)

<!-- RELATED:END -->
