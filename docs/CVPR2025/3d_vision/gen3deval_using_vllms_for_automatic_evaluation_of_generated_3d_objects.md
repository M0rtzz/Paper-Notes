---
title: >-
  [论文解读] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects
description: >-
  [CVPR 2025][3D视觉][3D生成评估] 本文提出Gen3DEval，一个基于vLLM微调的text-to-3D生成质量评估框架，通过对Llama3模型在合成+人工标注数据上微调，实现对3D物体外观、表面质量和文本一致性的自动评估，在与人类偏好对齐上显著超越GPT-4o等通用模型。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D生成评估
  - 视觉语言模型
  - 人类偏好对齐
  - ELO评分
  - 多视角渲染
---

# Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects

**会议**: CVPR 2025  
**arXiv**: [2504.08125](https://arxiv.org/abs/2504.08125)  
**代码**: https://shalini-maiti.github.io/gen3deval.github.io/ (项目页面)  
**领域**: 3D视觉  
**关键词**: 3D生成评估, 视觉语言模型, 人类偏好对齐, ELO评分, 多视角渲染

## 一句话总结
本文提出Gen3DEval，一个基于vLLM微调的text-to-3D生成质量评估框架，通过对Llama3模型在合成+人工标注数据上微调，实现对3D物体外观、表面质量和文本一致性的自动评估，在与人类偏好对齐上显著超越GPT-4o等通用模型。

## 研究背景与动机
1. **领域现状**：Text-to-3D生成近年发展迅速（扩散模型、NeRF、Gaussian Splatting），但缺乏标准化、与人类判断一致的评估指标。
2. **现有痛点**：PSNR/SSIM/Chamfer Distance需要ground truth数据，实际不可行（一个prompt可对应多种合理输出）；CLIP只评估文本一致性，忽略外观和表面质量；FID需要大规模标准分布，计算昂贵且不一致。
3. **核心矛盾**：Text-to-3D是一对多映射，不存在唯一参考，相似度指标本质上不适用；且现有指标维度单一，无法全面评估。
4. **本文目标** 构建一个无需ground truth、全面评估外观+表面质量+文本一致性、与人类偏好高度对齐的自动评估框架。
5. **切入角度**：GPT-4V虽能做3D评估但非专用模型且效果有限，作者认为需要针对3D质量评估进行专门微调。
6. **核心 idea**：用合成扰动数据+人工偏好标注微调vLLM，让模型学会从多视角渲染图中判断3D物体质量的三个维度。

## 方法详解

### 整体框架
Gen3DEval分两个阶段：Stage 1训练一个能进行成对比较的vLLM，Stage 2用该模型在Gen3DEval-Bench上做成对评比并计算ELO排名。输入为3D物体的多视角渲染图（最多8张，包括RGB和法线图），输出为哪个物体在指定维度上更好的判断。

### 关键设计

1. **两阶段vLLM训练（预训练+SFT）**:
    - 功能：让vLLM学会理解多视角3D渲染图并进行质量评判
    - 核心思路：预训练阶段冻结LLM和图像编码器，仅训练视觉-语言投影矩阵 $W_\theta$，用14万个3D artist网格的多视角渲染+文字描述做VQA训练。SFT阶段解冻投影矩阵和LLM，用成对比较数据进行指令微调，学习在外观/表面/文本一致性三个维度上做偏好判断。
    - 设计动机：两阶段策略确保先建立视觉-语言对齐，再专注于3D质量评估的具体任务，避免直接端到端训练的不稳定。

2. **合成扰动数据构建**:
    - 功能：大规模生成训练所需的成对比较数据
    - 核心思路：基于artist创建的高质量3D网格，通过Blender/NeRF/Gaussian Splatting引入可控扰动：Laplacian平滑、随机表面凸起、纹理模糊/接缝、透明度伪影、浮动元素、断裂组件等，模拟3D生成方法常见的缺陷。文本一致性数据通过多视角扩散模型生成不同caption的视图，用CLIP过滤低质量样本。
    - 设计动机：人工标注数据量有限（5K+样本），合成扰动能大规模扩充训练集，且能精确控制缺陷类型。

3. **多视角输入与图像编码器选择（CLIP）**:
    - 功能：从多角度全面捕捉3D物体质量信息
    - 核心思路：使用最多8张多视角RGB和法线图作为输入。对比了CLIP、DinoV2和Fit3D三种图像编码器及其组合。CLIP（336×336分辨率，ViT架构）在所有评估维度上表现最一致，特别是在OOD泛化上大幅领先。每张图产生576个视觉token。
    - 设计动机：单视角可能遗漏遮挡面和隐藏表面的问题；CLIP具有最好的泛化性能，因此选为默认编码器。

### 损失函数 / 训练策略
- 预训练使用next token prediction的最大似然目标，batch size 16，学习率1e-3，cosine scheduler，8×A100训练1天
- SFT使用相同的next token prediction目标，batch size 4，投影矩阵学习率2e-6，vLLM学习率1e-5，16×A100训练18小时
- 评估时通过成对比较+ELO评分系统生成最终排名

## 实验关键数据

### 主实验

| 方法 | 外观(Human) | 外观(OOD) | 表面(Synthetic) | 文本一致性(OOD) |
|------|-----------|----------|---------------|---------------|
| CLIP Score | 0.30 | 0.17 | 0.30 | 0.80 |
| GPT-4o | 0.59 | 0.69 | 0.54 | 0.55 |
| LLaVA-Qwen-7B | 0.54 | 0.54 | 0.51 | 0.58 |
| **Gen3DEval (CLIP)** | **0.90** | **0.89** | **0.99** | **0.86** |

### 消融实验（图像编码器）

| 编码器配置 | 外观(Human) | 外观(OOD) | 表面(OOD) | 文本(OOD) |
|-----------|-----------|----------|----------|----------|
| CLIP | **0.90** | **0.89** | 0.67 | **0.86** |
| CLIP + Fit3D | 0.90 | 0.78 | 0.57 | 0.53 |
| CLIP + DinoV2 | 0.86 | 0.78 | 0.51 | 0.74 |
| DinoV2 only | 0.77 | 0.54 | 0.61 | 0.58 |
| Fit3D only | 0.81 | 0.55 | 0.44 | 0.44 |

### 关键发现
- CLIP编码器在所有评估维度上表现最一致，特别是OOD泛化能力远超其他编码器，选择CLIP作为默认编码器
- Gen3DEval在外观评估上以大幅优势（0.90 vs GPT-4o的0.59）超越所有对比方法
- 在Gen3DEval-Bench上，Trellis排名第一，AssetGen第二，DreamFusion排名较低
- 多视角输入对不支持多图的模型（如BLIP、PaliGemma）影响严重，使用拼接网格替代效果很差

## 亮点与洞察
- **合成扰动策略**非常巧妙：直接对artist创作的高质量3D网格注入可控缺陷，既保证了数据质量又能大规模扩展，这种"从好到坏"的合成策略可迁移到其他质量评估任务
- **法线图引入**是关键创新：通过同时分析RGB和surface normal渲染，模型能评估几何质量而不仅仅是外观，这在3D评估中非常重要
- **ELO评分系统**的使用让成对比较结果转化为可排序的分数，类似棋类评分，比直接投票更鲁棒

## 局限与展望
- 对Janus face伪影的评估不够稳定，可能因为训练数据中此类样本不够多
- OOD表面评估仍有提升空间，受限于多样化标注的表面比较数据不足
- 对image-to-3D方法的评估会受到text-to-image pipeline质量的影响
- 8.35B参数量虽比GPT-4小很多，但部署成本仍不低，可以考虑蒸馏到更小模型
- 目前仅评估静态3D物体，未涉及动态场景和4D内容

## 相关工作与启发
- **vs GPT4VEval**: 使用通用GPT-4V做3D评估，未针对任务微调；Gen3DEval用专门数据微调8.35B模型，在各维度大幅超越
- **vs CLIP Score**: CLIP仅能评估文本-图像对齐，无法评估外观质量和表面质量；Gen3DEval统一了三个评估维度
- **vs T3Bench**: T3Bench提供了prompt基准但没有自动评估模型；Gen3DEval既提供基准（Gen3DEval-Bench）又提供评估模型
- 该工作的思路——"对领域任务微调vLLM做自动评估"——可迁移到其他生成任务的质量评估

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个专门微调vLLM做3D生成质量评估的工作，方向新但框架基于LLaVA
- 实验充分度: ⭐⭐⭐⭐ 多维度评估、多编码器消融、与多个模型对比，Gen3DEval-Bench设计合理
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题动机阐述充分，表格数据丰富
- 价值: ⭐⭐⭐⭐ 解决了3D生成评估的实际痛点，提供了标准化基准，对社区有推动作用

<!-- RELATED:START -->

## 相关论文

- [Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/3d_vision/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach](../../ICCV2025/3d_vision/how_far_are_ai-generated_videos_from_simulating_the_3d_visual_world_a_learned_3d.md)
- [PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)
- [Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)

<!-- RELATED:END -->
