---
title: >-
  [论文解读] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding
description: >-
  [ACL 2025][多模态][GUI grounding] 提出R-VLM，将传统目标检测中的区域提议（region proposal）和IoU感知损失引入VLM的GUI元素定位，通过两阶段放大推理和IoU加权交叉熵损失，在ScreenSpot和AgentStudio上平均提升13%的grounding准确率。
tags:
  - ACL 2025
  - 多模态
  - GUI grounding
  - 视觉语言模型
  - 区域感知
  - IoU损失
  - 两阶段放大
  - GUI自动化
---

# R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding

**会议**: ACL 2025  
**arXiv**: [2507.05673](https://arxiv.org/abs/2507.05673)  
**作者**: Joonhyung Park (KAIST), Peng Tang, Sagnik Das, Srikar Appalaraju, Kunwar Yashraj Singh, R. Manmatha, Shabnam Ghadar (AWS AI Labs)  
**代码**: 未公开  
**领域**: multimodal_vlm  
**关键词**: GUI grounding, 视觉语言模型, 区域感知, IoU损失, 两阶段放大, GUI自动化  

## 一句话总结

提出R-VLM，将传统目标检测中的区域提议（region proposal）和IoU感知损失引入VLM的GUI元素定位，通过两阶段放大推理和IoU加权交叉熵损失，在ScreenSpot和AgentStudio上平均提升13%的grounding准确率。

## 研究背景与动机

### 问题背景
GUI自动化Agent需要精确定位屏幕截图中的界面元素（按钮、图标、文本框等），这是所有后续操作执行的基础。现有的纯视觉GUI Agent直接从完整截图预测元素坐标，面临两个核心困难：（1）截图包含大量无关元素，布局复杂且多尺度；（2）VLM将坐标当作离散token用标准交叉熵训练，缺乏反映定位质量的学习信号。

### 已有工作的不足
- **SeeClick**等方法虽然在GUI grounding预训练上取得进展，但直接从全图预测坐标，小元素尤其容易定位失败
- 现有方法使用普通交叉熵损失训练坐标token，无法像目标检测中的IoU回归损失那样引导模型关注定位精度
- 作者对SeeClick在ScreenSpot上的预测做了IoU分布分析，发现预测的IoU值普遍偏低（集中在0-0.3区间），且小元素的准确率显著低于大元素

### 核心动机
将经典目标检测中已验证有效的两大策略——**区域提议+裁剪放大**和**IoU感知训练目标**——适配到VLM的GUI grounding场景中，弥补VLM在精细定位上的先天不足。

## 方法详解

### 整体框架
R-VLM在现有VLM（Qwen-VL）基础上引入两个模块：（1）两阶段放大grounding（推理阶段）+ 放大数据的指令微调（训练阶段）；（2）IoU感知加权交叉熵损失。两者互补——IoU损失让第一阶段预测更准确，更准确的region proposal反过来让第二阶段放大更有效。

### 关键设计1：两阶段放大grounding

**推理流程**：
1. 第一阶段：输入完整截图和用户指令，模型预测初始bounding box，作为region proposal
2. 根据初始预测的尺寸确定放大倍率$k$——元素越小，放大越多
3. 以初始预测为中心裁剪并放大该区域，送入模型进行第二阶段预测
4. 将第二阶段在放大视图中的坐标反变换回原图坐标，作为最终输出

**放大指令微调数据生成**：
- 不直接用GT框做裁剪区域，而是对GT施加随机扰动，生成GIoU > σ的噪声框来模拟第一阶段的不精确预测
- 裁剪放大后，配以特定指令模板（如"Given the zoomed-in view centered on the initial prediction, predict a detailed bounding box for [INSTRUCTION]"）
- 标签坐标相应更新为放大视图中的相对坐标

**关键优势**：该方法可以**无需训练**直接应用于任意VLM（training-free），仅在推理时执行两次前向传播即可获得显著提升。

### 关键设计2：IoU感知加权交叉熵损失

**核心思想**：在GT框周围生成$M$个伪GT框，根据每个伪框与真实GT的GIoU值为其分配不同的交叉熵权重。GIoU越高权重越接近1，GIoU越低权重越小，从而引导模型理解"高IoU预测比低IoU预测更好"的概念。

**损失函数**：
$$\mathcal{L}_{\text{IoU}}^{\text{CE}} = -\sum_{i=1}^{M} w_{\text{IoU}}^{(i)} \mathbf{b}^{(i)} \log \hat{\mathbf{b}}^{(i)} - \sum_{j=1}^{N} y_{\text{other}}^{(j)} \log \hat{y}_{\text{other}}^{(j)}$$

其中权重 $w_{\text{IoU}}^{(i)} = 1 + \frac{1}{2}\log(\text{GIoU}(\mathbf{b}^{(i)}, \mathbf{b}^{(0)}))$，使用对数尺度使低IoU伪框受到更大的惩罚。

**高效实现**：
- 将$M$个伪框拼接在GT框之后，**单次前向传播**完成所有伪框的损失计算
- 修改attention mask：防止伪框之间互相attend
- 修改RoPE位置编码：将GT框的位置编码复制给所有伪框，确保推理时模型仅输出单个坐标

### 关键设计3：协同效应
消融实验显示，IoU损失和放大grounding有互补效应：IoU损失改善了第一阶段预测的精度（使region proposal更靠近目标），从而让第二阶段放大更有效。

## 实验关键数据

### 表1：ScreenSpot GUI grounding准确率

| 方法 | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | 平均 |
|------|:-----------:|:-----------:|:------------:|:------------:|:--------:|:--------:|:----:|
| GPT-4V | 22.6 | 24.5 | 20.2 | 11.8 | 9.2 | 8.8 | 16.2 |
| CogAgent | 67.0 | 24.0 | 74.2 | 20.0 | 70.4 | 28.6 | 47.4 |
| SeeClick | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| **R-VLM** | **85.0** | **61.1** | **81.4** | **52.8** | **66.5** | **51.4** | **66.3 (+12.9)** |

Desktop Icon和Web Icon的提升尤为显著（分别+22.8和+18.9），这些场景中小图标密集排列，放大策略最为有效。

### 表2：AgentStudio GroundUI-1K grounding准确率

| 方法 | Web | Desktop | Mobile | 平均 |
|------|:---:|:-------:|:------:|:----:|
| GPT-4o | 7.5 | 8.3 | 26.3 | 13.4 |
| Claude-3.5-Sonnet | 13.0 | 14.0 | 26.3 | 17.3 |
| Gemini-1.5-pro | 31.2 | 24.3 | 51.3 | 35.2 |
| SeeClick | 64.3 | 44.3 | 73.7 | 61.1 |
| **R-VLM** | **76.5** | **65.3** | **79.7** | **74.1 (+13.0)** |

R-VLM在使用相同预训练数据和架构的条件下，大幅领先商用大模型和专用Agent模型。Desktop上21%的绝对提升最为突出。

### 表3：消融实验（ScreenSpot）

| IoU损失 | 放大微调 | 放大推理 | 平均准确率 |
|:-------:|:-------:|:-------:|:----------:|
| ✗ | ✗ | ✗ | 53.4 |
| ✗ | ✗ | ✓ | 61.7 (+8.3) |
| ✗ | ✓ | ✓ | 63.9 (+10.5) |
| ✓ | ✓ | ✓ | 66.4 (+12.9) |

仅在推理时加入放大即可获得8.3%提升，三个组件叠加后达到12.9%总提升。

### 表4：GUI导航任务

AITW上R-VLM平均action matching score达64.9%（+5.6%），click准确率71.0%（+4.6%）。Mind2Web上step success rate在Cross-Website设定下提升9.7%（16.4% → 26.1%），说明精确grounding直接转化为导航成功率的提升。

## 关键发现

- **IoU分布显著右移**：R-VLM的预测IoU分布从集中在0-0.3区间变为高IoU区间占主导，验证IoU损失确实让模型学会了"追求高IoU"
- **小元素提升最大**：按元素尺寸分组分析，baseline在小元素上准确率极低，R-VLM通过放大机制大幅改善了小元素定位
- **Training-free即有效**：在更强的UGround（10M数据、LLaVA-NeXT架构）上直接应用两阶段放大（无需重训），ScreenSpot平均提升1.8%，Multimodal-Mind2Web提升3.2%，证明该方法的通用性
- **放大步数的边际递减**：从2步到4步仅提升1.1%，但推理延迟翻倍（2.7s→5.6s/样本），2步是最佳性价比

## 亮点与洞察

- **传统检测思想的优雅迁移**：将Faster R-CNN时代的region proposal + IoU回归思路适配到VLM的token空间，解决了"VLM坐标是离散token无法直接优化IoU"的矛盾
- **高效的IoU损失实现**：通过attention mask修改和RoPE共享，将$M+1$次前向传播压缩为1次，几乎不增加训练开销
- **即插即用的推理策略**：两阶段放大不需要改训练流程、不需要改模型架构，任何VLM都可以直接使用
- **数据增强的巧妙设计**：用GIoU阈值控制噪声框的偏移程度来模拟真实的第一阶段预测误差，比直接用GT裁剪更贴近实际推理场景

## 局限性

- **第一阶段精度的天花板**：如果初始预测严重偏离目标（放大区域不包含目标），第二阶段无法纠正。作者提到未来可以生成多个region proposal来提升召回率
- **仅验证在Qwen-VL上**：主实验的IoU损失和放大微调仅在9.6B的Qwen-VL上验证，未在更大或更新的模型（如Qwen2-VL、InternVL）上实验
- **推理延迟翻倍**：两阶段放大需要两次前向传播，推理时间从1.4s增至2.7s/样本
- **GUI特定方法**：放大策略依赖GUI元素相对整张截图较小的先验，未验证在自然图像grounding上的效果
- **预训练数据固定**：使用的是SeeClick的1M预训练数据，未探索数据规模扩大带来的影响

## 相关工作与启发

- **SeeClick** (ACL 2024)：本文的直接基线，在Qwen-VL上用GUI grounding任务预训练，R-VLM在完全相同的数据和架构上实现大幅提升
- **CogAgent** (CVPR 2024)：引入专用高分辨率路径处理GUI截图，但grounding准确率仍不高
- **UGround** (ICLR 2025)：用10M数据在LLaVA-NeXT上大规模预训练，R-VLM的training-free放大策略在其上仍有3%+提升
- **Faster R-CNN** 系列：R-VLM的核心灵感来源，两阶段检测思路经过二十年仍在新场景中发挥作用
- **ZoomEye** (Shen et al. 2024)：类似的放大思路但用树结构搜索，R-VLM的方法更简洁——单次放大即可

**启发**：VLM在精细定位任务上的瓶颈不仅是模型能力问题，更是训练目标和推理策略的问题。传统CV中经过充分验证的范式（如region-based detection）在VLM时代依然有价值，关键在于如何适配token化的输出空间。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将目标检测经典方法适配到VLM是巧妙的，但放大思路本身已有先例
- 实验充分度: ⭐⭐⭐⭐ — 覆盖grounding和导航两类任务、4个benchmark、training-free验证和详细消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，预备分析充分铺垫了方法动机
- 价值: ⭐⭐⭐⭐ — 13%的grounding提升有实际意义，即插即用特性降低了使用门槛

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Aria-UI: Visual Grounding for GUI Instructions](aria-ui_visual_grounding_for_gui_instructions.md)
- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)
- [\[ICLR 2026\] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](../../ICLR2026/multimodal_vlm/grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)

<!-- RELATED:END -->
