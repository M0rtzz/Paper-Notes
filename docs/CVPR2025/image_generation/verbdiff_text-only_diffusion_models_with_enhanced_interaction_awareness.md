---
title: >-
  [论文解读] VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness
description: >-
  [CVPR 2025][图像生成][人物交互生成] 提出 VerbDiff，一个无需额外条件（如边界框）即可生成准确人物交互图像的文本到图像扩散模型，通过关系解耦引导（RDG）消除交互词偏差，利用交互区域模块（IR Module）从交叉注意力图中提取局部交互区域进行方向引导。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "人物交互生成"
  - "动词语义理解"
  - "关系解耦引导"
  - "交互区域定位"
  - "文本到图像扩散"
---

# VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness

**会议**: CVPR 2025  
**arXiv**: [2503.16406](https://arxiv.org/abs/2503.16406)  
**代码**: 无  
**领域**: 目标检测/图像生成  
**关键词**: 人物交互生成, 动词语义理解, 关系解耦引导, 交互区域定位, 文本到图像扩散

## 一句话总结

提出 VerbDiff，一个无需额外条件（如边界框）即可生成准确人物交互图像的文本到图像扩散模型，通过关系解耦引导（RDG）消除交互词偏差，利用交互区域模块（IR Module）从交叉注意力图中提取局部交互区域进行方向引导。

## 研究背景与动机

- 文本到图像扩散模型在描绘人物交互方面表现不佳，难以区分语义上不同的交互动词（如 "walking a bicycle" vs "riding a bicycle"）
- CLIP 存在强烈的物体偏差，倾向于关注提示中的物体名词而忽略动词的语义差异
- 现有方法依赖额外条件（LLM布局、边界框）来提供显式关系信息，但仍缺乏对交互语义的真正理解
- InteractDiffusion 即使有边界框辅助，在语义相近的交互词（如walking vs riding）共享相似bbox时仍然失败
- 生成图像的交互存在偏差，倾向于数据分布中频率最高的动词（如backpack总是生成wearing而非holding）
- HICO-DET 数据集中交互词分布极度长尾，常见动词主导生成结果
- CLIP相似度指标对交互动词的细微语义差异不够敏感
- 现有方法在无额外标注条件下难以定位生成图像中的具体交互区域

## 方法详解

### 整体框架

VerbDiff 基于 Stable Diffusion v1.4 构建，仅训练交叉注意力层。包含两个核心模块：**关系解耦引导（RDG）**利用频率锚文本和三元组损失消除交互偏差，**交互方向引导（IDG）**通过 IR 模块从交叉注意力图提取交互区域，引导模型关注局部交互细节。训练在HICO-DET上进行，仅需文本输入无需边界框。

### 关键设计

**设计一：关系解耦引导（RDG）**
- **功能**：消除生成图像中的交互偏差，增强对不同动词语义差异的理解
- **核心思路**：对每个人-物体对，定义频率锚词 $r^{anc} = \arg\max_{r \in R_o} \mathcal{C}(r|o)$（该对中出现最频繁的动词），构建锚文本 $T^{anc}$。通过三元组损失使生成图像特征 $f^{gen}$ 接近正确交互文本 $e^{gt}$ 而远离锚文本 $e^{anc}$：$\mathcal{L}_{\text{triple}} = \max(0, m + \text{sim}(f^{gen}, e^{gt}) - \text{sim}(f^{gen}, e^{anc}))$。同时用mask区域提取真实图像特征进行图像对齐损失，并乘以有效数 $\alpha(k)$ 平衡长尾分布
- **设计动机**：生成图像的偏差交互往往对应数据中最频繁的动词，通过显式推离锚文本特征实现语义区分

**设计二：交互区域模块（IR Module）**
- **功能**：从生成图像的交叉注意力图中自动提取交互区域，无需边界框
- **核心思路**：利用交叉注意力图 $\mathcal{A}_h, \mathcal{A}_r, \mathcal{A}_o$ 分别对应 h/r/o token，通过质心提取机制计算各token的中心点 $c_h, c_r, c_o$。定义交互中心 $c_{rel}$ 为三点质心，交互区域 $B_{rel}^{gen} = c_{rel} \pm \|c_h - c_o\|_2^2$
- **设计动机**：全图级别的特征对齐不够精细，聚焦交互发生的局部区域能更好地捕获交互细节

**设计三：交互方向引导（IDG）**
- **功能**：引导模型在局部交互区域内修改图像特征，使其更接近真实交互
- **核心思路**：从交互区域提取 $f_{rel}^{gt}$ 和 $f_{rel}^{gen}$，计算偏差特征 $f_{rel}^{bias} = f_{rel}^{gt} - f_{rel}^{gen}$，设计方向引导损失使全图级别的修改方向与交互区域的修改方向对齐：$\mathcal{L}_{\text{IDG}} = 1 - \frac{(f_{\mathcal{M}}^{gt} - f^{gen}) \cdot f_{rel}^{bias}}{|f_{\mathcal{M}}^{gt} - f^{gen}||f_{rel}^{bias}|}$
- **设计动机**：确保模型对图像的修改集中在交互区域而非整体布局，实现细粒度的交互语义对齐

### 损失函数

总损失 $\mathcal{L}_{\text{total}} = \lambda_1 \cdot \mathcal{L}_{\text{rec}} + \lambda_2 \cdot \mathcal{L}_{\text{RDG}} + \lambda_3 \cdot \mathcal{L}_{\text{IDG}}$，其中 $\lambda_1=1.0, \lambda_2=10, \lambda_3=0.8$。重建损失使用mask约束仅聚焦对应交互区域，RDG损失乘以长尾平衡因子。

## 实验关键数据

### 主实验：交互相似度和准确率

| 模型 | CLIP T2T | S-BERT T2T | HOI Acc Def.(Full) | KO.(Full) |
|------|----------|------------|-------------------|-----------|
| SD | 0.725 | 0.620 | 16.09 / 20.08 | 18.22 / 21.69 |
| GLIGEN | 0.683 | 0.554 | 15.88 / 17.83 | 17.91 / 19.35 |
| InteractDiffusion | 0.703 | 0.575 | 19.67 / 23.53 | 21.31 / 24.86 |
| **VerbDiff** | **0.733** | **0.633** | **22.59 / 27.05** | **24.79 / 28.43** |

### 消融实验：各损失组件贡献

| 设置 | CLIP T2T | S-BERT T2T | HOI Acc Def. | KO. |
|------|----------|------------|-------------|-----|
| $\mathcal{L}_{rec}$ only | 0.691 | 0.582 | 19.38 | 20.89 |
| +$\mathcal{L}_{triple}$+$\mathcal{L}_{align}$ | 0.700 | 0.589 | 20.32 | 21.87 |
| +$\mathcal{L}_{triple}$+$\mathcal{L}_{IDG}$ | 0.710 | 0.610 | 23.39 | 24.51 |
| **All (Full)** | **0.733** | **0.633** | **22.59** | **24.79** |

### 关键发现
- VerbDiff 在所有评估设置下全面超越需要显式边界框的 InteractDiffusion
- S-BERT指标比CLIP更能反映交互语义差异（VerbDiff在S-BERT上提升更显著）
- IDG贡献最大（+HOI Acc 3.0+），说明局部交互区域的聚焦至关重要
- 在多交互复杂提示下仍能准确区分不同交互动词，与DALL-E 3表现接近
- HOI准确率接近真实HICO-DET数据的水平（Def. 22.59 vs 26.52）

## 亮点与洞察

1. **频率锚文本的发现**：生成偏差对应最频繁动词，这一观察为解耦设计提供了清晰动机
2. **从交叉注意力图定位交互区域**：无需额外标注即可定位交互发生的位置，设计简洁有效
3. **引入S-BERT评估指标**：弥补了CLIP在交互语义细微区分上的不足
4. **仅训练交叉注意力层**：轻量高效，17小时训练即完成

## 局限与展望

- 基于SD v1.4的基础模型容量有限，与DALL-E 3等大模型仍有差距
- 仅在HICO-DET上训练和评估，泛化到其他HOI数据集未验证
- IR模块基于交叉注意力质心的简单几何计算，对复杂多人场景可能不够鲁棒
- 未来可结合更强的VLM和更大规模数据进一步提升交互理解

## 相关工作与启发

- 与InteractDiffusion需要显式边界框不同，VerbDiff仅依赖文本实现更好的交互理解
- 频率锚文本解耦思路可推广到其他存在类别不平衡偏差的生成任务
- 交叉注意力图的语义定位能力为无标注区域提取提供了新思路

## 评分

⭐⭐⭐⭐ — 问题定义精准，基于频率偏差的解耦思路新颖；实验对比充分，仅文本条件下超越需要边界框的方法令人印象深刻。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](scaling_down_text_encoders_of_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)

</div>

<!-- RELATED:END -->
