---
title: >-
  [论文解读] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training
description: >-
  [CVPR 2026][多模态][交错多模态生成] Wan-Weaver 提出规划器（VLM）+ 可视化器（DiT）的解耦架构，通过大规模文本代理数据训练规划器而非真实交错数据，在 OpenING 上 Overall 8.67 分超越 Nano Banana 的 8.85，在保持理解能力（MMMU 74.9）的同时实现 SOTA 交错文图生成。
tags:
  - CVPR 2026
  - 多模态
  - 交错多模态生成
  - 解耦训练
  - 多模态VLM
  - 视觉一致性
  - 规划-可视化
---

# Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training

**会议**: CVPR 2026  
**arXiv**: [2603.25706](https://arxiv.org/abs/2603.25706)  
**代码**: [https://doubiiu.github.io/projects/WanWeaver](https://doubiiu.github.io/projects/WanWeaver)  
**领域**: 多模态VLM  
**关键词**: 交错多模态生成、解耦训练、文本代理数据、视觉一致性、规划-可视化

## 一句话总结

Wan-Weaver 提出规划器（VLM）+ 可视化器（DiT）的解耦架构，通过大规模文本代理数据训练规划器而非真实交错数据，在 OpenING 上 Overall 8.67 分超越 Nano Banana 的 8.85，在保持理解能力（MMMU 74.9）的同时实现 SOTA 交错文图生成。

## 研究背景与动机

1. **领域现状**：交错多模态生成（interleaved text-image generation）需要模型根据用户指令生成穿插文字和图片的连贯内容，如图文教程、故事绘制等。GPT-4o+DALL-E3 通过流水线方式领先，开源方案（Anole、Emu3）差距较大。
2. **现有痛点**：(1) 高质量真实交错数据极度稀缺——网页抓取的图文数据质量差且版权风险高；(2) 联合训练文本理解和图像生成容易互相干扰——生成训练损害理解能力；(3) 长序列生成中视觉一致性难以保持——前面生成的角色在后面会"变脸"。
3. **核心矛盾**：交错生成需要同时具备"规划能力"（决定何时插图、图的内容描述）和"视觉一致性"（多张图中保持角色/风格一致），两者的训练信号和数据需求完全不同。
4. **本文目标**：通过解耦训练分别优化规划和视觉化能力，用合成文本代理数据替代稀缺的真实交错数据。
5. **切入角度**：将交错生成分为两个独立可训练的子任务——规划器只需学习"文本中哪里应该插图、图的详细描述是什么"，可以用纯文本代理数据训练；可视化器只需学习"根据描述和参考图生成一致的图片"。
6. **核心 idea**：解耦训练（Decoupled Training）+ 文本代理数据（textual-proxy）+ Dense Prompt Context Window（DPCW）注意力机制。

## 方法详解

### 整体框架

用户指令 → 规划器（QWen2.5-VL-32B-Think）生成含 `<imagine>` 标签的文本+密集图像描述 → 可视化器（Twin DiT）根据密集描述和前序视觉参考生成图像 → DPCW 注意力确保视觉一致性 → 输出交错图文内容。

### 关键设计

1. **解耦训练策略**

    - 功能：分别优化规划和视觉化，避免训练冲突
    - 核心思路：三阶段——(1) 冻结规划器只训练可视化器（文本到图、单图参考、多图参考三种一致性模式）；(2) 冻结可视化器只微调规划器（用文本代理数据，图片替换为密集描述）；(3) DPCW 微调让可视化器适应上下文窗口条件化。训练总计 9.6T token（可视化器）+ 35.72G token（规划器）
    - 设计动机：联合训练时视觉损失和文本损失会互相干扰——消融显示解耦训练的视觉损失曲线更平滑（从 ~0.25 降至 0.15 vs 联合训练的震荡）

2. **文本代理数据**

    - 功能：用纯文本模拟交错数据来训练规划器
    - 核心思路：将目标交错数据中的图片替换为 VLM 生成的密集描述，包裹在 `<imagine>` 标签中。三种数据来源：LLM 生成的用户查询对、VLM 围绕数据库图片生成的查询对、多图叙述（SigLIP聚类后精炼）。生成与理解数据比例 5:1
    - 设计动机：高质量真实交错数据不可获取，但文本代理数据可以无限生成——规划器只需学习"何时插图+描述什么"，不需要真正看到图片

3. **Dense Prompt Context Window (DPCW)**

    - 功能：让可视化器在去噪时关注上下文中的视觉参考
    - 核心思路：在密集 prompt 位置周围设置自注意力窗口，通过注意力 mask 策略让当前生成的图像能看到之前的视觉参考特征。使用 3D RoPE 编码时序位置
    - 设计动机：标准扩散生成只条件化于当前 prompt，无法利用前序图像的视觉信息来保持一致性

### 损失函数 / 训练策略

可视化器：Flow-matching 损失。规划器：标准自回归交叉熵。可视化器分三阶段递进训练（T2I → +SI2I → +MI2I）。

## 实验关键数据

### 主实验

| 方法 | OpenING Overall ↑ | WeaverBench Overall ↑ | MMMU (理解)↑ | GenEval (T2I)↑ | DPG (T2I)↑ |
|------|-------------------|----------------------|-------------|----------------|------------|
| Anole | 5.75 | 3.74 | - | - | - |
| Emu3 | 5.76 | - | - | - | - |
| Gemini+Flux | 7.23 | - | - | - | - |
| GPT-4o+DALL-E3 | 8.20 | - | - | - | - |
| Nano Banana | 8.85 | 8.38 | - | - | - |
| Bagel | - | - | 55.3 | 0.88 | 85.07 |
| **Wan-Weaver** | **8.67** | **8.43** | **74.9** | **0.89** | **87.21** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 解耦 vs 联合训练 | 视觉损失 0.15 vs 0.25 | 解耦更稳定 |
| 数据比例 0g1u | token acc ~0% | 纯理解数据无生成能力 |
| 数据比例 5g1u | 最优 | 生成为主+理解辅助 |
| T2I only | 基础文图对齐 | 无参考能力 |
| +SI2I | 外观保持 | 单图参考 |
| +MI2I | 长程视觉一致 | 完整能力 |

### 关键发现

- Wan-Weaver 保持了接近基座 QWen2.5-VL-32B 的理解能力（MMMU 74.9 vs 75.1），证明解耦训练有效避免了"生成损害理解"
- OpenING 8.67 接近甚至某些指标超越 GPT-4o+DALL-E3（8.20），表明开源方案已接近闭源天花板
- 图像编辑性能（ImgEdit 4.31）大幅超越专用编辑模型 Step1X-Edit（3.06）

## 亮点与洞察

- **文本代理数据的巧妙设计**：用密集描述替代真实图片来训练规划器，完全回避了交错数据稀缺的问题——是一种优雅的"数据降维"思路
- **解耦训练的工程价值**：规划器和可视化器可以独立迭代升级，不需要重新联合训练——系统维护成本大幅降低
- **理解+生成+编辑三合一**：同一个模型在理解(MMMU 74.9)、生成(GenEval 0.89)、编辑(ImgEdit 4.31)上都达到SOTA级别

## 局限与展望

- 用户必须预先指定生成图像的分辨率和宽高比，不能自适应根据内容决定
- 顺序生成瓶颈——所有已生成内容需要回馈模型，长序列时GPU内存消耗线性增长
- 生成能力的提升没有反哺理解能力——双向增强仍是开放问题
- 偶尔出现结构坍塌（如网格布局替代预期的独立图片），几何推理和符号接地仍有缺陷

## 相关工作与启发

- **vs GPT-4o+DALL-E3**: 闭源流水线方案 OpenING 8.20，Wan-Weaver 8.67——主要优势在多步一致性（8.56 vs 8.38）和内容完整性（9.41 vs 8.66）
- **vs Bagel/UniWorld**: 这些统一模型联合训练导致理解能力下降（MMMU 55-59），Wan-Weaver 通过解耦保持 74.9
- **vs Emu3**: 纯离散 token 方案 OpenING 仅 5.76，与 Wan-Weaver 差距源于视觉质量和一致性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 解耦训练+文本代理数据是对交错生成范式的重要创新
- 实验充分度: ⭐⭐⭐⭐⭐ OpenING+WeaverBench+单模态全面评测+详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰但训练细节较密集
- 价值: ⭐⭐⭐⭐⭐ 开源交错生成接近闭源水平的里程碑式工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [\[ICCV 2025\] WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation](../../ICCV2025/multimodal_vlm/wikiautogen_towards_multi-modal_wikipedia-style_article_generation.md)
- [\[CVPR 2026\] SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)

</div>

<!-- RELATED:END -->
