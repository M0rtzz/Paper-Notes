---
title: >-
  [论文解读] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries
description: >-
  [CVPR2026][多模态] 提出 NaiLIA，一种面向美甲设计图像的多模态检索方法，通过密集意图描述和调色板查询实现细粒度匹配，引入基于置信度分数的松弛对比损失（CRC loss）处理未标注正样本问题，在自建 NAIL-STAR 基准和 Marqo Fashion200K 上大幅超越现有方法。
tags:
  - "CVPR2026"
  - "多模态"
  - "dense intent description"
  - "palette query"
  - "对比学习"
  - "unlabeled positive"
  - "fashion AI"
  - "nail design"
---

# NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries

**会议**: CVPR 2026 Findings  
**arXiv**: [2603.05446](https://arxiv.org/abs/2603.05446)  
**代码**: [项目主页](https://nailia-94dpr.kinsta.page/)  
**领域**: others (多模态检索 / 时尚AI)  
**关键词**: multimodal retrieval, dense intent description, palette query, contrastive learning, unlabeled positive, fashion AI, nail design

## 一句话总结

提出 NaiLIA，一种面向美甲设计图像的多模态检索方法，通过密集意图描述和调色板查询实现细粒度匹配，引入基于置信度分数的松弛对比损失（CRC loss）处理未标注正样本问题，在自建 NAIL-STAR 基准和 Marqo Fashion200K 上大幅超越现有方法。

## 研究背景与动机

**市场需求驱动**：全球美甲沙龙市场规模约 110 亿美元，用户对按偏好搜索美甲设计图片有强烈需求，但现有搜索系统难以处理用户的多层次意图表达。

**图像检索优于生成**：美甲师反映 AI 生成图像常违反物理约束（如不可实现的装饰配件），多个美容平台已限制 AI 生成图像的使用，因此基于真实图像的检索更具实用价值。

**密集意图描述的挑战**：用户描述通常包含绘制元素（图案）、装饰元素（饰品）、主题（如"美人鱼风"）和整体印象（如"梦幻感"），这种多层次的抽象意图对现有视觉-语言模型构成挑战。

**颜色表达的不足**：时尚领域中微妙的颜色差异至关重要，但现有方法忽视了连续色彩输入（如 RGB 色值），仅依赖文本描述无法精确传达色调偏好。

**InfoNCE 损失的固有缺陷**：现有视觉-语言基础模型（CLIP、SigLIP 等）依赖 InfoNCE 损失，将所有非正样本视为负样本，但美甲图像间存在大量相似的未标注正样本，导致相似样本的相似度被错误地最小化。

**抽象层级偏差**：现有模型倾向于检索特定抽象层级的结果（通常偏向写实），例如将"贝壳灵感设计"理解为真实贝壳装饰，而非贝壳主题的艺术化设计。

## 方法详解

### 整体框架

NaiLIA 要解决的是"按用户的复杂意图检索美甲设计图"：用户既会用一段密集文字描述图案/饰品/主题/整体印象，又会给一组 RGB 调色板指定色调。系统把这套多层意图拆给三个模块处理——IPFM 融合文字意图与调色板得到语言-调色板表示 $\bm{l}_+$，VDFM 从三个角度理解候选图像得到视觉表示 $\bm{v}$，二者算相似度后由 CRAM 估计哪些"未标注但其实相似"的样本该被当成正样本，最后用松弛对比损失（CRC loss）把查询和图像对齐、按相似度排序输出。输入形式化为 $\bm{x} = \{\bm{x}_{\text{txt}}, \bm{x}_{\text{pal}}, X_{\text{img}}\}$，其中 $\bm{x}_{\text{txt}}$ 是密集意图描述，$\bm{x}_{\text{pal}} \in \mathbb{R}^{3 \times N_{\text{pal}}}$ 是调色板查询（零个或多个 RGB 颜色），$X_{\text{img}}$ 是待排序图像集。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["密集意图描述 + 调色板查询"]
    IMG["候选美甲设计图像集"]
    subgraph IPFM["1. IPFM 意图-调色板融合"]
        direction TB
        A["LLM 结构化描述<br/>多层描述 MDD + 名词短语 NNP"] --> C["以调色板为 query 交叉注意力<br/>放大颜色相关元素 → 表示 l₊"]
    end
    subgraph VDFM["2. VDFM 视觉设计融合"]
        direction TB
        D["三路视觉表示<br/>单模态 + 多模态对齐 + img2txt"] --> E["Transformer 融合 → 表示 v"]
    end
    Q --> IPFM
    IMG --> VDFM
    IPFM --> S["查询-图像相似度 S"]
    VDFM --> S
    subgraph CRAM["3. CRAM 基于置信度的松弛对齐"]
        direction TB
        F["MLLM 估置信度 c<br/>c ≥ θ 纳入未标注正样本集 Z"] --> G["CRC 损失<br/>L_P + λ·L_UP + λ·L_N"]
    end
    S --> CRAM
    CRAM --> OUT["按相似度排序的检索结果"]
```

### 关键设计

**1. IPFM 意图-调色板融合：把"说不清的颜色偏好"显式注入文字理解**

纯文本无法精确传达微妙色调，而时尚检索里颜色差异恰恰关键。IPFM 先用 LLM（GPT-4o）把原始描述结构化成多层设计描述（MDD）和归一化名词短语（NNP），再用 BEiT-3、SigLIP 等多个文本编码器得到语言表示 $(\bm{l}_{\text{txt}}, \bm{l}_{\text{MDD}}, \bm{l}_{\text{NNP}})$；调色板这边把 RGB 转到更贴合感知的 CIELAB 空间后过 Transformer 得到 $\bm{p}$。关键一步是以调色板 $\bm{p}$ 为 query 对语言表示做交叉注意力，选择性放大颜色相关的元素：$\bm{l}_{+} = \text{CrossAttn}(\bm{p}, \text{TFLayers}([\bm{l}_{\text{txt}}; \bm{l}_{\text{MDD}}; \bm{l}_{\text{NNP}}]))$，让"想要的颜色"真正参与意图编码。

**2. VDFM 视觉设计融合：三种视觉表示互补，补上抽象概念这一块**

视觉编码器擅长颜色形状纹理，却读不懂"美人鱼风""梦幻感"这类抽象设计意图。VDFM 同时取三种表示：DINOv2 给的单模态视觉表示 $\bm{v}_s$ 抓颜色/形状/纹理，BEiT-3 与 SigLIP 图像编码器给的多模态对齐表示 $\bm{v}_a$ 与语言对齐，再用 MLLM（GPT-4o、Qwen2-VL）把图像转写成设计元素/装饰/主题/印象的文字、经文本编码器得到 img2txt 意图结构表示 $\bm{v}_n$，专门捕获抽象概念和空间关系。三者经 Transformer 融合：$\bm{v}^{(i)} = \text{TFLayers}([\bm{v}_s^{(i)}; \bm{v}_a^{(i)}; \bm{v}_n^{(i)}])$。消融显示 $\bm{v}_a$ 最关键，少了它 R@1 直接掉 14.3pp。

**3. CRAM 基于置信度的松弛对齐：承认"相似图其实是正样本"，别再硬推开**

美甲图之间大量相互相似，InfoNCE 把所有非正样本一律当负样本，会错误地把这些相似样本的相似度压低。CRAM 用 MLLM（Qwen2-VL）对每对 $(i,j)$ 估一个置信度分数 $c_{ij} \in [0,1]$（输入查询的 NNP、候选图像及其 NNP），只要 $c_{ij} \geq \theta$ 就把这对加入未标注正样本集 $\mathcal{Z}$，交给损失函数松弛处理。

### 损失函数

基于 $\mathcal{Z}$ 提出 **Confidence-based Relaxed Contrastive (CRC) loss**：

$$\mathcal{L}_{\text{CRC}} = \mathcal{L}_P + \lambda_{UP} \mathcal{L}_{UP} + \lambda_N \mathcal{L}_N$$

- $\mathcal{L}_P = \sum_i (1 - S_{ii})^2$：正样本对的相似度应趋近 1
- $\mathcal{L}_{UP} = \sum_{(i,j) \in \mathcal{Z}} (\max(c_{ij} - S_{ij}, 0))^2$：未标注正样本的相似度应不低于其置信度分数
- $\mathcal{L}_N = \sum_{(i,j) \notin \mathcal{Z}} (\max(S_{ij}, 0))^2$：负样本的相似度应趋近 0

## 实验

### 基准数据集

**NAIL-STAR 基准**（自建）：10,625 张美甲设计图像，208 位标注者提供密集意图描述，平均句长 21.5 词，词汇量 7,014，调色板查询平均包含 2.0 个颜色，图像来自 42 种语言的用户（Pinterest），覆盖多元文化背景。训练/验证/测试 = 8,625/400/1,600。

### 主要结果

| 方法 | NAIL-STAR R@1 | NAIL-STAR MRR | Fashion200K R@1 | Fashion200K MRR |
|------|:---:|:---:|:---:|:---:|
| CLIP | 15.5 | 25.2 | 47.6 | 61.7 |
| SigLIP | 47.5 | 58.8 | 60.3 | 71.9 |
| BEiT-3 | 40.6 | 53.9 | 52.8 | 66.2 |
| BLIP-2 | 20.8 | 33.3 | 65.2 | 75.3 |
| **NaiLIA (desc-only)** | **49.5** | **61.0** | **73.8** | **82.0** |
| **NaiLIA (full)** | **56.4** | **67.6** | **74.6** | **82.7** |

- NAIL-STAR 上 NaiLIA (full) R@1 达 56.4%，比最优基线 SigLIP 高 **8.9pp**
- Marqo Fashion200K 上 R@1 达 74.6%，比最优基线 BLIP-2 高 **9.4pp**
- 所有差异在 p < 0.01 水平下统计显著

### 消融实验与关键发现

| 变体 | R@1 | 相比完整模型 |
|------|:---:|:---:|
| 完整模型 (a) | 56.4 | — |
| 去除 MDD (b) | 54.9 | -1.5 |
| 去除 NNP (c) | 54.5 | -1.9 |
| 去除 MDD+NNP (d) | 51.6 | -4.8 |
| 去除多模态对齐表示 (f) | 42.1 | **-14.3** |
| 去除 img2txt 表示 (g) | 54.0 | -2.4 |
| 使用 InfoNCE 替换 CRC (i) | 52.7 | -3.7 |
| 设 $\lambda_{UP}=0$ (j) | 54.5 | -1.9 |
| 固定 $c_{ij}=0.7$ (k) | 55.1 | -1.3 |

**关键发现**：

1. **多模态对齐表示最关键**：去除 $\bm{v}_a$ 导致 R@1 下降 14.3pp，是最重要的视觉表示组件
2. **CRC loss 的通用性**：在 CLIP 上替换 InfoNCE 为 CRC loss 也能提升 1.0pp R@1，说明 CRC loss 可作为通用检索损失函数
3. **调色板查询的广泛适用性**：为 CLIP 和 SigLIP 添加调色板输入分别提升 5.8pp 和 5.9pp R@1
4. **MLLM 估计置信度优于固定值**：动态估计 $c_{ij}$ 比固定 0.7 高 1.3pp，验证了 MLLM 作为置信度评估器的有效性

## 亮点

- 首次系统性地定义美甲设计语义检索任务（NAIL-STAR），结合密集意图描述与连续色彩调色板查询
- CRC loss 优雅地解决了对比学习中未标注正样本的问题，利用 MLLM 作为置信度估计器，思路可迁移至其他相似图像密集的检索任务
- Img2txt 意图结构表示的设计巧妙——将图像通过 MLLM 转为设计语义描述再编码，弥补了视觉编码器在抽象概念理解上的不足
- 构建了高质量的跨文化 NAIL-STAR 基准数据集（208 位标注者、42 种语言背景），并承诺公开

## 局限性

- 依赖多个大型模型（GPT-4o、Qwen2-VL、BEiT-3、SigLIP、DINOv2）进行预处理和推理，计算成本和延迟较高，实际部署可能受限
- 数据集规模 10,625 张相对较小，可能限制模型在更多样化设计上的泛化能力
- 聚焦于用户无关的检索设置，未考虑用户个性化偏好的建模
- MLLM 置信度估计在训练前预计算，无法随模型训练动态更新，可能存在估计偏差
- 应用场景较为垂直（美甲设计），虽声称 CRC loss 可通用，但仅在时尚领域数据集上验证

## 相关工作

- **时尚AI多模态检索**：EI-CLIP 引入时尚术语扩展 CLIP；CoSMo 处理参考图像+修改文本的组合查询；FashionViL/FAME-ViL 是时尚领域的代表性视觉-语言模型——本文的区别在于引入连续色彩输入和密集意图描述
- **视觉-语言基础模型**：CLIP、SigLIP、BEiT-3、BLIP-2 等通过对比学习实现跨模态对齐；AlphaCLIP 引入 alpha 通道关注感兴趣区域——本文不仅融合多编码器特征，还引入 img2txt 转换捕获抽象概念
- **对比学习中的噪声标签**：InfoNCE 的单标签监督本质上易受噪声影响——本文通过 MLLM 估计未标注正样本置信度来松弛对比损失

## 评分

- 新颖性: ⭐⭐⭐⭐ (CRC loss + 调色板融合 + img2txt 表示的组合具有创新性)
- 实验充分度: ⭐⭐⭐⭐ (两个数据集、完整消融、定性分析、统计显著性检验)
- 写作质量: ⭐⭐⭐⭐ (结构清晰、图示直观、任务定义规范)
- 价值: ⭐⭐⭐ (垂直应用场景限制了影响力，但 CRC loss 思路有一定通用价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ICML 2026\] Envy-Free Allocation of Indivisible Goods via Noisy Queries](../../ICML2026/others/envy-free_allocation_of_indivisible_goods_via_noisy_queries.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](../../ACL2025/others/towards_text-image_interleaved_retrieval.md)

</div>

<!-- RELATED:END -->
