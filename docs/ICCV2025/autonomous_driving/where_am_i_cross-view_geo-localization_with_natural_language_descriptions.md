---
title: >-
  [论文解读] Where am I? Cross-View Geo-localization with Natural Language Descriptions
description: >-
  [ICCV 2025][自动驾驶][Cross-View Geo-localization] 引入基于自然语言描述的跨视角地理定位新任务，构建覆盖3个城市3万+坐标的CVG-Text多模态数据集（街景+卫星+OSM+文本），并提出CrossText2Loc方法——通过扩展位置嵌入处理长文本和可解释检索模块提供定位理由，Top-1召回率提升超10%。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "Cross-View Geo-localization"
  - "Natural Language"
  - "Text-to-Image Retrieval"
  - "Satellite"
  - "OSM"
  - "LMM"
  - "Explainable Retrieval"
---

# Where am I? Cross-View Geo-localization with Natural Language Descriptions

**会议**: ICCV 2025  
**arXiv**: [2412.17007](https://arxiv.org/abs/2412.17007)  
**代码**: [yejy53.github.io/CVG-Text](https://yejy53.github.io/CVG-Text/)  
**领域**: Autonomous Driving / 跨视角地理定位  
**关键词**: Cross-View Geo-localization, Natural Language, Text-to-Image Retrieval, Satellite, OSM, LMM, Explainable Retrieval

## 一句话总结

引入基于自然语言描述的跨视角地理定位新任务，构建覆盖3个城市3万+坐标的CVG-Text多模态数据集（街景+卫星+OSM+文本），并提出CrossText2Loc方法——通过扩展位置嵌入处理长文本和可解释检索模块提供定位理由，Top-1召回率提升超10%。

## 研究背景与动机

### 问题场景

在GPS信号受干扰的场景（城市高楼遮挡、室内-室外过渡、紧急呼叫等），用户需要通过自然语言描述周围环境来确定位置。例如：
- 出租车乘客用语言告诉司机自己的位置
- 行人在紧急电话中描述周围环境供救援定位

### 现有方法的局限

**跨视角定位聚焦图像匹配**：Sample4G、SAFA等方法专注于街景图像到卫星图像的检索，但现实中用户可能只能提供文字描述

**文本定位局限于点云**：Text2Pose、Text2Loc等在3D点云中做文本定位，但点云获取成本高、存储开销大，难以全球规模部署

**卫星/OSM数据更实用**：卫星图像和OpenStreetMap具有全球覆盖、存储成本低的优势，但文本到卫星/OSM的跨视角检索此前未被研究

**长文本处理不足**：场景描述通常较长（平均126 tokens），而CLIP等模型固定最大序列长度77 tokens，会截断关键信息

## 方法详解

### CVG-Text数据集构建

#### 数据收集
- 覆盖3个城市：纽约（城市）、布里斯班（郊区）、东京（城市）
- 30,000+坐标点，每点包含：
    - 全景街景图像（2048×1024）和单视角街景
    - 卫星图像（512×512，zoom level 20，分辨率~0.12m）
    - OSM栅格瓦片（512×512，保留POI标识）

#### GPT-4o驱动的文本生成

采用渐进式场景分析策略：

1. **OCR预处理**：PaddleOCR提取街景中的文字信息（店名、公交站牌等），帮助GPT精确捕获关键定位线索，减少幻觉
2. **开放世界分割**：对街景图像进行语义分割，提供位置和语义细节；同时过滤移动物体（如车辆）上的OCR结果
3. **系统化提示**：引导GPT-4o按"道路特征→建筑标识→整体环境"的顺序渐进描述，使用前后左右等简单方向词
4. **质量控制**：格式过滤→GPT自查→人工专家审核（20%样本，10人×100小时，通过率77.6%）

文本统计：平均长度126 tokens，词汇丰富度（TTR）0.76，文本间相似度低（0.17），高质量。

### CrossText2Loc模型

#### 图像-文本对比学习

双流架构：文本编码器+视觉编码器，通过对比学习对齐跨域特征：

$$L_{itc} = \sum_{i=1}^n \sum_{j=1}^n -\log\frac{\exp(\text{sim}(v_i, t_j)/\tau)}{\sum_{k=1}^n \exp(\text{sim}(v_i, t_k)/\tau)}$$

其中 $\tau$ 为可学习温度参数。

#### 扩展位置嵌入（Extended Positional Embedding, EPE）

场景描述平均126 tokens，但CLIP限制77 tokens。通过线性插值扩展位置嵌入到 $N=300$ tokens：

$$P^*(x) = (1-(x-\lfloor x\rfloor)) \cdot P(\lfloor x\rfloor) + (x-\lfloor x\rfloor) \cdot P(\lceil x\rceil)$$

与LongCLIP不同，由于GPT生成的文本开头没有突出的短标题，采用**全文插值**而非知识保持拉伸。

#### 可解释检索模块（ERM）

推理时的可选模块，提升检索的可解释性和可信度：

1. **注意力热力图生成**：从起始层 $s$ 到输出层 $L$ 迭代累积非负梯度贡献：

$$R^{(l)} = R^{(l-1)} + \frac{1}{H}\sum_{h=1}^H \max(0, \nabla A_h^{(l)} \odot A_h^{(l)}) R^{(l-1)}$$

2. **LMM解释**：将文本和图像热力图输入GPT-4o，分析关键线索→比较推理→输出检索理由和置信度
3. **置信度重排**：将ERM置信度与相似度分数归一化求和重排Top-5结果

## 实验

### 主实验：跨视角文本检索定位

| 方法 | 纽约-卫星R@1 | 纽约-OSM R@1 | 布里斯班-卫星R@1 | 布里斯班-OSM R@1 | 东京-卫星R@1 | 东京-OSM R@1 |
|------|---|---|---|---|---|---|
| CLIP-L/14 | 35.08 | 31.50 | 34.08 | 32.50 | 28.08 | 21.00 |
| SigLIP-SO400M | 33.50 | 27.75 | 34.25 | 29.75 | 28.42 | 17.50 |
| BLIP | 34.58 | **52.92** | 34.50 | 43.00 | 29.75 | 30.67 |
| **Ours (w/o ERM)** | 46.25 | 59.08 | 43.58 | 46.08 | 36.83 | 34.33 |
| **Ours (w/ ERM)** | **50.33** | **62.33** | **47.58** | **48.75** | **41.75** | **36.92** |

**关键发现**：
- CrossText2Loc比最强基线BLIP在卫星检索上提升15.75%（纽约），在OSM检索上提升9.41%
- ERM重排进一步提升4-5% R@1，模拟了用户根据理由做决策的过程
- 纽约OSM效果最好（丰富的POI数据如公交站、店名），东京最差（CLIP对日文预训练不足）

### 消融实验：EPE模块效果

| 方法 | 卫星R@1 | OSM R@1 |
|------|--------|---------|
| CLIP | 35.08 | 31.50 |
| **CLIP + EPE** | **46.25** | **59.08** |
| SigLIP | 19.67 | 20.17 |
| **SigLIP + EPE** | **29.50** | **45.25** |

EPE在两种编码器上均带来显著提升，OSM检索提升尤为突出（+27.6%），证明长文本中的细节信息对POI匹配至关重要。

### 消融实验：文本生成质量对比

| 文本来源 | Len | TTR | Simi. | R@1-OSM | R@1-Sat |
|---------|-----|-----|-------|---------|---------|
| 直接GPT生成 | 108 | 0.74 | 0.22 | 25.17 | 38.00 |
| **CVG-Text（OCR+分割+提示链）** | **126** | **0.76** | **0.17** | **59.08** | **46.25** |

OCR辅助精确捕获街景文字→GPT减少幻觉→OSM检索提升33.9%。

### 跨视角检索的文本增强

| 查询方式 | OSM R@1 | 卫星R@1 |
|---------|---------|---------|
| 仅街景图像(Sample4G) | 27.10 | 91.70 |
| 仅文本 | 59.08 | 46.25 |
| **图像+文本融合** | **67.30** | **98.40** |

文本分支为传统跨视角检索提供互补信息，OSM检索提升40.2%。

## 亮点与洞察

1. **新任务定义**：首次提出用自然语言描述进行跨视角地理定位，填补了文本→卫星/OSM检索的研究空白
2. **高质量数据集**：CVG-Text通过OCR+分割+GPT-4o的渐进式管线，生成了比直接GPT生成质量高得多的场景描述
3. **长文本处理**：EPE用简单的线性插值扩展位置嵌入，但OSM检索提升27.6%，证明场景描述中的细节不容忽视
4. **可解释检索**：ERM不仅提供相似度分数，还给出自然语言检索理由和置信度——模拟了用户在实际应用中的决策过程
5. **实用价值明确**：文本检索可与图像检索互补，融合后卫星检索R@1达98.4%

## 局限性

1. 文本描述由GPT-4o生成，与真实用户的描述习惯有差距（真实用户描述更简短、更模糊）
2. 仅覆盖3个城市，地域多样性有限，不同地区的街景风格差异未充分考虑
3. ERM依赖GPT-4o进行推理解释，增加推理成本和延迟
4. 东京性能较差暴露了CLIP在非英语场景下的局限性
5. 检索范围 $M=100$（约10 km²），大范围检索的可扩展性需验证

## 相关工作

- **跨视角地理定位**：CVUSA、VIGOR、Sample4G——图像到图像检索
- **视觉语言导航与定位**：Text2Pose、Text2Loc——点云中的文本定位，成本高
- **LMM数据合成**：LatteCLIP——利用LMM合成文本用于无监督CLIP微调
- **多模态对齐**：CLIP、LongCLIP——文本-图像对比学习，但序列长度受限

## 评分

- 新颖性：⭐⭐⭐⭐⭐（开创性地定义文本到卫星/OSM的跨视角定位任务）
- 技术深度：⭐⭐⭐⭐（EPE简洁有效，ERM增加可解释性，但方法整体偏工程）
- 实验完整度：⭐⭐⭐⭐（多城市、多数据源、多消融，但缺少与其他文本定位方法的对比）
- 实用价值：⭐⭐⭐⭐⭐（紧急定位、行人导航等场景需求明确）
- 总体推荐：⭐⭐⭐⭐（新任务+新数据集的完整工作，有望开辟新研究方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](../../CVPR2026/autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](../../CVPR2026/autonomous_driving/vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[ICCV 2025\] Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations](beyond_one_shot_beyond_one_perspective_cross-view_and_long-horizon_distillation_.md)

</div>

<!-- RELATED:END -->
