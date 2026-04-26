---
title: >-
  [论文解读] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields
description: >-
  [ICCV 2025][3D视觉][3D语言场] 提出 GeoProg3D，首个支持城市级高保真3D场景自然语言交互的视觉编程框架，通过地理感知的城市级3D语言场（GCLF）和地理视觉API（GV-APIs），结合LLM推理引擎实现组合式地理空间推理，在新提出的952条查询的GeoEval3D基准上全面超越现有3D语言场和VLM方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D语言场
  - 城市级场景
  - 视觉编程
  - 组合推理
  - 地理信息
---

# GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields

**会议**: ICCV 2025  
**arXiv**: [2506.23352](https://arxiv.org/abs/2506.23352)  
**代码**: [snskysk/GeoProg3D](https://snskysk.github.io/GeoProg3D/)  
**领域**: 3d_vision  
**关键词**: 3D语言场, 城市级场景, 视觉编程, 组合推理, 地理信息

## 一句话总结

提出 GeoProg3D，首个支持城市级高保真3D场景自然语言交互的视觉编程框架，通过地理感知的城市级3D语言场（GCLF）和地理视觉API（GV-APIs），结合LLM推理引擎实现组合式地理空间推理，在新提出的952条查询的GeoEval3D基准上全面超越现有3D语言场和VLM方法。

## 研究背景与动机

**3D语言场**（3D Language Fields）等方法实现了通过自然语言查询与3D场景的交互（如LangSplat、LERF），但在扩展到城市级场景时面临两个根本性挑战：

**可扩展性不足**：现有方法（LangSplat、LERF等）主要面向室内场景，直接应用于超过 $1 \text{km}^2$ 的大规模城市数据时面临严重的内存和计算瓶颈。例如 LERF 和 LangSplat 在 UrbanScene3D（5×10⁶ m²）上直接OOM。

**任务多样性有限**：当前3D语言场主要局限于单词级的物体定位（grounding），无法满足城市应用中的多种需求——空间关系解释、物体计数与尺寸量化、地标识别等。城市场景中的查询天然具有组合性（如"距Chase银行100米内的红色店铺"），需要分解为多步推理。

**核心洞察**：将城市级3D语言场与视觉编程框架结合——用层次化3D Gaussian实现高效大规模场景重建和语言嵌入，用地理API提供丰富的操作接口，再由LLM动态组合这些API来应对多样化的查询需求。

## 方法详解

### 整体框架

GeoProg3D 包含两个核心组件和一个推理引擎：

1. **GCLF（地理感知城市级3D语言场）**：基于树结构层次化3D Gaussian构建，嵌入CLIP语言特征和地理坐标信息
2. **GV-APIs（地理视觉API）**：9个专用的图像与地理处理函数
3. **LLM推理引擎**：通过上下文学习（ICL）生成Python程序，动态组合GV-APIs操作GCLF

查询处理流程为两步：
- **Step 1 - 程序生成**：$z = \Pi(q, R)$，LLM $\Pi$ 根据查询 $q$ 和上下文示例 $R$ 生成Python程序 $z$
- **Step 2 - 程序执行**：$a = \Lambda(z; \mathcal{T})$，Python引擎 $\Lambda$ 在GCLF $\mathcal{T}$ 上执行程序获取答案 $a$

### 关键设计一：GCLF — 城市级3D语言场

**场景表示**：采用树结构3D Gaussian [Ren et al.]，学习Gaussian之间的嵌套关系。渲染时动态选择层级——在图像空间直径小于1像素的层级渲染Gaussian，远处场景用粗粒度表示，兼顾质量与效率。

| 场景 | LangSplat Gaussian数 | GCLF Gaussian数 | LangSplat速度(ms) | GCLF速度(ms) |
|------|---------------------|-----------------|-------------------|-------------|
| Center Blvd | 37,212 | 1,136,015 | 2.73 | 14.46 |
| UrbanScene3D | OOM | 37,813,418 | OOM | 20.83 |

虽然GCLF的Gaussian数量是LangSplat的数十倍，但渲染速度仅慢几倍（仍能实时），且在UrbanScene3D上LangSplat直接OOM而GCLF可正常运行。

**语言对齐**：继承LangSplat的方式，用场景特定自编码器压缩CLIP特征后嵌入Gaussian。推理时计算CLIP文本特征 $T(q)$ 与解码后语言嵌入 $D(\hat{l}(v))$ 在每个像素的余弦相似度来定位目标。

**地理参考（Georeferencing）**：半自动方式将3D Gaussian坐标对齐到OpenStreetMap的真实世界坐标——渲染4个小区域的俯视图，手动选取20+个地标点，用scikit-image计算坐标变换。对齐后可支持地标名称查询和真实世界度量（米为单位的距离和高度）。

### 关键设计二：GV-APIs — 9个地理视觉API

| # | 函数 | 功能 |
|---|------|------|
| 1 | `GetLandmarkSeg(query)` | 按地标名称获取分割区域 |
| 2 | `GetStructureSeg(query, area)` | 按结构名称获取分割（如"桥"） |
| 3 | `SegAround(area, distance)` | 获取指定距离内的周围区域 |
| 4 | `SegDirection(area, direction)` | 获取指定方向的区域 |
| 5 | `SegBetween(seg1, seg2)` | 获取两区域之间的区域 |
| 6 | `LargestSeg(segs)` | 通过聚类获取最大连通分割 |
| 7 | `MeasureDist(from, to)` | 计算真实世界距离（米） |
| 8 | `MeasureHeight(area)` | 计算真实世界高度（米） |
| 9 | `GetObjectSeg(query, area)` | 在区域内运行GroundingDINO检测 |

API 1-6 在GCLF的巨大城市空间中逐步缩小关注区域；7-8 利用地理参考信息进行真实世界度量；9 通过在GCLF渲染图上运行视觉基础模型实现细粒度检测。所有API在训练好的GCLF上工作，无需额外训练。

### 视觉编程

使用 GPT-3.5（gpt-3.5-turbo-instruct）作为LLM $\Pi$。仅需提供10-15个上下文示例即可使LLM有效使用GV-APIs，程序成功率超过90%。LLM能够生成示例中未出现的新结构程序，展现了结构级泛化能力。

**示例**：查询"距The View 100米内的红字广告牌"→ 生成三步程序：(1) `GetLandmarkSeg("The View")` 定位地标 → (2) `SegAround(area, 100)` 获取周围区域 → (3) `GetStructureSeg("Red-letter billboard", area)` 检索目标物。

## 实验

### 主实验一：定位任务（GRD）

| 场景 | 面积(m²) | LSeg | LERF | LangSplat | GCLF | GeoProg3D |
|------|---------|------|------|-----------|------|-----------|
| GoogleEarth | 2.4×10⁵ | 0.96% | 11.44% | 14.15% | 20.09% | **45.20%** |
| UrbanScene3D | 5.0×10⁶ | 4.65% | OOM | OOM | 6.98% | **30.23%** |

GeoProg3D 在定位准确率上大幅领先：GoogleEarth上45.2% vs LangSplat 14.15%（3倍提升）；UrbanScene3D上LERF和LangSplat直接OOM，GeoProg3D达到30.23%。

### 主实验二：多任务对比（SPR/CMP/CNT/MES）

| 方法 | SPR Acc↑ | CMP Acc↑ | CNT MAE↓ | MES-H MAE(m)↓ | MES-D MAE(m)↓ |
|------|----------|----------|----------|---------------|---------------|
| GPT-4o Vision | 24.77 | 2.63 | 3.02 | 158.16 | 195.29 |
| Llama-3.2 Vision | 54.84 | 28.49 | 2.54 | 88.06 | 133.20 |
| InternVL2.5-8B | 54.27 | 26.95 | 2.79 | 51.30 | 157.14 |
| GeoChat | 57.23 | 41.99 | 2.89 | 84.74 | 89.34 |
| TEOChat | 59.04 | 48.11 | 2.84 | 150.39 | 198.89 |
| **GeoProg3D** | **64.00** | **59.73** | **2.00** | **45.24** | **49.28** |

GeoProg3D 在5个任务上全面超越9个强基线（包括GPT-4o Vision、GeoChat等）。尤其在距离测量（MES-D）上误差仅49.28m，远低于GPT-4o的195.29m（4倍改进）。

### GeoEval3D 基准统计

- 952条人工标注查询-答案对，覆盖5个任务
- 场景面积超过 $3 \text{km}^2$，涵盖纽约和深圳
- 查询复杂度远高于此前数据集（词数多10倍以上）

## 亮点与洞察

1. **首个城市级3D组合推理框架**：将视觉编程范式从2D图像扩展到城市级3D场景
2. **模块化设计的优雅性**：GCLF负责场景表示+定位，GV-APIs提供操作接口，LLM负责推理——各组件职责清晰、可替换升级
3. **地理参考的关键价值**：将Gaussian坐标与OpenStreetMap对齐，使得系统能返回真实世界单位的度量结果（米）
4. 仅需10-15个示例即可让LLM有效使用API，体现了视觉编程的样本效率

## 局限性

1. **半自动地理参考依赖人工标注**：每个场景需手动选取20+地标点进行坐标对齐，限制了系统的全自动化部署
2. **GCLF的存储开销大**：相比LangSplat需要几十倍的Gaussian数量（3700万 vs 3万）
3. **LLM依赖**：生成程序的质量受限于LLM能力（当前使用GPT-3.5），复杂组合推理可能失败
4. 未讨论如何处理城市场景的动态变化（建筑拆建、时间变化等）

## 相关工作

- **城市级3D重建**：NeRF系（Block-NeRF等）、3DGS系（CityGaussian、Octree-GS） 
- **3D语言场**：LERF（NeRF+CLIP）、LangSplat（3DGS+CLIP+SAM）、LEGaussians
- **视觉编程**：ViPer/ViperGPT（2D图像）、CodeVQA、SayPlan（3D点云但仅室内）
- **地理感知VLM**：GeoChat、TEOChat、LHRS-BOT

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.2 |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](../../CVPR2025/3d_vision/perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)

<!-- RELATED:END -->
