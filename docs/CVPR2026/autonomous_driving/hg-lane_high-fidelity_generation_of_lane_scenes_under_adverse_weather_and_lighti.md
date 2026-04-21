---
title: >-
  [论文解读] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation
description: >-
  [CVPR 2026][自动驾驶][lane detection] 针对车道检测数据集（CULane/TuSimple）极端天气样本严重不足的问题，提出HG-Lane——一个无需重标注的两阶段扩散生成框架：Stage-I通过Control Information Fusion+Structure-aware Reverse Diffusion保留车道几何结构，Stage-II通过Appearance-aware Refinement调整光照风格，生成snow/rain/fog/night/dusk共30K图。CLRNet整体mF1提升+20.87%，snow场景+38.8%。
tags:
  - CVPR 2026
  - 自动驾驶
  - lane detection
  - adverse weather
  - 扩散模型
  - ControlNet
  - 数据增强
  - CULane
  - TuSimple
---

# HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation

**会议**: CVPR 2026  
**arXiv**: [2603.10128](https://arxiv.org/abs/2603.10128)  
**代码**: [zdc233/HG-Lane](https://github.com/zdc233/HG-Lane)  
**领域**: 车道检测 / 数据生成  
**关键词**: lane detection, adverse weather, diffusion model, ControlNet, data augmentation, CULane, TuSimple

## 一句话总结

针对车道检测数据集（CULane/TuSimple）极端天气样本严重不足的问题，提出HG-Lane——一个无需重标注的两阶段扩散生成框架：Stage-I通过Control Information Fusion+Structure-aware Reverse Diffusion保留车道几何结构，Stage-II通过Appearance-aware Refinement调整光照风格，生成snow/rain/fog/night/dusk共30K图。CLRNet整体mF1提升+20.87%，snow场景+38.8%。

## 研究背景与动机

车道检测是自动驾驶的基础感知任务，当前主流数据集CULane和TuSimple主要在晴天/白天场景下采集，极端天气（雪、雨、雾）和低光照（夜间、黄昏）的样本严重不足。这导致车道检测模型在恶劣条件下性能急剧下降——而这些恰恰是最需要可靠检测的场景。

现有解决思路面临两个核心矛盾：

**实际采集成本极高**：在真实的暴雪、暴雨、浓雾中驾驶采集数据不仅危险，且地理和季节限制大。即使采集到，仍需逐帧标注车道线，每帧标注成本远高于普通目标检测

**现有生成方法丢失车道语义**：直接使用风格迁移（CycleGAN等）或无条件生成模型转换天气风格时，生成图像中的车道线位置、形状往往发生偏移甚至消失。原始标注与生成图像不再对齐，需要重新标注，相当于回到了原点

**根本需求**：一种能在改变天气/光照外观的同时，严格保留车道线几何结构的生成方法——使得原始标注可以直接复用于生成图像，实现"零标注成本"的数据增强。

## 方法详解

### 整体流程

HG-Lane采用两阶段生成流水线，对每张输入的正常天气图像$I$：

- **Stage-I**：Structure-aware Reverse Diffusion → 生成保留车道结构的天气变换图像$I'$
- **Stage-II**（仅night/dusk）：Appearance-aware Refinement → 对$I'$进一步调整光照风格

两个阶段均使用**预训练**的ControlNet模型，无需针对车道数据fine-tune。

### 模块1：Control Information Fusion（控制信息融合）

**核心问题**：如何构建一个既能指导天气生成、又能精确保留车道位置的控制信号？

**方案**：将三种互补信息融合为单一控制图$C_0$：

$$C_0 = \text{Canny}(I) \oplus (\text{LaneAnnotation}(I) \odot \text{ColorMask})$$

具体步骤：

1. **Canny边缘图** $E = \text{Canny}(I)$：提取原图的全局结构信息（道路边界、车辆轮廓、路标等），为扩散模型提供场景级布局约束
2. **车道标注着色图**：取原始车道线标注（2D坐标点集），按车道类别赋予不同颜色，渲染为彩色蒙版$L_{\text{color}}$。不同颜色帮助模型区分左/右车道、实/虚线
3. **融合叠加**：将Canny边缘图与着色车道标注进行通道级叠加，得到融合控制图$C_0$

**设计动机**：单独使用Canny边缘时，车道线作为细线段容易在扩散过程中被忽略；单独使用车道标注缺乏场景全局结构。两者融合后，车道线在控制图中既有全局上下文又有显式的强信号。

### 模块2：Stage-I — Structure-aware Reverse Diffusion

**架构**：基于Stable Diffusion + Canny-ControlNet的条件生成流程。

**输入**：
- 控制图$C_0$（融合后的Canny+lane信息）
- Category-specific text prompt：根据目标天气类别使用特定的文本提示，如"A road scene with lane markings during heavy snowfall"

**生成过程**：

$$\epsilon_\theta(z_t, t, c_{\text{text}}, C_0) = \text{SD}(z_t, t, c_{\text{text}}) + \text{ControlNet}(z_t, t, C_0)$$

Canny-ControlNet在latent space的每一步去噪中注入结构约束，确保生成结果的边缘分布与$C_0$高度一致。由于$C_0$中包含了显式的车道标注信息，车道线的位置和形状被强制保留。

**Category-specific Prompts**：针对6种天气/光照条件分别设计文本提示：
- Snow: 强调路面积雪、飘落雪花的视觉特征
- Rain: 强调路面反光、雨滴模糊
- Fog: 强调远处能见度降低、朦胧感
- Night: 强调整体暗光、车灯照明
- Dusk: 强调天空渐暗、暖色调光线
- Shadow: 强调局部遮挡阴影

### 模块3：Stage-II — Appearance-aware Refinement

**动机**：Stage-I使用Canny-ControlNet主要控制结构，对全局色调/亮度的控制力不足。尤其是night和dusk场景，需要大幅度改变图像整体亮度和色温，仅靠text prompt难以实现自然的光照变换。

**方案**：仅对night和dusk场景应用第二阶段——使用**InstructPix2Pix ControlNet**对Stage-I的输出进行光照风格调整。

$$I_{\text{final}} = \text{IP2P-ControlNet}(I', c_{\text{instruction}})$$

其中$c_{\text{instruction}}$为编辑指令，如"Make it look like nighttime with street lights"。InstructPix2Pix天然支持保持图像结构的同时修改外观属性，与Stage-I的结构保留目标互补。

**为何snow/rain/fog不需Stage-II**：这些天气变化主要是叠加效果（雪花、雨滴、雾气），Stage-I的text prompt足以引导，无需额外的全局光照调整。

### 数据集构建

基于CULane训练集（约88K张），为每种天气类别生成5000张图像，共30K张（5000×6类），构建HG-Lane Benchmark。生成图像直接复用原始车道线标注。

## 实验关键数据

### 主实验：车道检测性能提升（CULane测试集）

以CLRNet（CVPR 2022，主流车道检测器）为基线：

| 训练数据 | 整体mF1 | Snow | Rain | Fog | Night | Dusk | Shadow |
|----------|---------|------|------|-----|-------|------|--------|
| CULane原始 | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| +HG-Lane 30K | **+20.87%** | **+38.8%** | +18.2% | **+26.84%** | **+21.5%** | +15.7% | +13.2% |

Snow场景提升最为显著（+38.8%），因为原始CULane几乎不含雪景样本。

### 跨检测器泛化性

在多个车道检测器上验证HG-Lane数据的通用性，均获得显著提升，说明生成数据的增益不依赖于特定模型架构。

### 车道标注保留质量

通过在生成图像上直接使用原始标注评估车道检测IoU，验证车道线位置未发生偏移。定量指标显示生成图像与原始标注的平均IoU保持在95%以上。

### 消融实验

| 配置 | mF1提升 |
|------|---------|
| 仅Canny控制（无lane融合） | +11.3% |
| 仅lane标注控制（无Canny） | +8.7% |
| Canny+lane融合（Stage-I完整） | +17.5% |
| Stage-I + Stage-II（night/dusk） | **+20.87%** |

Control Information Fusion相比单一控制信号提升明显。Stage-II对night/dusk场景贡献约3.4%的额外增益。

### 与现有方法对比

与CycleGAN、UNIT等风格迁移方法相比，HG-Lane在车道保留质量和下游检测性能上全面领先。传统方法生成的图像中车道线经常变形或消失，导致原始标注不可用。

## 亮点与洞察

- **"零标注成本"的实用价值**：整个流程不需要任何额外标注——原始车道标注直接复用。这对标注成本极高的自动驾驶场景意义重大
- **融合控制图设计巧妙**：Canny提供全局布局，着色lane标注提供显式车道信号，两者互补。比单独使用任一信号效果好30-50%
- **两阶段分治策略合理**：结构保留（Stage-I）与外观调整（Stage-II）解耦处理，避免单一模型同时处理两个目标时的trade-off
- **全部使用预训练模型，无需fine-tune**：ControlNet和InstructPix2Pix均使用公开预训练权重，复现门槛低，且不依赖车道场景的训练数据
- **Snow +38.8%说明数据缺口的影响**：原始数据集中最缺乏的类别提升最大，充分证明数据不均衡是当前车道检测的核心瓶颈之一

## 局限与展望

1. **生成多样性受限于Canny边缘**：控制图以原图的Canny边缘为基础，生成图像的场景布局与原图高度一致。无法生成"全新场景"的极端天气图像，多样性受限于原始数据集的场景分布
2. **Stage-II仅处理night/dusk**：其他天气条件（如rain+night组合）未设计专门的refinement流程。多种恶劣条件叠加的场景（如雪夜、雾夜）可能需要更复杂的多阶段处理
3. **仅验证2D车道检测**：未在3D车道检测（如OpenLane）或BEV车道感知任务上验证。极端天气对3D感知的影响可能更复杂
4. **生成质量的定量评估有限**：主要通过下游检测性能间接评估生成质量，缺乏FID/IS等生成质量指标的系统报告
5. **扩散生成速度较慢**：生成30K张图像的计算开销可观。若需更大规模的数据增强（如百万级），计算成本可能成为瓶颈

## 相关工作与启发

- **ControlNet (ICCV 2023)**：提供了结构化条件控制的基础能力 → HG-Lane创新地将Canny边缘与任务特定标注融合作为控制信号
- **InstructPix2Pix (CVPR 2023)**：基于指令的图像编辑 → HG-Lane将其用于Stage-II的光照风格调整，是一种巧妙的工程化应用
- **CycleGAN/UNIT**：传统风格迁移方法 → 无法保留细粒度车道结构，HG-Lane通过显式控制信号解决了这一根本缺陷
- **CLRNet (CVPR 2022)**：主流车道检测器 → HG-Lane生成的数据对其提升最为显著
- **ACGEN (CVPR 2024)**：另一种条件生成用于自动驾驶的工作 → HG-Lane专注于车道检测任务，控制信号设计更加任务特定
- **启发**：此"融合控制图+两阶段生成"的范式可推广到其他需要精确保留标注信息的数据增强任务，如交通标志检测、路面标线检测等

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3.5 | 核心组件（ControlNet、IP2P）均为已有方法，创新在于融合控制图的设计和两阶段分治策略，工程创新为主 |
| 实用性 | 4.5 | 零标注成本、全预训练权重、30K benchmark开源，实际应用价值极高 |
| 实验充分度 | 4.0 | 多检测器验证、消融完整、与风格迁移对比充分，缺少FID等独立生成质量评估 |
| 写作质量 | 3.5 | 方法描述清晰，两阶段流程结构化好，但部分细节（prompt具体内容、超参选择）不够充分 |

<!-- RELATED:START -->

## 相关论文

- [TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)
- [Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)

<!-- RELATED:END -->
