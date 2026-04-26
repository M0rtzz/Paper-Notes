---
title: >-
  [论文解读] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data
description: >-
  [NeurIPS 2025][遥感][遥感基础模型] GeoLink将OpenStreetMap矢量数据直接融入遥感基础模型预训练，通过异构GNN编码OSM数据并设计多粒度跨模态学习目标（区域-图像级对比 + 对象-patch级融合），在127万样本对上高效预训练后，7个分类和4个分割/变化检测benchmark全面超越现有RS FM。
tags:
  - NeurIPS 2025
  - 遥感
  - 遥感基础模型
  - OpenStreetMap
  - 多模态预训练
  - 异构图神经网络
  - 跨模态对齐
---

# GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data

**会议**: NeurIPS 2025  
**arXiv**: [2509.26016](https://arxiv.org/abs/2509.26016)  
**代码**: [GitHub](https://github.com/bailubin/GeoLink_NeurIPS2025)  
**领域**: 遥感  
**关键词**: 遥感基础模型, OpenStreetMap, 多模态预训练, 异构图神经网络, 跨模态对齐

## 一句话总结

GeoLink将OpenStreetMap矢量数据直接融入遥感基础模型预训练，通过异构GNN编码OSM数据并设计多粒度跨模态学习目标（区域-图像级对比 + 对象-patch级融合），在127万样本对上高效预训练后，7个分类和4个分割/变化检测benchmark全面超越现有RS FM。

## 研究背景与动机

**领域现状**：遥感基础模型（RS FM）已在多尺度、多时相、多传感器方向取得进展，但地面级地理空间数据的融入仍不足。

**现有痛点**：现有将OSM用于RS的方法多采用间接策略（转标签/知识图谱/合成文本），人工密集、任务特化且丢失空间信息。

**核心矛盾**：RS图像与OSM数据存在巨大模态鸿沟（数据结构/内容/空间粒度不同），但OSM提供的位置语义、结构化知识和社会经济信息是纯视觉分析无法获取的。

**本文目标** 设计地理空间显式方法，直接利用OSM原始矢量元素为RS FM注入地理上下文。

**切入角度**：将OSM建模为异构图，通过GNN编码后与RS ViT编码器进行多粒度交互。

**核心 idea**：用OSM的异构图结构作为RS自监督预训练的多粒度监督信号，同时支持掩码高效训练和多模态下游融合。

## 方法详解

### 整体框架

GeoLink含三个编码器：(1) ViT-L RS图像编码器输出patch编码；(2) GATConv异构GNN OSM编码器输出节点编码（点/线/面三类）；(3) Two-way Transformer融合编码器生成混合编码。预训练阶段同时掩码两种模态，通过三个SSL目标联合优化。

### 关键设计

1. **异构OSM图构建与编码**:
    - 功能：将OSM矢量地图建模为异构图，节点为点/折线/多边形，边为拓扑空间关系
    - 核心思路：用BERT对OSM标签键值对编码，按全局频率加权平均 $\sigma_V = \sum w_i h_i / \sum w_i$；用Delaunay三角化等拓扑关系构建边
    - 设计动机：OSM自由标签系统需语言模型处理未见值；拓扑关系比距离更稳健

2. **区域-图像级对比对齐**:
    - 功能：全局层面对齐RS和OSM表示
    - 核心思路：Set2Set分别聚合三类节点→类型注意力加权→OSM区域编码 $\varepsilon_G$；RS mean pooling→$\varepsilon_I$；InfoNCE对比损失 $\mathcal{L}_{cont}$
    - 设计动机：对比学习可将OSM结构化语义传递给图像编码器

3. **对象-patch融合 + 空间一致性约束**:
    - 功能：细粒度跨模态关联学习
    - 核心思路：Two-way Transformer + 正弦位置嵌入解决空间模糊性；一致性损失 $\mathcal{L}_{cst} = \frac{1}{N}\sum\|\varepsilon_{OR}^m - \sigma_V^m\|^2$ 强制掩码节点的融合表示与原始特征一致
    - 设计动机：基于地理学第一定律——空间上下文与掩码对象属性强相关

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{rec} + 0.01\mathcal{L}_{cont} + 0.01\mathcal{L}_{cst}$。RS 75%掩码+MAE重建，OSM 20%节点掩码。仅60 epoch预训练（vs Scale-MAE 800 epoch），4×RTX6000，batch 2640。

## 实验关键数据

### 主实验

| 任务 | 数据集 | 指标 | GeoLink | 之前SOTA | 提升 |
|------|--------|------|---------|----------|------|
| kNN分类 | RESISC-45 | Top-1 | **87.33%** | 85.42%(Scale-MAE) | +1.91% |
| Fine-tuning | EuroSAT | Top-1 | 98.30% | 98.27%(MMEarth) | 持平 |
| 分割(FT) | AI4SmallFarms | mIoU | **47.29%** | 45.98%(Scale-MAE) | +1.31% |
| 变化检测(FT) | SpaceNet7 | mIoU | **64.07%** | 63.22%(Scale-MAE) | +0.85% |
| UV识别(多模态) | UV数据集 | mIoU | **81.68%** | 80.09%(Scale-MAE+OSM) | +1.59% |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无OSM预训练 | 分类/分割均下降 | OSM预训练显著增强RS编码器 |
| 无对比损失 | 分类性能下降 | 区域级跨模态对齐核心 |
| 无一致性损失 | 融合质量下降 | 精细空间约束必要 |

### 关键发现
- kNN协议下优势最显著——学到了结构化RS表示空间
- 训练样本有限时优势更明显（高数据效率）
- 多模态融合使混淆UFZ类别显著更可分
- 空间相关性在多模态地理数据融合中起关键作用

## 亮点与洞察
- 首个直接利用OSM原始矢量数据进行RS FM预训练的框架
- 异构GNN处理OSM数据的设计优雅——三类节点+拓扑边+BERT标签编码
- 仅60 epoch预训练即收敛，训练效率极高
- 多粒度学习目标设计合理：全局对比+局部位置感知融合互补

## 局限与展望
- 预训练数据可能存在地域偏差（依赖OSM标注覆盖度）
- 仅使用RGB波段，未扩展到多光谱/SAR
- 融合编码器增加额外计算开销

## 相关工作与启发
- **vs Scale-MAE/CROMA**: 关注多尺度/多传感器但忽视地面地理知识；GeoLink补充了这一空白
- **vs 间接OSM利用**: 传统方法丢失空间信息；GeoLink直接图编码保留完整

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个直接融合OSM到RS FM预训练
- 实验充分度: ⭐⭐⭐⭐ 7+4个benchmark，多协议评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图精美
- 价值: ⭐⭐⭐⭐ RS FM多模态化新方向，代码开源

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](../../ICCV2025/remote_sensing/rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)
- [\[NeurIPS 2025\] EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)

<!-- RELATED:END -->
