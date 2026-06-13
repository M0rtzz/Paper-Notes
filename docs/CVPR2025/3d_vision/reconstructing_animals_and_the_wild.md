---
title: >-
  [论文解读] Reconstructing Animals and the Wild
description: >-
  [CVPR 2025][3D视觉][单图3D重建] 本文提出RAW方法，用LLM自回归解码CLIP图像嵌入为结构化的组合式3D场景表示（动物+自然环境），创新性地引入CLIP投影头替代离散的资产名称预测，使模型能在更大规模的资产集合上泛化，首次实现了从单张自然图像同时重建动物和环境。 领域现状：3D动物重建已有大量工作…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "单图3D重建"
  - "动物姿态"
  - "自然场景"
  - "LLM逆图形学"
  - "合成数据"
---

# Reconstructing Animals and the Wild

**会议**: CVPR 2025  
**arXiv**: [2411.18807](https://arxiv.org/abs/2411.18807)  
**代码**: [https://raw.is.tue.mpg.de/](https://raw.is.tue.mpg.de/)  
**领域**: 3D视觉 / 场景重建  
**关键词**: 单图3D重建, 动物姿态, 自然场景, LLM逆图形学, 合成数据

## 一句话总结
本文提出RAW方法，用LLM自回归解码CLIP图像嵌入为结构化的组合式3D场景表示（动物+自然环境），创新性地引入CLIP投影头替代离散的资产名称预测，使模型能在更大规模的资产集合上泛化，首次实现了从单张自然图像同时重建动物和环境。

## 研究背景与动机

**领域现状**：3D动物重建已有大量工作，从2D姿态估计发展到3D形变模型（如SMAL），但几乎所有方法都只重建动物本体，忽略了环境上下文。逆图形学方法近来借助LLM实现了从图像到图形代码的推断（如IG-LLM）。

**现有痛点**：（1）现有动物重建忽略环境——但动物行为分析需要环境信息（遮挡、物理边界、自然交互）；（2）自然场景重建极具挑战——树木、灌木、岩石等不像人造物体有规则几何形状；（3）组合式自然场景的3D地面真实数据极度缺乏。

**核心矛盾**：IG-LLM用离散token预测资产名称，当资产集合扩大时（本文需要包含多种动物和植物），模型会混淆外观相似但语义不同的物体（如把老虎误认为灌木），因为离散token之间缺乏语义距离度量。

**本文目标**：从单张自然图像生成可编辑、可动画化的组合式3D场景，同时包含动物和植物/地形。

**切入角度**：用连续的CLIP嵌入替代离散资产名称，使LLM能通过语义相似度检索资产，而非记忆token序列。

**核心 idea**：在LLM中添加特殊的[CLIP] token，当生成到资产外观时绕过离散tokenizer，直接将hidden state投影到CLIP空间，用余弦相似度与资产库中渲染图的CLIP嵌入匹配。

## 方法详解

### 整体框架
基于指令微调的LLaMA-7B + 冻结CLIP视觉编码器。输入图像经CLIP编码后投影到LLM token空间，LLM自回归生成结构化图形代码：先生成场景级属性（太阳参数、大气），再按像素面积从大到小生成每个物体（位置、高度、旋转、外观）。生成的代码可直接在Blender中渲染。

### 关键设计

1. **CLIP投影头（替代离散资产名称）**:

    - 功能：将连续语义替代离散token来预测物体外观/身份
    - 核心思路：引入特殊token [CLIP]，当LLM预测到该token时，将当前hidden state通过线性层投影到CLIP嵌入空间。训练目标是与该资产在相应yaw角下渲染图的CLIP嵌入的余弦相似度。推理时通过最近邻搜索匹配资产库
    - 设计动机：离散资产名称token之间无语义距离（"tiger"和"bush"在token空间等距），导致大规模资产集上混淆严重。CLIP空间中相似外观的资产自然接近，提供了有意义的梯度信号

2. **百万级合成数据集**:

    - 功能：提供组合式3D场景的图像-代码训练对
    - 核心思路：基于Infinigen框架修改，预生成6000个资产（鸟类、食肉动物、食草动物、灌木、岩石、树木各1000个），在10000个不同场景中各渲染100张图像，共100万张。每个非定向资产（树、石、灌木）按5°增量标注72个yaw方向，有效资产数达432000
    - 设计动机：真实自然场景3D数据无法大规模获取，程序化生成是唯一可行路径。多视角渲染增加数据多样性

3. **结构化图形代码表示**:

    - 功能：将3D场景表示为可解析、可编辑、可渲染的代码序列
    - 核心思路：场景代码依次包含太阳/大气参数、地面纹理的[CLIP]嵌入、物体列表（按像素面积排序），每个物体有像素数、位置、高度、旋转[ROT]和外观[CLIP]
    - 设计动机：从大到小排序使模型先关注显著物体再处理背景，图形代码格式可直接用于Blender渲染和编辑

### 损失函数 / 训练策略
三个损失：标准next-token prediction（文本部分）、旋转矩阵的MSE损失（经对称正交化后）、CLIP嵌入的余弦相似度+范数正则化。

## 实验关键数据

### 主实验

| 方法 | LPIPS↓ | CLIP相似度↑ | BioCLIP↑ | DINOv2↑ |
|------|--------|------------|----------|---------|
| IG-LLM (baseline) | 0.720 | 0.748 | 0.421 | 0.833 |
| + CLIP head (RAW) | **0.696** | **0.762** | **0.456** | **0.842** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 离散名称 vs CLIP嵌入 | CLIP版本在所有指标上提升，且避免了语义无关的错误（老虎→灌木） |
| 仅合成训练→真实图像 | 能成功泛化到真实自然图像，验证跨域迁移 |
| 物体排序（按像素大小） | 有助于模型优先关注显著物体 |

### 关键发现
- CLIP投影头消除了离散token的语义混淆问题，使模型能在6000+资产上正确匹配
- 仅在合成数据上训练就能泛化到真实自然图像中的动物+环境重建
- 结构化输出可直接编辑和动画化，支持下游行为分析

## 亮点与洞察
- [CLIP] token的设计优雅——用连续语义嵌入替代离散token的思路可推广到任何需要从大型库中检索的LLM任务
- 首次将自然环境纳入动物3D重建，填补了计算行为学的重要空白
- 资产按yaw角分72个方向标注的做法简单有效地解决了非定向物体的外观歧义

## 局限与展望
- 场景复杂度受LLM上下文长度限制（最多25个物体）
- 重建几何较为粗糙，使用预定义资产库而非精确形状估计
- 动物姿态估计精度有限，未使用专门的parametric body model
- 可与细粒度动物重建方法（如SMAL）结合，先粗后精

## 相关工作与启发
- **vs IG-LLM**: 直接扩展，核心贡献是CLIP投影头解决了大资产集的可扩展性问题
- **vs SMAL/3D动物重建**: 互补关系——本文关注整体场景布局而非精细形状，两者结合是未来方向
- **vs Infinigen**: 本文在Infinigen基础上做了大量简化以实现百万级数据生成

## 评分
- 新颖性: ⭐⭐⭐⭐ CLIP投影头的idea巧妙，首次重建动物+环境
- 实验充分度: ⭐⭐⭐ 合成数据评估为主，真实图像主要是定性展示
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 对计算行为学和自然场景理解有开创意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2025\] Reconstructing Humans with a Biomechanically Accurate Skeleton](reconstructing_humans_with_a_biomechanically_accurate_skeleton.md)
- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
