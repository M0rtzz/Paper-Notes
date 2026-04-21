---
title: >-
  [论文解读] Reconstructing Animals and the Wild
description: >-
  [CVPR 2025][3D视觉][3D场景重建] 提出 RAW（Reconstruct Animals and the Wild），首次从单张图像同时重建自然场景中的动物及其环境，通过训练 LLM 将 CLIP 嵌入解码为结构化的图形程序表示，引入 CLIP 投影头解决大规模资产识别的扩展性问题。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D场景重建
  - 动物重建
  - LLM逆图形学
  - 合成数据
  - 组合式场景表示
---

# Reconstructing Animals and the Wild

**会议**: CVPR 2025  
**arXiv**: [2411.18807](https://arxiv.org/abs/2411.18807)  
**代码**: [https://raw.is.tue.mpg.de/](https://raw.is.tue.mpg.de/)  
**领域**: 3D视觉 / 场景重建  
**关键词**: 3D场景重建, 动物重建, LLM逆图形学, 合成数据, 组合式场景表示

## 一句话总结

提出 RAW（Reconstruct Animals and the Wild），首次从单张图像同时重建自然场景中的动物及其环境，通过训练 LLM 将 CLIP 嵌入解码为结构化的图形程序表示，引入 CLIP 投影头解决大规模资产识别的扩展性问题。

## 研究背景与动机

- 3D 场景重建是计算机视觉的基础问题，但现有工作主要聚焦于人造环境（如室内场景），对自然场景（树木、灌木、岩石、动物）缺乏有效方法
- 动物行为分析（计算行为学）需要理解动物在 3D 环境中的位置和交互，但现有方法仅重建孤立的动物，忽略环境上下文
- 自然场景的复杂性远超人造环境：动物可能与背景融合、物体距离差异大、光照条件多变、基本不具备几何规则性
- 获取自然场景的 3D 标注数据极其困难，需要依靠合成数据生成
- 直接用 LLM 以离散 token 形式预测资产名称时，模型无法扩展到大规模资产集合——经常混淆相似资产

## 方法详解

### 整体框架

RAW 基于 IG-LLM 框架，使用指令微调的 LLaMA-7b 作为基座模型，冻结的 CLIP 视觉编码器作为图像 tokenizer。输入为单张自然场景图像的 CLIP 嵌入，输出为结构化图形程序代码，描述场景属性（太阳参数、大气条件）和各物体（位置、高度、旋转、外观）。代码可直接在 Blender 中渲染。

### 关键设计

**设计一：CLIP 投影头（连续资产表示）**

- **功能**：解决 LLM 在大规模资产集合中的身份混淆问题
- **核心思路**：引入特殊 token [CLIP]，当 LLM 生成该 token 时，将隐藏状态通过线性投影层映射到 CLIP 嵌入空间，而非经过离散化的 tokenizer。训练时以渲染资产图像的 CLIP 编码作为目标。推理时通过最近邻检索获取对应资产
- **设计动机**：离散 token 之间没有语义距离度量（"老虎"和"灌木"在 token 空间等距），而 CLIP 空间中视觉外观相似的资产自然接近，便于模型学习连续外观估计

**设计二：高效合成数据生成**

- **功能**：生成百万级带 3D 标注的自然场景训练数据
- **核心思路**：基于 Infinigen 框架进行优化：(1) 限制资产类型为 6 类（岩石、灌木、树木、食肉动物、食草动物、鸟类）；(2) 预生成 6,000 个唯一资产并实例化复用；(3) 全场景填充而非仅相机视角区域；(4) 按偏航角分 72 个方向扩展非可定向资产。在 10,000 个场景中生成 100 万张图像
- **设计动机**：原始 Infinigen 生成单张图像需 4.5 小时且无 3D 标注，简化后大幅提升效率

**设计三：物体按视觉显著性排序**

- **功能**：引导模型先关注最显著的物体再处理背景细节
- **核心思路**：图形程序中的物体按占据像素数从大到小排序，编码场景级属性（太阳参数、大气条件）在序列开头，最多包含 25 个物体
- **设计动机**：LLM 的因果注意力机制下，序列顺序决定了信息流向和注意力分配

### 损失函数

- 标准 next-token 预测的交叉熵损失
- [NUM] token 的连续旋转参数回归使用 MSE 损失
- [CLIP] token 的语义嵌入预测使用 CLIP 空间的余弦相似度损失

## 实验关键数据

### 定性结果

- 仅在合成数据上训练，成功泛化到真实世界自然图像
- 能同时重建动物和环境（树木、灌木、岩石等）
- 重建结果可编辑、可动画化、可集成到图形引擎

### 关键发现

- 没有 CLIP 投影头时，LLM 经常将老虎与灌木、鸟类与岩石混淆
- CLIP 投影头使模型能够扩展到包含 432,000 个有效资产（含方向变体）的大规模集合
- 按视觉显著性排序的物体序列比随机排序训练更稳定
- 合成到真实的域间迁移效果良好，归功于 CLIP 编码器的域不变特性

## 亮点与洞察

1. **首次同时重建动物与自然环境**：填补了计算行为学中环境上下文缺失的空白
2. **LLM 做逆图形学的巧妙扩展**：CLIP 投影头将离散 token 预测转换为连续语义空间匹配，优雅解决了资产身份扩展性问题
3. **组合式可编辑表示**：输出的图形程序代码天然支持编辑和模拟

## 局限与展望

- 资产粒度较粗，动物形状为模板级别而非精细重建
- 场景复杂度被限制在 25 个物体以内（受限于上下文长度）
- 合成数据的光照和材质分布与真实世界仍有差距
- 未来可结合更精细的动物形状模型和更大的上下文长度实现更详细的重建

## 相关工作与启发

- **IG-LLM** [Kulits et al.] 将反图形学视为 LLM 支持的归纳程序合成
- **Infinigen** [Raistrick et al.] 提供了程序化自然场景生成工具
- **SMAL** [Zuffi et al.] 提供了多物种 3D 可变形动物模型
- 本文为 LLM 驱动的逆图形学在自然场景中的应用提供了新范例

## 评分

⭐⭐⭐⭐ — 首次同时重建动物与自然环境的创新问题定义，CLIP 投影头解决资产扩展性问题的方案优雅，合成到真实的泛化效果令人印象深刻。但动物形状粗糙且场景复杂度受限。

<!-- RELATED:START -->

## 相关论文

- [Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [Reconstructing Humans with a Biomechanically Accurate Skeleton](reconstructing_humans_with_a_biomechanically_accurate_skeleton.md)
- [Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)
- [PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)

<!-- RELATED:END -->
