---
title: >-
  [论文解读] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation
description: >-
  [CVPR2026][图像分割][SAR语义分割] 提出首个十亿参数级SAR视觉基础模型CrossEarth-SAR，通过物理引导的稀疏MoE架构结合SAR物理描述子，在22个跨域语义分割基准中的20个取得SOTA，部分multi-gap场景超越已有方法10%+ mIoU。
tags:
  - CVPR2026
  - 图像分割
  - SAR语义分割
  - 视觉基础模型
  - 混合专家(MoE)
  - 域泛化
  - 遥感
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**会议**: CVPR2026  
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)  
**代码**: [VisionXLab/CrossEarth-SAR](https://github.com/VisionXLab/CrossEarth-SAR)  
**领域**: 语义分割  
**关键词**: SAR语义分割, 视觉基础模型, 混合专家(MoE), 域泛化, 遥感

## 一句话总结

提出首个十亿参数级SAR视觉基础模型CrossEarth-SAR，通过物理引导的稀疏MoE架构结合SAR物理描述子，在22个跨域语义分割基准中的20个取得SOTA，部分multi-gap场景超越已有方法10%+ mIoU。

## 背景与动机

1. **SAR全天候观测优势**：SAR不受天气和光照限制，是灾害监测、环境监测、城市管理等时间敏感应用的关键工具，但其语义理解远比可见光域困难。
2. **SAR固有成像挑战**：相干成像引入乘性散斑噪声、侧视几何导致叠掩/前缩/阴影等空间畸变、雷达后向散射而非颜色导致语义歧义——这三者共同破坏了现代视觉模型的基本假设。
3. **极端域碎片化问题**：SAR数据因传感器平台（Sentinel-1/ALOS-2/Capella）、频段（C/L/X）、极化模式（HH/HV/VH/VV）、入射角等参数差异，呈严重域碎片化，模型跨域迁移时性能灾难性下降。
4. **现有基础模型不适配SAR**：SatMAE、SkySense等地理空间基础模型主要面向光学多光谱数据，架构和预训练策略未考虑SAR后向散射物理和噪声特性。
5. **缺乏统一DG评测标准**：SAR跨域语义分割领域缺少系统化的域泛化基准，阻碍了方法的公平比较和推进。
6. **大规模SAR标注数据稀缺**：高质量SAR语义分割标注获取困难，限制了大模型训练。

## 方法详解

### 整体框架

CrossEarth-SAR在DINOv2 ViT骨干上集成物理引导稀疏MoE，将原始FFN替换为包含路由器和多个专家的MoE模块。输入SAR图像复制为3通道后经ViT编码，同时计算3个SAR物理描述子辅助路由，最终由Mask2Former解码器生成分割预测。提供S/B/L三种规模（90M/300M/1.3B参数，激活参数20M/80M/300M）。

### SAR物理描述子

设计SAR Physical Operator $g_{\text{sar}}(\cdot)$计算三个物理描述子，为路由器提供稳定的物理先验：

- **方向熵 $H_{DE}$（成像几何）**：通过Sobel梯度方向直方图的熵量化图像结构规则性，低值→明确线性边缘结构，高值→复杂不规则结构
- **等效视数 ENL（雷达系统）**：$\text{ENL} = (\mu/\sigma)^2$，度量散斑噪声强度，高值→散斑弱/统计稳定，低值→噪声波动大
- **局部粗糙度 $R_{LR}$（目标散射）**：分块均值方差，捕获纹理变异性，高值→复杂纹理，低值→平滑纹理

三者拼接为 $s = [H_{DE}, \text{ENL}, R_{LR}] \in \mathbb{R}^3$，在各ViT block中与token嵌入拼接后送入路由器。

### 物理引导稀疏MoE

- **路由器**：将token嵌入 $Z$ 与描述子 $S$ 拼接后计算softmax得分 $\pi = \text{softmax}(W_r[Z \| S] + b_r)$，选Top-k个专家
- **Token级MoE聚合**：$\tilde{z} = \sum_{k \in \mathcal{I}} g_k \cdot E_k(z)$，其中门控权重为归一化的路由得分
- **最优配置**：n=6个专家、Top-k=1激活，实现计算效率与容量的平衡

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{BC}}$$

- $\mathcal{L}_{\text{seg}}$：Mask2Former分割损失
- $\mathcal{L}_{\text{BC}} = \lambda_{\text{BC}} \cdot n \sum_{k=1}^{n} f_k p_k$（负载均衡损失，$\lambda_{\text{BC}}=0.005$），防止专家坍塌

### 数据与基准

- **CrossEarth-SAR-200K**：整合公开SAR数据（全监督40K）+ 收集数据（弱监督伪标签160K），覆盖6大洲数百城市，图像统一裁剪/缩放至512×512
- **22个DG基准**：跨8类域差（区域/极化/复数值/区域+极化/区域+平台/区域+微波频段/区域+极化+频段/区域+平台+频段），基于6个公开数据集构建

## 实验关键数据

### 单域差场景（Tab.2）

| 方法 | 参数量 | N2S | VV2F | HH2F | C(r)2R | 12基准均值 |
|------|--------|-----|------|------|--------|-----------|
| DINOv2 (Baseline) | 300M | 32.3 | 65.7 | 56.8 | 71.3 | 55.5 |
| DINOv3 | 300M | 33.7 | 48.3 | 50.6 | 69.9 | 53.0 |
| MTP | 300M | 30.6 | 30.4 | 36.0 | 70.8 | 44.7 |
| **CrossEarth-SAR-L** | **1.3B(300M)** | **37.8** | **73.8** | **72.3** | **76.4** | **61.9** |
| **CrossEarth-SAR-L*** | **1.3B(300M)** | **38.0** | **73.9** | **71.8** | **76.9** | **62.7** |

- 单域差12个基准均值：CrossEarth-SAR-L*达62.7%，超越Baseline +7.2%
- HH2F提升最大：+15.5% mIoU（56.8→72.3）

### 多域差场景（Tab.3）

| 方法 | 参数量 | F2A | A2F | O2D | S2A | D2F | W2D | 10基准均值 |
|------|--------|-----|-----|-----|-----|-----|-----|-----------|
| DINOv2 (Baseline) | 300M | 13.4 | 15.5 | 17.8 | 55.9 | 26.0 | 16.7 | 24.3 |
| **CrossEarth-SAR-L*** | **1.3B(300M)** | **16.1** | **27.0** | **23.1** | **57.9** | **26.5** | **25.6** | **28.5** |

- 多域差10个基准均值：+4.2%；A2F基准提升最大：+11.5%

### 消融实验（Tab.5-6）

- **伪标签有效性**：仅40K全监督45.1% → 加入160K弱监督59.4%（+14.3%）
- **MoE设计**：无负载均衡无描述子61.1% → 完整方案62.4%（+1.3%）
- **物理描述子**：三个描述子各自对不同域差有独立贡献，组合使用效果最佳
- **专家数n**：3→4→5→6，性能单调上升（60.9→62.4）
- **Top-k选择**：k=1最优（62.4），k=2/3反而下降

## 亮点

- **首个十亿参数级SAR基础模型**：稀疏MoE使参数量扩至十亿级而保持推理代价可控（仅激活300M参数）
- **物理引导路由机制**：三个SAR物理描述子解决MoE在SAR异构数据上的路由不稳定问题，设计巧妙且物理意义明确
- **系统性贡献**：同时推出200K大规模预训练数据集+22个DG基准+S/B/L三种模型规格，形成完整的研究基础设施
- **22个基准中20个SOTA**：覆盖单/双/三域差的全面验证，部分场景提升超过10% mIoU

## 局限与展望

- 伪标签依赖光学配对图像的CrossEarth模型生成，质量受限于光学-SAR匹配精度，部分场景标注可靠性存疑
- 多域差场景（如D2O、D2F）提升有限甚至略低于Baseline，说明三域差泛化仍是开放问题
- 预训练需要16×A100 80GB，计算资源门槛极高，限制了社区复现和应用
- 物理描述子为手工设计的3维向量，信息量有限；可探索可学习的物理特征提取
- Top-k=1意味着每token仅激活一个专家，多专家协同能力未被充分利用
- 仅评测语义分割任务，未验证在目标检测、变化检测等其他SAR任务上的泛化性

## 与相关工作的对比

| 维度 | CrossEarth-SAR | CrossEarth（光学DG）| SARATR-X（SAR目标识别）| SatMAE/SkySense（光学FM）|
|------|---------------|-------------------|---------------------|------------------------|
| 模态 | SAR专用 | 光学 | SAR | 光学/多光谱 |
| 任务 | 跨域语义分割 | 跨域语义分割 | 目标识别 | 多任务 |
| 架构 | MoE + ViT | ViT | HiViT | ViT |
| 参数量 | 1.3B（稀疏） | 300M | 60M | 300M |
| 物理先验 | SAR描述子引导路由 | 无 | 无 | 无 |
| DG基准 | 22个/8类域差 | 光学DG | 无系统DG评测 | 无系统DG评测 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 物理引导MoE路由在SAR领域是首创，三描述子设计有物理洞察
- 实验充分度: ⭐⭐⭐⭐⭐ — 22个基准覆盖8类域差，消融全面，可视化充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，但表格过多略显冗长
- 价值: ⭐⭐⭐⭐ — 完整生态（模型+数据+基准）对SAR社区有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)
- [SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [A Mixed Diet Makes DINO An Omnivorous Vision Encoder](a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)

<!-- RELATED:END -->
