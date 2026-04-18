---
title: >-
  [论文解读] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement
description: >-
  [CVPR 2026][医学图像][乳腺超声分割] 利用简单外观描述（"dark oval"等）驱动 Grounding DINO + SAM 免训练生成乳腺超声伪标签，再通过双教师不确定性-熵加权融合与自适应反向对比学习精炼伪标签质量，仅 2.5% 标注即达到甚至超过全监督上界。
tags:
  - CVPR 2026
  - 医学图像
  - 乳腺超声分割
  - VLM伪标签
  - 外观提示
  - 双教师框架
  - 不确定性融合
  - 反向对比学习
---

# A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement

**会议**: CVPR 2026  
**arXiv**: [2603.06167](https://arxiv.org/abs/2603.06167)  
**代码**: 无  
**领域**: 医学图像分割 / 半监督学习  
**关键词**: 乳腺超声分割, VLM伪标签, 外观提示, 双教师框架, 不确定性融合, 反向对比学习

## 一句话总结

利用简单外观描述（"dark oval"等）驱动 Grounding DINO + SAM 免训练生成乳腺超声伪标签，再通过双教师不确定性-熵加权融合与自适应反向对比学习精炼伪标签质量，仅 2.5% 标注即达到甚至超过全监督上界。

## 研究背景与动机

**领域现状**：乳腺超声（BUS）分割是早期乳腺癌诊断的关键步骤。全监督深度学习方法已取得良好效果，但依赖大量像素级标注，标注成本极高且需要专家放射科医师。

**现有痛点**：

1. 半监督方法在极少标注（如 2.5%）下教师模型欠训练，伪标签质量差且结构碎片化
2. 主流强-弱数据增强策略针对 RGB 自然图设计，不适合灰度、含斑点噪声的超声图像
3. VLM（Grounding DINO + SAM）可提供外部伪标签，但医学术语提示（"tumor"、"high density"）零样本定位效果不稳定，因为 VLM 缺乏医学域语义

**核心矛盾**：极少标注下需要高质量外部伪标签，但 VLM 直接用医学提示效果差。

**本文目标** 如何免训练地获得结构一致的 BUS 伪标签，并在双教师框架中有效利用。

**切入角度**：BUS 病灶具有一致的外观特征——暗色椭圆/圆形区域。用简单自然语言外观描述替代医学术语提示，可绕过域鸿沟实现跨域迁移。

**核心 idea**：用外观描述而非医学术语驱动 VLM 生成伪标签，再由双教师融合 + 反向对比学习精炼。

## 方法详解

### 整体框架

分两阶段：(1) **APPG**——用 LLM 将医学特征转译为外观描述，驱动 Grounding DINO 检测 + SAM 分割，免训练生成伪标签；(2) **伪标签精炼**——先用伪标签预训练冻结的静态教师 $T^A$ 捕获粗结构先验，再在双教师半监督框架中通过 UEWF 融合两路伪标签、AURCL 增强边界判别力。学生模型 $S$ 以标注数据监督损失 $\mathcal{L}_s$、融合伪标签无监督损失 $\mathcal{L}_u$ 和对比损失 $\mathcal{L}_c$ 联合训练。

### 关键设计

1. **APPG（外观提示伪标签生成）**

    - 用 LLM（GPT-5）将通用乳腺肿瘤医学特征转译为简洁外观描述："dark oval"、"dark round"、"dark lobulated"
    - 将描述送入 Grounding DINO 得到边界框 $b_i^u = \text{VLM}_{\text{DINO}}(x_i^u, \text{aprmpt})$，再由 SAM 生成分割掩码 $\hat{y}_i^0 = \text{SAM}(x_i^u, b_i^u)$
    - 全程免训练，利用自然图像和超声图像之间的外观共性完成跨域迁移
    - 通过面积阈值（前景 > 1%）过滤无效伪标签，仅保留结构有效的样本

2. **UEWF（不确定性-熵加权融合）**

    - 静态教师 $T^A$（VLM 伪标签预训练后冻结）和动态教师 $T^B$（EMA 更新）分别生成软伪标签 $\hat{\mathbf{y}}_i^A$、$\hat{\mathbf{y}}_i^B$
    - 用 Shannon 熵 $\mathcal{H}(\hat{\mathbf{y}}(\mathbf{p})) = -\sum_c \hat{\mathbf{y}}_c(\mathbf{p}) \log(\hat{\mathbf{y}}_c(\mathbf{p}) + \epsilon)$ 量化逐像素不确定性
    - 经 patch-wise 平均池化（$k=14$）平滑后取倒数为置信度权重 $\mathbf{w}_{A,B} = \frac{1}{\mathbf{E}_{A,B}^{\text{smooth}} + \epsilon}$
    - 加权融合：$\hat{\mathbf{y}}_i^F = \frac{\mathbf{w}_A \cdot \hat{\mathbf{y}}_i^A + \mathbf{w}_B \cdot \hat{\mathbf{y}}_i^B}{\mathbf{w}_A + \mathbf{w}_B + \epsilon}$

3. **AURCL（自适应不确定性引导反向对比学习）**

    - 对学生的低置信度像素（动态 top-K 阈值 $\tau_i = \max(\text{top-K}(\mathbf{C}_i, K), 0.2)$ 选取），将预测概率翻转 $\tilde{\mathbf{p}}_i(u,v) = 1 - \mathbf{p}_i(u,v)$，生成"反向视图"
    - 在 patch 级提取原始和反向视图特征，用 InfoNCE 对比损失拉近同位置正对、推远不同位置负对
    - 迫使网络在模糊边界区域学习更具判别性的表征

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_s + \lambda_u \mathcal{L}_u + \lambda_c \mathcal{L}_c$$

其中 $\mathcal{L}_s$ 和 $\mathcal{L}_u$ 均为 BCE + Dice，$\mathcal{L}_c$ 为 AURCL 对比损失。$\lambda_u=1$，$\lambda_c=0.5$。ResNet-34 骨干，输入 224×224，Adam（momentum 0.995），ReduceLROnPlateau 调度，batch size 8（标注/未标注各半），100 epochs。不使用数据增强。

## 实验关键数据

### 主实验

| 数据集 | 标注比例 | Dice(%) | IoU(%) | vs 之前 SOTA | 提升 |
|--------|----------|---------|--------|-------------|------|
| BUSI | 2.5% | **72.72** | **63.20** | BCP 58.93 | +13.79 |
| BUSI | 10% | **77.40** | **67.13** | Text-semiseg 75.06 | +2.34 |
| BUSI | 20% | **78.38** | **68.60** | Text-semiseg 75.83 | +2.55 |
| UBB | 2.5% | **75.75** | **65.89** | Text-semiseg 59.76 | +15.99 |
| UBB | 10% | **75.95** | **65.70** | Text-semiseg 74.70 | +1.25 |
| UBB | 20% | **78.15** | **68.05** | Text-semiseg 75.55 | +2.60 |
| BUSI 全监督 | 100% | 81.68 | 73.74 | — | — |

UBB 数据集 2.5% 标注下，本文 75.75% Dice 甚至超过 100% 标注全监督 U-Net（74.81%）。

### 消融实验

| 组件 | Dice(%) | 增量 |
|------|---------|------|
| Baseline (U-Net, 2.5%) | 50.00 | — |
| + 静态教师（VLM伪标签预训练） | 67.34 | +17.34 |
| + 双教师 EMA | 71.20 | +3.86 |
| + UEWF | 71.89 | +0.69 |
| + Patch-wise 平滑 | 72.20 | +0.31 |
| + AURCL | **72.72** | +0.52 |

VLM 对比：MediClipV2 仅 28.74% Dice，UniversalSeg 30.68%，本文 72.72%。

### 关键发现

- APPG 贡献最大（+17.34% Dice），提供稳定的外部结构先验
- 外观描述提示显著优于医学术语和放射学属性描述——"dark oval" 比 "tumor" 的检测框更准确
- Patch-wise 平滑比 pixel-wise 更鲁棒
- UBB 跨设备数据集上优势更大（2.5% 下 +15.99%），证明框架的泛化能力

## 亮点与洞察

- 仅需一句简单外观描述即可跨域迁移 VLM 到任意医学模态，范式可泛化到皮肤镜、甲状腺超声、内镜息肉等
- 2.5% 标注即超越全监督，极端低标注场景下优势巨大
- 反向对比学习关注不确定区域的思路新颖，与常规对比学习只关注可靠区域形成互补
- 外观描述作为跨域桥梁的思路值得推广到其他医学少标注场景

## 局限与展望

- 当病灶外观高度异质时（如浸润性病灶形态多变），简单外观描述可能不够
- 未探索更强的 VLM（如 Grounded SAM 2），升级 VLM 可能进一步提升伪标签质量
- 仅在二分类（病灶/背景）上验证，多类分割场景未涉及
- 不使用数据增强的策略限制了在其他域的适用性

## 相关工作与启发

- **vs PH-Net (CVPR'24)**：通过 patch-wise hardness 挖掘困难区域，但仍依赖模型自身伪标签，2.5% 下 Dice 仅 55.13%，远低于本文
- **vs Text-semiseg (MICCAI'25)**：引入文本驱动多平面视觉交互增强伪标签，10%/20% 下有竞争力但 2.5% 下仅 56.85%，说明文本引导在极少标注下不如外观提示 + VLM 免训练方案
- **vs CSC-PA (CVPR'25)**：跨样本原型对齐增强语义一致性，2.5% 仅 58.78%
- 启发：双教师不确定性融合机制可迁移到其他半监督检测/分割任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 外观提示驱动 VLM 免训练生成伪标签的思路简洁且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、三种标注比例、充分消融、跨模态泛化验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，流程图直观，消融逐步递进
- 价值: ⭐⭐⭐⭐ 2.5% 标注超越全监督的实用价值高，范式可推广
