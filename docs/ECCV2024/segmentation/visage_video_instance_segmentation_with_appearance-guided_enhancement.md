---
title: >-
  [论文解读] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
description: >-
  [ECCV 2024][图像分割][视频实例分割] 针对在线视频实例分割(VIS)中现有方法过度依赖位置信息导致的关联错误，提出VISAGE通过从骨干特征中显式提取外观嵌入、结合对比学习和简化tracker来增强实例关联准确性，在YTVIS和OVIS基准上取得SOTA。
tags:
  - ECCV 2024
  - 图像分割
  - 视频实例分割
  - 外观引导
  - 查询匹配
  - 对比学习
  - 记忆库
---

# VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement

**会议**: ECCV 2024  
**arXiv**: [2312.04885](https://arxiv.org/abs/2312.04885)  
**代码**: https://github.com/KimHanjung/VISAGE (有)  
**领域**: 分割  
**关键词**: 视频实例分割, 外观引导, 查询匹配, 对比学习, 记忆库

## 一句话总结

针对在线视频实例分割(VIS)中现有方法过度依赖位置信息导致的关联错误，提出VISAGE通过从骨干特征中显式提取外观嵌入、结合对比学习和简化tracker来增强实例关联准确性，在YTVIS和OVIS基准上取得SOTA。

## 研究背景与动机

视频实例分割需要在视频序列中对不同实例进行分类、分割和跟踪。近年来，基于查询的检测器(如Mask2Former)推动了在线VIS方法的快速发展，主要有query-propagation和query-matching两种策略。

然而，作者通过深入分析发现了一个关键问题：**现有方法严重依赖位置信息，忽略了外观信息。** 具体表现为：
1. 在镜头切换场景下，物体位置突变导致追踪失败
2. 在轨迹交叉场景下，基于位置的匹配导致ID交换
3. 通过水平翻转图像生成伪视频进行实验验证：即使物体外观差异明显，现有模型仍出现关联错误（因为它们倾向于维持前一帧预测的空间顺序）

这些观察表明，外观信息是物体匹配的关键维度，尤其在位置线索不足以区分身份的条件下。

## 方法详解

### 整体框架

VISAGE基于Mask2Former检测器构建，分为三个核心组件：
1. 检测器：标准的backbone + transformer encoder + transformer decoder结构
2. 外观引导增强模块：从backbone特征中通过mask pooling提取外观查询
3. 简化追踪器：联合使用object embedding和appearance embedding进行匈牙利匹配

### 关键设计

1. **外观查询提取(Appearance Query Extraction)**：
   - 核心操作：使用object query预测的mask对backbone特征图进行平均池化(mask pooling)，得到appearance query
   - 设计动机：传统跟踪方法(如RoIPool/RoIAlign)早已使用特征图提取实例特征，但现代query-based方法丢失了这一能力
   - 将两种query分别投射为appearance embedding $\mathbf{e}_a \in \mathbb{R}^{N \times C}$ 和 object embedding $\mathbf{e}_i \in \mathbb{R}^{N \times C}$
   - 关键发现：使用backbone特征显著优于使用transformer encoder特征（AP: 55.1 vs 51.4），因为backbone保留了更丰富的视觉信息

2. **对比学习增强嵌入区分度**：
   - 对object embedding和appearance embedding分别施加对比损失
   - 让同一实例在不同帧中的嵌入更接近，不同实例的嵌入更分离
   - 关键差异：与之前方法(IDOL, CTVIS)不同的是，将两种embedding分开处理，让每种嵌入保持各自特性，在匹配时互补
   - 对比损失权重设为2.0，与检测器原始损失加权求和

3. **简化追踪器与记忆库**：
   - 匹配得分：$\mathbf{s} = (1-\alpha) \cdot \cos(\mathbf{e}_i^t, \mathbf{m}_i^t) + \alpha \cdot \cos(\mathbf{e}_a^t, \mathbf{m}_a^t)$
   - 使用匈牙利算法进行最优分配，α=0.75在推理时使用
   - 记忆库大小 W=5，读取记忆嵌入时通过时间加权和置信度加权：$\mathbf{m}^t = \sum_{w=1}^{W} \mathbf{e}^{t-w} s^{t-w} \times \frac{W}{w}$
   - 不使用NMS、不需要tracklet初始化/删除阈值等heuristic操作，大幅简化pipeline

### 损失函数 / 训练策略

- 基础损失：延续Mask2Former的损失函数和权重
- 额外损失：对appearance embedding和object embedding分别加对比损失，权重均为2.0
- Backbone: ResNet-50，COCO预训练初始化，采用COCO联合训练策略
- 批大小：16 videos
- 训练设备：4块NVIDIA A6000 GPU

## 实验关键数据

### 主实验

| 方法 | 设置 | YTVIS19 AP | YTVIS21 AP | OVIS AP |
|------|------|-----------|-----------|---------|
| MinVIS | online | 47.4 | 44.2 | 25.0 |
| IDOL | online | 49.5 | 43.9 | 30.2 |
| GenVIS | online | 50.0 | 47.1 | 35.8 |
| DVIS | online | 51.2 | 46.4 | 31.0 |
| TCOVIS | online | 52.3 | 49.5 | 35.3 |
| CTVIS | online | 55.1 | 50.1 | 35.5 |
| **VISAGE** | **online** | **55.1** | **51.6** | **36.2** |

### 消融实验

| 外观信息 | 记忆库 | YTVIS19 AP | AP50 | AP75 | AR1 | AR10 |
|---------|-------|-----------|------|------|-----|------|
| ✗ | ✗ | 49.9 | 71.4 | 54.7 | 47.0 | 58.7 |
| ✗ | ✓ | 50.2 | 72.1 | 54.7 | 47.3 | 60.7 |
| ✓ | ✗ | 53.4 | 76.8 | 58.7 | 49.8 | 61.2 |
| ✓ | ✓ | **55.1** | **78.1** | **60.6** | **51.0** | **62.3** |

外观权重α分析（α=0为仅位置，α=1为仅外观）：

| α | YTVIS19 AP | OVIS AP |
|---|-----------|---------|
| 0.00 | 53.4 | 32.2 |
| 0.25 | 53.6 | 34.5 |
| 0.50 | 54.5 | 34.8 |
| **0.75** | **55.1** | **36.2** |
| 1.00 | 24.9 | - |

### 关键发现

1. **外观引导提升巨大**：加入外观信息带来3.5 AP的提升（49.9→53.4），比记忆库的增益(0.3 AP)大10倍以上
2. **在复杂场景(OVIS)上效果更显著**：OVIS的AP从32.2提升到36.2(+4.0)，因为频繁遮挡使位置线索更不可靠
3. **外观和位置缺一不可**：α=1.0（仅外观）时性能暴跌至24.9 AP，说明二者协同工作
4. **简化tracker反而更好**：不使用NMS等heuristic的简化tracker(55.1 AP)优于传统tracker+NMS(54.4 AP)
5. **Backbone特征优于Encoder特征**：用Backbone生成外观查询比用Transformer Encoder高3.7 AP

## 亮点与洞察

1. **问题发现精准**：通过翻转图像的简单实验优雅地证明了"位置偏见"现象
2. **方法极其简洁**：核心创新仅是一个mask pooling操作 + 加权匹配公式，但效果显著
3. **打破常规**：query-based方法通常认为query已包含足够信息，但VISAGE证明显式提取外观特征是必要且有效的
4. **简化>复杂**：去除大量heuristic超参数(NMS阈值、tracklet管理阈值等)后性能反而提升

## 局限性 / 可改进方向

1. **仅基于ResNet-50**：未在更强backbone(如Swin-L)上验证，可能还会有更大的提升空间
2. **外观特征的全局性不足**：当前mask pooling的外观特征是"局部平均"，可能丢失细粒度纹理信息
3. **记忆库大小固定**：W=5的窗口可能不适用于所有视频长度，长视频可能需要更大的记忆
4. **未探索在线学习**：外观嵌入的对比学习仅在训练时进行，推理时不更新

## 相关工作与启发

- MinVIS开创了仅训练检测器、推理时做帧间匹配的范式
- CTVIS使用记忆库进行训练时的对比学习，增强了判别能力
- 传统跟踪器(如RoIPool/RoIAlign)提取实例特征的方式给了VISAGE灵感
- 外观引导的思想可以推广到其他query-based视觉任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题洞察深刻，但方法本身(mask pooling + 对比学习)并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ — 四个基准全面对比，消融详尽(特征来源、α值、tracker设计)
- 写作质量: ⭐⭐⭐⭐⭐ — 翻转图像实验的proof of concept设计精妙，算法流程清晰
- 价值: ⭐⭐⭐⭐ — 提供了简单有效的VIS改进方案，具有很强的实践价值
