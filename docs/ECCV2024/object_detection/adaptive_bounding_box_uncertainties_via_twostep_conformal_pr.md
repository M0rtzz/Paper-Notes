---
title: >-
  [论文解读] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction
description: >-
  [ECCV 2024][目标检测][目标检测] 本文提出一种两步共形预测框架用于多目标检测的不确定性量化：第一步生成类别标签的共形预测集合以处理分类错误，第二步基于集成和分位数回归生成自适应的边界框不确定性区间，在保证覆盖率的同时提供实际可用的紧致预测区间。
tags:
  - ECCV 2024
  - 目标检测
  - 共形预测
  - 不确定性量化
  - 边界框回归
  - 安全关键系统
---

# Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction

**会议**: ECCV 2024  
**arXiv**: [2403.07263](https://arxiv.org/abs/2403.07263)  
**代码**: https://github.com/alextimans/conformal-od (有)  
**领域**: 目标检测  
**关键词**: 目标检测, 共形预测, 不确定性量化, 边界框回归, 安全关键系统

## 一句话总结
本文提出一种两步共形预测框架用于多目标检测的不确定性量化：第一步生成类别标签的共形预测集合以处理分类错误，第二步基于集成和分位数回归生成自适应的边界框不确定性区间，在保证覆盖率的同时提供实际可用的紧致预测区间。

## 研究背景与动机
安全关键应用（如自动驾驶、移动机器人）需要对模型的预测不确定性进行精确量化。现有的目标检测不确定性方法（贝叶斯推断、MC Dropout、深度集成等）需要大量修改模型架构或训练过程，且**无法提供覆盖率保证**。共形预测（Conformal Prediction, CP）作为一种分布无关的不确定性量化框架，可以提供后验的、与模型无关的概率保证，但将其应用于目标检测面临两个核心挑战：（1）边界框预测依赖于类别标签，**分类错误会导致选错共形分位数**，使得覆盖率保证失效；（2）标准CP产生的**固定宽度区间不能自适应物体大小**，导致小物体过度覆盖而大物体覆盖不足。本文的切入角度是设计两步式共形流水线：先用CP处理分类不确定性，再将其传播到边界框区间构造中。核心idea：用类条件共形预测同时保证分类和定位的覆盖率，且使区间自适应于物体大小。

## 方法详解

### 整体框架
整体框架是一个两步顺序式共形预测流水线：（1）**第一步——分类CP**：对目标检测器的分类头应用共形预测，生成类别标签预测集合 $\hat{C}_L(X_{n+1})$，保证真实类别以 $(1-\alpha_L)$ 概率被包含；（2）**第二步——回归CP**：对每个边界框坐标分别构造共形预测区间，使用第一步得到的标签集合选择正确的类条件分位数，保证4个坐标的联合覆盖率 $(1-\alpha_B)$。整体覆盖率保证为 $(1-\alpha_L)(1-\alpha_B)$。

### 关键设计
1. **自适应边界框方法 (Box-Ens / Box-CQR)**:
    - 功能：生成随物体大小自适应调整的预测区间
    - 核心思路：Box-Ens（共形集成）使用归一化残差作为非一致性分数 $s = |c^k - \hat{c}^k| / \hat{\sigma}(X)$，其中 $\hat{\sigma}$ 为集成检测器预测的标准差，生成的区间可以按模型不确定性缩放。Box-CQR（共形分位数回归）训练额外的分位数回归头预测上下分位数 $\hat{Q}_{\alpha_B/2}$ 和 $\hat{Q}_{1-\alpha_B/2}$，区间宽度由分位数预测自然决定
    - 设计动机：标准CP（Box-Std）产生固定宽度区间，对大物体不够大、对小物体过于保守，破坏了覆盖率在不同大小物体间的平衡

2. **两步共形流水线与ClassThr**:
    - 功能：将分类不确定性传播到边界框区间，使覆盖率保证扩展到被错误分类的物体
    - 核心思路：使用类条件共形分类器(ClassThr)生成标签预测集合 $\hat{C}_L(X_{n+1}) = \{y \in \mathcal{Y}: \hat{\pi}_y(X_{n+1}) \geq 1 - \hat{q}_L^y\}$，然后用max策略从标签集合中选择边界框分位数 $\hat{q}_B^k = \max\{\hat{q}_B^{k,y}\}_{y \in \hat{C}_L(X_{n+1})}$。设置 $\alpha_L=0.01$ 使得 $(1-\alpha_L)(1-\alpha_B) \approx (1-\alpha_B)$
    - 设计动机：先前工作只在正确分类的物体上提供保证，在多类别场景（如自动驾驶中car/person/bicycle共存）中实用性有限。两步法将覆盖率保证扩展到所有被检测到的物体

3. **多重检验校正 (Max-Rank)**:
    - 功能：解决对4个坐标分别做CP时的多重检验问题
    - 核心思路：对$m$个坐标分别做CP相当于并行进行$m$个假设检验，朴素Bonferroni校正过于保守。采用Westfall & Young排列校正的改进版max-rank方法，在秩空间中操作，利用坐标间的正相关结构获得更紧致的校正
    - 设计动机：Bonferroni假设独立性，但边界框坐标天然高度相关（它们共同参数化一个框），利用这种相关结构可以避免过度保守

### 损失函数 / 训练策略
CP是后验方法，不需要修改模型训练。Box-CQR方法需要对检测器添加分位数回归头并用分位数损失训练。集成方法需要训练多个独立的检测器。关键超参数：$\alpha_L=0.01$（标签覆盖率99%），$\alpha_B=0.1$（边界框覆盖率90%），IoU阈值0.5用于匈牙利匹配。

## 实验关键数据

### 主实验
与先前方法在COCO数据集上的对比（目标覆盖率90%，跨类别平均）：

| 方法 | 检测器 | MPIW (双侧) | 覆盖率 (双侧) | MPIW (单侧) | 覆盖率 (单侧) |
|------|--------|------------|--------------|------------|--------------|
| Deep Ensembles | 5×Faster R-CNN | 12.31 | 0.21 ❌ | 74.15 | 0.49 ❌ |
| GaussianYOLO | YOLOv3 | 7.00 | 0.08 ❌ | 87.07 | 0.35 ❌ |
| Andéol et al. (Best) | Faster R-CNN | N/A | - | 87.62 | 0.91 ✓ |
| **Box-Std (Ours)** | Faster R-CNN | 55.47 | 0.88 ✓ | **85.42** | **0.88** ✓ |
| **Box-Std (Ours)** | Sparse R-CNN | **41.92** | 0.89 ✓ | 77.33 | 0.89 ✓ |

### 消融实验

| 配置 | 覆盖率 | MPIW | 说明 |
|------|--------|------|------|
| Box-Std (固定宽度) | 达标 | 最小 | 效率最高但小/大物体覆盖不均 |
| Box-Ens (集成自适应) | 达标 | 稍大 | 大中小物体覆盖更均衡 |
| Box-CQR (分位数自适应) | 达标 | 中等 | 大物体覆盖改善显著 |
| Top (单类别标签) | ❌ 不达标 | 最小 | 依赖分类准确率，无保证 |
| Naive (密度水平集) | ❌ 标签不达标 | 小 | 对模型校准敏感 |
| **ClassThr (共形阈值)** | ✓ 达标 | 中等 | 唯一同时满足标签和框保证的方法 |

### 关键发现
- 深度集成和GaussianYOLO等传统不确定性方法**严重欠覆盖**（覆盖率仅0.08-0.49），证明缺乏保证的方法在安全关键场景不可靠
- Box-Ens在不同物体大小间的覆盖率最均衡：大物体覆盖率显著提升，仅以略微增大的MPIW为代价
- ClassThr生成的平均标签集大小 $\leq 4$，说明标签CP的开销很小，不会导致边界框区间过度膨胀
- max-rank校正相比Bonferroni产生显著更紧致的区间，验证了利用坐标相关结构的价值

## 亮点与洞察
- 提出了端到端的安全边界框不确定性框架：后验、高效、可泛化，不需要修改底层检测器
- 两步法的设计非常优雅：通过将分类不确定性显式传播到定位不确定性中，提供了真正实用的安全保证
- 类条件保证的选择比边际保证更强更实用，避免了不同类别间覆盖率不均的问题
- 覆盖率的可控权衡：用户可以通过调节 $\alpha_L$ 和 $\alpha_B$ 灵活平衡分类和定位的安全需求

## 局限与展望
- 只对**检测到的真正例**提供保证，不处理漏检（假阴性），这是CP框架本身的限制
- max策略选择分位数偏保守，导致ClassThr方法的区间倾向于过度覆盖，可以探索基于混淆矩阵的加权分位数选择策略
- 交换性假设要求数据分布不变，在分布漂移场景（如天气变化）下保证可能失效
- 目前仅在2D检测上验证，3D检测、实例分割等扩展需要进一步探索
- Box-Ens需要训练多个检测器，计算成本较高

## 相关工作与启发
- 与Andéol et al.的核心区别：扩展到多类别设置并处理分类错误，先前工作只在单类别且正确分类的物体上提供保证
- 启发：CP框架为黑盒模型提供了极有价值的后验保证工具，特别适合无法修改模型架构的部署场景
- 两步顺序CP的思想可推广到其他多阶段预测任务（如先分割后分类），其中上游不确定性需要传播到下游
- max-rank多重检验校正可用于任何涉及多维共形预测的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 两步共形框架和自适应边界框方法是有意义的方法创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种检测器、多种基线、1000次trial统计，非常详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，符号体系完整，逻辑推导清晰，讨论部分对实践者非常有价值
- 价值: ⭐⭐⭐⭐ 对自动驾驶等安全关键领域有直接应用价值，且框架具有良好的可扩展性

<!-- RELATED:START -->

## 相关论文

- [REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)
- [Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)
- [Rectify the Regression Bias in Long-Tailed Object Detection](rectify_the_regression_bias_in_long-tailed_object_detection.md)

<!-- RELATED:END -->
