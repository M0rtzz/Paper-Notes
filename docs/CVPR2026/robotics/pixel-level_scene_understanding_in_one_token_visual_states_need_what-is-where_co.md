---
title: >-
  [论文解读] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition
description: >-
  [CVPR 2026][机器人][视觉状态表示] 本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息…
tags:
  - "CVPR 2026"
  - "机器人"
  - "视觉状态表示"
  - "自监督学习"
  - "机器人策略学习"
  - "瓶颈token"
  - "全局-局部重建"
---

# Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition

**会议**: CVPR 2026  
**arXiv**: [2603.13904](https://arxiv.org/abs/2603.13904)  
**代码**: [项目主页](https://seokminlee-chris.github.io/CroBo-ProjectPage)  
**领域**: 机器人 / 自监督学习  
**关键词**: 视觉状态表示, 自监督学习, 机器人策略学习, 瓶颈token, 全局-局部重建

## 一句话总结
本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息，在Franka Kitchen和DMC机器人策略学习benchmark上达到SOTA。

## 研究背景与动机
1. **领域现状**：自监督视觉表示学习（MAE、DINO、DINOv2等）在图像分类和语义分割等任务上表现优异，但这些方法没有显式考虑"好的视觉状态表示应该编码什么"。对于机器人的序列决策任务，需要将原始视觉观察压缩为紧凑的视觉状态。
2. **现有痛点**：(1) 对比学习方法（DINO）学习的是全局语义特征，对空间位置不敏感；(2) MAE通过patch级重建学习局部特征，但没有全局场景理解；(3) SiamMAE/CropMAE学习patch级对应关系，不适合需要单个紧凑表示的下游任务；(4) ToBo引入了瓶颈token概念，但基于时序帧重建，未显式约束空间组成编码。
3. **核心矛盾**：机器人需要检测场景中的微妙变化（如机械臂移动了几厘米），这要求视觉状态同时编码"是什么物体"和"在哪里"。但现有SSL方法要么丢失空间信息（对比学习的增强不变性），要么没有单一紧凑表示（patch级方法输出tokens序列）。
4. **本文目标** 学习一个单token的视觉状态表示，使其编码完整的"what-is-where"场景组成——哪些语义实体存在、它们在哪里、如何排列。
5. **切入角度**：如果一个紧凑表示能够从极少线索中恢复出任意局部裁剪区域的内容，那它必然编码了整个场景的语义和空间信息。
6. **核心 idea**：用单个[CLS] token作为全局瓶颈，重建90%遮蔽的局部裁剪视图，迫使瓶颈token成为像素级场景组成的压缩表示。

## 方法详解

### 整体框架
CroBo想解决的问题是：怎么让一个单独的token装下整个场景的"什么物体在什么位置"。它的做法是把这个目标转化成一个重建难题——从一张图里采样单帧，先裁出一个大的全局源视图$\mathbf{x}^g$（尺度[0.5, 1.0]），再从源视图内部裁出一个小的局部目标视图$\mathbf{x}^l$（尺度[0.3, 0.6]）。两个视图共用一个Siamese编码器，但待遇不同：源视图完整输入、不遮蔽，编码后留下一个[CLS] token当作"全局瓶颈"；目标视图则被狠狠遮蔽90%，只剩零星几个可见patch。最后一个Transformer解码器拿着源视图的[CLS] token加目标视图的少量可见patch，去重建目标视图里被遮掉的那些patch。整条pipeline没有任何时序输入，全靠"局部是全局的子区域"这个空间关系把信息逼进瓶颈token。

### 关键设计

**1. 全局-局部重建目标：让单个token成为重建小图的唯一靠山**

痛点很直接——既然想要一个紧凑表示编码"what-is-where"，就得设计一个任务，让模型不编码这些信息就完不成。CroBo的安排是：目标视图$\mathbf{x}^l$是源视图的一个子区域，90%遮蔽后只剩约14个可见patch，解码器手里实际只有两路信号，一是源视图压出来的[CLS] token（全局场景上下文），二是目标这十几个可见patch（局部线索）。可见patch太稀疏，光凭它们根本拼不出被遮区域，于是解码器被迫死死依赖[CLS] token——而要让[CLS]能指引"这块该填什么、填在哪",它就只能事先把"每个物体是什么、在场景的哪个位置"全编码进去。和传统MAE的区别正在这里：MAE是同一视图内75%遮蔽、可见patch还能提供足够局部上下文，瓶颈不紧；CroBo的跨视图+90%极端遮蔽把局部线索抽干，全局瓶颈才真正成为重建的唯一靠山。

**2. Siamese编码器配高遮蔽率：用共享权重和信息瓶颈双重逼迫**

既然重建要同时读源视图和目标视图，两路的表示空间必须对齐，否则[CLS] token和可见patch说的不是同一种"语言"。CroBo用同一个ViT编码两路解决对齐：源视图全部patch进去产出$N$个patch token加1个[CLS] token，目标视图只把约10%的可见patch送进去。真正的杠杆是遮蔽率——它从MAE标准的75%一路推到90%，消融里95%还能再涨。背后的逻辑是设计1的延伸：遮蔽率一旦偏低，解码器就有机会绕过瓶颈、只靠目标视图残存的可见patch蒙混重建，那[CLS] token就学不到东西；把可见信息抽到只剩一两成，才能堵死这条捷径、强制信息全部走瓶颈。

**3. Crop而非Time构建视图对：用空间子集换一个明确的重建目标**

构建源-目标视图对有三种选法，CroBo逐一对比：Time（取不同帧）、Crop（同帧不同裁剪）、Time+Crop（既换帧又裁剪）。结论是Crop一致最好，因为目标视图就是源视图的空间子集，"该重建成什么样"是确定且well-defined的；Time会引入运动和光照变化，让重建目标变得模糊；Time+Crop把两种歧义叠加，效果反而最差。值得点一句的是，CroBo训练只用单帧、并不吃时序数据，之所以仍在视频数据集上训练，纯粹是为了和ToBo等基于视频的前作做公平对比——这也间接说明，"学动态表示"未必非要喂视频，把单帧的空间组成理解学扎实就够了。

举个具体画面：一张CLEVR场景图里有两个青色球体，目标视图把它俩都裁进去又90%遮掉，可见patch里完全看不到球。CroBo靠[CLS] token里编码的全局组成，仍能在重建结果里把这两个被完全遮挡的球放回正确位置——这正是"单token装下what-is-where"在起作用的直观证据。

### 损失函数 / 训练策略
训练目标就是被遮patch上的MSE重建损失：$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{\mathbf{x}}_i^l - \mathbf{x}_i^l \|_2^2$，其中$\mathcal{M}$是被遮patch集合，重建目标用归一化像素值。骨干为ViT-S/16编码器配8层、512维的解码器，AdamW优化，batch size 1536，在Kinetics-400上训400 epochs。

## 实验关键数据

### 主实验（Franka Kitchen成功率% + DMC归一化分数）

| 方法 | Knob on | Light on | Sdoor | Ldoor | Micro | walker/stand | walker/walk | reacher/easy |
|------|---------|----------|-------|-------|-------|-------------|-------------|-------------|
| MAE | 12.0 | 24.3 | 71.5 | 12.8 | 10.0 | - | - | - |
| DINOv2 | 25.0 | 46.6 | 87.8 | 17.6 | 21.8 | 81.2 | 38.2 | 87.6 |
| ToBo | 58.4 | 80.6 | 98.4 | 44.2 | 51.2 | 87.0 | 77.7 | 87.5 |
| **CroBo** | **65.6** | **87.6** | **99.4** | 41.2 | **64.8** | **92.0** | **80.8** | **95.8** |
| Δ SOTA | +7.2 | +7.0 | +1.0 | -3.0 | +13.6 | +5.0 | +3.1 | +8.3 |

### 消融实验

| 配置 | Knob1 | Light | Sdoor | Ldoor | Micro |
|------|-------|-------|-------|-------|-------|
| Time（时序帧） | 44.4 | 67.6 | 96.4 | 36.8 | 49.6 |
| **Crop（空间裁剪）** | **57.6** | **81.6** | **98.6** | **36.8** | **50.4** |
| Time+Crop | 38.2 | 62.8 | 90.8 | 22.2 | 28.4 |
| 遮蔽率75% | 41.4 | 70.6 | 94.0 | 22.6 | 35.0 |
| 遮蔽率90% | 57.6 | 81.6 | 98.6 | 36.8 | 50.4 |
| 遮蔽率95% | **59.0** | **86.6** | **99.4** | **41.2** | **58.0** |

### 关键发现
- **Crop显著优于Time**：同帧裁剪提供well-defined的重建目标，时序帧引入过多歧义。这颠覆了"需要视频才能学习动态表示"的直觉
- **遮蔽率越高越好**：从75%到95%，所有任务性能持续提升，说明更少的目标线索迫使模型更充分利用瓶颈token
- **ViT-S超越所有ViT-L baseline**：小模型（65.0%平均成功率）超过了所有大模型baseline（最好63.3%），增益来自更好的表示而非更大的模型
- **感知直线性分析**：CroBo在DAVIS视频上的表示曲率仅75.4°，远低于DINOv2的103.28°，说明CroBo的表示在时序上更线性，更好地编码了"什么在哪里移动"
- **重建可视化**：即使目标视图90%被遮蔽，CroBo能正确恢复被完全遮挡的物体（如CLEVR中两个青色球体都不可见但位置被正确重建），证明瓶颈token确实编码了完整的what-is-where信息

## 亮点与洞察
- **"不需要视频就能学动态表示"的发现**非常反直觉：CroBo仅用单帧的空间裁剪训练，在机器人动态任务上优于基于视频时序的方法。这说明空间组成理解是动态理解的基础
- **极高遮蔽率作为信息瓶颈的设计**巧妙结合了两个思想：MAE的重建学习 + 信息瓶颈原理。遮蔽率越高，瓶颈越紧，表示越被迫编码全局信息
- **感知直线性指标**为表示质量提供了新的度量维度：好的视觉状态表示应该使得观察轨迹在表示空间中近似线性，便于预测和控制

## 局限与展望
- 仅在Franka Kitchen和DMC两个仿真环境中评估，缺乏真实机器人实验
- 瓶颈token是单个token，信息容量有限，对于更复杂的场景可能不足
- 仅使用ViT-S/B/L评估，未探索更大或更新的视觉骨干（如ViT-G、DINOv2-L+）
- 重建目标使用像素MSE，可能不是最优的，感知损失或特征级重建可能更好
- Ldoor任务上不如ToBo，暗示某些需要强时序关联的任务仍需时序信息

## 相关工作与启发
- **vs ToBo**: CroBo的直接前身，ToBo用时序帧重建，CroBo用空间裁剪重建，后者更清晰
- **vs CropMAE**: 类似的空间裁剪思路，但CropMAE学习patch级对应关系，CroBo将信息压缩为单token
- **vs SiamMAE**: SiamMAE也是Siamese编码器+视频帧对，但没有瓶颈token设计
- **vs DINOv2**: 通用表示但不保留精细空间信息，在机器人任务上明显不如CroBo

## 评分
- 新颖性: ⭐⭐⭐⭐ what-is-where的概念清晰，全局-局部重建设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark、完整消融、重建可视化、感知直线性分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事逻辑性强，从"什么是好的视觉状态"出发推导出方法设计
- 价值: ⭐⭐⭐⭐ 对机器人视觉表示学习有重要启发，单帧训练胜时序训练的发现有广泛影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] Towards Training-Free Scene Text Editing](towards_training-free_scene_text_editing.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[CVPR 2026\] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)

</div>

<!-- RELATED:END -->
