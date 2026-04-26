---
title: >-
  [论文解读] MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding
description: >-
  [ICCV 2025][自动驾驶][驾驶视频理解] 提出 MCAM，通过驾驶状态有向无环图 (DSDAG) 构建视觉-语言模态间的因果结构，结合多层级特征提取和因果分析模块，用于自车级别驾驶视频理解中的行为描述与原因推理。
tags:
  - ICCV 2025
  - 自动驾驶
  - 驾驶视频理解
  - 因果分析
  - 有向无环图
  - 视觉-语言
  - 自车行为理解
---

# MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding

**会议**: ICCV 2025  
**arXiv**: [2507.06072](https://arxiv.org/abs/2507.06072)  
**代码**: [GitHub](https://github.com/SixCorePeach/MCAM)  
**领域**: 自动驾驶  
**关键词**: 驾驶视频理解, 因果分析, 有向无环图, 视觉-语言, 自车行为理解

## 一句话总结

提出 MCAM，通过驾驶状态有向无环图 (DSDAG) 构建视觉-语言模态间的因果结构，结合多层级特征提取和因果分析模块，用于自车级别驾驶视频理解中的行为描述与原因推理。

## 研究背景与动机

自动驾驶视频理解旨在将驾驶行为从视觉数据翻译为文本叙述——不仅描述"做了什么"，还需解释"为什么这样做"。现有方法面临三个核心问题：

**浅层因果关系挖掘**：现有视觉到语言方法（如 ADAPT、DriveGPT4）停留在概率相关性层面，缺乏对驾驶行为的深层因果推理。例如，看到前车刹车灯亮就判断"因为前车减速而停车"，但真正的原因可能是远处的红灯。

**跨模态虚假相关**：视觉特征和语言描述之间存在虚假相关。比如在某些场景中，路边的停车标志和车辆无关的停靠场景在特征空间中会产生错误关联，模型可能据此生成不准确的推理。

**忽略自车级别因果建模**：现有工作主要关注事件级别的因果关系，而非以自车状态转换为核心的因果理解。自车从安全状态出发，因环境变化面临潜在危险，采取驾驶动作回归安全状态——这一完整闭环缺乏形式化建模。

作者的核心洞察：驾驶行为理解可以形式化为一个状态转换图——从初始安全状态 $X_s$ 出发，环境变化 $Z$ 引发潜在危险 $W$，驾驶动作 $Y$ 使车辆转向新的安全状态 $X_e$。任务是逆向推理：给定观察到的行为和结果，找出环境中哪些因素 $V$ 是关键影响因子。

## 方法详解

### 整体框架

MCAM 包含三个组件：
1. **多层级特征提取器 (MFE)**：从视频中提取全局和局部特征
2. **因果分析模块 (CAM)**：基于 DSDAG 构建因果关系
3. **视觉-语言 Transformer (VLT)**：整合视觉特征和文本信息生成描述和推理

### 关键设计

1. **多层级特征提取器 (MFE)**：

    - **核心思路**：并行使用 VidSwin Transformer（捕获全局依赖）和 3DResNet（提取局部特征），分别处理整个视频、首帧和末帧。
    - **输入处理**：视频帧 $B \times F \times H \times W \times 3$，VidSwin 输出全局特征、3DResNet 输出局部特征。通过下采样和 1×1 卷积对齐尺寸后用线性层融合。
    - **设计动机**：Transformer 擅长全局建模但局部特征提取不足，CNN 善于捕获局部模式。双通路并行提取互补特征。首帧和末帧的分别提取对应 DSDAG 中的初始/终止状态。

2. **因果分析模块 (CAM)**：

    - **驾驶状态 DAG (DSDAG)**：将驾驶过程建模为有向无环图，包含节点：初始安全状态 $X_s$、环境 $Z$、驾驶动作 $Y$、潜在危险 $W$、终止安全状态 $X_e$。
    - **因果推理形式化**：基于 Pearl 的因果框架，使用 do-演算建模干预效果。关键等式：
        - 动作由状态和环境决定：$Y_c = F_Y(Z_\xi | U_s)$
        - 危险由不采取动作时的环境决定：$W = F_W(Z_\xi | U_s, do(Y_c = \emptyset))$
        - 终止状态：$X_e = F_{X_e}(Z_\xi | U_s, do(Y_c = c))$
    - **特征解耦与融合**：MFE 输出的 6 组特征（首/末帧 × 全局/局部，全片 × 全局/局部）通过不同的线性层分别投影为初始状态特征 $F_{init}$、终止状态特征 $F_{end}$、潜在危险特征 $F_{pot}$、动作特征 $F_{act}$ 和原始特征 $F_{ori}$。
    - **注意力加权**：拼接因果特征 $H = \text{Concat}(F_{init}, F_{end}, F_{pot}, F_{act})$，计算注意力权重 $\alpha = \text{Softmax}(W_H H + b_H)$，加权原始特征 $F = \alpha \odot F_{ori}$。
    - **设计动机**：将视频特征解耦为驾驶状态图中的不同成分（状态、环境、动作、危险），使模型能显式推理哪些环境因素真正驱动了驾驶行为，减少虚假相关。

3. **视觉-语言 Transformer (VLT)**：

    - **核心思路**：使用 MLP 将因果特征对齐到文本嵌入空间，通过 Transformer 解码器生成描述和推理文本。
    - **稀疏注意力掩码**：使用稀疏约束 $L_{sparse} = \lambda \sum_{i,j} |V_{(i,j)}|$ 限制因果特征和词嵌入的关系矩阵稀疏性，防止模型产生幻觉。
    - **设计动机**：直接生成可能导致视觉内容和文本不匹配（幻觉问题），稀疏约束确保只有真正相关的视觉特征影响特定词汇的生成。

### 损失函数 / 训练策略

- **信号损失**：$L_{signal} = \frac{1}{2N} \sum_i |y_i - \hat{y}_i| + (y_i - \hat{y}_i)^2$（L1 + L2 混合）
- **文本生成损失**：交叉熵 + KL 散度：$L_{caption} = -\frac{1}{N}\sum_i \sum_c y_{i,c}\log(\hat{y}_{i,c}) + \beta \cdot D_{KL}(P \| Q)$
- **总损失**：$L_{total} = L_{signal} + L_{caption}$
- 批大小 16，训练 40 epoch，BDD-X 学习率 0.0003，CoVLA 学习率 0.0001
- 所有视频帧预处理为 224×224，采样 32 帧
- 单块 A100 80GB GPU 训练

## 实验关键数据

### 主实验

**BDD-X 数据集（Table 2）：**

| 方法 | 叙述 B4 | 叙述 CIDEr | 推理 B4 | 推理 CIDEr | 参数量 | FPS |
|------|--------|-----------|--------|-----------|--------|-----|
| DriveGPT4 | 30.0 | 214.0 | 9.4 | 102.7 | 7.85B | — |
| RAG-Driver | 34.3 | 260.8 | 11.1 | 109.1 | 7.08B | — |
| Baseline(ADAPT) | 33.4 | 241.6 | 8.2 | 75.5 | 620.2M | 365 |
| **MCAM** | **35.7** | **252.0** | **9.1** | **94.1** | **885.3M** | **336** |

**CoVLA 数据集（Table 3）：**

| 方法 | B1 | B4 | CIDEr | METEOR | ROUGE | 参数量 |
|------|-----|-----|-------|--------|-------|--------|
| Baseline | 81.9 | 74.2 | 236.9 | 48.8 | 80.7 | 620.2M |
| **MCAM** | **82.6** | **75.3** | **275.4** | **50.2** | **81.9** | **885.3M** |

### 消融实验

**模块组合消融（Table 4，BDD-X）：**

| 配置 | 叙述 B4 | 叙述 CIDEr | 推理 B4 | 推理 CIDEr | 参数量 |
|------|--------|-----------|--------|-----------|--------|
| VidSwin + VLT | 32.9 | 235.8 | 8.0 | 74.3 | 582.2M |
| 3DResNet + VLT | 32.8 | 221.8 | 6.4 | 55.3 | 494.0M |
| MFE + VLT | 33.4 | 218.1 | 7.1 | 79.8 | 845.8M |
| VidSwin + CAM + VLT | 34.2 | 242.7 | 8.2 | 80.3 | 588.3M |
| **MFE + CAM + VLT (MCAM)** | **35.3** | **251.6** | **9.0** | **92.9** | **885.3M** |

### 关键发现

- CAM 对推理任务的提升（CIDEr 75.5 → 92.9）远大于叙述任务（241.6 → 252.0），证明因果建模对解释性推理更关键
- MFE（全局+局部）比单一编码器效果更好，但 CAM 可以在仅 VidSwin 的基础上也带来显著提升（CIDEr 235.8 → 242.7）
- MCAM 参数量仅 885.3M，远小于 LLM-based 方法（7B+），但性能具有竞争力
- 定性分析显示 MCAM 能正确识别远处红灯而非近处刹车灯作为停车原因（减少虚假相关）

## 亮点与洞察

- **DSDAG 的引入为驾驶行为理解提供了结构化因果框架**——将模糊的"理解"任务转化为显式的状态转换推理问题
- **轻量级设计**：不依赖 LLM，用约 900M 参数达到接近 7B 模型的性能，FPS 336 适合实际部署
- **因果解耦策略**：将视频特征投影为不同的因果成分（状态/环境/动作/危险），使因果推理可解释
- 首次将因果分析结构引入自车级别视频理解任务

## 局限与展望

- 数据集标注存在噪声（CoVLA 由 LLaMA-7B 生成标签，存在错误标注），影响模型上限
- DSDAG 的状态定义较为简化，未建模复杂的多车交互因果链
- VLT 使用的 Transformer 解码器在生成长文本时容易"跑偏"
- 未与最新的 MLLM（如 GPT-4V、Qwen-VL）进行对比
- 因果分析模块的注意力权重可解释性有待进一步验证

## 相关工作与启发

- Pearl 的结构因果模型在计算机视觉中的又一次成功应用
- 与 CMCIR 等事件级因果方法不同，MCAM 聚焦于自车行为这一更实用的维度
- 因果特征解耦思路可启发其他场景理解任务——如行人意图预测、交通事故分析
- 稀疏注意力掩码防幻觉的策略值得在多模态生成任务中推广

## 评分

- **新颖性**: ⭐⭐⭐⭐ DSDAG 和因果分析模块的结合是有创意的设计，但因果推理的实际效果主要体现在注意力加权上
- **实验充分度**: ⭐⭐⭐ 两个数据集、消融覆盖模块组合，但未与最新 MLLM 对比，缺少因果分析可解释性验证
- **写作质量**: ⭐⭐⭐ 因果建模部分公式定义较多但与实际实现的对应关系不够清晰
- **价值**: ⭐⭐⭐⭐ 轻量化方案实用，因果分析思路对驾驶行为理解有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[CVPR 2025\] Online Video Understanding: OVBench and VideoChat-Online](../../CVPR2025/autonomous_driving/online_video_understanding_ovbench_and_videochat-online.md)
- [\[CVPR 2025\] MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction](../../CVPR2025/autonomous_driving/maskgwm_a_generalizable_driving_world_model_with_video_mask_reconstruction.md)
- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)

<!-- RELATED:END -->
