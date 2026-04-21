---
title: >-
  [论文解读] Generating Multimodal Driving Scenes via Next-Scene Prediction
description: >-
  [CVPR 2025][自动驾驶][多模态场景生成] 提出 UMGen，一个统一的多模态驾驶场景生成框架，将自车动作、地图、交通参与者和图像四种模态进行 token 化，通过帧间时序自回归（TAR）和帧内有序自回归（OAR）两阶段策略逐场景生成，同时引入动作感知地图对齐（AMA）模块保持自车运动与地图的一致性，可自主生成长达 60 秒的连贯驾驶序列。
tags:
  - CVPR 2025
  - 自动驾驶
  - 多模态场景生成
  - 自回归模型
  - 驾驶仿真
  - 下一场景预测
  - 时序一致性
---

# Generating Multimodal Driving Scenes via Next-Scene Prediction

**会议**: CVPR 2025  
**arXiv**: [2503.14945](https://arxiv.org/abs/2503.14945)  
**代码**: https://yanhaowu.github.io/UMGen (项目页)  
**领域**: 自动驾驶 / 场景生成  
**关键词**: 多模态场景生成, 自回归模型, 驾驶仿真, 下一场景预测, 时序一致性

## 一句话总结

提出 UMGen，一个统一的多模态驾驶场景生成框架，将自车动作、地图、交通参与者和图像四种模态进行 token 化，通过帧间时序自回归（TAR）和帧内有序自回归（OAR）两阶段策略逐场景生成，同时引入动作感知地图对齐（AMA）模块保持自车运动与地图的一致性，可自主生成长达 60 秒的连贯驾驶序列。

## 研究背景与动机

**领域现状**：生成式模型在自动驾驶中用于创建多样化的驾驶场景，尤其是数据集中罕见或未覆盖的场景，也可用于构建闭环仿真系统来安全测试自动驾驶系统。

**现有痛点**：现有方法通常只能生成有限的模态组合。GUMP 和 TrafficGen 只生成自车动作和 agent 轨迹，但没有地图演化（地图是静态的），限制了真实感；DriveDreamer 和 GAIA-1 能生成图像但无法预测交通参与者运动，缺乏对 agent 行为的细粒度控制。没有方法能同时生成并保持所有关键模态的一致性。

**核心矛盾**：多模态场景生成面临两个挑战：(1) 将所有模态的 token 展开为一个长序列进行 vanilla AR 建模会导致计算量爆炸；(2) 在同一帧内不同模态之间缺乏模态间一致性约束，容易出现冲突。

**本文目标** 如何在统一框架下同时生成四种关键模态（自车动作、地图、agent、图像），保证多模态一致性和时序连贯性，同时控制计算开销？

**切入角度**：将场景生成问题分解为帧间预测和帧内预测两个子问题，分别用 TAR 和 OAR 处理，避免在超长 token 序列上做全局注意力。同时利用自车动作对地图做仿射变换来保持两者一致性。

**核心 idea**：用"时序并行 + 模态内顺序"的两级自回归策略取代 vanilla 的全 token 自回归，大幅降低多模态场景生成的计算复杂度，同时通过 AMA 模块显式约束自车-地图一致性。

## 方法详解

### 整体框架

UMGen 的 pipeline 如下：给定过去 T 帧的多模态场景序列，(1) 将各模态（自车动作、地图、agent、图像）通过离散化或 VQ-GAN 转为 token；(2) Ego-action 预测模块预测下一帧的自车动作；(3) AMA 模块根据预测的自车动作对地图特征做仿射变换对齐；(4) TAR 模块通过因果注意力并行聚合每个 token 位置的时序信息；(5) OAR 模块以 GPT 风格按固定模态顺序（ego→map→agent→image）自回归生成帧内 token；(6) 解码 token 得到下一帧场景。

### 关键设计

1. **帧间时序自回归（TAR）**:

    - 功能：捕获每个 token 位置在时间维度上的演化规律
    - 核心思路：对经过 AMA 对齐的 T 帧 token 嵌入 $\bar{\mathbf{e}}_{1:T}$，沿时间维度对每个 token 位置 $i$ 做因果自注意力 $\bar{\mathbf{e}}_{T+1}^i = \text{CSA}(\bar{\mathbf{e}}_1^i, ..., \bar{\mathbf{e}}_T^i)$，然后在帧内用双向自注意力做初始的跨模态信息交换。各 token 位置之间并行处理，计算复杂度为 $O(T \times N)$ 而非 vanilla AR 的 $O((T \times N)^2)$。
    - 设计动机：每个 token 位置在相邻帧中通常对应相同的物理位置/物体，因此按位置做时序注意力能高效捕获运动和变化趋势，同时避免在 $T \times N$ 长度序列上做全局注意力的巨大开销。

2. **帧内有序自回归（OAR）**:

    - 功能：在单帧内按模态因果顺序生成 token，保证模态间一致性
    - 核心思路：以 TAR 的输出 $\mathbf{h}_{T+1}$ 作为时序先验，结合已生成的前序 token $\mathbf{o}_{T+1}^{1:i-1}$ 做因果自注意力预测当前 token $\mathbf{o}_{T+1}^i$。生成顺序为 ego→map→agent→image，反映了物理因果链：自车动作改变可观测地图，影响周围 agent 行为，最终反映在相机图像中。
    - 设计动机：模态间存在因果依赖关系（如自车转弯→地图旋转→agent 让行→图像变化），按此顺序自回归生成能显式建模这种依赖，防止跨模态冲突。

3. **动作感知地图对齐（AMA）**:

    - 功能：根据自车动作对地图特征做几何变换，为下一帧地图预测提供强先验
    - 核心思路：将地图 token 嵌入重排为 H×W 空间特征，利用预测的自车动作（$\theta, dx, dy$）构造仿射变换矩阵生成采样网格，通过双线性插值对地图做旋转和平移变换，然后与原始地图特征相加得到变换后的地图嵌入。
    - 设计动机：相邻帧的地图在自车坐标系下的变化主要由自车运动引起，通过显式的仿射变换可以低成本地传递地图信息，大幅降低地图生成难度。

### 损失函数 / 训练策略

总损失为 OAR 和 TAR 两个交叉熵损失之和：$\mathcal{L}_{total} = CE(\mathbf{p}^{OAR}_{T+1}, \mathbf{z}_{T+1}) + CE(\mathbf{p}^{TAR}_{T+1}, \mathbf{z}_{T+1})$。训练时每次随机取 21 帧序列，在 32 张 RTX 4090 上训练 300 个 epoch（约 2 天）。推理时使用 Top-k 采样策略生成 token。

## 实验关键数据

### 主实验

在 nuPlan 和 WOMD 数据集上的初始场景生成 MMD 指标对比：

| 方法 | Position↓ | Heading↓ | Speed↓ | 数据集 |
|------|-----------|----------|--------|--------|
| TrafficGen | 0.83 | 0.82 | 0.90 | WOMD |
| SceneDM | 0.39 | 0.37 | 0.62 | WOMD |
| **UMGen** | **0.17** | **0.22** | **0.35** | **WOMD** |
| TrafficGen | 3.29 | 1.04 | 4.34 | nuPlan |
| **UMGen** | **0.42** | **0.35** | **0.73** | **nuPlan** |

UMGen 在所有 MMD 指标上大幅超越对比方法，说明其生成的场景分布更接近真实数据。

### 消融实验

| 配置 | Agent MMD↓ | Agent CR↓ | 说明 |
|------|-----------|-----------|------|
| Full model | 0.31 | 0.018 | 完整模型 |
| w/o TAR | 0.45 | 0.032 | 去掉时序建模，场景连贯性下降 |
| w/o OAR | 0.38 | 0.041 | 去掉帧内顺序生成，碰撞率大幅上升 |
| w/o AMA | 0.34 | 0.025 | 地图对齐缺失，空间一致性降低 |

### 关键发现

- OAR 对减少模态间冲突至关重要（去掉后碰撞率翻倍），验证了帧内模态顺序建模的必要性。
- TAR 相比 vanilla AR 在推理效率上有显著优势：per-token 推理时间降低约 60%，峰值显存降低约 40%。
- UMGen 能生成长达 60 秒的连贯多模态驾驶序列，展现了良好的时序稳定性。
- 通过控制 ego-action 输入，可以生成用户指定的场景（如转弯、直行），为仿真测试提供灵活性。

## 亮点与洞察

- **两级 AR 的分治策略**：TAR 并行处理时序、OAR 顺序处理模态，将 $O((TN)^2)$ 的注意力复杂度降为 $O(T \times N)$，这是一个通用的长序列多模态生成加速方案，可迁移到视频-音频联合生成等任务。
- **模态顺序的物理因果性**：ego→map→agent→image 的生成顺序不是随意的，而是反映了真实世界的因果链，这种 domain knowledge 的注入使生成质量大幅提升。
- **AMA 模块的简洁有效**：仅用一个仿射变换就实现了自车-地图一致性维护，成本极低但效果显著。

## 局限与展望

- 图像生成依赖 VQ-GAN，分辨率和质量有限，未来可考虑使用扩散模型替代。
- 当前 agent 数量通过 padding 固定，对 agent 动态出现/消失的建模还不够灵活。
- 仅展示了前视图生成，是否能扩展到多视图一致的场景生成值得探索。
- 没有在闭环仿真中验证生成场景对自动驾驶策略学习的实际效用。

## 相关工作与启发

- **vs GAIA-1**: GAIA-1 也做 AR 驾驶视频生成但只有图像和 ego-action 两个模态，UMGen 扩展到四模态且效率更高。
- **vs GUMP**: GUMP 做 agent 轨迹生成但地图是静态的，UMGen 引入地图模态和 AMA 模块使场景更真实。
- **vs DriveDreamer**: DriveDreamer 用两个独立网络分别生成地图和视频缺乏模态一致性，UMGen 的 OAR 模块在统一框架下保证一致性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 四模态统一生成 + TAR/OAR 双层 AR 是新颖的方案设计
- 实验充分度: ⭐⭐⭐ 定性结果丰富但定量评估主要限于 MMD，缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为自动驾驶仿真提供了有前景的多模态生成方案

<!-- RELATED:START -->

## 相关论文

- [Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)
- [SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)

<!-- RELATED:END -->
