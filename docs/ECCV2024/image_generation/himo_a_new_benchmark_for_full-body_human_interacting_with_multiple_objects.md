---
title: >-
  [论文解读] HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects
description: >-
  [ECCV 2024][图像生成][人物-物体交互] 提出 HIMO，首个大规模全身人体与多物体交互的 4D MoCap 数据集（3.3K序列，4.08M帧），并附带详细的文本描述和时间段分割标注，提出双分支条件扩散模型及自回归管线，生成协调的多物体交互动作序列。
tags:
  - ECCV 2024
  - 图像生成
  - 人物-物体交互
  - 多物体
  - 动作生成
  - 扩散模型
  - 时序分割
---

# HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects

**会议**: ECCV 2024  
**arXiv**: [2407.12371](https://arxiv.org/abs/2407.12371)  
**代码**: [项目主页](https://lvxintao.github.io/himo) (有)  
**领域**: 图像生成  
**关键词**: 人物-物体交互, 多物体, 动作生成, 扩散模型, 时序分割

## 一句话总结

提出 HIMO，首个大规模全身人体与多物体交互的 4D MoCap 数据集（3.3K序列，4.08M帧），并附带详细的文本描述和时间段分割标注，提出双分支条件扩散模型及自回归管线，生成协调的多物体交互动作序列。

## 研究背景与动机

生成人类与物体的交互（HOI）对数字人、游戏、AR/VR、机器人和具身AI至关重要。然而，现有的数据集和模型几乎全部限于**单物体交互**，忽视了日常生活中普遍存在的多物体操作场景。

**现有数据集的局限**：

| 数据集 | 手部 | 多物体 | 文本 | 时序分割 | 序列数 | 帧数 |
|--------|------|--------|------|----------|--------|------|
| GRAB | ✓ | ✗ | ✗ | ✗ | 1,334 | - |
| BEHAVE | ✗ | ✗ | ✗ | ✗ | 321 | 15K |
| InterCap | ✗ | ✗ | ✗ | ✗ | 223 | 67K |
| ARCTIC | ✓ | ✗ | ✗ | ✗ | 339 | 2.1M |
| CHAIRS | ✓ | ✗ | ✗ | ✗ | 1,390 | - |
| **HIMO** | ✓ | ✓ | ✓ | ✓ | 3,376 | 4.08M |

核心缺口：

**无多物体交互数据**：现有全部数据集仅涉及单一物体

**缺乏文本标注**：多数HOI数据集无文本描述，限制了文本驱动生成研究

**缺乏时序分割**：长交互序列未被分解为原子操作，无法实现细粒度时间线控制

HIMO 旨在同时解决以上三个缺口。

## 方法详解

### 整体框架

工作包含两大部分：
1. **HIMO 数据集构建**：混合捕捉系统 + SMPL-X拟合 + 文本/时序标注
2. **HOI 合成模型**：双分支扩散模型 + 互注意力模块 + 自回归组合

提出两个新任务：
- **HIMO-Gen**：给定整段文本描述，生成完整的多物体交互序列
- **HIMO-SegGen**：给定分段文本描述（时间线控制），生成可组合的交互序列

### 关键设计

#### 1. 混合动捕系统

**身体捕捉**：
- Optitrack 光学动捕系统，20台 PrimeX-22 红外摄像机
- 每位受试者穿戴含41个反光标记点的动捕服
- 2K分辨率，120fps

**手指捕捉**：
- Noitom PNS 惯性手套，精确记录精细手指姿势
- 惯性方案避免了手-物体和物体-物体遮挡
- 频繁重新校准确保捕捉质量

**时空对齐**：
- 空间：手背上的定位板提供手腕旋转信息，整合全身和手指运动
- 时间：Tentacle Sync 设备为 Optitrack 和 PNS 提供时间码同步

**物体设计**：
- 53种常见家居物品，选自 ContactDB 和 Sketchfab
- 调整大小后3D打印，确保mesh geometry与实物一致
- 每个物体粘贴 3-6 个 12.5mm 反光球标记用于刚体追踪
- 关键原则：标记粘贴位置需减少对操作的副作用

**采集规模**：
- 34位受试者
- 2或3个物体的预定义组合
- 每个交互任务执行3次（不同操作方式）
- 总计：3,376序列，9.44小时，4.08M帧
- 其中2,486序列涉及2个物体，890序列涉及3个物体

#### 2. SMPL-X 参数拟合

采用 SMPL-X 表达式人体模型，参数包括：
- 全局朝向 $g \in \mathbb{R}^3$
- 身体姿态 $\theta_b \in \mathbb{R}^{21 \times 3}$
- 手指姿态 $\theta_h \in \mathbb{R}^{30 \times 3}$
- 根平移 $t \in \mathbb{R}^3$
- 体形参数 $\beta \in \mathbb{R}^{10}$

优化目标：
$$\mathbb{E} = \alpha \mathbb{E}_j + \lambda \mathbb{E}_s + \gamma \mathbb{E}_r$$

- $\mathbb{E}_j$：关节位置匹配损失（MoCap关节与SMPL-X回归关节的L2距离）
- $\mathbb{E}_s$：平滑项（相邻帧关节位置差的L2范数）
- $\mathbb{E}_r$：正则化项（$\|\theta_b\|_2^2 + \|\theta_h\|_2^2$）
- 权重：$\alpha=1, \lambda=0.1, \gamma=0.01$

#### 3. 文本与时序标注

**文本标注**规范：
- 描述整个交互过程，强调操作顺序（"First...then...finally..."）
- 描述交互模式（如 "picking up", "rotating", "pouring"）
- 标注参与的身体部位（左手/右手）

**时序分割**规范：
- 将长序列拆分为细粒度的原子操作
- 例："用茶壶倒茶到茶杯" → "举起茶壶" + "倒茶入杯" + "放下茶杯"
- HOI序列和对应文本同时分割为等份数段
- 平均每序列2.60个分段

#### 4. HIMO-Gen：双分支扩散模型

**数据表示**：
- 人体运动 $\boldsymbol{H} \in \mathbb{R}^{T \times D_h}$：全局关节位置 $P^i \in \mathbb{R}^{52 \times 3}$，6D连续旋转 $Q^i \in \mathbb{R}^{52 \times 6}$，平移 $t^i \in \mathbb{R}^3$
- 多物体运动 $\boldsymbol{O} \in \mathbb{R}^{T \times D_o}$：相对旋转 $\mathbf{R}^i \in \mathbb{R}^6$，全局平移 $\mathbf{T}^i \in \mathbb{R}^3$
- 物体几何 $\boldsymbol{G} \in \mathbb{R}^{1024 \times 3}$：使用 BPS 表示，采样1024点计算与基点集的方向向量

**模型架构**：
- **人体分支**：输入 = 噪声人体运动 $\boldsymbol{H}_t$ + 初始状态掩码 $\boldsymbol{H}_0$ + 文本嵌入 + 时间步嵌入
- **物体分支**：输入 = 噪声物体运动 + 初始物体状态掩码 + 物体几何嵌入 + 文本嵌入
- **文本编码**：CLIP 编码器 + 线性层
- **去噪网络**：$\ell_{enc} = 8$ 层 Transformer，4头注意力

**互注意力模块**（mutual interaction module）：
每个 Transformer 块包含两个注意力层和一个前馈层：
1. 自注意力层：分别嵌入人体/物体特征
2. 互注意力层：人体分支的 K,V 来自物体特征，物体分支的 K,V 来自人体特征

$$\boldsymbol{H}^{(i+1)} = FF(\text{softmax}(\frac{\mathbf{Q}_h \mathbf{K}_o^T}{\sqrt{C}} \mathbf{V}_o))$$
$$\boldsymbol{O}^{(i+1)} = FF(\text{softmax}(\frac{\mathbf{Q}_o \mathbf{K}_h^T}{\sqrt{C}} \mathbf{V}_h))$$

**物体成对距离损失**（$\mathcal{L}_{dis}$）：
- 显式约束交互物体间的相对距离模式
- 如切苹果时刀逐渐接近苹果表面，维持一段时间后离开
- 使用 L2 损失使生成结果与GT的物体间距保持一致

### 损失函数 / 训练策略

**总损失**：
$$\mathcal{L} = \lambda_{vel}\mathcal{L}_{vel} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_{pen}\mathcal{L}_{pen} + \lambda_{dis}\mathcal{L}_{dis}$$

- $\mathcal{L}_{vel}$：关节速度损失
- $\mathcal{L}_{pos}$：关节位置损失
- $\mathcal{L}_{pen}$：人-物体穿透损失
- $\mathcal{L}_{dis}$：物体成对距离损失
- 权重：$\lambda_{vel} = \lambda_{pos} = \lambda_{pen} = 1$，$\lambda_{dis} = 0.1$

**训练配置**：
- Adam优化器，学习率0.0001，权重衰减0.99
- HIMO-Gen最大运动长度300帧，HIMO-SegGen最大100帧
- 最大文本长度分别40和15
- 训练时间：HIMO-Gen约9小时，HIMO-SegGen约12小时（单A100 GPU，batch size 128）
- 数据拆分：训练/测试/验证 = 0.8/0.15/0.05

#### 5. HIMO-SegGen：自回归生成管线

- 用时序分割后的短序列-文本对训练 HIMO-Gen 模型
- 修改条件输入为**过去几帧**（而非仅第一帧）
- 推理时：取前一段生成结果的最后几帧作为下一段的条件，迭代生成

实验发现条件帧数为10帧效果最佳。

## 实验关键数据

### 主实验

**2-objects 分区定量比较**：

| 方法 | R-Precision(Top3) ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|------|---------------------|-------|-----------|-------------|-------------|
| Real | 0.7988 | 0.0176 | 3.5659 | 11.3973 | - |
| IMoS | 0.5013 | 7.5890 | 8.7402 | 7.0033 | 0.9920 |
| MDM | 0.6052 | 6.8457 | 8.0187 | **11.3891** | 1.2880 |
| priorMDM | 0.5891 | 7.8517 | 7.2509 | 12.5799 | 1.5911 |
| **HIMO-Gen** | 0.6369 | **1.4811** | 3.6491 | 11.6603 | 1.7863 |
| **HIMO-SegGen** | **0.6404** | 4.2004 | **3.6077** | 15.7317 | **2.0495** |

**3-objects 分区定量比较**：

| 方法 | R-Precision(Top3) ↑ | FID ↓ | MM-Dist ↓ |
|------|---------------------|-------|-----------|
| IMoS | 0.4662 | 4.9902 | 7.7702 |
| MDM | 0.5025 | **4.5713** | 6.3144 |
| priorMDM | 0.5137 | 4.8210 | 5.8900 |
| **HIMO-Gen** | **0.5350** | 4.7712 | **5.0866** |

### 消融实验

互注意力模块和物体距离损失的消融在补充材料中。基于论文的讨论：

| 设计选择 | 效果 |
|---------|------|
| 无互注意力 | 人体和物体动作时空不对齐 |
| 无物体距离损失 | 物体间出现不合理接触 |
| 条件帧数=10 | 最佳过渡平滑度（vs 5/15/20） |
| 随机打乱物体顺序 | 减轻物体输入顺序的影响 |

### 关键发现

1. **HIMO-Gen 在 FID 上大幅领先**：2-objects上FID仅1.48，比次优的MDM(6.85)降低78%
2. **HIMO-SegGen 提升文本对齐**：R-Precision和MM-Dist优于HIMO-Gen，归功于更精练的条件描述
3. **互注意力模块关键**：保证人体和物体运动的协调性
4. **泛化能力**：模型可应用于未见过的物体几何，也可合成训练集中不存在的新HOI组合
5. **3-objects更具挑战性**：所有方法在3-objects上性能均下降，FID差距缩小

## 亮点与洞察

1. **填补重要数据空白**：首个多物体+全身+文本+时序分割的4D HOI数据集，每个维度都有独特贡献
2. **时序分割标注的创新价值**：允许将复杂操作分解为原子动作，支持灵活的时间线控制和动作组合
3. **混合动捕系统设计精良**：光学（身体精度）+ 惯性（手指抗遮挡）的组合方案平衡了精度和可行性
4. **物体成对距离损失**：巧妙地用简单的L2约束建模物体间的距离模式，有效减少穿透
5. **自回归组合管线简洁高效**：仅需修改条件输入为过去几帧即可实现平滑过渡

## 局限与展望

1. **数据集规模仍有限**：3.3K序列相比图像/视频领域仍然较小
2. **物体种类受限**：53种物品，主要集中在餐饮和厨房场景
3. **2-3个物体**：实际场景可能涉及更多物体的同时操作
4. **SMPL-X拟合精度**：MoCap数据到参数化模型的拟合不可避免存在误差
5. **自回归误差累积**：长序列的分段生成可能导致漂移
6. **缺乏手-物体接触模型**：虽然有穿透损失，但缺乏精细的接触力建模

## 相关工作与启发

- **GRAB / ARCTIC**：全身HOI数据集的先驱，但限于单物体
- **MDM (Motion Diffusion Model)**：本文基础框架的来源
- **priorMDM**：人-人交互生成方法，双分支设计的灵感来源
- **GMD**：关键帧条件的可控运动生成
- **启发**：时序分割+自回归的范式可推广到更复杂的多智能体交互场景

## 评分

- **创新性**：★★★★★ — 数据集填补重要空白，多物体+时序分割是全新设定
- **实用性**：★★★★☆ — 对游戏/VR/机器人有直接应用价值，但数据规模仍有待扩展
- **实验充分度**：★★★★☆ — 多基线比较+泛化实验+消融，评估协议参照成熟框架
- **写作质量**：★★★★☆ — 结构清晰，系统介绍完整，但方法部分稍显冗长

<!-- RELATED:START -->

## 相关论文

- [Visual Persona: Foundation Model for Full-Body Human Customization](../../CVPR2025/image_generation/visual_persona_foundation_model_for_full-body_human_customization.md)
- [MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [Towards Reliable Advertising Image Generation Using Human Feedback](towards_reliable_advertising_image_generation_using_human_fe.md)

<!-- RELATED:END -->
