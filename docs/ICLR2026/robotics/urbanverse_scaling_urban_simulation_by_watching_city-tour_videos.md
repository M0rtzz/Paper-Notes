---
title: >-
  [论文解读] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos
description: >-
  [ICLR 2026][机器人][城市仿真] UrbanVerse是一个数据驱动的real-to-sim系统，将众包城市旅拍视频转化为物理感知的交互式仿真场景，包含10万+标注3D资产和自动场景构建流水线，在IsaacSim中生成160个高质量场景，训练的PPO导航策略在真实世界零样本转移中成功率达89.7%，完成337m长距离任务仅需2次人工干预。
tags:
  - ICLR 2026
  - 机器人
  - 城市仿真
  - real-to-sim
  - 具身智能
  - 3D资产库
  - 导航策略
---

# UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos

**会议**: ICLR 2026  
**arXiv**: [2510.15018](https://arxiv.org/abs/2510.15018)  
**代码**: [urbanverseproject.github.io](https://urbanverseproject.github.io/)  
**领域**: 机器人学 / 仿真  
**关键词**: 城市仿真, real-to-sim, 具身智能, 3D资产库, 导航策略

## 一句话总结
UrbanVerse是一个数据驱动的real-to-sim系统，将众包城市旅拍视频转化为物理感知的交互式仿真场景，包含10万+标注3D资产和自动场景构建流水线，在IsaacSim中生成160个高质量场景，训练的PPO导航策略在真实世界零样本转移中成功率达89.7%，完成337m长距离任务仅需2次人工干预。

## 研究背景与动机
城市空间中的具身AI智体（如配送机器人、四足机器人）正在快速发展。训练这类智体需要大量多样化的高保真城市环境，但现有仿真方案存在根本性矛盾：

- **手工制作场景**: 如CARLA仅有15个场景，不可扩展，人力成本高
- **程序化生成场景**: 如MetaUrban/UrbanSim使用硬编码规则，产生的场景偏离真实世界分布（如随机停放的滑板车不符合真实parking模式）
- **被动真实数据**: 如城市旅拍视频有丰富的多样性，但缺乏动作标签和交互性
- **3D重建方法**: 如3DGS可从视频重建场景，但产出的是静态纹理网格，无语义和物理属性

核心矛盾：规模与真实感的矛盾。简单增加数量（程序化生成）不能带来泛化——如果场景不忠实反映真实世界分布，数量再多也无效。

UrbanVerse的核心idea：**从真实城市旅拍视频中提取场景语义和布局，用高质量3D资产实例化为物理可交互的仿真场景**——"digital cousin"范式。这结合了真实数据的多样性和仿真的交互性。

## 方法详解

### 整体框架
UrbanVerse由两大支柱组成：
1. **UrbanVerse-100K**: 标注了物理属性的10万+城市3D资产库
2. **UrbanVerse-Gen**: 从视频到仿真的自动流水线

输入：YouTube城市旅拍视频 → 输出：IsaacSim中的物理可交互仿真场景

### 关键设计

1. **UrbanVerse-100K资产库 (Section 3.1)**:
   从Objaverse的80万嘈杂3D资产中，通过三阶段半自动化流水线筛选和标注：
   
    - **资产筛选**: 10名标注员用Three.js查看器标注3周，过滤掉破损网格、缺纹理、纸片状、尺度异常等8类质量问题，保留15.8万可用资产
    - **城市本体构建**: 基于OpenStreetMap标签结构，扩展ADE20K、Cityscapes等数据集类别，建立3层667个叶类别的城市语义本体
    - **属性标注**: 用GPT-4.1对每个资产的缩略图和4个旋转快照进行33个属性标注（语义、可供性、物理属性如质量、摩擦力），API总成本$1,334
   
   最终资产库包含：102,530个GLB物体 + 288个PBR地面材质 + 306个HDRI天空图

2. **UrbanVerse-Gen场景构建流水线 (Section 3.2)**:
   定义统一的3D城市场景图 $\mathcal{V} = \langle\mathcal{O}, \mathcal{G}, \mathcal{S}\rangle$（物体/地面/天空节点）
   
   三阶段处理：
    - **场景蒸馏**: 从视频中提取语义和3D布局。使用MASt3R估计度量深度和相机位姿，YoloWorld+SAM2进行开放词汇物体解析，Mask2Former做路面/人行道分割。跨帧融合得到持久物体节点（含类别、质心、3D包围盒、朝向、外观裁切）
    - **资产匹配与多样化**: 为每个节点从UrbanVerse-100K中检索 $k_{cousin}$ 个匹配资产。三步：CLIP语义匹配 → 包围盒几何过滤(mBBD) → DINOv2外观排序。地面用像素MSE匹配PBR材质，天空用HSV直方图匹配HDRI
    - **仿真场景生成**: 在UrbanSim(IsaacSim)中实例化：拟合地面平面并贴材质 → 应用HDRI天空图 → 放置物体（质心对齐+碰撞检测+物理属性赋值）

3. **Digital Cousin多样化策略**:
   每个视频场景可生成 $k_{cousin}=5$ 个外观不同但布局一致的digital cousin变体，通过替换不同的匹配资产实现。这种布局内多样性与布局间多样性互补，增强策略泛化。

4. **PPO导航策略训练**:

    - Actor-Critic架构，连续动作空间
    - 观测：RGB图像(135×240) + 目标相对位置
    - 3层CNN编码器[16,32,64] + 3层MLP(128)
    - 奖励设计：到达奖励(+2000) + 碰撞惩罚(-200) + 位置跟踪(粗+细) + 速度奖励
    - 训练时每次加载16个场景，每100个episode换一批场景

### 损失函数 / 训练策略
PPO优化，学习率1e-4（自适应），γ=0.99，GAE τ=0.95，PPO clip ε=0.2，KL阈值0.01，熵系数0.002，1500 epoch，单张L40S GPU混合精度训练。

## 实验关键数据

### 主实验
**场景构建保真度** (KITTI-360, 45序列, 平均198.7m)：

| SfM | 场景解析器 | 类别(%) | 资产(%) | 距离(m) | 朝向(°) | 体积(m³) | mAP25 |
|-----|----------|---------|---------|---------|---------|---------|-------|
| MASt3R | YoWorldSAM2 | **93.1** | **75.1** | **1.4** | 19.8 | **0.8** | **28.2** |
| VGGT | YoWorldSAM2 | 91.5 | 70.6 | 2.1 | 20.1 | 1.3 | 9.4 |

**CraftBench泛化测试** (10个艺术家设计场景)：

| 方法 | SR(%) | CT | RC(%) |
|------|------|-----|-------|
| MBRA | 35.6 | 25.6 | 52.9 |
| S2E | 33.1 | 27.7 | 55.7 |
| PPO-UrbanSim | 9.1 | 31.5 | 19.4 |
| **PPO-UrbanVerse** | **41.9** | 35.5 | **62.4** |

**零样本Sim-to-Real** (16个真实城市场景)：

| 方法 | 轮式SR(%) | 四足SR(%) |
|------|----------|----------|
| NoMad | 33.3 | 37.5 |
| S2E | 47.9 | 58.6 |
| PPO-UrbanSim | 18.8 | 18.8 |
| **PPO-UrbanVerse** | **77.1** | **89.7** |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 1 layout → 32 layouts | SR: 低→41.9% | 场景数量scaling power law成立 |
| 1 cousin → 5 cousins | SR: 低→更高 | 布局内多样性也很重要 |
| UrbanVerse vs PG场景 | 人类评分3.58 vs 2.9/5 | 70%以上用户偏好UrbanVerse |
| 预训练+目标场景微调 | SR: 0%→80% | Real-to-sim-to-real闭环有效 |

### 关键发现
- **Scaling power law存在**: 场景数量和digital cousin数量与性能之间呈幂律关系，线性拟合R²高
- **真实分布至关重要**: 同等数量的程序化生成场景几乎无法提升泛化（PG曲线平坦）
- **PPO-UrbanVerse超越导航基础模型**: 简单PPO策略在UrbanVerse场景上训练，超越了NoMad、CityWalker等大规模预训练的视觉导航基础模型
- **零样本迁移极其strong**: 四足Go2在真实世界达到89.7%成功率，超越S2E +31.1%
- **337m长距离任务**: 仅2次人工干预完成公共街道上的长距离导航任务
- **人类评价**: UrbanVerse自动生成的场景获评3.58/5，而艺术家手工场景4.08/5，差距不大

## 亮点与洞察
- **完整的pipeline**: 从视频采集→资产库构建→场景生成→策略训练→真实世界部署，形成闭环
- **10万级资产库**: 解决了3D资产的质量和规模问题，这是独立的重要贡献
- **Scaling law的发现**: 在具身AI领域验证了data scaling law的存在，为"更多场景=更好策略"提供了定量证据
- **两类机器人验证**: 同一策略在轮式和四足上都有效，说明学到的是环境理解而非特定运动学
- **24国160场景**: 跨文化、跨地理的多样性是真实世界泛化的关键
- **Real-to-sim-to-real闭环**: 针对已知部署环境，拍一段视频→生成仿真→微调策略→部署，实用价值极高

## 局限与展望
- Digital cousin仍与真实场景有gap——资产替换不可能完美匹配原始物体
- 朝向误差(19.8°)仍较大，对精确导航可能有影响
- 仅使用PPO，未探索更先进的强化学习算法
- 仅评估了导航任务，操作(manipulation)任务未涉及
- 场景动态性不足——没有行人、车辆等动态障碍物的运动
- HDRI天空图提供的光照是静态的，无法模拟时间推移
- 依赖YouTube视频的CreativeCommons授权，数据来源受限

## 相关工作与启发
- **Digital Cousins (Dai et al., 2024)**: 室内场景的多变体生成，UrbanVerse将此扩展到大规模室外城市场景
- **MetaUrban / UrbanSim (Wu et al., 2025)**: UrbanVerse的仿真平台基础，本文解决了其程序化生成的局限性
- **ViNT / NoMad (Shah et al., 2023; Sridhar et al., 2024)**: 视觉导航基础模型，但基于被动数据训练，缺乏交互式学习
- **S2E (He et al., 2025)**: 在仿真中训练绕障策略并迁移到真实世界，但场景规模和多样性不如UrbanVerse
- **Data Scaling Laws (Lin et al., 2025)**: 模仿学习的数据扩展法则，UrbanVerse在RL+仿真场景中发现了类似规律
- 启发：众包视频是一种几乎无限的仿真素材来源，值得在其他领域（如室内、工厂）推广

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [Instruction-based Image Manipulation by Watching How Things Move](../../CVPR2025/robotics/instruction-based_image_manipulation_by_watching_how_things_move.md)
- [UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)

<!-- RELATED:END -->
