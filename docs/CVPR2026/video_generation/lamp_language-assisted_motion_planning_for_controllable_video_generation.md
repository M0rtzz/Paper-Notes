---
title: >-
  [论文解读] LAMP: Language-Assisted Motion Planning for Controllable Video Generation
description: >-
  [CVPR 2026][视频生成] 提出LAMP框架，将运动控制建模为语言到程序合成问题：设计电影摄影启发的运动DSL，训练LLM将自然语言描述转化为结构化运动程序，再确定性映射为3D对象和相机轨迹来条件化视频生成，首次实现从自然语言同时生成对象和相机运动。
tags:
  - CVPR 2026
  - 视频生成
  - 运动控制
  - LLM规划
  - 领域特定语言
  - 电影摄影
---

# LAMP: Language-Assisted Motion Planning for Controllable Video Generation

**会议**: CVPR 2026  
**arXiv**: [2512.03619](https://arxiv.org/abs/2512.03619)  
**代码**: [项目主页](https://cyberiada.github.io/LAMP/)  
**领域**: 视频生成  
**关键词**: 视频生成, 运动控制, LLM规划, 领域特定语言, 电影摄影

## 一句话总结

提出LAMP框架，将运动控制建模为语言到程序合成问题：设计电影摄影启发的运动DSL，训练LLM将自然语言描述转化为结构化运动程序，再确定性映射为3D对象和相机轨迹来条件化视频生成，首次实现从自然语言同时生成对象和相机运动。

## 研究背景与动机

视频生成已取得显著进展，但运动控制——指定对象动态和相机轨迹——仍受限于有限的用户交互方式。现有方法大多依赖文本、从视频提取的标注、或简单的2D绘制界面，难以表达复杂的电影化运动。

核心痛点：对象运动和相机轨迹本质上是耦合的（相机通常相对运动对象定义），同时指定两者需要高级空间规划和心理想象能力。例如，编排追逐场景需要同时协调跑者路径和追踪相机。

现有方法的局限：
- 直接从文本回归3D坐标困难：语言到运动的映射是多模态的、结构受限的
- 先前方法仅关注布局生成或相机轨迹合成，不统一对象和相机运动
- 缺乏可迭代编辑的界面

LAMP的核心idea：利用LLM的程序合成能力，将运动控制转化为**语言条件化的程序合成**问题——LLM生成符号化运动程序而非连续坐标，然后确定性地映射为3D轨迹。

## 方法详解

### 整体框架

1. 自然语言描述输入 → 2. LLM运动规划器生成DSL运动程序 → 3. 确定性转换为3D对象和相机轨迹 → 4. 渲染为控制视频 → 5. 条件化预训练视频扩散模型生成最终视频

### 关键设计

1. **电影摄影启发的运动DSL**:
    - 功能：提供可解释、可组合的运动表示
    - 核心思路：基于CameraBench分类体系，定义四种基础运动原语：
        - **Free-form**：无约束6-DoF运动
        - **Orbit track**：相机围绕目标物体环绕
        - **Tail track**：相机跟随物体运动
        - **Rotation track**：相机原地旋转跟踪
    - 每个原语由修饰符参数化：平移控制（lat, vert, depth）、旋转控制（yaw, pitch, roll）、时间/风格提示（speed_fast, ease_in, jitter_low），以key-value对形式表达
    - 运动序列由最多4个运动标签组成，跨4个时间段实现变化
    - 设计动机：符号化表示带来数据效率、可解释性和组合性——复杂运动从简单原语组合涌现

2. **程序化训练语料构建**:
    - 功能：提供大规模文本-运动配对数据训练LLM
    - 核心思路：构建400K文本-运动样本（100K自由运动 + 100K物体相对运动 × 原始+LLM改写）。过程：采样并组合运动原语 → DSL程序 → 确定性转换为3D轨迹 → 模板文本描述 → LLM改写增加语言多样性
    - 覆盖27个粗类（3运动类型×3方向）和343个细类，旋转角度密集采样[-180°, 180°]
    - 设计动机：自动化生成避免了大规模人工标注，且数据分布可控——常见电影运动频率高，复杂组合运动频率低

3. **LLM运动规划与层次化分解**:
    - 功能：从自然语言生成对象和相机的符号化运动程序
    - 核心思路：将联合概率分解为 $p(s_{cam}, s_{obj} | t) = p(s_{obj} | t_{obj}) \cdot p(s_{cam} | s_{obj}, t_{cam})$，先生成对象运动再条件化生成相机运动。微调VLM（Qwen2.5-VL）在400K语料上学习DSL程序生成
    - 设计动机：分解反映了电影制作的层次结构——对象运动定义场景动态，相机根据对象调整以保持构图。支持迭代精修（如"把相机放低一点"）

### 损失函数 / 训练策略

LLM规划器采用标准自回归训练。推理时DSL程序确定性映射为3D轨迹，渲染为控制视频（2D bounding box投影 + 全局方块投影），与文本/首帧一起输入VACE视频生成器。

## 实验关键数据

### 主实验 — DataDoP相机轨迹评估

| 模型 | 修正F1 | CLaTr Score | CLaTr FID |
|------|--------|-------------|-----------|
| CCD (预训练) | — | 5.29 | 357.8 |
| ET (预训练) | — | 2.46 | 609.9 |
| GenDoP (DataDoP训练) | 0.400 | 36.18 | 22.7 |
| **LAMP (预训练)** | **0.763** | **36.29** | 66.9 |
| **LAMP (ft DataDoP)** | **0.776** | **36.52** | 67.2 |

### ET数据集评估

LAMP在简单(pure)和复杂(mixed)分割上一致超越所有基线的F1分数

### 消融实验

| 配置 | 说明 |
|------|------|
| 无DSL (直接回归) | 性能显著下降 |
| 无微调 (零样本) | 基础能力可用但精度低 |
| 完整LAMP | 最优性能 |

### 关键发现

- LAMP在未经DataDoP训练的情况下，修正F1就超过了在该数据集上训练的GenDoP（0.763 vs 0.400），证明DSL表示的强泛化性
- 符号化程序比直接坐标回归更高效，需要的数据更少
- 迭代精修能力（如"稍微缩小""相机再低一点"）是独特优势——用户无需昂贵的视频合成即可调整运动

## 亮点与洞察

- 将运动生成重新定义为程序合成而非坐标回归，是架构层面的思路转变
- DSL设计与电影摄影惯例对齐，使生成的运动具有专业电影感
- 解耦设计允许在视频合成前迭代修改运动，大幅降低创作成本
- 首次统一了对象和相机运动的自然语言控制

## 局限与展望

- 目前仅支持单对象场景（3D bounding box），多对象交互场景未处理
- 运动序列限制为4个时间段，更长的复杂运动需要扩展
- 最终视频质量仍受限于底层视频生成模型（VACE）
- CLaTr FID相对GenDoP较高（66.9 vs 22.7），说明轨迹的真实感仍有提升空间

## 相关工作与启发

- **vs GenDoP**: GenDoP用GPT生成详细导演描述来指导自回归相机路径生成，但LLM角色仅限于辅助描述；LAMP让LLM直接输出可执行的运动程序
- **vs ET**: ET使用LLM生成的电影描述来指导扩散模型预测轨迹；LAMP跳过扩散直接用DSL确定性映射
- **vs CameraCtrl/EPiC**: 这些方法仅控制相机，假设对象静态；LAMP统一控制两者

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将运动控制重新定义为程序合成，DSL设计与电影学结合的创新性强
- 实验充分度: ⭐⭐⭐⭐ 在多个基准上定量对比，包含消融和用户研究
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法层次分明，图示直观
- 价值: ⭐⭐⭐⭐⭐ 对视频生成的可控性研究有重大推动，框架设计优雅且可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_fewstep_controllable_video_generation.md)
- [\[CVPR 2026\] Training-free Motion Factorization for Compositional Video Generation](training-free_motion_factorization_for_compositional_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

</div>

<!-- RELATED:END -->
