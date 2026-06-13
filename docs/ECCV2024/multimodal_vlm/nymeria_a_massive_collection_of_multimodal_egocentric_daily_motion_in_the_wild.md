---
title: >-
  [论文解读] Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild
description: >-
  [ECCV 2024][多模态VLM][自我中心视角] Nymeria 是目前世界最大的野外人体运动数据集（300 小时、264 名参与者），首次提供同步定位的多设备多模态自我中心数据（Project Aria 眼镜+腕带+动捕服），并配套 310.5K 句层次化运动语言描述。 领域现状： 随着智能眼镜和可穿戴设备的兴起（M…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "自我中心视角"
  - "人体运动数据集"
  - "多模态"
  - "运动-语言"
  - "日常活动"
---

# Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild

**会议**: ECCV 2024  
**arXiv**: [2406.09905](https://arxiv.org/abs/2406.09905)  
**代码**: [https://www.projectaria.com/datasets/nymeria](https://www.projectaria.com/datasets/nymeria)  
**领域**: 多模态VLM  
**关键词**: 自我中心视角, 人体运动数据集, 多模态, 运动-语言, 日常活动

## 一句话总结

Nymeria 是目前世界最大的野外人体运动数据集（300 小时、264 名参与者），首次提供同步定位的多设备多模态自我中心数据（Project Aria 眼镜+腕带+动捕服），并配套 310.5K 句层次化运动语言描述。

## 研究背景与动机

**领域现状**: 随着智能眼镜和可穿戴设备的兴起（Meta、Google Glass、Apple Vision Pro 等），以用户自身运动为上下文的人体中心上下文化 AI 成为重要研究方向。然而，当前研究受限于数据：实际数据规模/多样性/模态有限，仿真数据缺乏真实性。

**现有痛点**:
   - **野外地面真值难获取**: 光学标记动捕受视线遮挡限制，仅能在有限空间捕获；惯性动捕存在累积漂移
   - **多设备对齐困难**: 不同设备的时间和空间对齐需要高精度同步，现有方案依赖视觉/音频线索，精度和可靠性有限，且采集过程中需频繁中断以校正时钟漂移
   - **语言标注稀疏**: 现有运动-语言数据集描述简短、缺乏场景上下文，规模远不及 LLM 训练所需
   - 现有数据集要么规模小、要么仅室内、要么缺乏参数化人体表示、要么没有第一人称视角

**核心矛盾**: 自我中心视角下全身运动的"观测不足"（只能看到手和部分身体）使得运动估计本质上是欠定问题，需要大量真实数据来学习运动先验，但现有数据集在规模、多样性和模态丰富度上远不够。

**本文目标** 构建一个多方面远超现有数据集的人体运动数据集，涵盖：野外全身动捕 + 多模态自我中心数据 + 精确同步定位 + 大规模层次化语言描述。

**切入角度**: 利用 XSens 惯性动捕服 + Project Aria 智能眼镜 + 自制 miniAria 腕带的多设备组合，配合自研亚毫秒同步硬件和全局优化对齐算法。

**核心 idea**: 通过软硬件综合创新，以前所未有的规模采集同步定位的多模态自我中心日常运动数据，并建立从精细运动叙述到活动摘要的多层次语言标注。

## 方法详解

### 整体框架

Nymeria 数据集构建流程: 硬件设计与同步 → 数据采集（20 场景、264 人、50 地点）→ 数据处理（动捕重定向 + SLAM 定位 + 全局对齐）→ 层次化语言标注 → 基准评测

### 关键设计

1. **多设备同步与定位系统**:

   **功能**: 将 XSens 动捕服（17 个惯性传感器）、Project Aria 眼镜（RGB + 灰度 + 眼动追踪相机 + IMU 等）、两个 miniAria 腕带，通过统一时间信号实现亚毫秒级同步，并通过 SLAM + 优化定位到统一 3D 坐标系。

   **核心思路**: 
    - 自研同步设备提供统一时间戳给所有设备，XSens 与 Aria 对齐精度 ≤1 个运动帧（4.2ms）
    - Project Aria MPS 提供毫米级 6DoF 定位（VIO + SLAM + 视觉-惯性 bundle adjustment）
    - 通过 HandEye 标定将 XSens 漂移轨迹与 MPS 精确轨迹对齐：假设头部到设备的刚性变换 $T_{HD}$ 恒定，将轨迹切分为 4.2ms 短段，优化局部速度匹配

   **设计动机**: 多设备数据的价值取决于对齐精度。商用设备缺乏统一同步协议，现有方案的视觉/音频线索不够可靠。定制同步硬件 + 优化算法是唯一能在长时间野外采集中保持高精度的方案。

2. **全身运动重定向 (Motion Retargetting)**:

   **功能**: 将 XSens 骨骼运动重定向到具有 159 个关节的参数化人体模型。

   **核心思路**: 利用 XSens 输出的 79 个解剖标志点全局位置，求解逆运动学优化问题：

    $\arg\min_{\phi, \theta_{0\cdots T-1}, \mathbf{v}^{0\cdots K-1}} \sum_{t=0}^{N-1} \sum_{i=0}^{K-1} \|T^i(\phi, \theta_t)\mathbf{v}^i - \mathbf{p}_t^i\|^2$

   其中 $\phi$ 为身份参数（全局缩放 + 骨骼长度），$\theta$ 为姿态参数（6DoF 全局 + 52 个欧拉角），$\mathbf{v}^i$ 为标志点局部偏移。使用 Levenberg-Marquardt 算法求解，并加入参数限制和身体碰撞惩罚。

   **设计动机**: XSens 的简化骨骼模型存在自穿透和不自然姿态等问题。通过解剖学启发的人体模型 + 碰撞约束优化，可以获得更真实的运动表示。

3. **层次化运动-语言描述 (Hierarchical Motion-Language Description)**:

   **功能**: 为运动数据提供三个粒度的自然语言标注。

   **核心思路**: 25 名标注员通过同步播放**自我中心视频 + 第三人称视频 + 3D 运动渲染**三个视角，按从精细到粗略三个层次标注：
    - **运动叙述** (Motion Narration): ≤5 秒片段，详细描述全身姿态、手臂/腿部动作、注意力方向
    - **原子动作** (Atomic Action): ≤5 秒片段，用动词简洁描述动作
    - **活动摘要** (Activity Summarization): 30 秒片段，一句话概括主要活动

   **设计动机**: 
    - 三视角同步播放让标注员对运动有全面理解（第一人称看手部交互，第三人称/3D 渲染看全身）
    - 层次化标注支持不同研究需求：细粒度运动生成、动作识别、活动理解
    - 上下文描述（包含物体、环境信息）而非孤立动作标签，更贴近 LLM 时代的需求

### 数据规格与统计

- **规模**: 300 小时日常活动，1200 个序列，平均 15 分钟/序列
- **参与者**: 264 人（48.5% 女性、51.4% 男性），多元种族
- **场景**: 20 个场景类型（室内 18 种 + 户外 5 种），50 个地点（47 个住宅 + 3 个校园区域）
- **传感器数据**: 260M 身体姿态、201.2M 图像、11.7B IMU 样本、10.8M 注视点
- **轨迹**: 头部 399Km + 双腕 1053Km
- **语言**: 310.5K 句、8.64M 词、6545 词汇量

## 实验关键数据

### 主实验 — 与现有数据集对比

| 数据集 | 时长(h) | 姿态帧(M) | 平均序列长(min) | 参与者 | 语言句数(K) | 词汇量 | 头戴 | 腕带 | 户外 | 注视 |
|-------|--------|----------|--------------|-------|-----------|-------|------|------|------|------|
| AMASS | 42 | 0.9 | 0.22 | 346 | - | - | ✗ | ✗ | ✗ | ✗ |
| HPS | 4.5 | 0.5 | 8.2 | 7 | - | - | ✓ | ✗ | ✓ | ✗ |
| EgoBody | 2 | 0.4 | 1 | 36 | - | - | ✓ | ✗ | ✗ | ✓ |
| HumanML3D | 28.6 | 2.9 | 0.12 | - | 45.0 | 5371 | ✗ | ✗ | ✗ | ✗ |
| EgoExo4D | 88.8 | 9.6 | 2.6 | 740 | 432 | 4405 | ✓ | ✗ | ✓ | ✗ |
| **Nymeria** | **300** | **260** | **15** | **264** | **310.5** | **6545** | **✓** | **✓** | **✓** | **✓** |

### 基准评测 — 运动追踪/生成/语言

| 任务 | 方法 | 数据 | MPJPE(cm)↓ | Hand PE(cm)↓ | MPJVE(cm/s)↓ | FID↓ |
|------|------|------|-----------|-------------|-------------|------|
| 3点追踪 | AvatarPoser | AMASS | 4.20 | 2.34 | 28.23 | - |
| 3点追踪 | AvatarPoser | Nymeria(real) | 7.97 | 6.25 | 16.71 | - |
| 3点追踪 | AvatarPoser | Nymeria(synth) | 7.31 | 3.47 | 16.63 | - |
| 3点生成 | BoDiffusion | AMASS | 3.63 | - | - | - |
| 3点生成 | BoDiffusion | Nymeria | 7.98 | - | - | 2.32 |
| 1点生成 | EgoEgo | Nymeria | 13.22 | - | - | 5.14 |
| VQ-VAE | 2PQ-16384CB | Nymeria | 3.449cm(mm) | - | - | - |

### 关键发现

- **Nymeria 的运动更难**: AvatarPoser 在 AMASS 上 MPJPE 4.20cm，在 Nymeria 上升至 7.97cm——因为包含爬楼梯、运动、不平坦地形等复杂动作
- **真实 vs 合成输入的差距小**: real (7.97) vs synthetic (7.31)，说明设备追踪质量高
- **3 点 > 1 点**: BoDiffusion (7.98) vs EgoEgo (13.22)，腕带数据提供重要约束
- **运动 Token 化可行**: VQ-VAE 在 Nymeria 上重建质量接近 AMASS，支持 LLM 式运动生成
- **运动到文本的初步探索**: MotionGPT 在 Nymeria 子集上 BLEU@1=42.22、CIDEr=37.27，因描述复杂度高于 HumanML3D，性能低于原论文——但表明数据足以训练有效模型
- **平均每句 27.8 词**，远超现有运动-语言数据集的描述长度

## 亮点与洞察

1. **工程壮举**: 264 人、50 地点、300 小时、多设备亚毫秒同步——数据集构建本身就是巨大的技术和运营成就
2. **模态完备性前所未有**: 参数化全身运动 + 自我中心RGB/灰度/眼动 + 腕带 + IMU + 注视 + 3D 场景点云 + 6DoF 定位——全部精确同步
3. **层次化语言标注设计精妙**: 三视角同步标注保证质量，从运动叙述到活动摘要的粒度层次支持多种研究方向
4. **20 个场景覆盖丰富日常**: 从做饭、健身、远足到派对布置、西蒙说游戏——不是空洞的走路/跑步
5. **全局对齐算法巧妙**: 将 HandEye 标定问题应用于 XSens 漂移轨迹与 SLAM 轨迹的对齐

## 局限与展望

1. **动捕服影响自然性**: 穿戴惯性动捕服和腕带会改变外观和限制某些动作范围
2. **XSens 精度受限于标定和体型测量**: 不准确的体型测量会导致自穿透和非自然姿态
3. **场景覆盖不完整**: 缺少公共场景（如公交、商店、医院等），15% 录制为户外活动比例偏低
4. **无手部精细运动**: 与 Motion-X 不同，Nymeria 未包含手指和面部表情
5. **语言标注成本高**: 25 名标注员仅完成了部分数据的精细标注（运动叙述仅 38.6h/300h）
6. **隐私限制**: 去标识化（面部模糊）限制了面部相关研究的使用

## 相关工作与启发

- **AMASS**: 开创性的大规模运动数据集，统一多个标记动捕数据为 SMPL，但缺乏场景上下文和自我中心视角——Nymeria 填补了这一空白
- **EgoExo4D**: 大规模自我中心+第三人称数据集，但缺少参数化全身运动和腕带数据——Nymeria 提供了完整的运动真值
- **Motion-X**: 大规模全身运动数据集（含面部/手指），但来自单目视频估计，精度低于惯性动捕
- **HumanML3D**: 最重要的运动-语言数据集，但规模和描述复杂度远不及 Nymeria
- **启发**: 数据集论文的核心价值在于覆盖空白维度 + 极致规模 + 详实的基线评测

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 多个"第一"：最大野外运动数据集、首个多设备同步自我中心数据集、最大运动-语言数据集
- **实验充分度**: ⭐⭐⭐⭐ 提供了追踪/生成/VQ-VAE/运动到文本四类基线，但每类仅 1-2 个方法
- **写作质量**: ⭐⭐⭐⭐ 数据集论文写作规范，统计详实，但篇幅很长
- **实用价值**: ⭐⭐⭐⭐⭐ 公开数据集 + 完整工具链，将显著推动自我中心运动理解领域的发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ECCV 2024\] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)
- [\[NeurIPS 2025\] Reading Recognition in the Wild](../../NeurIPS2025/multimodal_vlm/reading_recognition_in_the_wild.md)
- [\[ACL 2025\] Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts](../../ACL2025/multimodal_vlm/donate_or_create_comparing_data_collection.md)
- [\[ICCV 2025\] Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images](../../ICCV2025/multimodal_vlm/visual_chronicles_using_multimodal_llms_to_analyze_massive_collections_of_images.md)

</div>

<!-- RELATED:END -->
