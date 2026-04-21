---
title: >-
  [论文解读] DexVLG: Dexterous Vision-Language-Grasp Model at Scale
description: >-
  [ICCV 2025][机器人][灵巧抓取] 提出DexVLG——首个大规模视觉-语言-灵巧抓取模型，构建了包含174K物体、1.7亿抓取姿态的DexGraspNet 3.0数据集（带部件级语义标注），结合VLM和Flow Matching姿态预测头，在仿真中实现76%+零样本执行成功率，并在真实世界中完成语义对齐的灵巧抓取。
tags:
  - ICCV 2025
  - 机器人
  - 灵巧抓取
  - 视觉语言模型
  - Flow Matching
  - 语义部件抓取
  - 大规模数据集
---

# DexVLG: Dexterous Vision-Language-Grasp Model at Scale

**会议**: ICCV 2025  
**arXiv**: [2507.02747](https://arxiv.org/abs/2507.02747)  
**代码**: 无  
**领域**: 机器人 / 灵巧抓取  
**关键词**: 灵巧抓取, 视觉语言模型, Flow Matching, 语义部件抓取, 大规模数据集

## 一句话总结
提出DexVLG——首个大规模视觉-语言-灵巧抓取模型，构建了包含174K物体、1.7亿抓取姿态的DexGraspNet 3.0数据集（带部件级语义标注），结合VLM和Flow Matching姿态预测头，在仿真中实现76%+零样本执行成功率，并在真实世界中完成语义对齐的灵巧抓取。

## 研究背景与动机
- **领域现状**：Vision-Language-Action (VLA) 模型在机器人领域发展迅速，但由于数据采集困难，进展主要局限于简单夹爪末端执行器
- **现有痛点**：类人灵巧手的功能性抓取（即根据语义指令抓取物体特定部位）研究严重缺乏——既缺少大规模训练数据，也缺少有效的模型架构
- **核心矛盾**：灵巧手的自由度高(>20 DoF)、抓取姿态空间极大，传统方法难以覆盖大量物体和部件的语义对齐抓取
- **切入角度**：Data-driven方案——先大规模生成高质量、语义对齐的灵巧抓取数据，再训练大模型来学习语言引导的抓取姿态预测
- **核心idea**：构建超大规模的部件级灵巧抓取数据集(DexGraspNet 3.0)，结合VLM理解自然语言指令和RGBD输入，通过Flow Matching预测灵巧手的抓取姿态

## 方法详解

### 整体框架
DexVLG由两大部分构成：(1) DexGraspNet 3.0数据集的构建管线——从Objaverse筛选物体、SAMesh语义分割、GPT-4o标注部件名称和物体尺寸、基于能量优化的抓取姿态合成；(2) DexVLG模型——VLM编码器处理RGBD图像和语言指令，Flow Matching姿态预测头生成灵巧手的抓取姿态参数。

### 关键设计
1. **DexGraspNet 3.0数据集构建**:

    - 功能：从Objaverse的800K+物体中筛选并标注174K个物体，为每个物体的每个语义部件合成抓取姿态，总计1.7亿个抓取姿态
    - 核心流程：
        - GPT-4o六视图查询过滤低质量/不合适物体
        - Trimesh提取网格 → ManifoldPlus水密化 → CoACD凸分解
        - SAMesh语义分割 → GPT-4o Set-of-Marks标注部件名称
        - GPT-4o估计物体合理尺寸并缩放(对角线20-50cm)
        - 基于部件几何的初始手势对齐 → 梯度优化合成抓取姿态
    - 设计动机：数据规模是泛化的基础。部件级标注使模型能理解"抓手柄""抓瓶盖"等语义指令

2. **LP-based Differentiable Force Closure (LP-DFC)**:

    - 功能：改进原始DFC能量优化目标，合成更自然的抓取姿态
    - 核心思路：在每个时间步，先固定手姿态，用线性规划求解最优接触力大小：
      $\min_{\mathbf{f}} \|G(\mathbf{f} \odot c)\|_2$, s.t. $\max_i(\mathbf{f})_i = 1$, $(\mathbf{f})_i \geq 0$
      然后根据净力矩P和接触力f重新缩放DFC能量
    - 设计动机：原始DFC假设等量接触力，导致拇指对指时手指倾斜等伪影；LP-DFC考虑不同接触力大小，生成更符合物体几何的自然姿态

3. **部件对齐的手势初始化**:

    - 功能：根据物体部件的几何特征，对灵巧手的初始姿态进行语义对齐
    - 核心思路：将物体部件分为4类——盖状(lid-like)、盘状(disk-like)、L形(L-shaped)、杆状(shaft-like)，对每类设计特定的手掌位置和朝向对齐策略。Wrap抓取(7个接触点，5指尖+手掌)和Pinch抓取(4个接触点，拇指+食指+中指+手掌)两种模式
    - 设计动机：梯度优化对初始姿态非常敏感(DexGraspNet论文已指出)，部件对齐的初始化注入了强先验，使优化后的姿态更自然且语义可区分

4. **VLM + Flow Matching姿态预测**:

    - 功能：接收RGBD图像和语言指令，预测灵巧手的抓取姿态参数
    - 核心思路：VLM编码视觉和语言输入，Flow Matching作为去噪模块生成抓取姿态（而非DDPM/DDIM扩散）
    - 设计动机：Flow Matching范式比传统扩散更易学习姿态生成，实验验证其显著优于DDPM和DDIM

### 损失函数 / 训练策略
- 抓取合成的正则化能量：$E_{reg} = \omega_{limit}E_{limit} + \omega_{pen}E_{pen} + \omega_{spen}E_{spen} + \omega_{dir}E_{dir}$
    - 关节角限制能量、手-物穿透能量、手自穿透能量、接触方向对齐能量（余弦相似度）
- 每个物体每个部件生成2×5000个初始化姿态（Wrap + Pinch模式各5000）
- 仿真渗透检查进一步过滤低质量姿态
- 桌面场景相机设置：距桌面中心80cm、俯视45°均匀采样

## 实验关键数据

### 主实验 (仿真Benchmark)

| 数据 | LVIS-Seen | | Unseen | | SamPart3D | |
|------|-----------|---|--------|---|-----------|---|
| | Suc↑ | PGA↑ | Suc↑ | PGA↑ | Suc↑ | PGA↑ |
| Wrap grasp | 87.7 | 62.1 | 79.1 | 36.6 | 76.3 | 52.0 |
| Pinch grasp | 71.8 | 20.2 | 54.8 | 15.2 | 50.6 | 21.3 |

DexVLG在未见物体(Unseen)上仍保持79.1%的Wrap抓取成功率和76.3%的零样本泛化率。

### 消融实验

**去噪范式消融**:

| 方法 | LVIS-Seen | | Unseen | | SamPart3D | |
|------|-----------|---|--------|---|-----------|---|
| | Suc↑ | PGA↑ | Suc↑ | PGA↑ | Suc↑ | PGA↑ |
| DDPM | 51.9 | 7.8 | 34.1 | 10.9 | 40.7 | 5.5 |
| DDIM | 57.7 | 12.5 | 39.6 | 10.4 | 35.2 | 8.5 |
| **Flow Matching** | **75.3** | **39.1** | **54.0** | **18.3** | **53.4** | **27.0** |

**数据集质量评估**:

| 方法 | 规模↑ | 穿透↓(mm) | 自穿透↓(mm) | Q1↑ |
|------|-------|----------|------------|-----|
| DexGraspNet | 1.32M | 13.5 | 0.93 | 0.114 |
| Multi-GraspLLM | 120k | 7.1 | - | 0.091 |
| **Ours-Wrap** | **103M** | **1.75** | **0.19** | 0.085 |
| **Ours-Pinch** | **67M** | **1.42** | **0.22** | 0.067 |

### 关键发现
- Flow Matching显著优于DDPM/DDIM：成功率分别提升23.4%/17.6%（LVIS-Seen），PGA提升31.3%/26.6%
- Wrap抓取在所有数据集上一致优于Pinch抓取（成功率高15.9-24.3%），表明全手包裹更稳定
- DexGraspNet 3.0在规模上比前辈大两个数量级(170M vs 1.32M)，且穿透度大幅降低(1.75mm vs 13.5mm)
- 部件对齐初始化比随机初始化产生的抓取姿态更自然、语义更可区分
- Q1稳定性指标略低于纯power grasp数据集，因为部件对齐本身就不以稳定性为唯一目标

## 亮点与洞察
- 数据驱动路线的极致体现：174K物体、1.7亿姿态、部件级标注——数据规模本身就是核心贡献
- GPT-4o在数据管线中的多处应用（物体筛选、部件标注、尺寸估计）展示了LLM在机器人数据构建中的潜力
- LP-DFC对DFC的改进虽小但关键——从等力假设到变力建模，显著改善拇指对指场景的姿态质量
- 4类部件几何分类(lid/disk/L-shaped/shaft)虽不严格穷尽，但覆盖了大多数功能性部件形状
- 从数据集构建到模型训练到仿真评估的完整管线具有很强的工程价值

## 局限与展望
- 缓存中仅含Appendix，主体方法和实验细节缺失——VLM编码器架构和训练策略的细节未知
- 真实世界实验描述不足，仅提到"成功的部件对齐抓取"但无量化结果
- Pinch抓取成功率较低(50-72%)，对精细操作场景的适用性受限
- 数据合成完全在仿真中进行，sim-to-real gap可能影响实际部署
- 仅使用LEAP Hand这一类灵巧手，对其他手型(如Shadow Hand, Allegro)的泛化未验证
- 物体尺寸裁剪到20-50cm限制了对极小/极大物体的适用性

## 相关工作与启发
- DexGraspNet系列(1.0→2.0→3.0)展示了灵巧抓取数据集的迭代演进路径
- SoFar和OmniSpatial的部件-几何关系先验被有效利用
- Flow Matching作为动作生成范式的优越性在灵巧操作中得到验证，可能成为VLA系统的标准选择
- 部件对齐初始化的思路可推广到其他需要语义引导姿态合成的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个大规模语言引导灵巧抓取模型+数据集，但方法论上主要是已有技术的组合
- 实验充分度: ⭐⭐⭐ 缓存仅含Appendix消融，主实验无法全面评估；仿真benchmark设计合理但真实世界验证不足
- 写作质量: ⭐⭐⭐ Appendix细节丰富，但主体内容缺失影响完整性评估
- 价值: ⭐⭐⭐⭐⭐ 数据集规模和质量本身具有极高的社区价值，将推动灵巧抓取领域的后续研究

<!-- RELATED:START -->

## 相关论文

- [NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)
- [EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)
- [CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](../../NeurIPS2025/robotics/cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)

<!-- RELATED:END -->
