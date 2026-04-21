---
title: >-
  [论文解读] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
description: >-
  [NeurIPS 2025][3D视觉][3D场景合成] 提出SceneWeaver，首个用于3D场景合成的反思型智能体框架，通过标准化可扩展的工具接口统一多种场景生成范式，并利用reason-act-reflect闭环迭代优化，在物理合理性、视觉真实感和语义对齐上全面超越现有方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D场景合成
  - LLM智能体
  - 自反思
  - 工具调用
  - 具身AI
---

# SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent

**会议**: NeurIPS 2025  
**arXiv**: [2509.20414](https://arxiv.org/abs/2509.20414)  
**代码**: [项目主页](https://scene-weaver.github.io/)  
**领域**: 3D视觉  
**关键词**: 3D场景合成, LLM智能体, 自反思, 工具调用, 具身AI

## 一句话总结
提出SceneWeaver，首个用于3D场景合成的反思型智能体框架，通过标准化可扩展的工具接口统一多种场景生成范式，并利用reason-act-reflect闭环迭代优化，在物理合理性、视觉真实感和语义对齐上全面超越现有方法。

## 研究背景与动机
室内3D场景合成是计算机视觉和图形学的长期研究课题。随着具身AI的兴起，对场景的需求从"视觉逼真"扩展到了"物理可交互"和"指令可控"——场景需要在仿真器中可交互，并能精确响应任务特定的用户指令。

现有方法各有所长但无一能全面满足需求：（1）**数据驱动生成模型**（如ATISS、DiffuScene、PhyScene）从3D-FRONT等数据集学习，能产生较真实的布局，但受限于预定义的场景类型和数据集质量/多样性；（2）**基于规则的系统**（如ProcThor、Infinigen）通过手工约束确保物理合理性，但逻辑僵硬、难扩展；（3）**LLM方法**（如LayoutGPT、Holodeck、I-Design）具有开放词汇理解和语义灵活性，但空间推理能力不足，常产生物理不合理的结果。

更重要的是，近期一些work开始将场景生成分解为多阶段pipeline（先LLM生成布局，再2D生成模型精调等），但这些pipeline是**"静态的"**——规划和执行由固定prompt和硬编码逻辑驱动，缺乏根据生成反馈进行自适应决策的能力，也难以灵活集成新的合成工具。

SceneWeaver的核心idea是构建一个**反思型（reflective）智能体框架**：将现有场景合成方法抽象为模块化工具，通过自反思规划器动态选择工具、根据反馈迭代精调场景、在物理约束下执行修改。这种闭环设计使得智能体能够识别语义不一致、调用针对性工具、在连续迭代中更新环境。

## 方法详解

### 整体框架
SceneWeaver由两个核心组件构成：（1）**标准化可扩展工具接口**——将多种场景生成方法按生成粒度分类为模块化工具；（2）**自反思规划器**——给定用户查询$q$和工具集$\mathcal{D}$，在$T$次迭代中通过reason-act-reflect范式逐步优化场景状态$s_t$。每步接收上一步的反思$v_{t-1}$，选择工具$d_t$执行精调，物理感知执行器处理碰撞和约束。

### 关键设计
1. **标准化场景合成工具接口**:

    - **场景初始化器（Scene Initializer）**：生成全场景布局作为起点。包含三类：数据驱动生成模型（可扩展但限于预定义类型）、real-to-sim方法（高质量但多样性有限）、LLM方法（开放词汇但空间推理弱）
    - **微场景实现器（Microscene Implementer）**：添加细节小物体（如桌上的键盘、显示器）。包含LLM方法（语义多样但空间误差大）和2D引导方法（视觉真实但受限于2D-3D提升）
    - **细节精调器（Detail Refiner）**：修正错误，包括基于规则的约束放置、6D姿态精调、LLM驱动的语义不合理物体移除
    - 每个工具通过**标准化工具卡**描述，包含功能描述、适用场景、使用约束、输入参数等字段，确保新工具的即插即用
    - 设计动机：没有单一方法能同时满足真实感、物理合理性和可控性，因此通过标准接口组合多方法的互补优势

2. **基于反馈的自反思规划（Feedback-driven Self-reflective Planning）**:

    - **反思生成**：调用MLLM（如GPT-4）对当前场景$s_t$生成反思$v_t$，包括物理指标（碰撞分数、边界违反）和感知指标（视觉真实感、功能性、布局连贯性、用户查询对齐、完整性），并附自然语言改进建议
    - **自反思规划**：规划器根据当前上下文和记忆$m_t = (d_{t-l:t-1}, s_{t-l:t-1}, v_{t-l:t-1})$，总结当前问题→排序候选工具→选择最合适的工具$d_t$→生成工具特定指令。工具置信度根据历史表现动态调整——失败降低置信度，反复失败触发重新规划
    - 设计动机：闭环设计使智能体能从失败中学习和适应。记忆长度设为1以避免幻觉

3. **物理感知执行（Physics-aware Execution）**:

    - 基于Infinigen和Blender构建执行器
    - 每次迭代：加载前序场景 → 应用工具$d_t$的布局修改 → 从混合资产库（Objaverse、3D-Future、Infinigen等）检索3D mesh → 调整物体放置满足关系约束 → 运行固定步物理优化解决碰撞和边界违反
    - 设计动机：大多数工具输出的是3D包围盒布局，需要物理执行器将其转化为可交互的3D场景，同时确保物理合理性

### 损失函数 / 训练策略
SceneWeaver不涉及端到端训练。核心是利用预训练MLLM（GPT-4）的推理能力，通过精心设计的prompt和工具卡引导场景生成。最大迭代次数设为10，记忆长度设为1。

## 实验关键数据

### 主实验（常见房间类型）

| 方法 | 卧室#Obj | 卧室#OB↓ | 卧室#CN↓ | 卧室Real.↑ | 卧室Comp.↑ | 客厅#Obj | 客厅Real.↑ | 客厅Comp.↑ |
|------|---------|---------|---------|----------|----------|---------|----------|----------|
| PhyScene | 3.3 | 0.1 | 0.3 | 5.7 | 4.0 | 8.0 | 5.2 | 3.3 |
| LayoutGPT | 5.4 | 1.0 | 1.3 | 7.5 | 4.2 | 8.4 | 6.4 | 3.6 |
| I-Design | 9.6 | 0.0 | 0.0 | 8.6 | 6.1 | 9.7 | 8.4 | 5.9 |
| Holodeck | 32.2 | 0.0 | 38.5 | 8.6 | 6.2 | 23.0 | 8.9 | 8.1 |
| **SceneWeaver** | **14.0** | **0.0** | **0.0** | **9.2** | **9.4** | **17.3** | **9.1** | **8.7** |

### 消融实验

| 配置 | #Obj | #OB↓ | #CN↓ | Real.↑ | Func.↑ | Lay.↑ | Comp.↑ |
|------|------|------|------|--------|--------|-------|--------|
| 无反思模块 | 25.0 | 0.0 | 0.0 | 8.0 | 8.3 | 6.3 | 6.3 |
| 无物理优化 | 27.3 | 0.7 | 2.0 | 8.3 | 9.3 | 6.7 | 7.7 |
| 多步规划(非迭代) | 29.3 | 0.0 | 0.0 | 8.3 | 7.7 | 7.0 | 7.3 |
| **完整SceneWeaver** | **34.7** | **0.0** | **0.0** | **9.0** | **9.3** | **7.3** | **7.7** |

### 关键发现
- 开放词汇场景生成的平均指标上，SceneWeaver全面领先：#Obj=36.5, Real.=8.8, Func.=9.4, Comp.=8.0，且物理约束零违反
- Holodeck虽然物体数量最多（#Obj=32.2），但碰撞严重（#CN=38.5），说明单纯堆物体不等于高质量场景
- 移除反思模块导致语义质量显著下降，自反思迭代优于一次性多步规划
- 工具组合的消融证明初始化器+实现器+修改器的完整组合效果最佳，三类工具互补
- 数据驱动方法在物理指标上优于LayoutGPT，说明纯LLM难以确保物理合理性
- 人类偏好评估中SceneWeaver获得87-95%的偏好率，且在多样性维度上同样领先

## 亮点与洞察
- 将LLM Agent范式引入3D场景合成是一个非常自然且有效的设计。reason-act-reflect闭环使得系统能从生成反馈中学习，比固定pipeline灵活鲁棒
- 标准化工具卡的设计使得新工具可以即插即用，极大提升了框架的可扩展性。随着未来更多场景生成方法的出现，SceneWeaver可以持续受益
- 人类评估结果显示SceneWeaver在与所有基线的成对比较中获得85%以上的偏好，验证了自动评估与人类判断的一致性
- 基于OpenManus平台实现的ReAct-style推理链，使得整个系统的规划过程透明可解释
- 在8种开放词汇房间类型上的测试展示了强大的泛化能力，平均物体数36.5远超基线同时保持零物理违规
- 记忆长度设为1的设计虽然简单但有效避免了LLM幻觉问题，体现了实用主义的工程智慧

## 局限与展望
- 依赖GPT-4进行反思和工具选择，推理成本较高，单场景生成需要多轮API调用，速度和成本是瓶颈
- 资产检索质量受限于现有3D模型库（Objaverse/3D-Future），某些罕见物体类别可能找不到合适的mesh
- 物理优化是基于规则的简化版本，无法处理复杂的物理交互场景（如铰链、弹簧等）
- 场景规模主要限于单房间，多房间或建筑级场景的生成尚未探索
- 人类评估仅20人参与，评估规模较小
- 工具卡的描述质量直接影响规划器的决策，需要精心设计
- 反思评估依赖俯视图渲染而非3D感知评估，在复杂垂直结构场景上可能不够准确

## 相关工作与启发
- **vs Holodeck**: Holodeck物体数量多但碰撞严重（#CN=38.5 vs 0.0），说明缺少物理优化和反思机制的局限性
- **vs I-Design**: I-Design物理指标好(零碰撞)但物体少、完整性低；SceneWeaver在保证零碰撞的同时物体多3-4倍
- **vs LayoutGPT**: 纯LLM生成物理合理性差（#OB=1.0, #CN=1.3），验证了单独LLM不足以确保空间推理正确性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将反思型LLM Agent框架用于3D场景合成，标准化工具接口设计具有前瞻性
- 实验充分度: ⭐⭐⭐⭐⭐ 常见房型和开放词汇两个setting，8种房间类型，多维度指标，消融和人类评估全面
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但方法部分稍长，可适当精简
- 价值: ⭐⭐⭐⭐⭐ 对具身AI场景构建有直接价值，统一框架思路有望推广到更多3D生成任务

<!-- RELATED:START -->

## 相关论文

- [One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [Matrix3D: Large Photogrammetry Model All-in-One](../../CVPR2025/3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)
- [3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)

<!-- RELATED:END -->
