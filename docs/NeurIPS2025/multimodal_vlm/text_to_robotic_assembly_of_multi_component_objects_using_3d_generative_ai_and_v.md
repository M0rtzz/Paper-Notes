---
title: >-
  [论文解读] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
description: >-
  [NeurIPS 2025][多模态VLM][VLM推理] 提出了一个端到端流水线，将自然语言输入通过3D生成式AI转化为网格模型，再利用VLM的零样本多模态推理自动分解为多组件3D模型（结构件+面板件），最终由机器人臂自动装配成物理对象，并支持用户通过对话反馈调整组件分配。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM推理"
  - "机器人装配"
  - "3D生成式AI"
  - "多组件物体"
  - "人机协作"
---

# Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.02162](https://arxiv.org/abs/2511.02162)  
**代码**: 暂无（依赖Autodesk内部平台）  
**领域**: 多模态VLM  
**关键词**: VLM推理, 机器人装配, 3D生成式AI, 多组件物体, 人机协作

## 一句话总结

提出了一个端到端流水线，将自然语言输入通过3D生成式AI转化为网格模型，再利用VLM的零样本多模态推理自动分解为多组件3D模型（结构件+面板件），最终由机器人臂自动装配成物理对象，并支持用户通过对话反馈调整组件分配。

## 研究背景与动机

3D生成式AI（如DreamFusion、Get3D、Latte3D）已能从文本生成各种3D几何体，但将其转化为物理对象面临两大挑战：

**缺乏组件级表示**：现有3D生成模型生成的是整体网格（monolithic mesh），缺少机器人装配所需的组件级分解——即不知道哪些部分是结构骨架、哪些是功能表面（如椅子的座面、灯的灯罩）。

**功能感知的组件分配**：将网格分解为预定义组件需要同时推理几何形状和物体功能。例如，凳子需要在座面上放置水平面板以形成平坦表面，而灯需要在灯罩框架上放置面板以散射光线。这种推理对传统基于规则的方法来说过于复杂。

**用户偏好的多样性**：面板分配可能因用户偏好不同而有多种合理方案，需要支持人在回路的交互式调整。

现有工作局限：
- 3D打印流水线（Sketch2Prototype, Style2Fab）只支持单材料增材制造
- 部件感知生成模型（PartGen, StructureNet）关注几何重建而非装配
- 3D分割方法未考虑机器人可达性和功能角色

## 方法详解

### 整体框架

系统分为五个阶段：(1) 文本输入 → (2) 3D网格生成（Autodesk Project Bernini）→ (3) 网格体素化为结构组件 → (4) VLM推理分配面板组件 → (5) UR20机器人臂装配。

### 关键设计

1. **网格离散化（Mesh Discretization）**：定义两类装配组件——结构组件（承重骨架，立方体）和面板组件（附着在结构上的平面）。将AI生成的网格按结构组件尺寸体素化，形成主承重框架。关键创新在于面板组件的放置取决于物体的功能和几何形状。

2. **VLM功能感知部件选择**：使用Google Gemini 2.5 Pro，输入三要素——物体描述（理解功能）、轴测图（理解几何）、组件类型（理解面板功能），零样本推理出需要面板的部件。例如对"椅子"输出 "Parts = seat, backrest"。系统提示约束VLM只选择满足功能的最小部件集。

3. **VLM几何感知标签映射**：第一步的VLM输出是抽象部件名，第二步需定位到3D模型的具体网格面。方法：合并共面面、为每个面分配唯一整数标签（排除朝内的垂直面和朝下水平面——机器人臂无法到达），对标签化网格的轴测渲染图再次调用VLM，将部件名映射到具体标签号。

4. **人在回路的对话式反馈**：用户可通过自然语言修改VLM的初始分配（如"我只想要座面有面板，靠背不要"），VLM接收用户反馈和标签化网格图像，输出更新后的标签集合。无需任何任务特定训练，完全依赖VLM的零样本推理能力。

5. **机器人装配执行**：多组件装配导出为坐标列表 $C = \{(x_i, y_i, z_i, r_{x_i}, r_{y_i}, r_{z_i})\}$ 和组件类型列表 $T = \{t_0, t_1\}$。按从底到顶排序并保持连通性，UR20机器人臂配备Robotiq夹爪执行pick-and-place序列。

### 损失函数 / 训练策略

- **无需训练**：完全基于VLM（Gemini 2.5 Pro）的零样本推理
- 系统提示精心设计以约束输出格式和最小化选择
- 排除不可达面（朝内/朝下）作为预处理，防止VLM生成无法执行的方案

## 实验关键数据

### 主实验（用户偏好评估）

| 方法 | 椅子(%) | 桌子(%) | 灯(%) | 架子(%) | 垃圾桶(%) | **平均(%)** |
|------|--------|--------|------|--------|----------|------------|
| **VLM（本文）** | 96.9 | 100.0 | 81.3 | 100.0 | 75.0 | **90.6** |
| Rule-based | 18.8 | 100.0 | 34.4 | 100.0 | 43.8 | 59.4 |
| Random | 0.0 | 0.0 | 0.0 | 6.3 | 6.3 | 2.5 |

32位参与者×5个物体×3种方法 = 480个判断。用户可多选。

### 统计显著性检验（McNemar配对检验）

| 对比 | $\chi^2$ | p值 | 结论 |
|------|---------|------|------|
| VLM vs. Rule-based | 38.11 | <0.001 | VLM显著优于规则 |
| VLM vs. Random | 137.11 | <0.001 | VLM显著优于随机 |
| Rule-based vs. Random | 88.17 | <0.001 | 规则优于随机 |

所有比较经Bonferroni校正后仍显著（$p^* < 0.017$）。

### 关键发现

- 规则方法（所有朝上表面放面板）在桌子和架子上与VLM表现相当（都是100%），但在几何复杂的物体（椅子18.8%、灯34.4%）上大幅落后
- 用户反馈显示存在多种合理方案（如仅座面有面板、仅灯罩有面板），验证了人在回路的必要性
- 机器人装配无一将面板放在不可达位置，VLM输出符合制造约束
- 不需要任何任务特定训练，完全依赖VLM的预训练多模态知识

## 亮点与洞察

1. **端到端的文本→物理对象流水线**：首次将3D生成AI、VLM推理和机器人装配串联成完整管线
2. **零样本VLM推理用于制造决策**：展示了VLM的几何+功能推理能力在工业场景中的潜力，无需微调
3. **渐进式推理分解**：将复杂的组件分配分解为两步VLM任务（先选部件名、再映射标签号），降低了单次推理的难度
4. **预处理排除不可达面**：巧妙利用领域知识约束VLM的输出空间

## 局限与展望

- 仅支持两种预定义组件类型（结构件+面板），未扩展到铰链、把手、不同材质等
- 评估限于5种常见物体和简单prompt，复杂/非常规物体（如雕塑、乐器）未被测试
- 依赖Autodesk闭源平台（Project Bernini、Fusion 360），可复现性受限
- VLM的标签映射可能在高度对称或标签密集的网格上出错
- 装配序列仅按底到顶排序，未考虑更复杂的装配约束

## 相关工作与启发

- CLIPort、SayCan等展示了VLM在机器人操作中的语言接地能力，本文进一步扩展到装配几何生成
- PartGen、StructureNet等部件感知模型关注几何建模，本文关注功能推理
- 启示：VLM的世界知识足以进行零样本的制造决策推理，关键在于合理的任务分解和输出约束设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 端到端管线新颖，首次将VLM用于功能感知的装配组件分配
- 实验充分度: ⭐⭐⭐ 用户研究规模适中但物体种类有限，缺少VLM推理失败案例分析
- 写作质量: ⭐⭐⭐⭐ 流水线描述清晰，prompt设计透明
- 实用价值: ⭐⭐⭐ 概念验证阶段，受限于闭源工具和有限组件类型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](../../ICCV2025/multimodal_vlm/instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
