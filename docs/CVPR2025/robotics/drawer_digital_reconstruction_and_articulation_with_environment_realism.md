---
title: >-
  [论文解读] DRAWER: Digital Reconstruction and Articulation with Environment Realism
description: >-
  [CVPR 2025][机器人][digital twin] 提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。
tags:
  - CVPR 2025
  - 机器人
  - digital twin
  - articulated objects
  - Gaussian splatting
  - SDF
  - game engine
  - real-to-sim-to-real
  - robotic manipulation
---

# DRAWER: Digital Reconstruction and Articulation with Environment Realism

**会议**: CVPR 2025  
**arXiv**: [2504.15278](https://arxiv.org/abs/2504.15278)  
**机构**: UIUC / University of Washington / Allen Institute for AI / Cornell University
**领域**: 机器人学 / 3D 重建  
**关键词**: digital twin, articulated objects, Gaussian splatting, SDF, game engine, real-to-sim-to-real, robotic manipulation

## 一句话总结
提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。

## 研究背景与动机

1. **领域现状**：从真实世界数据创建虚拟数字复制品在游戏、机器人、虚拟现实等领域有巨大潜力。现有方法要么只关注外观建模忽视物理交互，要么优先交互性但牺牲真实感。
2. **现有痛点**：
   - **NeRF/3DGS 方法**：渲染质量好但几何精度不足，"漂浮"的 Gaussian 未与底层几何对齐
   - **Neural SDF 方法**：几何精度好但渲染质量落后，且体积渲染慢
   - **URDFormer**：可估计铰接但依赖预定义的 asset 库，物理真实度受限
   - 没有方法同时实现：高保真渲染 + 精细几何 + 物理交互 + 实时性能
3. **核心矛盾**：外观保真度和几何精度之间的矛盾，以及静态重建和可交互性之间的鸿沟。
4. **切入角度**：双场景表示——SDF 负责几何精度，Gaussian splatting 负责渲染质量；将场景分解为可交互组件，自动推理铰接类型和铰链位置。
5. **核心 idea 一句话**：SDF + Gaussian 双表示（几何+外观） + 铰接推理 + amodal 形状补全 = 完整可交互数字孪生。

## 方法详解

### 整体管线
输入多视角 posed 图像（来自单视频）→ 双场景表示重建 → 铰接体识别与推理 → amodal 形状估计 + 隐藏区域纹理生成 → 可交互数字孪生

### 关键设计

1. **双场景表示（Dual Scene Representation）**
   - **Neural SDF 分支**：
     - 映射 3D 点和视角到 RGB 颜色和符号距离
     - 通过体积渲染监督学习
     - 提供精确几何（高质量 mesh 提取）
   - **Gaussian Splatting 分支**：
     - 实时渲染（>30 FPS）
     - 光栅化渲染，不需要逐点采样
     - 提供高保真外观
   - **耦合策略**：Gaussian 的位置和法线由 SDF 表面约束，确保几何一致性

2. **铰接推理模块**
   - **铰接类型识别**：区分旋转（revolute）和平移（prismatic）两种铰接类型
   - **铰链位置估计**：推理铰链轴的位置和方向
   - 与 3DOI（铰接预测基础模型）对比，EA-Score 达 0.994 vs. 0.861

3. **Amodal 形状估计与隐藏纹理生成**
   - 做什么：重建物体被遮挡部分的形状和纹理
   - 核心问题：打开抽屉/柜门后露出的内部表面在原视频中不可见
   - 解决：使用 SDF 进行 amodal 形状补全 + 纹理 inpainting
   - 效果：创建完整的可交互物体模型

4. **游戏引擎集成**
   - 自动导出到 Unreal Engine
   - 支持物理碰撞、射击交互、开关动画
   - 实时运行

### 应用示例

1. **交互式游戏**
   - 第一人称视角自由移动
   - 射击球体产生真实碰撞
   - 打开/关闭抽屉柜门

2. **Real-to-Sim-to-Real 机器人迁移**
   - 将重建场景导入 Isaac Sim
   - 运动规划生成训练数据
   - 3D Diffusion Policy 学习策略
   - 直接迁移到 Franka Emika Panda 真实机器人

## 实验关键数据

### 铰接推理对比
| 方法 | 总物体数 | 正确预测↑ | 旋转物体数 | 旋转正确↑ | EA-Score↑ |
|------|---------|----------|-----------|----------|----------|
| 3DOI | 80 | 78 | 59 | 57 | 0.861 |
| **DRAWER** | 80 | **78** | 59 | **58** | **0.994** |

### 全管线对比（vs. URDFormer / Digital Cousin）
| 方法 | Precision | Recall | 视觉保真度 | 几何精度 |
|------|-----------|--------|-----------|---------|
| URDFormer | 中（依赖 asset 库） | 低 | 中 | 低 |
| Digital Cousin | 中 | 中 | 中 | 中 |
| **DRAWER** | **高** | **高** | **高** | **高** |

### 铰接运动仿真
- 与 KlingAI 对比，使用 Earth Mover's Distance 度量运动轨迹质量
- DRAWER 的物理仿真运动轨迹比 KlingAI 生成的"高出一个数量级"

### 消融实验
| 组件 | 渲染质量 | 几何质量 | 交互兼容性 |
|------|---------|---------|-----------|
| 仅 SDF mesh | 中 | 高 | ✓ |
| + Gaussian splatting | 高 | 高 | ✓ |
| + 铰接推理 | 高 | 高 | ✓（可动） |
| + Amodal 补全 | 高 | 高 | ✓（完整） |

## 亮点与洞察
- **双表示** SDF+Gaussian 取长补短：SDF 的几何精度 + Gaussian 的渲染速度与质量
- **端到端全自动**：从视频到可交互数字孪生无需人工干预
- **Real-to-Sim-to-Real** 闭环验证：证明了方法在机器人领域的实用价值
- **EA-Score 0.994**（接近完美）表明铰接推理极其准确
- 从学术原型（场景重建）到实际应用（游戏/机器人）的完整链路
