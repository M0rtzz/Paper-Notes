---
title: >-
  [论文解读] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation
description: >-
  [AAAI 2026][自动驾驶][对抗攻击] 提出首个针对立体匹配模型的3D全表面纹理物理对抗攻击，通过立体对齐渲染模块和区域感知的融合攻击（merging attack），使对抗车辆在深度图中与背景无缝融合，导致自动驾驶感知系统严重失效。
tags:
  - AAAI 2026
  - 自动驾驶
  - 对抗攻击
  - 立体匹配
  - 双目深度估计
  - 3D纹理伪装
  - 物理对抗样本
---

# Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation

**会议**: AAAI 2026  
**arXiv**: [2511.14386](https://arxiv.org/abs/2511.14386)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 对抗攻击, 立体匹配, 双目深度估计, 3D纹理伪装, 物理对抗样本

## 一句话总结

提出首个针对立体匹配模型的3D全表面纹理物理对抗攻击，通过立体对齐渲染模块和区域感知的融合攻击（merging attack），使对抗车辆在深度图中与背景无缝融合，导致自动驾驶感知系统严重失效。

## 研究背景与动机

现有物理对抗攻击（PAE）主要针对单目深度估计（MDE），且多采用2D patch形式，在立体匹配驱动的双目深度估计（SM-BDE）场景下存在根本局限：

**2D patch 局限性**：仅影响目标物体的局部区域，在不同视角下效果不稳定，且破坏了立体相机所需的物理视差一致性

**现有SM攻击不足**：PASM 和 Adv-DM 简单假设左右视图间存在直接位移关系，忽略了真实立体相机的视差几何约束，在物理部署时失效

**隐藏攻击不完整**：现有hiding attack仅将目标深度推到无穷远，物体轮廓仍清晰可见，攻击意图暴露

**SM-BDE在自动驾驶中日益普及**（百度Apollo、Waymo、Mobileye），其安全性亟需评估

## 方法详解

### 整体框架

系统包含三个核心模块：
1. **立体对齐3D渲染模块**：利用3D目标检测获取车辆位姿，在球坐标系下参数化左右相机视点，独立渲染带对抗纹理的3D车辆
2. **融合攻击纹理生成模块**：通过边界深度提取→区域分割→分区域优化实现深度融合
3. **端到端纹理优化**：可微渲染管线反向传播损失，迭代优化全表面纹理 $\theta$

给定立体背景图像对 $b = (b^l, b^r)$，将纹理映射到3D网格 $O$ 并合成到场景中：

$$x_\theta = S(R(O, \theta, k), b, m)$$

优化目标为使SM模型 $F$ 的深度预测逼近背景目标深度 $d_t$：

$$\theta = \arg\min_\theta \mathcal{L}(F(x_\theta^l, x_\theta^r), d_t)$$

### 关键设计

**（1）立体对齐3D渲染（Stereo-Aligned 3D Rendering）**

与单目渲染不同，SM攻击必须满足三个约束：
- 左右视图中对抗物体保持几何一致外观
- 背景上下文在双视图中保持连贯
- 相机视点遵循物理精确的立体基线

通过3D检测获取车辆包围盒 $\text{bbox} = \{t_x, t_y, t_z, t_l, t_w, t_h, t_r, t_c\}$，在球坐标系下参数化渲染视点 $k = \{\text{dist}, \text{elev}, \text{azim}\}$，为左右相机 $(k_l, k_r)$ 分别渲染，确保视差一致性。

**（2）融合攻击（Merging Attack）纹理生成**

三步流程实现深度融合：

**步骤一：边界深度提取** — 对目标mask $m$ 做max pooling扩展得到边界mask $m_{bg}$，提取周围背景深度：

$$m_{bg} = \text{Maxpool}(m) - m, \quad d_{bg} = d \cdot m_{bg}$$

**步骤二：区域分割** — 计算平均背景深度 $d_{bg}^{avg}$，在物体左右边界找到与之最接近的参考点，将车辆深度图水平分为上下两个区域。下部更贴近地面背景，上部与背景深度差异更大。

**步骤三：分区优化** — 对上下区域分别对齐到各自的局部背景深度：

$$\mathcal{L}_{\text{merge}}(\theta) = \text{MSE}(d_{obj}^{up}, d_{bg}^{up}) + \text{MSE}(d_{obj}^{bt}, d_{bg}^{bt})$$

**（3）出现攻击（Appearing Attack）**

互补策略——最小化目标感知深度使其"逼近"相机，迫使周围车辆紧急制动：

$$\mathcal{L}_{\text{appear}}(\theta) = \text{MSE}(d_{obj}, D_{max})$$

### 损失函数 / 训练策略

总损失包含三项：

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{merge}}(\theta) + \alpha \mathcal{L}_{\text{nps}}(\theta) + \beta \mathcal{L}_{\text{tv}}(\theta)$$

- $\mathcal{L}_{\text{nps}}$：不可打印度损失，约束颜色在打印可复现范围内（$\alpha=5$）
- $\mathcal{L}_{\text{tv}}$：总变差平滑损失，抑制高频噪声（$\beta=0.1$）
- 采用EoT增强鲁棒性：随机扰动光源位置 $[-3,3]$m，环境光强度 $[0.3, 0.9]$，注入高斯噪声模拟雨雾
- Adam优化器，100 epochs，lr=0.01 余弦衰减至 $1e^{-4}$

## 实验关键数据

### 主实验

**Table 1: 五个SM模型上融合攻击效果对比**

| 方法 | PSMNet | GA-Net | RAFT-Stereo | CREStereo | AnyStereo |
|------|--------|--------|-------------|-----------|-----------|
| | $\mathcal{E}_{blend}↓$ / $\mathcal{E}_{cover}↑$ / $\mathcal{E}_{sys}↑$ | 同左 | 同左 | 同左 | 同左 |
| Benign | 0.631/0.013/0 | 0.641/0.012/0 | 0.786/0.012/0 | 0.677/0.017/0 | 0.572/0.093/0 |
| PASM | 0.475/0.154/0.13 | 0.411/0.088/0.12 | 0.502/0.148/0.07 | 0.431/0.094/0.15 | 0.471/0.124/0.15 |
| Adv-DM | 0.510/0.176/0.04 | 0.449/0.075/0.12 | 0.614/0.143/0.05 | 0.444/0.077/0.17 | 0.480/0.119/0.09 |
| **Ours** | **0.058/0.553/0.74** | **0.069/0.588/0.69** | **0.082/0.571/0.62** | **0.071/0.598/0.70** | **0.056/0.576/0.76** |

本文在所有五个模型上全面领先：$\mathcal{E}_{blend}$降低约10倍，$\mathcal{E}_{cover}$提升约4倍，Apollo系统碰撞率从<0.15提升至0.62-0.76。

**Table 2: 真实物理环境评估（3D打印1:30模型 + iPhone双目）**

| 条件 | $\mathcal{E}_{blend}↓$ (Benign→Adv) | $\mathcal{E}_{cover}↑$ (Benign→Adv) |
|------|------|------|
| 正午 | 0.481→0.087 | 0.036→0.519 |
| 日落 | 0.536→0.067 | 0.042→0.577 |
| 侧面视角 | 0.557→0.071 | 0.030→0.581 |
| 12m距离 | 0.517→0.074 | 0.035→0.504 |

### 消融实验

**Table 4: 模块消融（PSMNet，含 $\mathcal{L}_{nps}$ + $\mathcal{L}_{tv}$）**

| 配置 | $\mathcal{E}_{blend}↓$ | $\mathcal{E}_{cover}↑$ |
|------|------|------|
| None | 0.631 | 0.015 |
| SAR only | 0.403 | 0.541 |
| Merge only | 0.611 | 0.024 |
| **Full** | **0.051** | **0.587** |

SAR是有效攻击SM的基础（无SAR则 $\mathcal{E}_{cover}$ 极低），Merge是实现隐蔽融合的关键（无Merge则 $\mathcal{E}_{blend}$ 仍高）。两者协同才能同时实现高覆盖率和低可见性。

### 关键发现

1. **物理可部署性验证**：1:30比例模型车 + iPhone双目在真实环境中验证了跨光照、视角、距离的鲁棒性
2. **全角度鲁棒性**：0°-330° heading angle 下 $\mathcal{E}_{blend} < 0.09$，远优于patch方法在侧面视角失效的问题
3. **系统级威胁**：接入Apollo全栈感知规划后碰撞率最高达0.76

## 亮点与洞察

1. **首创性**：首个针对SM的3D全表面纹理对抗攻击，解决了MDE攻击在BDE下失效的根本问题
2. **融合攻击概念**：从"隐藏"升级到"融合"，通过分区域深度对齐消除物体轮廓，比简单推远深度更具隐蔽性
3. **实验完备性**：数字仿真（CARLA）+ 物理真实（3D打印）+ 系统级（Apollo）三层评估

## 局限与展望

1. 物理实验使用1:30缩放模型，实际车辆尺寸的纹理印刷精度和耐候性需进一步验证
2. 当前评估以静态场景为主，动态行驶中的时序一致性未充分验证
3. 防御方法（如立体一致性检验）的对抗鲁棒性未讨论
4. 融合攻击依赖背景深度的准确估计，复杂背景下的泛化性存疑

## 相关工作与启发

- **PAE发展**：2D patch（Eykholt 2018）→ 局部3D patch（Liu 2024, Cheng 2021）→ 本文全表面3D纹理
- **SM模型覆盖全面**：PSMNet → GA-Net → RAFT-Stereo → CREStereo → AnyStereo，跨架构泛化
- **对自动驾驶安全评估的启发**：揭示SM-BDE系统的安全脆弱性，为防御研究提供重要对照

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ★★★★☆ | 首个3D纹理对SM的物理攻击，融合攻击概念创新 |
| 技术深度 | ★★★★☆ | 立体对齐渲染和分区域优化设计精巧 |
| 实验质量 | ★★★★★ | 5模型+CARLA+物理验证+Apollo系统级 |
| 写作质量 | ★★★★☆ | 结构清晰，图示丰富 |
| 实用价值 | ★★★★☆ | 揭示双目感知安全隐患，有防御研究参考价值 |

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](../../CVPR2025/3d_vision/defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](../../CVPR2026/3d_vision/spectral_defense_against_resourcetargeting_attack.md)

<!-- RELATED:END -->
