---
title: >-
  [论文解读] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics
description: >-
  [CVPR 2025][偏振逆渲染] NeISF++ 将偏振逆渲染从仅支持介电体扩展到同时支持导体和介电体，通过引入二元控制变量 $m$ 的广义 pBRDF 模型、复折射率建模和 DoLP 几何初始化，在合成导体场景上法线角度误差降至 1.789°（比 NeISF 的 10.303°低 83%）。 1. 领域现状：基于偏振的…
tags:
  - "CVPR 2025"
  - "偏振逆渲染"
  - "导体"
  - "复折射率"
  - "Stokes向量"
  - "pBRDF"
---

# NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics

**会议**: CVPR 2025  
**arXiv**: [2411.10189](https://arxiv.org/abs/2411.10189)  
**代码**: 待发布  
**领域**: 其他  
**关键词**: 偏振逆渲染、导体、复折射率、Stokes向量、pBRDF

## 一句话总结

NeISF++ 将偏振逆渲染从仅支持介电体扩展到同时支持导体和介电体，通过引入二元控制变量 $m$ 的广义 pBRDF 模型、复折射率建模和 DoLP 几何初始化，在合成导体场景上法线角度误差降至 1.789°（比 NeISF 的 10.303°低 83%）。

## 研究背景与动机

1. **领域现状**：基于偏振的逆渲染（如 NeISF、PANDORA）利用偏振相机捕捉的 Stokes 向量信息来恢复几何、材质和光照。但现有方法只支持介电体（如塑料、玻璃），无法处理金属等导体材质。
2. **现有痛点**：(1) 导体没有次表面散射（漫反射项为零），现有 pBRDF 模型直接崩溃；(2) 导体的 Fresnel 反射需要复折射率 $ior = \eta - ki$，现有方法只支持实数折射率；(3) 导体的强镜面反射使基于法向量一致性（NCC）的几何初始化失效。
3. **核心矛盾**：真实场景中导体和介电体常常共存（如金属底座+塑料外壳），但现有方法只能处理其中一类——需要统一框架。
4. **本文目标**：设计同时支持导体和介电体的偏振逆渲染方法。
5. **切入角度**：引入二元标记 $m$（$m=0$ 导体，$m=1$ 介电体）控制 pBRDF 中漫反射项的开关，并用 DoLP（线偏振度）替代法向量做几何初始化——DoLP 对光照强度不变，对导体鲁棒。
6. **核心 idea**：广义 pBRDF + 复折射率 + DoLP 初始化。

## 方法详解

### 整体框架

多视角偏振图像 → DoLP 计算 → VolSDF 几何初始化（用 DoLP 替代法向量一致性）→ 联合优化 SDF 几何 $\mathbb{S}$、pBRDF 材质场 $\mathbb{B}$（输出粗糙度 $r$、反照率 $a$、实折射率 $\eta$、消光系数 $k$）和入射 Stokes 场 $\mathbb{L}$ → 渲染偏振图像与观测对比优化。

### 关键设计

1. **广义偏振 BRDF (pBRDF)**

    - 功能：统一建模导体和介电体的偏振反射
    - 核心思路：在 Baek pBRDF 基础上引入二元标记 $m$：$\mathbf{s} = \int_\Omega m \cdot \mathbf{R}^{cam}_{dif} \cdot \mathbf{M}_{dif} \cdot \mathbf{s}^r_{dif} + \mathbf{R}^{cam}_{spec} \cdot \mathbf{M}_{spec} \cdot \mathbf{s}^r_{spec} \, d\omega_i$。当 $m=0$（导体）时漫反射项消失，只保留镜面项
    - 设计动机：导体的物理本质是没有次表面散射——电磁波在金属表面完全反射。$m$ 的引入直接映射了这个物理事实

2. **复折射率建模**

    - 功能：准确描述导体的 Fresnel 反射特性
    - 核心思路：材质场输出复折射率 $ior = \eta - ki$，其中 $\eta \in \mathbb{R}^3$（实部，决定反射率）和 $k \in \mathbb{R}^3$（虚部/消光系数，决定吸收）。利用 PyTorch 的复数自动微分支持
    - 设计动机：金属的颜色来自波长依赖的复折射率——铜的红色、金的黄色都是 $k(\lambda)$ 的体现。不建模 $k$ 就无法还原金属颜色

3. **DoLP 几何初始化**

    - 功能：为导体场景提供鲁棒的几何先验
    - 核心思路：DoLP $\rho = \sqrt{s[1]^2 + s[2]^2}/s[0]$ 对光照强度不变（因为分子分母同时缩放），用 $\rho$ 的一致性替代传统 NCC 法向量一致性来初始化 VolSDF
    - 设计动机：导体的强镜面反射使 NCC 失效（同一点在不同视角的颜色完全不同），但 DoLP 反映的是偏振程度而非亮度

### 损失函数 / 训练策略

初始化：$\mathcal{L}_{init} = \lambda_\rho L_1(\rho, \hat\rho) + \lambda_I L_1(I, \hat I) + \lambda_{Eik} L_{Eik}$。联合优化：$\mathcal{L}_{joint} = \lambda_{\rho_s} L_1(\rho_s, \hat\rho) + \lambda_s L_1(s, \hat s) + \lambda_{Eik} L_{Eik}$。需要用户提供导体-介电体掩码。

## 实验关键数据

### 主实验

| 方法 | Helmet 法线误差↓ | Stanford Scan 法线误差↓ |
|------|-----------------|----------------------|
| VolSDF | 8.829° | 11.754° |
| PANDORA | 13.212° | 21.740° |
| NeRO | 5.001° | 13.352° |
| NeISF | 10.303° | 14.022° |
| **NeISF++** | **1.789°** | **6.487°** |

### 消融实验

| 配置 | Helmet 法线误差↓ | 说明 |
|------|-----------------|------|
| w/o DoLP 初始化 | 2.400° | DoLP 初始化贡献 0.6° |
| w/ DoLP 初始化 | **1.789°** | 完整模型 |
| VolSDF (传统NCC) | 8.829° | NCC 在导体上失效 |
| VolSDF-DoLP | 4.715° | 仅靠 DoLP 已大幅提升 |

### 关键发现

- NeISF++ 在导体场景（Helmet）上法线误差仅 1.789°，比 NeISF 的 10.303° 低 83%——证明导体建模是必需的
- DoLP 初始化本身就将 VolSDF 从 8.829° 降至 4.715°，是一个有效的独立技巧
- 材质分解精度：粗糙度 MAE 从 NeISF 的 0.2075 降至 0.0161，提升一个数量级

## 亮点与洞察

- **物理驱动的方法设计**：$m$ 标记、复折射率、DoLP 都直接对应导体的物理特性——不是盲目堆模块，而是每个设计都有物理根据
- **DoLP 的鲁棒性是一个有价值的发现**：DoLP 对光照不变的特性在导体以外的强反射场景（如湿地面）也可能有用
- **PyTorch 复数自动微分的工程价值**：证明现代框架已原生支持复数反向传播，降低了复折射率建模的工程门槛

## 局限与展望

- 需要手动标注导体-介电体掩码——最费时的步骤
- 介电体区域仍假设固定折射率 1.5，不进行估计
- 仅支持线偏振，圆偏振信号被忽略
- 目前没有自动材质分割能力

## 相关工作与启发

- **vs NeISF**: 仅支持介电体。NeISF++ 通过 $m$ 标记和复折射率扩展到导体
- **vs NeRO**: 基于环境图的逆渲染，不利用偏振信息。NeISF++ 利用偏振额外约束更准确
- **vs PANDORA**: 同样用偏振但假设全是介电体，导体场景误差 13.2° vs NeISF++ 1.8°

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将偏振逆渲染扩展到导体+介电体混合场景
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据验证，但场景数量偏少
- 写作质量: ⭐⭐⭐⭐ 物理推导清晰
- 价值: ⭐⭐⭐⭐ 偏振逆渲染的重要扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)
- [\[ICML 2025\] Improving the Effective Receptive Field of Message-Passing Neural Networks](../../ICML2025/others/improving_the_effective_receptive_field_of_message-passing_neural_networks.md)

</div>

<!-- RELATED:END -->
