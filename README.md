<p align="right">
  <a href="#-english-version">English</a> | 
  <a href="#-中文版本">中文</a>
</p>

# SH-WRNN: Implicit Spherical Harmonics Weight Field Routing Neural Network
# SH-WRNN: Implicit Spherical Harmonics Weight Field Routing Neural Network

[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.8+](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)

> **Breaking the Matrix Monopoly:** A Brain-Inspired, Continuous-Field Architecture for Next-Generation Edge AI.

SH-WRNN (Implicit Spherical Harmonics Weight Field Routing Neural Network) is a radical departure from standard parameter-heavy Multi-Layer Perceptrons (MLPs). By leveraging the orthogonal completeness of **Spherical Harmonics (SH)**, this framework decouples neural connection weight size from actual training parameters. 

Instead of optimizing millions of localized discrete weights, SH-WRNN models synapses as an **Implicit Spherical Harmonics Weight Field**—a continuous parametric surface topologically equivalent to flattening a local patch of a 3D spherical manifold into a 2D rectangular space. Controlled by just a few dozen global coefficients, it achieves extreme parameter compression during training and unparalleled inference speedups via **Surface Baking**.

---

## 🌟 Core Highlights & Innovations

### 1. Implicit Spherical Harmonics Weight Field
In standard networks, growing the layer size demands a quadratic explosion of parameters (\(O(N \times M)\)). SH-WRNN solves this by compressing the entire connection field into just **16 continuous Spherical Harmonics coefficients** (4th-degree). The dense matrix is implicitly generated on-the-fly, acting as a massive geometric regularizer that forces the network to generalize deep topological patterns.

### 2. Data-Dependent Surface Path-Finding
We redefine dynamic routing. Input features are anchored at fixed latitudinal tracks on our flattened parametric surface. When a specific data sample passes through, it utilizes its own semantic content to dynamically steer its longitudinal destination. The active synapse weight is instantly sampled based on the "landscape height" at that precise coordinate intersection, shifting the concept of neural computation from discrete dot-products to fluid, geometric map look-ups.

### 3. Asymmetric Training-Inference Topology (Surface Baking)
* **Training Phase**: The model utilizes analytical trigonometry and continuous SH fields to optimize global geometric representations under an ultra-low parameter budget.
* **Inference Phase (The Game Changer)**: Upon convergence, the continuous SH weight field is **baked** into static, highly structured discrete look-up sub-surfaces. Transcendental functions are fully eliminated. Inference is reduced to ultra-fast memory slicing.

---

## 📈 Hardcore Benchmark (MNIST Proof of Concept)

We validated this paradigm on the MNIST dataset using a pure, single-sample fluid connection stream (\(784 \to 32 \to 10 \to 10\)). 

The **Surface Baking** mechanism shows a phenomenal asymmetric speedup, demonstrating clear viability for hardware with large L3 caches (e.g., modern edge CPUs) by effectively bypassing GPU memory-bandwidth constraints:

| Metric | Continuous Manifold (Training Mode) | Baked Surface (Inference Mode) | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Accuracy (1 Epoch)** | **91.05%** | **90.83%** | *Negligible loss (-0.22%)* |
| **Total Inference Time**| 34.82s | **9.74s** | **🚀 3.57x Faster** |

---

## 🛠️ Project Structure & Usage

```bash
├── sh_routing_core.py      # Explicit 4th-degree Real Spherical Harmonics engine & Layers
├── train_manifold.py       # End-to-end training loop with dynamic coordinate projection
├── bake_and_infer.py       # The Baking script & Ultra-fast memory-slicing inference model
└── README.md
```

### Quick Start: Training & Baking

```python
# 1. Train the continuous model
model = FlatSHSpaceNetwork3D(max_degree=4).to(device)
# ... Train for 1 epoch -> hits ~91% Accuracy

# 2. Bake the continuous field into high-speed static surfaces
from bake_and_infer import BakedSHSpaceNetwork3D
baked_model = BakedSHSpaceNetwork3D(trained_model=model).to(device)

# 3. Enjoy 3.5x+ sub-millisecond execution!
logits = baked_model(test_images)
```

---

## 🔮 Future Horizon: The Low-Power Hardware Vision

Current LLMs and Deep Learning models are heavily bottlenecked by GPU memory bandwidth and power consumption. Human brains, conversely, maintain high intelligence with massive pathways but extremely sparse activation (~1%), resulting in ultra-low power consumption.

SH-WRNN aims at this future. By offloading complex continuous fields into baked coordinate tables, we shift the compute load from massive parallel arithmetic (GEMM) to high-speed memory indexing. With architectures boasting **large L3 caches**, this paradigm opens up a path where **CPUs can counter-attack GPUs** in edge intelligence, paving the way for ubiquitous, zero-latency local AI.

## 🤝 Collaborative Research

This project is developed by independent researchers striving to push the boundaries of Geometric Deep Learning. If you are interested in expanding this toward Large Language Models (LLMs), Audio Foundational Models, or custom hardware compilers, feel free to open an issue or reach out for collaboration!


---

# 🇨🇳 中文版本
# SH-WRNN: 隐式球谐权重场-寻路神经网络

[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.8+](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)

> **打破矩阵算力垄断：** 一种面向下一代边缘端 AI 的类脑、连续场几何网络架构。

SH-WRNN（隐式球谐权重场-寻路神经网络）是对传统依赖庞大参数量的全连接层（MLP）的一次底层范式颠覆。通过引入**球谐函数（Spherical Harmonics, SH）**的正交完备性，本架构成功将神经网络的连接矩阵体积与实际可训练的参数量进行了深度解耦。

在训练阶段，SH-WRNN 无需优化数以百万计的局部离散权重，而是将突触连接建模为一个**隐式球谐权重场**——这是一个在拓扑上等同于**将三维球面流形的局部斑块展平为二维矩形参数空间**的连续曲面。该架构仅由十几个全局球谐系数控制，在训练时实现了恐怖的参数压缩比，并通过独特的**曲面烘焙（Surface Baking）**技术在推理时获得了颠覆性的速度提升。

---

## 🌟 核心亮点与技术创新

### 1. 隐式球谐权重场 (Implicit Spherical Harmonics Weight Field)
在传统网络中，网络层规模的扩大会带来参数量的平方级爆炸（\(O(N \times M)\)）。SH-WRNN 彻底打破了这一魔咒，将整个稠密连接场压缩进区区 **16 个连续的全局球谐系数**（4阶）。高维稠密矩阵由该场实时隐式生成，充当了极其强大的几何正则化器，强迫网络提炼全局流形的深层拓扑共性。

### 2. 内容驱动的曲面寻路 (Data-Dependent Surface Path-Finding)
我们重新定义了动态路由机制。输入特征被预先锚定在展平曲面固定的“纬度轨道”（行）上。当特定的数据样本通过时，它会根据自身的内容特征，动态引导并喷射出属于该样本的“经度目的地”（列）。突触权重通过查询该经纬度交汇处的“地势高度”瞬间计算得出。网络计算从此告别了死板的点积矩阵乘法，变成了一种流体般的几何地图“看图找路”。

### 3. 非对称训练-推理拓扑与离线烘焙 (Surface Baking)
* **训练阶段**：模型利用解析三角函数和球谐连续场，在极低的参数预算下，强迫梯度流提炼全局几何流形表征。
* **推理阶段（核心大招）**：模型收敛后，连续的球谐权重场将被一次性**烘焙（Baking）**成高度结构化的离散空间查找表。超越函数计算被完全消除，前向传播被简化为极速的内存切片与索引。

---

## 📈 硬核基准测试 (MNIST 概念验证)

我们在 MNIST 数据集上验证了这一范式，使用了一个纯净的、单样本流体连接网络（架构：\(784 \to 32 \to 10 \to 10\)）。

**曲面烘焙（Baking）**机制展现出了现象级的非对称加速表现。这一结果有力地证明了该架构在拥有**大 L3 缓存的现代边缘端 CPU** 上的运行可行性，成功绕过了高昂的 GPU 内存带宽瓶颈：

| 评估指标 | 连续几何流形 (训练模式) | 离线烘焙曲面 (推理模式) | 性能变化 / 加速比 |
| :--- | :---: | :---: | :---: |
| **测试集准确率 (1 Epoch)** | **91.05%** | **90.83%** | *精度损失微乎其微 (-0.22%)* |
| **测试集总推理时间**| 34.82s | **9.74s** | **🚀 速度暴涨 3.57 倍** |

---

## 🛠️ 项目结构 & 核心用法

```bash
├── sh_routing_core.py      # 显式4阶实球谐引擎与寻路网络层实现
├── train_manifold.py       # 端到端连续流形训练脚本（动态坐标投影）
├── bake_and_infer.py       # 离线曲面烘焙与极速内存切片推理脚本
└── README.md
```

### 快速上手：训练与烘焙

```python
# 1. 初始化并训练连续流形模型
model = FlatSHSpaceNetwork3D(max_degree=4).to(device)
# ... 训练 1 个 epoch -> 测试集准确率冲上 ~91%

# 2. 将连续球谐场快速离线烘焙为高维静态曲面
from bake_and_infer import BakedSHSpaceNetwork3D
baked_model = BakedSHSpaceNetwork3D(trained_model=model).to(device)

# 3. 享受超过 3.5 倍的亚毫秒级极致推理！
logits = baked_model(test_images)
```

---

## 🔮 未来愿景：面向未来的低功耗硬件设计

当今的大模型（LLM）和深度学习高度受制于 GPU 的内存带宽和恐怖的功耗。然而，人类大脑拥有万亿级的神经通路，同时却只有约 1% 的极低神经元激活率，实现了极高智能与极低功耗的完美统一。

SH-WRNN 正是瞄准这一未来而生。通过将复杂的连续权重场离线烘焙为高速坐标查找表，我们将计算负载从密集的高并发通用矩阵乘法（GEMM）释放出来，转化为高速的内存寻址与切片。在拥有**大 L3 缓存的现代 CPU 架构**上，这种非对称范式将为 **CPU 在边缘智能领域反超 GPU** 开辟一条全新的几何美学道路，让无延迟、低能耗的本土具身智能成为可能。

## 🤝 合作与研究

本项目由独立研究人员自发推进，旨在打破常规，探索几何深度学习的边界。如果您有兴趣将该机制推广至大语言模型（LLM）、音频大模型或定制化的硬件编译器开发，欢迎提交 Issue 或直接联系我们展开合作！

