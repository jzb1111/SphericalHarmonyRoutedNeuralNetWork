# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:52:53 2026

@author: jzb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import numpy as np

# 固化随机种子，确保网络初始化、球谐系数采样以及数据流在多设备上的绝对可复现性
torch.manual_seed(42)

# =====================================================================
# 🌌 1. 全张量化、4阶 16通道显式实球谐基底矩阵计算 (保持绝对稳定)
# =====================================================================
def compute_explicit_sh_basis_4degree(theta, phi):
    """
    【数学原理与物理背景】
    本函数实现了在球面上对 4 阶实球谐函数（Real Spherical Harmonics）的显式解析计算。
    球谐函数是拉普拉斯算子在单位球面上的本征函数，常用于计算机图形学中的环境光渲染和 3D 几何特征建模。
    传统方法使用复杂的循环迭代，而在深度学习中，我们通过高效的 PyTorch 全张量化（Fully Tensorized）
    技术，将多维张量直接送入三角函数组合，消除了传统循环的开销，完美释放 GPU 的并发算力。

    【输入参数说明】
    :param theta: 偏振角 (Polar Angle) 张量，通常映射在 [0, pi] 之间，控制球面的纬度方向。
    :param phi:   方位角 (Azimuthal Angle) 张量，通常映射在 [0, 2*pi] 之间，控制球面的经度方向。
    :return Y:    采样得到的 16 通道球谐基底矩阵，形状为 [M, 16]，每一列代表一个特定的球谐基底。
    """
    M = theta.shape[0]
    device = theta.device
    
    # 强制将精度对齐为 float32，避免半精度（float16）在大规模指数乘法中产生溢出或数值下溢
    t = theta.to(torch.float32)
    p = phi.to(torch.float32)
    
    # 【三角泛函多倍角预计算】提前计算好所需的多倍角，避免在后续的 16 个基底计算中进行重复的三角函数开销
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    cos_p, sin_p = torch.cos(p), torch.sin(p)
    cos_2p, sin_2p = torch.cos(2.0 * p), torch.sin(2.0 * p)
    cos_3p, sin_3p = torch.cos(3.0 * p), torch.sin(3.0 * p)

    # 初始化纯净的零矩阵，准备填充 0阶 到 3阶（共 1 + 3 + 5 + 7 = 16 通道）的连续几何基底
    Y = torch.zeros((M, 16), device=device, dtype=torch.float32)
    
    # -----------------------------------------------------------------
    # 0 阶基底 (l = 0): 常数项，代表单位球面的均匀各向同性分量
    # -----------------------------------------------------------------
    Y[:, 0] = 0.5 * math.sqrt(1.0 / math.pi)
    
    # -----------------------------------------------------------------
    # 1 阶基底 (l = 1): 包含 3 个通道，分别对应 3D 空间中的 X, Y, Z 正交方向
    # -----------------------------------------------------------------
    Y[:, 1] = 0.5 * math.sqrt(3.0 / math.pi) * sin_t * sin_p
    Y[:, 2] = 0.5 * math.sqrt(3.0 / math.pi) * cos_t
    Y[:, 3] = 0.5 * math.sqrt(3.0 / math.pi) * sin_t * cos_p
    
    # -----------------------------------------------------------------
    # 2 阶基底 (l = 2): 包含 5 个通道，捕捉更高频的球面弯曲和对称性
    # -----------------------------------------------------------------
    Y[:, 4] = 0.25 * math.sqrt(15.0 / math.pi) * (sin_t**2) * sin_2p
    Y[:, 5] = 0.5 * math.sqrt(15.0 / math.pi) * sin_t * cos_t * sin_p
    Y[:, 6] = 0.25 * math.sqrt(5.0 / math.pi) * (3.0 * cos_t**2 - 1.0)
    Y[:, 7] = 0.5 * math.sqrt(15.0 / math.pi) * sin_t * cos_t * cos_p
    Y[:, 8] = 0.25 * math.sqrt(15.0 / math.pi) * (sin_t**2) * cos_2p
    
    # -----------------------------------------------------------------
    # 3 阶基底 (l = 3): 包含 7 个通道，提供细粒度的球面局部几何变化表征
    # -----------------------------------------------------------------
    Y[:, 9] = 0.125 * math.sqrt(35.0 / (2.0 * math.pi)) * (sin_t**3) * sin_3p
    Y[:, 10] = 0.25 * math.sqrt(105.0 / math.pi) * (sin_t**2) * cos_t * sin_2p
    Y[:, 11] = 0.125 * math.sqrt(21.0 / (2.0 * math.pi)) * sin_t * (5.0 * cos_t**2 - 1.0) * sin_p
    Y[:, 12] = 0.25 * math.sqrt(7.0 / math.pi) * (5.0 * cos_t**3 - 3.0 * cos_t)
    Y[:, 13] = 0.125 * math.sqrt(21.0 / (2.0 * math.pi)) * sin_t * (5.0 * cos_t**2 - 1.0) * cos_p
    Y[:, 14] = 0.25 * math.sqrt(105.0 / math.pi) * (sin_t**2) * cos_t * cos_2p
    Y[:, 15] = 0.125 * math.sqrt(35.0 / (2.0 * math.pi)) * (sin_t**3) * cos_3p
    
    return Y

# =====================================================================
# 🌌 2. 可学习球谐表征网络层 (连续空间采样的动态权重生成器)
# =====================================================================
class LearnableSphericalHarmonicsLayer(nn.Module):
    def __init__(self, num_degrees=4):
        """
        【设计思路】
        该层不保存传统的、离散固定的全连接权重矩阵。相反，它保存的是球谐基底的“线性组合系数”。
        在正向传播时，它接收连续的空间坐标（行和列），通过调用上述的实球谐引擎在球面进行空间采样，
        将采样点与可学习系数结合，动态“吹出”一层纯净、平滑的连续权重。
        """
        super().__init__()
        self.num_coeffs = 16  # 4阶对应的固定基底通道数
        
        # 使用基于通道维度的标准凯明（Kaiming）正态分布初始化球谐系数，防止网络深层出现梯度崩塌
        self.sh_coeffs = nn.Parameter(torch.randn(self.num_coeffs, 1) * math.sqrt(2.0 / self.num_coeffs))

    def forward(self, row, col):
        """
        【坐标映射机理】
        为了让球谐引擎平滑运行，需要将来自物理特征的输入坐标（通常在 0 到 1 之间）线性缩放到单位球面上。
        这里将 row 和 col 映射至偏振角 theta 与方位角 phi。
        
        【优化要点】
        为了防止 sigmoid 产生过度饱和导致采样点在边界扎堆（退化为固定常数），
        缩放公式设计为：theta = (0.10 + 0.30 * row) * pi。
        这意味着采样范围被锁定在球面最具有几何区分度的中间黄金流形带 [0.1*pi, 0.4*pi] 之间，
        极大地捍卫了动态特征表达的多样性。
        """
        theta = (0.10 + 0.30 * row) * torch.pi
        phi = (0.10 + 0.30 * col) * torch.pi
        
        # 展平进行全并行球谐采样计算 -> 输出形状为 [M, 16]
        Y = compute_explicit_sh_basis_4degree(theta, phi)
        
        # 🔧 修复机制：通过矩阵乘法融合可学习系数，并进行显式展平，吐出干净的一维高维空间权重信号
        return torch.matmul(Y, self.sh_coeffs).reshape(-1)

# =====================================================================
# 🌿 3. SH-Routing v3.1：回归全局特征大格局的流体空间路由层
# =====================================================================
class Flat3DSpaceRouteConnect(nn.Module):
    def __init__(self, in_features, nex_activate_dim, sph_har_n=4):
        """
        【路由架构设计理念】
        传统的 Linear 层是通过矩阵乘法静态映射特征。
        而 SH-Routing 则是让前一层的输入特征，动态地为每一个 Batch、每一个样本“喷射”出它们专属的
        3D 寻路 Key。这些 Key 作为列坐标，与输入的行坐标交织成一个动态网格，再由球谐引擎实时采样生成
        一个“量身定制”的权重面板，通过批矩阵乘法（BMM）实现流体信号传导。
        """
        super().__init__()
        self.in_features = in_features                  # 输入特征维度（如第一层的 784）
        self.nex_activate_dim = nex_activate_dim          # 输出或下一层桥梁维度
        
        # 嵌套我们第一部分定义的球谐几何采样层
        self.Sph_har_engine = LearnableSphericalHarmonicsLayer(num_degrees=sph_har_n)
        
        # 样本路由投影器：每个样本根据自己的输入特征，独立计算出自己在高维隐式空间中的定位寻路坐标
        self.route_projector = nn.Linear(in_features, nex_activate_dim)
        self.bias = nn.Parameter(torch.zeros(self.nex_activate_dim))

    def forward(self, activate_ind, pre_layer_input):
        """
        【输入维度追踪】
        :param activate_ind:     行坐标，已在外部被统一对齐为 [Batch, in_features] 形状。
        :param pre_layer_input: 输入特征张量，形状为 [Batch, in_features]。
        """
        B, N = pre_layer_input.shape
        
        # 计算特征缩放因子（缩放特征维度的平方根），类似于 Transformer 中的 Scaled Dot-Product
        scale_factor = 1.0 / math.sqrt(float(N))
        
        # 🚀 1. 每个样本独立、无同质化地喷射出自己的专属高维寻路坐标 [Batch, nex_activate_dim]
        # 通过 sigmoid 将坐标柔和地限制在 (0, 1) 的开放区间内
        keys = torch.sigmoid(self.route_projector(pre_layer_input)) 
        
        # =====================================================================
        # 🪐 强健的多样本流形自适应对齐（彻底摆脱传统的 if-else 动态类型分支）
        # =====================================================================
        # 此时输入的 activate_ind 在所有层中都统一具备标准的 2D 形状 [B, N]
        # 我们只需在最末尾轴上补上一维通道轴，即可利用 PyTorch 机制极其优雅、无冲突地广播至 3D 空间
        row_grid = activate_ind.unsqueeze(-1).expand(B, N, self.nex_activate_dim)
        
        # 同理，列坐标（专属 Key）在第 1 轴进行扩展，与行坐标交织，扩张为：[B, N, nex_activate_dim]
        col_grid = keys.unsqueeze(1).expand(B, N, self.nex_activate_dim)
        
        # 2. 🚀 零循环、全并行球谐连续权重采样
        # 展平送入球谐引擎计算，将多维张量压缩为一维长向量打入 GPU
        flat_weight = self.Sph_har_engine(row_grid.reshape(-1), col_grid.reshape(-1))
        
        # 从长向量中瞬间恢复出三维对齐的样本专属动态权重面板：[Batch, in_features, nex_activate_dim]
        weight_matrix = flat_weight.reshape(B, N, self.nex_activate_dim)
        
        # 3. 批矩阵乘法 (BMM) 瞬间完成特征传导
        # pre_layer_input.unsqueeze(1) 形状为 [B, 1, in_features]
        # 与 weight_matrix [B, in_features, nex_activate_dim] 作用，输出 [B, 1, nex_activate_dim]
        res = torch.bmm(pre_layer_input.unsqueeze(1), weight_matrix).squeeze(1)
        
        # 加入缩放因子控制方差，并叠加上可学习偏置，完成平滑的几何信号流转
        res = res * scale_factor + self.bias
        return res, keys

# =====================================================================
# 📦 4. 组装第三代单体球谐寻路网络 (全局特征完美对齐静态版)
# =====================================================================
class FlatSHSpaceNetwork3D(nn.Module):
    def __init__(self, max_degree=4):
        """
        【网络拓扑缩减说明】
        将特征宽度设为较窄的级联层（784 -> 5 -> 5 -> 10），使其高度聚焦于球谐引擎的物理特征对齐。
        """
        super().__init__()
        self.layer1 = Flat3DSpaceRouteConnect(in_features=784, nex_activate_dim=32, sph_har_n=max_degree)
        self.layer2 = Flat3DSpaceRouteConnect(in_features=32, nex_activate_dim=3, sph_har_n=max_degree)
        self.layer3 = Flat3DSpaceRouteConnect(in_features=3, nex_activate_dim=10, sph_har_n=max_degree)
        
        # 使用平滑的 SiLU（Swish）作为非线性门控门，相比 ReLU 它在原点附近具备连续二阶导数，更契合球谐函数的平滑连续特性
        self.non_linear_gate = nn.SiLU()

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        
        # 🔧 终极优化的静态对齐机制：在第一层初始化时，一出生就为其垫上 Batch 维度 [B, 784]
        # 这消除了层与层之间输入张量维度的“基因突变”，是保证 TorchScript 100% 静态编译成功的关键
        first_activate_ind_lis = torch.linspace(0, 1, 784, device=device).unsqueeze(0).expand(B, 784)
        
        # Layer 1 推进：从 784 维的高维特征图谱中抽丝剥茧，提取出 5 维几何骨架
        layer1_res, layer1_keys = self.layer1(first_activate_ind_lis, x)
        layer1_res = self.non_linear_gate(layer1_res)
        
        # Layer 2 推进：直接传入具有 Batch 属性的 layer1_keys，内部无痛执行多样本 3D 广播机制
        layer2_res, layer2_keys = self.layer2(layer1_keys, layer1_res)
        layer2_res = self.non_linear_gate(layer2_res)
        
        # Layer 3 推进：将特征直接收拢至 10 个分类类别中，吐出最终的 Logits
        layer3_res, layer3_keys = self.layer3(layer2_keys, layer2_res)
        
        logits = layer3_res
        return logits, [layer1_keys, layer2_keys, layer3_keys]

# =====================================================================
# 🚀 5. 端到端大 Batch 强健训练闭环与编译固化流水线
# =====================================================================
if __name__ == '__main__':
    # 定义基本的数据流展平转换组件
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # 加载经典的 MNIST 手写数字数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # 🔓 维度封印解除！现在支持任意的大 Batch 训练（这里设为 32，可随意调大，告别报错）
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并部署至指定的硬件加速设备
    model = FlatSHSpaceNetwork3D(max_degree=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"\n[🌿 SH-Routing Net v3.1 任意Batch大 Batch 优化版启动]")
    print(f"当前加速设备: {device} | 第一层全局输入特征宽度: 784 维全特征直接映射 \n")
    model.train()

    running_loss = 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True) # 使用 set_to_none=True 清空梯度，能省下可观的显存带宽
        
        # 前向传播，获取预测输出与各层的寻路 Key
        output, all_keys = model(x_batch)
        
        # 100% 纯净分类 CrossEntropyLoss 计算
        loss = criterion(output, y_batch)
        
        # 反向传播与梯度步进更新
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 每 100 步打印当前的即时分类损失
        if (batch_idx + 1) % 100 == 0:
            print(f"Step {batch_idx+1:04d} | 纯净分类 Loss_CE: {loss.item():.4f}")
            
        # 每 2000 步进行阶段性平均评估，并优雅抽样首个样本的独立物理自由坐标
        if (batch_idx + 1) % 2000 == 0:
            print(f"\n" + "="*70)
            print(f"训练进度: {batch_idx+1}/{len(train_loader)} | 近2000步平均分类 Loss: {running_loss / 2000:.4f}")
            
            # 🔍 优雅解包：专门抽取当前 Batch 中索引为 0 的“第一个样本”的专属坐标，防止 Batch 混同混淆
            k1_show = all_keys[0].detach().cpu().numpy()
            k2_show = all_keys[1].detach().cpu().numpy()
            print(f"-> 抽样首个样本在 Layer1 喷射出的 5通道自由坐标: {np.round(k1_show, 3)}")
            print(f"-> 抽样首个样本在 Layer2 喷射出的 5通道自由坐标: {np.round(k2_show, 3)}")
            print("="*70 + "\n")
            running_loss = 0.0

    # =====================================================================
    # 🪐 完美的二进制全包固化：使用 Script 机制将公式与权重融为一体
    # =====================================================================
    model.eval()
    print(f"\n[💾 开启全二进制编译固化阶段]...")
    
    try:
        # 由于我们彻底移除了动态 if 分支，并将第一层行坐标做了 Batch 维度的静态对齐，
        # 现在的网络结构在 TorchScript 看来具有极度完美的静态确定性。
        compiled_model = torch.jit.script(model)
        
        # 保存为独立的二进制 .pt 文件。它将数学公式和权重融为一体，可在 C++ 等无 Python 环境下直接推理。
        torch.jit.save(compiled_model, "sh_routed_network.pt")
        print(f"[💾 固化大胜] 成功保存全包式静态编译模型: sh_routed_network.pt")
    except Exception as e:
        # 极为稳健的防呆底线保护机制
        torch.save(model.state_dict(), "sh_routed_network.pth")
        print(f"[💾 常规固化] JIT编译提示({e})，已安全切换为标准权重字典保存(sh_routed_network.pth)。")
