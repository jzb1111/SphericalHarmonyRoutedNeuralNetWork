# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:28:21 2026

@author: jzb
"""

import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
torch.set_default_device('cpu')
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
    device = torch.device('cpu')#theta.device
    
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

class BakedSHSpaceNetwork3D(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.device = torch.device('cpu')#next(trained_model.parameters()).device
        self.non_linear_gate = nn.SiLU()
        
        # 1. 继承原模型的轻量级坐标投影器 (这些在推理时依然需要，用来动态生成寻路 Key)
        self.route_projector1 = trained_model.layer1.route_projector
        self.route_projector2 = trained_model.layer2.route_projector
        self.route_projector3 = trained_model.layer3.route_projector
        
        self.bias1 = trained_model.layer1.bias
        self.bias2 = trained_model.layer2.bias
        self.bias3 = trained_model.layer3.bias
        
        # =====================================================================
        # 🪐 2. 核心大招：离线烘焙 (Baking) 过程
        # =====================================================================
        print("[💾 正在将连续球谐场烘焙为离线流体曲面...]")
        with torch.no_grad():
            # 提取原模型中训练好的 16 维球谐系数 [16, 1]
            c1 = trained_model.layer1.Sph_har_engine.sh_coeffs
            c2 = trained_model.layer2.Sph_har_engine.sh_coeffs
            c3 = trained_model.layer3.Sph_har_engine.sh_coeffs
            
            # 建立固定的行坐标 (纬度 theta)
            theta_L1 = (0.10 + 0.30 * torch.linspace(0, 1, 784, device=self.device)) * torch.pi
            theta_L2 = (0.10 + 0.30 * torch.linspace(0, 1, 32, device=self.device)) * torch.pi
            theta_L3 = (0.10 + 0.30 * torch.linspace(0, 1, 10, device=self.device)) * torch.pi
            
            # 为了让动态推理实现“查找表”般的极致速度，我们将经度 phi (0到1) 离散化为精细的网格
            # 这里选择 512 个采样点，足以完美还原 4 阶球谐曲面的光滑低频波形
            self.grid_res = 512
            phi_grid = (0.10 + 0.30 * torch.linspace(0, 1, self.grid_res, device=self.device)) * torch.pi
            
            # --- 烘焙第一层矩阵曲面 ---
            # 广播网格计算全图基底
            t1_mat = theta_L1.unsqueeze(1).expand(784, self.grid_res)
            p1_mat = phi_grid.unsqueeze(0).expand(784, self.grid_res)
            Y1 = compute_explicit_sh_basis_4degree(t1_mat.reshape(-1), p1_mat.reshape(-1))
            # 烘焙结果：[784, 512]，代表 784个特征在 512个物理坐标上的连续权重全景图
            self.register_buffer('baked_surface_L1', torch.matmul(Y1, c1).reshape(784, self.grid_res))
            
            # --- 烘焙第二层矩阵曲面 ---
            t2_mat = theta_L2.unsqueeze(1).expand(32, self.grid_res)
            p2_mat = phi_grid.unsqueeze(0).expand(32, self.grid_res)
            Y2 = compute_explicit_sh_basis_4degree(t2_mat.reshape(-1), p2_mat.reshape(-1))
            # 烘焙结果：[32, 512]
            self.register_buffer('baked_surface_L2', torch.matmul(Y2, c2).reshape(32, self.grid_res))
            
            # --- 烘焙第三层矩阵曲面 ---
            t3_mat = theta_L3.unsqueeze(1).expand(10, self.grid_res)
            p3_mat = phi_grid.unsqueeze(0).expand(10, self.grid_res)
            Y3 = compute_explicit_sh_basis_4degree(t3_mat.reshape(-1), p3_mat.reshape(-1))
            # 烘焙结果：[10, 512]
            self.register_buffer('baked_surface_L3', torch.matmul(Y3, c3).reshape(10, self.grid_res))
            
        print("[✨ 烘焙大功告成！已成功固化高维空间静态映射表]")

    def _query_baked_weight(self, baked_surface, keys, target_dim):
        """ 🚀 高速查表网格线性插值或近似索引用法 """
        # keys 形状: [B, target_dim]，值在 0 ~ 1 之间
        # 将 0~1 的连续寻路坐标映射到离散的网格索引 [0, grid_res - 1]
        idx = (keys * (self.grid_res - 1)).long().clamp(0, self.grid_res - 1) # [B, target_dim]
        
        # 利用高级索引（Advanced Indexing）瞬间切片，提取专属权重矩阵
        # baked_surface: [In_dim, 512]
        # 经过切片后直接喷射出：[B, In_dim, target_dim] 的独家样本权重，跳过所有球谐计算！
        B = keys.shape[0]
        weight_matrix = baked_surface.unsqueeze(0).expand(B, -1, -1) # [B, In_dim, 512]
        
        # 收集对应坐标处的烘焙特征
        idx_expanded = idx.unsqueeze(1).expand(-1, baked_surface.shape[0], -1) # [B, In_dim, target_dim]
        dynamic_weight = torch.gather(weight_matrix, dim=2, index=idx_expanded)
        return dynamic_weight

    def forward(self, x):
        B, N = x.shape
        scale_factor = 1.0 / math.sqrt(N)
        
        # --- Layer 1 高速推理 ---
        layer1_keys = torch.sigmoid(self.route_projector1(x)) # [B, 32]
        W1 = self._query_baked_weight(self.baked_surface_L1, layer1_keys, target_dim=32) # [B, 784, 32]
        res1 = torch.bmm(x.unsqueeze(1), W1).squeeze(1) * scale_factor + self.bias1
        res1 = self.non_linear_gate(res1)
        
        # --- Layer 2 高速推理 ---
        layer2_keys = torch.sigmoid(self.route_projector2(res1)) # [B, 10]
        W2 = self._query_baked_weight(self.baked_surface_L2, layer2_keys, target_dim=10) # [B, 32, 10]
        res2 = torch.bmm(res1.unsqueeze(1), W2).squeeze(1) * (1.0 / math.sqrt(32)) + self.bias2
        res2 = self.non_linear_gate(res2)
        
        # --- Layer 3 高速推理 ---
        layer3_keys = torch.sigmoid(self.route_projector3(res2)) # [B, 10]
        W3 = self._query_baked_weight(self.baked_surface_L3, layer3_keys, target_dim=10) # [B, 10, 10]
        logits = torch.bmm(res2.unsqueeze(1), W3).squeeze(1) * (1.0 / math.sqrt(10)) + self.bias3
        
        keys=[layer1_keys, layer2_keys, layer3_keys]
        return logits, keys

# 使用方法示例：
# inference_model = BakedSHSpaceNetwork3D(model).to(device)
# outputs = inference_model(x_batch) # 这时的前向传播极快，且彻底去除了三角函数运算
if __name__ == '__main__':
    # 1. 载入 10,000 张独立测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =====================================================================
    # 🪐 一键无依赖读取：直接加载编译好的全包 pt 文件
    # =====================================================================
    model_path = "sh_routed_network.pt" # 换成你本地的文件名
    print(f"正在异步唤醒全包式编译球谐寻路大模型: {model_path} ...")
    
    # 独立免类声明直接加载
    model = torch.jit.load(model_path).to(device)
    model = BakedSHSpaceNetwork3D(model).to(device)
    model.eval()

    correct_predictions = 0
    total_predictions = 0
    
    print("\n[📊 测试集 10,000 张全新图片盲测大考开始...]")
    t1=time.time()
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            # 运行前向寻路推理
            test_output, _ = model(x_test)
            
            # 计算准确率
            _, predicted_label = torch.max(test_output, dim=1)
            correct_predictions += (predicted_label == y_test).sum().item()
            total_predictions += y_test.size(0)
    t2=time.time()
    duration=t2-t1
    accuracy_percentage = (correct_predictions / total_predictions) * 100.0
    print("\n" + "="*65)
    print("✨【测试大考最终成绩】")
    print(f"-> 独立编译模型: {model_path}")
    print(f"-> 全新测试集分类准确率: {accuracy_percentage:.2f}%")
    print(f"-> 总共用时{duration}s")
    print("="*65 + "\n")
