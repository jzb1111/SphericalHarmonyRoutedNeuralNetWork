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
    def __init__(self, trained_model, grid_res=256):
        super().__init__()
        self.device = next(trained_model.parameters()).device
        self.non_linear_gate = nn.SiLU()
        self.grid_res = grid_res
        
        # 1. 继承原模型的轻量级坐标投影器与偏置
        self.route_projector1 = trained_model.layer1.route_projector
        self.route_projector2 = trained_model.layer2.route_projector
        self.route_projector3 = trained_model.layer3.route_projector
        
        self.bias1 = trained_model.layer1.bias
        self.bias2 = trained_model.layer2.bias
        self.bias3 = trained_model.layer3.bias
        
        # 获取各层的特征维度用来做前向计算
        self.in_dim_L1 = trained_model.layer1.in_features
        self.in_dim_L2 = trained_model.layer2.in_features
        self.in_dim_L3 = trained_model.layer3.in_features
        self.out_dim_L3 = trained_model.layer3.nex_activate_dim

        # =====================================================================
        # 🪐 2. 终极烘焙：将各层球谐场渲染为 100% 纯正的 [512, 512] 各向同性全景地图
        # =====================================================================
        print(f"[💾 正在唤醒终极双向流体烘焙，网格分辨率: {grid_res}x{grid_res} ...]")
        with torch.no_grad():
            c1 = trained_model.layer1.Sph_har_engine.sh_coeffs
            c2 = trained_model.layer2.Sph_har_engine.sh_coeffs
            c3 = trained_model.layer3.Sph_har_engine.sh_coeffs
            
            # 建立全景均匀网格（同时用于纬度 theta 和 经度 phi）
            grid_coord = (0.10 + 0.30 * torch.linspace(0, 1, self.grid_res, device=self.device)) * torch.pi
            
            # 广播生成各向同性的正方形交叉网格矩阵 [grid_res, grid_res]
            t_mat = grid_coord.unsqueeze(1).expand(self.grid_res, self.grid_res)
            p_mat = grid_coord.unsqueeze(0).expand(self.grid_res, self.grid_res)
            
            # 全并行计算全景基底场
            Y_grid = compute_explicit_sh_basis_4degree(t_mat.reshape(-1), p_mat.reshape(-1))
            
            # 将三层不同的球谐场系数直接烙印在 [512, 512] 的静态正方形全景画卷上！
            # 推理时它们是纯静态数据字典，超越函数计算量彻底为 0
            self.register_buffer('baked_map_L1', torch.matmul(Y_grid, c1).reshape(self.grid_res, self.grid_res))
            self.register_buffer('baked_map_L2', torch.matmul(Y_grid, c2).reshape(self.grid_res, self.grid_res))
            self.register_buffer('baked_map_L3', torch.matmul(Y_grid, c3).reshape(self.grid_res, self.grid_res))
            
        print("[✨ 全景双向流体烘焙大功告成！已固化 100% 纯动态路由查找表]")

    def _query_2d_baked_map(self, baked_map, row_keys, col_keys):
        """ 🚀 真正的完全体：双向动态连续坐标查表索引切片 """
        B = row_keys.shape[0]
        N = row_keys.shape[1]      #% 输入轴特征宽度
        M = col_keys.shape[1]      #% 输出轴特征宽度
        
        # 将行与列的双向连续值全部量化映射为全景图的离散网格整数索引
        row_idx = (row_keys * (self.grid_res - 1)).long().clamp(0, self.grid_res - 1) # [B, N]
        col_idx = (col_keys * (self.grid_res - 1)).long().clamp(0, self.grid_res - 1) # [B, M]
        
        # 扩展行索引与列索引，以便利用高级索引在 [512, 512] 图上瞬间切出 [B, N, M] 专属权重
        # 1. 抽取行：[512, 512] -> 根据 row_idx 抽取出 [B, N, 512]
        row_idx_expanded = row_idx.unsqueeze(-1).expand(-1, -1, self.grid_res)
        extracted_rows = torch.gather(baked_map.unsqueeze(0).expand(B, -1, -1), dim=1, index=row_idx_expanded)
        
        # 2. 抽取列：[B, N, 512] -> 根据 col_idx 抽取出 [B, N, M] 的终极动态权重
        col_idx_expanded = col_idx.unsqueeze(1).expand(-1, N, -1)
        dynamic_weight = torch.gather(extracted_rows, dim=2, index=col_idx_expanded)
        
        return dynamic_weight

    def forward(self, x):
        B, _ = x.shape
        
        # =====================================================================
        # --- Layer 1 高速推理 (Layer 1 的行是固定的 linspace 均匀纬度) ---
        # =====================================================================
        # 现场给第一层构造连续的固定行编码 [B, 784]
        l1_static_row = torch.linspace(0, 1, self.in_dim_L1, device=self.device).unsqueeze(0).expand(B, -1)
        layer1_keys = torch.sigmoid(self.route_projector1(x)) # [B, 32]
        
        W1 = self._query_2d_baked_map(self.baked_map_L1, l1_static_row, layer1_keys) # [B, 784, 32]
        res1 = torch.bmm(x.unsqueeze(1), W1).squeeze(1) * (1.0 / math.sqrt(self.in_dim_L1)) + self.bias1
        res1 = self.non_linear_gate(res1)
        
        # =====================================================================
        # --- Layer 2 高速推理 (🚨 真正的灵魂：行的纬度直接用上一层的 layer1_keys！) ---
        # =====================================================================
        layer2_keys = torch.sigmoid(self.route_projector2(res1)) # [B, 3] (或10)
        
        W2 = self._query_2d_baked_map(self.baked_map_L2, layer1_keys, layer2_keys) # [B, 32, 3]
        res2 = torch.bmm(res1.unsqueeze(1), W2).squeeze(1) * (1.0 / math.sqrt(self.in_dim_L2)) + self.bias2
        res2 = self.non_linear_gate(res2)
        
        # =====================================================================
        # --- Layer 3 高速推理 (🚨 行的纬度用上一层的 layer2_keys！) ---
        # =====================================================================
        layer3_keys = torch.sigmoid(self.route_projector3(res2)) # [B, 10]
        
        W3 = self._query_2d_baked_map(self.baked_map_L3, layer2_keys, layer3_keys) # [B, 3, 10]
        logits = torch.bmm(res2.unsqueeze(1), W3).squeeze(1) * (1.0 / math.sqrt(self.in_dim_L3)) + self.bias3
        keys=[layer1_keys,layer2_keys,layer3_keys]
        return logits,keys

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
