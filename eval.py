# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:57:22 2026

@author: jzb
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
torch.set_default_device('cpu')
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
    model_path = "sh_routed_network - 32-10-10.pt" # 换成你本地的文件名
    print(f"正在异步唤醒全包式编译球谐寻路大模型: {model_path} ...")
    
    # 独立免类声明直接加载
    model = torch.jit.load(model_path).to(device)
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
