import torch
import torch.nn as nn


# 定义简单模型
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(10, 5)
        self.classifier = nn.Linear(5, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


model = DemoModel()

# 创建优化器（默认1个参数组）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 测试打印各参数组学习率
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i} lr: {group['lr']:.4f}")
