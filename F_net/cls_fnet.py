import torch
from torch import nn
import torch.nn.functional as F

# 假设我们有一个简单的文本分类数据集
batch_size = 32
seq_length = 128
dim = 512  # 每个token的嵌入维度

# 创建FNet模型
dim = 512
depth = 6
mlp_dim = 2048
dropout = 0.1

model = FNet(dim, depth, mlp_dim, dropout)

# 创建一个简单的分类头
class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # 假设我们取序列的第一个token的嵌入作为句子的表示
        x = x[:, 0]
        x = self.fc(x)
        return x

num_classes = 10  # 假设我们有10个类别
classification_head = ClassificationHead(dim, num_classes)

# 假设我们有输入数据
input_data = torch.randn(batch_size, seq_length, dim)

# 前向传播
output = model(input_data)
logits = classification_head(output)

# 计算损失
labels = torch.randint(0, num_classes, (batch_size,))
loss = F.cross_entropy(logits, labels)

# 反向传播和优化
loss.backward()
# optimizer.step()  # 假设我们已经定义了一个优化器

print("Loss:", loss.item())
