import torch
import numpy as np

pred = np.array([[-0.4089, -1.2471, 0.5907],
                [-0.4897, -0.8267, -0.7349],
                [0.5241, -0.1246, -0.4751]])
label = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [1, 0, 1]])

pred = torch.from_numpy(pred).float()
label = torch.from_numpy(label).float()

## 通过BCEWithLogitsLoss直接计算输入值（pick）
crition1 = torch.nn.BCEWithLogitsLoss()
loss1 = crition1(pred, label)
print(loss1)

crition2 = torch.nn.MultiLabelSoftMarginLoss()
loss2 = crition2(pred, label)
print(loss2)

##  通过BCELoss计算sigmoid处理后的值
crition3 = torch.nn.BCELoss()
loss3 = crition3(torch.sigmoid(pred), label)
print(loss3)