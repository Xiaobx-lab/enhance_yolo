from pyexpat import model
import torch
from nets.yolo import YoloBody
import numpy as np

if __name__ == "__main__": 
    model = YoloBody(3,3,anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=20, bilinear=False)
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model_data/yolo_weights.pth'
    model_dict      = model.state_dict() # 读取当前模型参数
    pretrained_dict = torch.load(model_path, map_location = device) # 读取预训练模型
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict) # 是用预训练的模型更新当前参数
    model.load_state_dict(model_dict) # 加载模型参数

    # 显示没有匹配上的key
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    x = torch.Tensor(1,3,416,416)
    outputs = model(x)