import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from utils.deeplearning import seg_model
from utils import train_net
from dataset import MyDataset
from dataset import train_transform, val_transform

from ai_hub import inferServer
import json
import base64
import cv2
from io import BytesIO
from torch.cuda.amp import autocast
from torch.autograd import Variable as V
import segmentation_models_pytorch as smp

class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

    #数据前处理
    def pre_process(self, data):
        #json process
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        bytesIO = BytesIO()

        img = Image.open(BytesIO(bytearray(img)))
        img=np.array(img)
        img = img.astype(np.float32)

        transform=val_transform
        img = transform(image=img)['image']
        img=img.unsqueeze(0)
        return img
    
       #数据后处理
    def post_process(self, data):
        pred = data.squeeze().cpu().data.numpy()
        pred = np.argmax(pred,axis=0)
        pred = np.uint8(pred+1)

        # pred_save=Image.fromarray(pred)
        # pred_save=pred_save.convert('L')
        # pred_save.save('./test.png')

        # print('pred:', pred)
        pred = cv2.imencode('.png', pred)[1]
        img_encode = np.array(pred).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data,'utf-8')
        return bast64_str

    #模型预测：默认执行self.model(preprocess_data)，一般不用重写,如需自定义，可覆盖重写
    def predict(self, data):
        with torch.no_grad():
            ret = self.model(data.cuda())
            return ret

# 训练
if __name__=="__main__":
    param = {}
    Image.MAX_IMAGE_PIXELS = 1000000000000000

    #指定GPU
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # device_ids=[0, 1]
    #不指定，使用全部
    device_ids = [i for i in range(torch.cuda.device_count())]

    model = seg_model()
    param['model_name'] = model.model_name
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    # 参数设置
    param['epochs'] = 40       # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
    param['batch_size'] = 64    # 批大小
    param['lr'] = 8e-3         # 学习率
    param['gamma'] = 0.94        # 学习率衰减系数
    param['step_size'] = 1        # 学习率衰减起始epoch
    param['momentum'] = 0.5       # 动量
    param['weight_decay'] = 5e-4    # 权重衰减
    param['disp_inter'] = 1       # 显示间隔(epoch)
    param['save_inter'] = 10       # 保存间隔(epoch)
    param['iter_inter'] = 150     # 显示迭代间隔(batch)
    param['min_inter'] = 20
    param['save_log_dir'] = os.path.join('save/log', param['model_name'])      # 日志保存路径
    param['save_ckpt_dir'] = os.path.join('save/model', param['model_name'])    # 权重保存路径
    param['load_ckpt_dir'] = None

    # 准备数据集
    train_dir = './data_sets/train/'
    val_dir = './data_sets/val/'
    train_data = MyDataset(train_dir, upsamp = 0, transform=train_transform)
    valid_data = MyDataset(val_dir, upsamp = False, transform=train_transform)
    train_net(param, model, train_data, valid_data, device=device_ids)#local参数为本地训练时，保存模型和可视化