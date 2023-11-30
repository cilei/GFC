from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")  # 创建SummaryWriter，为log文件起目录

for i in range(100):
    writer.add_scalar("y=2x",2*i,i)  # 第一个参数相当于标题，第二个参数就相当于纵坐标的值，第三个参数就相当于横坐标的值

writer.close()
