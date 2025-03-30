import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np


plt.rcParams['font.sans-serif'] = 'Times New Roman'

# matplotlib.rcParams['font.sans-serif'] = ['SimHei']

fontsize=16

# 变量类别个数
categories_num = 13

def process(data: list):
    # 把数据变成能画雷达图的形式
    data[0] = 1 / data[0]
    data[1] = 1 / data[1]
    data[2] = 1 / data[2]
    data[6] = data[6] / 5
    data[7] = 1 / data[7]
    data[8] = 1 / data[8]
    data[9] = data[9] / 1.1
    data[10] = 1 / data[10]
    data[11] = 1 / data[11]
    data[12] = 1 / data[12]
    data.append(data[0])

    return data

x_labels = [
'句子长度',  # 取倒数
'hsk等级',  # 取倒数
'依存树深度',  # 取倒数
'词汇添加',
'词汇删除',
'词汇重排',
'词汇熟悉度',
'POS熵',  # 取倒数
'词熵',  # 取倒数
'标点个数',
'短句长度',  # 取倒数
'oov比值',
'oov种类比值'
]

x_labels = [
'length',  # 取倒数
'hsk',  # 取倒数
'D-tree',  # 取倒数
'add',
'delete',
'reorder',
'familiar',
'POS entropy',  # 取倒数
'word entropy',  # 取倒数
'punctuation',
'short length',  # 取倒数
'oov number',
'oov word'
]

x_labels = []

# 设置每个点的角度值
angles = [n / float(categories_num) * 2 * pi for n in range(categories_num)]
angles += angles[:1]

# 读取数据
result = pd.read_excel('result.xlsx')
data_my = process(result['我的数据集'].tolist())
data_ifly_zero = process(result['ifly_zero'].tolist())
data_ali_zero = process(result['ali_zero'].tolist())

# 初始化极坐标网格
ax_1 = plt.subplot(111, polar=True)
# ax_1 = plt.subplot(121, polar=True)
# ax_2 = plt.subplot(122, polar=True)
# ax_3 = plt.subplot(223, polar=True)
# ax_4 = plt.subplot(224, polar=True)

# 设置x轴的标签
# plt.xticks(angles[:-1], x_labels, color='grey', size=8)

ax_1.set_rlabel_position(0)
ax_1.plot(angles, data_my, linewidth=1, color='b')
ax_1.fill(angles, data_my, 'b', alpha=0.1)
ax_1.set_xticks(angles[:-1])
ax_1.set_xticklabels(x_labels, fontsize=fontsize)
ax_1.set_ylim(0, 1.5)
ax_1.set_title('CHSS vs. iFLYTEK Spark', fontsize=fontsize)

ax_1.set_rlabel_position(0)
ax_1.plot(angles, data_ifly_zero, linewidth=1, color='r')
ax_1.fill(angles, data_ifly_zero, 'r', alpha=0.1)
ax_1.set_xticks(angles[:-1])
ax_1.set_ylim(0, 1.5)

# ax_2.set_rlabel_position(0)
# ax_2.plot(angles, data_my, linewidth=1, color='b')
# ax_2.fill(angles, data_my, 'b', alpha=0.1)
# ax_2.set_xticks(angles[:-1])
# ax_2.set_xticklabels(x_labels, fontsize=fontsize)
# ax_2.set_ylim(0, 1.5)
# ax_2.set_title('CHSS vs. Qwen', fontsize=fontsize)
#
# ax_2.set_rlabel_position(0)
# ax_2.plot(angles, data_ali_zero, linewidth=1, color='r')
# ax_2.fill(angles, data_ali_zero, 'r', alpha=0.1)
# ax_2.set_xticks(angles[:-1])
# ax_2.set_ylim(0, 1.5)

# ax_3.set_rlabel_position(0)
# ax_3.plot(angles, data_my, linewidth=1, color='b')
# ax_3.fill(angles, data_my, 'b', alpha=0.1)
# ax_3.set_xticks(angles[:-1])
# ax_3.set_xticklabels(x_labels)
# ax_3.set_ylim(0, 1.5)
# ax_3.set_title('my_dataset vs ifly')
#
# ax_3.set_rlabel_position(0)
# ax_3.plot(angles, data_ifly_zero, linewidth=1, color='r')
# ax_3.fill(angles, data_ifly_zero, 'r', alpha=0.1)
# ax_3.set_xticks(angles[:-1])
# ax_3.set_ylim(0, 1.5)

plt.show()
