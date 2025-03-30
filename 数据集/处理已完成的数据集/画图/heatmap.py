import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math

# 配色 https://blog.csdn.net/qq_43201025/article/details/131043840
# 参数 https://blog.csdn.net/qq_43201025/article/details/131042439

def heatmap():
    excel = pd.read_excel('result_copy.xlsx')

    data = excel.iloc[:11, [1, 2, 3, 4]]

    print(data)

    # data.index = ['句子长度', 'hsk等级', '依存树深度', '词汇添加', '词汇删除', '词汇重排', '词汇熟悉度', 'POS熵',
    #               '词熵', '标点个数', '短句长度']
    data.index = ['词汇添加', '词汇删除', '词汇重排', 'hsk等级', '词汇熟悉度', '词熵', '句子长度', '依存树深度', 'POS熵',
                   '标点个数', '短句长度']

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    sns.heatmap(data=data,  # 指定绘图数据
                cmap='RdBu_r',  # 指定填充色
                linewidths=.1,  # 设置每个单元格边框的宽度
                annot=True,  # 显示数值
                vmax=1.5,
                vmin=0,
                fmt='.2f',
                annot_kws={"fontsize": 14}
                )

    plt.show()


if __name__ == '__main__':
    heatmap()
