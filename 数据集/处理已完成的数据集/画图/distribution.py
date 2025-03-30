import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import seaborn as sns


# 设置字体
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def distribution_two(name, param='familiar_ratio', scope=[0,4], my_dataset_name='血友病数据集', xlabel=None):
    # 二合一
    param = param
    bins = 50
    xlabel = xlabel
    font_size = 18
    a, b = scope[0], scope[1]

    data_my = '../abs_data/My_Dataset.xlsx'
    data_my = pd.read_excel(data_my)[param]
    y_my, bins_my, _ = plt.hist(data_my, density=True, bins=bins, range=(a, b))
    centers_my = (bins_my[:-1] + bins_my[1:]) / 2

    data_CSS = '../abs_data/CSS.xlsx'
    data_CSS = pd.read_excel(data_CSS)[param]
    y_CSS, bins_CSS, _ = plt.hist(data_CSS, density=True, bins=bins, range=(a, b))
    centers_CSS = (bins_CSS[:-1] + bins_CSS[1:]) / 2

    data_CSS_Wiki = '../abs_data/CSS_Wiki.xlsx'
    data_CSS_Wiki = pd.read_excel(data_CSS_Wiki)[param]
    y_CSS_Wiki, bins_csswiki, _ = plt.hist(data_CSS_Wiki, density=True, bins=bins, range=(a, b))
    centers_CSS_Wiki = (bins_csswiki[:-1] + bins_csswiki[1:]) / 2

    data_MCTS = '../abs_data/MCTS.xlsx'
    data_MCTS = pd.read_excel(data_MCTS)[param]
    y_MCTS, bins_mcts, _ = plt.hist(data_MCTS, density=True, bins=bins, range=(a, b))
    centers_MCTS = (bins_mcts[:-1] + bins_mcts[1:]) / 2

    df_heatmap = pd.DataFrame()
    df_heatmap.index = centers_my

    for index in range(len(centers_my)):
        centers_my[index] = round(centers_my[index], 2)

    df_heatmap[my_dataset_name] = y_my
    df_heatmap['CSSWiki'] = y_CSS_Wiki
    df_heatmap['CSS'] = y_CSS
    df_heatmap['MCTS'] = y_MCTS

    centers_MCTS, y_MCTS = smooth(centers_MCTS, y_MCTS)
    centers_CSS_Wiki, y_CSS_Wiki = smooth(centers_CSS_Wiki, y_CSS_Wiki)
    centers_CSS, y_CSS = smooth(centers_CSS, y_CSS)
    centers_my, y_my = smooth(centers_my, y_my)

    plt.cla()

    ax_1 = plt.subplot(211)
    ax_2 = plt.subplot(212)

    ax_1.plot(centers_my, y_my)
    ax_1.plot(centers_CSS_Wiki, y_CSS_Wiki)
    ax_1.plot(centers_CSS, y_CSS)
    ax_1.plot(centers_MCTS, y_MCTS)
    ax_1.set_ylabel('density', fontsize=font_size)
    ax_1.set_xlabel(xlabel, fontsize=font_size)
    ax_1.legend([my_dataset_name, 'CSSWiKi', 'CSS', 'MCTS'], fontsize=font_size)

    sns.heatmap(data=df_heatmap.T,  # 指定绘图数据
                cmap='RdBu_r',  # 指定填充色
                linewidths=.1,  # 设置每个单元格边框的宽度
                annot=False,  # 显示数值
                vmax=1.5,
                vmin=0,
                square=True,
                cbar=True,  # 是否绘制颜色条
                ax=ax_2,
                cbar_kws={'location': 'bottom', 'shrink': 0.5},
                annot_kws = {"fontsize": 14}
                )
    plt.xticks(rotation=0)

    plt.show()

def smooth(x, y):
    """
    用于平滑，接受原本的 x, y
    返回平滑后的 x, y
    """
    x_new = np.linspace(min(x), max(x), 300)
    y_new = make_interp_spline(x, y)(x_new)
    return x_new, y_new


if __name__ == '__main__':
    # 标点那边的比值要改一下
    # 要不要增加OOV比值？？？
    # distribution_two(param='add', name='add', scope=[0, 1], xlabel='词汇添加')
    # distribution_two(param='delete', name='delete', scope=[0, 1])
    # distribution_two(param='reorder', name='reorder', scope=[0, 1])

    distribution_two(param='hsk_ratio', name='HSK', scope=[0, 3.0])
    # distribution_two(param='familiar_ratio', name='familarity', scope=[0, 4.0])
    # distribution_two(param='word_entropy_ratio', name='word entropy', scope=[0, 2.0])

    # distribution_two(param='length_ratio', name='length', scope=[0, 2.5])
    # distribution_two(param='depth_ratio', name='D-Tree', scope=[0, 2.5])
    # distribution_two(param='pos_ratio', name='POS entropy', scope=[0, 2.0])
    # distribution_two(param='dot_count_ratio', name='punctuation', scope=[0, 4.0])
    # distribution_two(param='length_short_ratio', name='short sentence length', scope=[0, 2.5])

    """
    'length_ratio'
    'hsk_ratio'
    'familiar_ratio'
    'depth_ratio'
    'pos_ratio'
    'word_entropy_ratio'
    'add'
    'delete'
    'reorder'
    'dot_count_ratio'
    'length_short_ratio'
    """

