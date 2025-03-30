from matplotlib import pyplot as plt
import pandas as pd

# plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 定义图像的分布，如果是 4行4列，那么后面就应该只一次性生存四行的图像
fig, axes = plt.subplots(4, 4, sharey=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3},
                         figsize=(10, 8))

def sub_plot(axes, data=[[1,2],[3,4],[5,6],[7,8]], title=['','','',''], y_label=None, ylim=(0,1)):
    i = 0
    # axes[3].legend(['before', 'after'], loc=2)
    axes[0].set_ylabel(y_label, rotation=0, fontsize=14, labelpad=35)
    for axe in axes:
        axe.set_ylim(ylim)
        axe_bars = axe.bar([1, 1.25], data[i], width=0.2, edgecolor='black', color=['#f07673', '#7998ad'])
        # '#800020', '#dcd2c6' edgecolor='black',
        for bar in axe_bars:
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            axe.text(x, height+ylim[1]/10, f'{height.round(2)}', ha='center')

        axe.set_title(title[i], x=0.5, y=1.2)
        axe.spines['top'].set_visible(False)
        axe.spines['bottom'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.spines['left'].set_visible(False)
        axe.set_xticks([])
        axe.set_yticks([])

        i += 1

def data_dict_construct(path='前后数据对比.xlsx'):
    # 读取excel中的数据，组成带有数据的字典
    data = pd.read_excel(path)
    data_dict = {}
    i = 1
    while i < 19:
        key = data.columns.tolist()[i].replace('c_', '')
        data_list = []
        for data_c, data_s in zip(data[data.columns.tolist()[i]], data[data.columns.tolist()[i+1]]):
            data_list.append([data_c, data_s])
        data_dict[key] = data_list
        i += 2
    return data_dict

if __name__ == '__main__':
    data = data_dict_construct()

    # 按行生成图像
    sub_plot(axes[0], title=['血友病数据集', 'CSS', 'CSSWiKi', 'MCTS'], y_label='句子长度', data=data['length'], ylim=(0, 80))
    sub_plot(axes[1], y_label='hsk难度', data=data['hsk'], ylim=(0, 5))
    sub_plot(axes[2], y_label='依存树深度', data=data['depth'], ylim=(0, 7))
    sub_plot(axes[3], y_label='词汇熟悉度', data=data['familiar'], ylim=(0, 1))

    # sub_plot(axes[0], title=['血友病数据集', 'CSS', 'CSSWiKi', 'MCTS'], y_label='词性熵', data=data['pos'], ylim=(0, 5))
    # sub_plot(axes[1], y_label='词熵', data=data['word_entropy'], ylim=(0, 8))
    # sub_plot(axes[2], y_label='标点个数', data=data['dot_count'], ylim=(0, 5))
    # sub_plot(axes[3], y_label='短句长度', data=data['length_short'], ylim=(0, 20))

    plt.legend()
    plt.show()