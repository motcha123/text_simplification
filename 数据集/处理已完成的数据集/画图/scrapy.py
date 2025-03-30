from matplotlib import pyplot as plt
import pandas as pd

plt.rc('font',family='Times New Roman')

fig, axes = plt.subplots(4, 4, sharey=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3},
                         figsize=(4, 4))

def sub_plot(axes, data=[[1,2],[3,4],[5,6],[7,8]], title=['','','',''], y_label=None, ylim=(0,1)):
    i = 0
    # axes[3].legend(['before', 'after'])
    axes[0].set_ylabel(y_label, rotation=0, fontsize=8, labelpad=20)
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
    sub_plot(axes[0], title=['HCSS', 'CSS', 'CSSWiKi', 'MCTS'], y_label='length', data=data['length'], ylim=(0, 80))
    sub_plot(axes[1], y_label='hsk', data=data['hsk'], ylim=(0, 5))
    sub_plot(axes[2], y_label='depth', data=data['depth'], ylim=(0, 5))
    sub_plot(axes[3], y_label='familiar', data=data['familiar'], ylim=(0, 1))

    # sub_plot(axes[0], y_label='pos', data=data['pos'], ylim=(0, 5))
    # sub_plot(axes[1], y_label='word_entropy', data=data['word_entropy'], ylim=(0, 6))
    # sub_plot(axes[2], y_label='dot_count', data=data['dot_count'], ylim=(0, 5))
    # sub_plot(axes[3], y_label='length_short', data=data['length_short'], ylim=(0, 20))
    plt.show()