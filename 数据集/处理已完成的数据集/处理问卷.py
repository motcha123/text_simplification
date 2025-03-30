from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

path = r'D:\edge_download\问卷结果\问卷结果\分析\analyze.xlsx'

def paint_num():
    # 按数量画
    data = pd.read_excel(path, sheet_name=-1)
    A = data['A'].tolist()
    B = data['B'].tolist()
    C = data['C'].tolist()
    D = data['D'].tolist()

    more_B = 0
    more_A = 0
    equal_AB = 0
    for index in range(len(A)):
        if B[index] > A[index]:
            more_B += 1
        elif A[index] > B[index]:
            more_A += 1
        else:
            equal_AB += 1

    print('more_B', more_B, 'percentage', str(more_B/160 * 100)+'%')
    print('more_A', more_A, 'percentage', str(more_A/160 * 100)+'%')
    print('equal_AB', equal_AB, 'percentage', str(equal_AB/160 * 100)+'%')




    # 画图
    legends_list = ['A', 'B', 'C', 'D']
    data = [A, B, C, D]

    i = 0
    x = range(len(A))
    width = 1
    # 将bottom_y元素都初始化为0
    bottom_y = [0] * len(A)
    for y in data:
        plt.bar(x, y, width, bottom=bottom_y, label=legends_list[i])
        # 累加数据计算新的bottom_y
        bottom_y = [a + b for a, b in zip(y, bottom_y)]
        i += 1

        if i == 2:
            break

    plt.legend(fontsize=14)
    plt.xlabel('题目编号', fontsize=18)
    plt.ylabel('人数', fontsize=18)
    # plt.title('Stacked bar')
    plt.show()


def paint_per():
    # 按 A B 的比例画，且只统计 A B
    data = pd.read_excel(path, sheet_name=-1)
    A = data['A_per'].tolist()
    B = data['B_per'].tolist()

    more_B = 0
    more_A = 0
    equal_AB = 0
    for index in range(len(A)):
        if B[index] > A[index]:
            more_B += 1
        elif A[index] > B[index]:
            more_A += 1
        else:
            equal_AB += 1

    print('more_B', more_B, 'percentage', str(more_B / 160 * 100) + '%')
    print('more_A', more_A, 'percentage', str(more_A / 160 * 100) + '%')
    print('equal_AB', equal_AB, 'percentage', str(equal_AB / 160 * 100) + '%')

    # 画图
    legends_list = ['A', 'B']
    data = [A, B]

    x = range(len(A))
    width = 1
    # 将bottom_y元素都初始化为0
    bottom_y = [0] * len(A)
    i = 0
    for y in data:
        plt.bar(x, y, width, bottom=bottom_y, label=legends_list[i])
        # 累加数据计算新的bottom_y
        bottom_y = [a + b for a, b in zip(y, bottom_y)]
        i += 1


    plt.legend()
    plt.title('Stacked bar')
    plt.show()

if __name__ == '__main__':
    paint_num()
