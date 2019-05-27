from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from utils.threshold import load_pkl


def argument_error_analysis(attention):

    tri_typess = ["Other", "Theme", "Cause", "Instrument", "ToLoc",
                  "FromLoc", "Site", "AtLoc", ]

    x=[1,2,3,4,5,6]
    y=[3,2,1]

    confusion = attention
    # confusion[0][0] = 300
    print(confusion)

    print(np.shape(confusion))
    # 热度图，后面是指定的颜色块，gray也可以，gray_x反色也可以
    plt.imshow(confusion, cmap=plt.cm.Reds, vmax=4)
    indices = range(len(confusion))
    indicesy = range(len(confusion[0]))
    # 坐标位置放入
    # 第一个是迭代对象，表示坐标的顺序
    # 第二个是坐标显示的数值的数组，第一个表示的其实就是坐标显示数字数组的index，但是记住必须是迭代对象
    plt.xticks(indicesy, x)
    plt.yticks(indices, y)
    # 热度显示仪？就是旁边的那个验孕棒啦
    # plt.colorbar()
    # 就是坐标轴含义说明了
    plt.xlabel('chemical mentions')
    plt.ylabel('disease mentions')
    # 显示数据，直观些
    # for first_index in range(len(confusion)):
    #     for second_index in range(len(confusion[first_index])):
    #         plt.text(first_index, second_index, confusion[first_index][second_index])

    # 显示
    plt.show()

def topk():
    x=[100,200,300,400,500,600]
    y=[57.55,59.01,56.46,60.09,58.38,56.92]
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("K")  # X轴标签
    plt.ylabel("F(%)")  # Y轴标签
    plt.title("the performance comparison")  # 图标题
    plt.show()  # 显示图
    plt.savefig("topk_k.jpg")  # 保存图


if __name__ == "__main__":
    attention = load_pkl('entity_align')
    # temp=[2.702659,3.7815566,3.170545 ]
    # a.append(temp)
    # temp = [1.0777982,1.203137,1.1610005]
    # a.append(temp)
    # temp = [1.0904192,0.62009156,1.1182997]
    # a.append(temp)
    # temp = [3.3131166,4.273107,3.8353167]
    # a.append(temp)
    # temp = [2.820662,2.4218264,3.0692577]
    # a.append(temp)
    # temp = [1.5665073,1.5812267,1.740691]
    a=[]
    temp = [3.170545, 1.1610005, 1.1182997, 3.8353167, 3.0692577, 1.740691]
    a.append(temp)
    temp=[3.7815566,1.203137,0.62009156,4.273107,2.4218264,1.5812267]
    a.append(temp)

    temp = [2.702659, 1.0777982, 1.0904192, 3.3131166, 2.820662, 1.5665073]
    a.append(temp)
    print(a)

    argument_error_analysis(a)

    # topk()