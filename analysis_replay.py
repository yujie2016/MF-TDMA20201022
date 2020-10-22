import numpy as np


# 分析重放攻击，先生成200 * 200 的相关矩阵，其中每一维代表10帧中0载波的一个时隙的调制后的数据

# 从replay-attack-muti.txt 文件中读取文件形成200个元素列表
def read_replay_data(file_name):
    f = open(file_name, "r")
    line = f.readline()
    frame_datas = []
    while line:
        data = line.split(":")
        # print(len(data))
        if len(data) == 2:
            frame_data = data[1]
            # 将字符串转换成list
            frame_data = frame_data[1:-2].split(",")
            temp_data = [float(e) for e in frame_data]
            # print(frame_data)
            frame_datas.append(np.array(temp_data))
        line = f.readline()
    f.close()
    return frame_datas



frame_datas = read_replay_data("./attack/replay-attack.txt")
# print(len(frame_datas))
# for i in range(len(frame_datas)):
#     print(frame_datas[i].shape[0])
# 规整数据为最大长度
def format_length(data):
    # 获取为度最大值
    frame_datas = []
    max_lengh = 0
    for i in range(len(data)):
        if data[i].shape[0] >= max_lengh:
            max_lengh = data[i].shape[0]
    # 将列表中的元素进行扩展
    # for j in range(len(data)):
    #     # print(data[j].shape)
    #     add_length = max_lengh - data[j].shape
    #     nums = np.zeros((add_length,))
    #     data = np.append(data[j], nums)
    #     frame_datas.append(data)
    # return frame_datas
    return max_lengh


max_length = format_length(frame_datas)
# print(max_length)
# 对数据进行扩展更新
for i in range(len(frame_datas)):
    add_length = max_length - frame_datas[i].shape[0]
    # print(add_length)
    nums = np.zeros((add_length,))
    data = np.append(frame_datas[i], nums)
    frame_datas[i] = data

# print(frame_datas[199].shape)


# 对数据进行相关行求取，同时除以最大样本点数做归一化处理
# print((frame_datas[199] * frame_datas[199]).sum()/max_length)
relative_list = []
print(len(frame_datas))
print(len(frame_datas[0]))
for i in range(len(frame_datas)):
    array_a = frame_datas[i]
    for j in range(len(frame_datas)):
        array_b = frame_datas[j]
        # 做向量乘法求相关性，并且做归一化处理，保留小数点后5为
        relate_num = round((array_a * array_b).sum()/max_length, 4)
        relative_list.append(relate_num)
print(len(relative_list))
relative_matrix = list(np.array(relative_list).reshape(200, 200))
# print(list(relative_matrix))
print(len(relative_matrix))
print(len(relative_matrix[0]))
# 保存相关矩阵
f = open("./attack/relative_matrix.txt", "w")
for i in range(len(relative_matrix)):
    f.write(str(list(relative_matrix[i])) + "\n")
f.close()

# 先计算每一时隙自相关数据的范围，再根据矩阵找到在此范围的数据的坐标，判断横纵坐标的对应关系，从而确定重放时隙。重放时隙要在原始时隙之后出现
# 1 计算原始时隙的自相关取值范围
self_relative = []
for i in range(len(frame_datas)):
    rela = round((frame_datas[i] * frame_datas[i]).sum()/max_length, 4)
    if rela != 0:
        self_relative.append(rela)

# print(min(self_relative))
top = max(self_relative)
bottom = min(self_relative)

# 在相关矩阵中找到范围内的所有值对应的横纵坐标
array = np.array(relative_list).reshape(200, 200)
pos = np.where((array >= bottom) & (array <=top))
# print(len(pos[0]))

for i in range(len(pos[0])):
    if pos[0][i] < pos[1][i]:
        # print("(" + str(pos[0][i]) + "," + str(pos[1][i]) + ")")
        print(str(pos[0][i]//20) + " - " + str(pos[0][i] % 20) + "------>" + str(pos[1][i]//20) + " - " + str(pos[1][i] % 20))