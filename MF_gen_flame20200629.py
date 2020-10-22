# 根据子站生成的数据形成每帧的比特流文件

def generate_frame():
    f = open("./NCC/NCC.txt", "r")
    data_f = open("./NCC/Frame_data.txt", "w")
    line = f.readline()
    index = 0
    while line:
        # print(line)
        eles = line.split(":")
        # 确定 帧 的下标
        if eles[0] == "init":
            index = 0
        else:
            index = eles[0]
        rule = eval(eles[1])
        # 循环读取计划表中的值，再将生成的二进制数据写入文件中
        content = ""
        for i in range(len(rule)):
            for j in range(len(rule[i])):
                if isinstance(rule[i][j], int):
                    content += "_:"
                else:
                    file_name = ("./RCST/RCST-%s.txt" % rule[i][j])
                    f_temp = open(file_name, "r")
                    line_temp = f_temp.readline()
                    while line_temp:
                        sub_eles = line_temp.split("|")
                        sub_frame = sub_eles[0].split(":")[1]
                        sub_fr = sub_eles[2]
                        sub_slot = sub_eles[3]
                        sub_data = sub_eles[4]
                        if int(sub_frame) == int(index) and int(sub_fr) == i and int(sub_slot) == j:
                            content += (str(sub_data) + ":")
                        line_temp = f_temp.readline()
                    f_temp.close()
            content += "\n"
        data_f.write("frame:" + str(index) + "\n" + content)

        line = f.readline()
    f.close()
    data_f.close()

# generate_frame()

# 创建生成单用户MF-TDMA情况下的比特流数据，用于单用户的调制，从内容数据中直接读取载波、时隙和数据，直接循环生成内容单用户帧信息
def gen_ue_frame(uename):
    f = open("./RCST/RCST-"+ uename + ".txt", "r")
    filename = "./NCC/" "Frame_"+ uename +".txt"
    data_f = open(filename, "w")
    line = f.readline()
    # 先将数据读取，将帧数、波、时隙域数据进行统计
    data_info = []
    datas = []
    index = 0
    while line:
        line_data = line.split("|")
        # print(line_data)
        frame_num = int(line_data[0].split(":")[1])
        data_frq = int(line_data[2])
        data_slot = int(line_data[3])
        data = int(line_data[4])
        data_info.append((frame_num,data_frq,data_slot))
        datas.append(data)
        # print(frame_num,data_frq,data)
        line = f.readline()
        index += 1
    # print(data_info)
    # print(data)
    # print(index+1)
    for i in range((index+2)):
        content = ""
        frame_info = "frame:" + str(i) + "\n"
        for j in range(10):
            for k in range(20):
                data = ""
                if (i, j, k) in data_info:
                    index = data_info.index((i, j, k))
                    data = str(datas[index])
                else:
                    data = "_"
                content += data + ":"
            content += "\n"
        # print(frame_info + content)
        data_f.write(frame_info + content)
    f.close()
    data_f.close()

gen_ue_frame("UE9")