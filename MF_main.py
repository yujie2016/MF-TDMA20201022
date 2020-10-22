from MF_mainstation_class import *
from MF_substation_class import *
import time
from MF_gen_flame import *
from QPSK import *
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# # 定义载频的带宽，单位是hz
# BAND_WIDTH = 2 * 1000 * 1000 * 1000
#
# # 定义载频个数
# FREQUENCY_NUM = 8
#
# # 计算每个载频的带宽，单位是hz
# FREQUENCY_WIDTH = BAND_WIDTH / FREQUENCY_NUM
#
# # 定义帧时长，时间为s
# FLAME_TIME = 0.128
#
# # 定义帧中包含的时隙个数
# SLOT_NUM = 128
#
# # 计算每个时隙的时长，以及时隙每秒包含的数据量
# SLOT_TIME = FLAME_TIME / SLOT_NUM
# SLOT_COUNT = 1 / SLOT_TIME
#
# # 计算时隙中包含的数据量 ,单位是bit
# SLOT_DATA = BAND_WIDTH / SLOT_COUNT
#
# # 在时隙之间进行跳频
# # 因为 载频带宽 远大于 符号的传输速率，则认为载频之间近似正交
#
# # 定义链路层原始数据传输速率，单位为bit
# DATA_SPEED = 128

# 一般情况下，子站数量比跳频个数要多，图示整个接入过程。主要用于时钟控制


# 定义网络，需要一个主站，控制20个子站，同时将资源分配划分为10个载波，一帧划分为20个时隙
# 定义一个主站
maintain = Mainstation(20, 10, 20)
# 循环定20个主站，其中物理层的最大传输单元和传输速率
for i in range(20):
    index = i + 1
    username = "UE" + str(index)
    MTU = 8 * random.randint(5, 10)
    rate = MTU * random.randint(13, 15) + random.randint(0, 7)
    substation = Substation(username, MTU, rate)
    substation.priority = random.randint(1, 5)
    substation.receive_rules(maintain.rules, maintain.receipt)
    maintain.add_substation(substation)

# print(maintain.substations[10].name)

# 创建一个恶意子站
# username = "UEevil"
# MTU = 8*random.randint(5, 10)
# rate = MTU * random.randint(9, 10) + random.randint(0, 7)
# evil_substation = Substation(username, MTU, rate)
# evil_substation.receive_rules(maintain.rules, maintain.receipt)
# evil_substation.status = 0
# evil_substation.priority = 4


# 定义交互流程的函数，入参为当前帧需要入网的个数，保证子站随机入网
# 改进式子站随机入网函数
def new_init_action(maintain, num, batch):
    w = 0
    try_time = 0
    while w < num:
        if try_time == 50:
            break
        r = random.randint(0, 19)
        if maintain.substations[r].status == 0:
            maintain.substations[r].status = 1
            w += 1
        else:
            try_time += 1
            continue


# 修改子站状态变更
def status_change(maintain):
    for ww in range(len(maintain.substations)):
        if maintain.substations[ww].status == 1:
            maintain.substations[ww].status = 2
        elif maintain.substations[ww].status == 2:
            maintain.substations[ww].status = 3
        elif maintain.substations[ww].status == 3:
            maintain.substations[ww].status = 0


# 每个子站更新rule
def sub_update_rules(maintain):
    for i in range(len(maintain.substations)):
        maintain.substations[i].receive_rules(maintain.rules, maintain.receipt)


def find_sub(name):
    str = name.split("UE")[1]
    pos = int(str) - 1
    return pos


def find_status(maintain, pos):
    str = ""
    if maintain.substations[pos].status - 1 == 1:
        str = "login"
    elif maintain.substations[pos].status - 1 == 2:
        str = "control"
    return str


def pop(maintain, name):
    n = np.mat(maintain.rules)
    pos = np.argwhere(n == name)
    for i in range(len(pos)):
        maintain.rules[pos[i][0]][pos[i][1]] = 0


def clear_data(maintain, batch, name):
    a = ""
    file_name = ("./RCST/RCST-%s.txt" % name)
    f_temp = open(file_name, "r")
    line_temp = f_temp.readline()
    while line_temp:
        sub_eles = line_temp.split("|")
        sub_frame = sub_eles[0].split(":")[1]
        if sub_frame != batch:
            a += line_temp
        else:
            continue
    w_temp = open(file_name, "w")
    w_temp.write(a)


def write_receipt_file(name,content):
    f = open("./response/RCST-" + name + ".txt", "a+")
    f.writelines(content)
    f.close()


# 每一个子站进行rules解析
def profile_r(maintain, batch):
    for r in range(len(maintain.substations)):
        maintain.substations[r].receive_rules(maintain.rules, maintain.receipt)
        maintain.substations[r].profile_rule(batch)
        # maintain.update_rule_by_substation(maintain.substations[r])
        sub_update_rules(maintain)
# 对恶意子站的编辑
#     if evil_substation.status == 1:
#         # print("test1")
#         evil_substation.signal_frequency = 0
#         evil_substation.signal_slot = random.randint(0, 19)
#         if maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] == 1 or \
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] == 2:
#             maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = evil_substation.name
#         else:
#             old_name = maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot]
#             sub_index = find_sub(old_name)
#             if evil_substation.priority > maintain.substations[sub_index].priority:
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = evil_substation.name
#                 print(find_status(maintain, sub_index) + maintain.substations[sub_index].name + " connect failed")
#                 maintain.substations[sub_index].status = maintain.substations[sub_index].status - 1
#             else:
#                 new_name = maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] + "+UEevil"
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = new_name
#         # 把对应的时隙更新为子站名
#         login_fpdu = evil_substation.source_data.logon_FPDU()
#         content = "batch" + ":" + str(batch) + "|status:" + str(evil_substation.status) + "|" + str(
#             evil_substation.signal_frequency) + "|" + str(evil_substation.signal_slot) + "|" + login_fpdu + "\n"
#         evil_substation.write_file(content)
#         evil_substation.status = 2
#     elif evil_substation.status == 2:
#         # print("test2")
#         evil_substation.signal_frequency = 0
#         evil_substation.signal_slot = random.randint(0, 19)
#         if maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] == 1 or \
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] == 2:
#             maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = evil_substation.name
#         else:
#             old_name = maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot]
#             sub_index = find_sub(old_name)
#             if evil_substation.priority > maintain.substations[sub_index].priority:
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = evil_substation.name
#                 print(find_status(maintain, sub_index) + maintain.substations[sub_index].name + " connect failed")
#                 maintain.substations[sub_index].status = maintain.substations[sub_index].status - 1
#             else:
#                 new_name = maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] + "+UEevil"
#                 maintain.rules[evil_substation.signal_frequency][evil_substation.signal_slot] = new_name
#         control_fpdu = evil_substation.source_data.control_FPDU()
#         content = "batch" + ":" + str(batch) + "|status:" + str(evil_substation.status) + "|" + str(
#             evil_substation.signal_frequency) + "|" + str(evil_substation.signal_slot) + "|" + control_fpdu + "\n"
#         evil_substation.write_file(content)
#         evil_substation.status = 3
#     elif evil_substation.status == 3:
#         # print("test3")
#         slot_num = math.ceil(evil_substation.data_length / evil_substation.phy_mtu)
#         # print("slot_num:")
#         print(" " + evil_substation.name + "占用时隙数：" + str(slot_num))
#         m = np.mat(maintain.rules[1:])
#         m = np.transpose(m)
#         init_index = 0
#         for j in range(len(m)):
#             # fr_index = random.randint(0, 8)
#             fr_index = 1
#             while fr_index:
#                 # print("test")
#                 if maintain.rules[fr_index][j] == 0:
#                     maintain.rules[fr_index][j] = evil_substation.name
#                 else:
#                     old_name = maintain.rules[fr_index][j]
#                     sub_index = find_sub(old_name)
#                     if evil_substation.priority > maintain.substations[sub_index].priority:
#                         maintain.rules[fr_index][j] = evil_substation.name
#                         pop(maintain, old_name)
#                         print(old_name + " connect failed")
#                         maintain.substations[sub_index].status = 3
#                     else:
#                         new_name = maintain.rules[fr_index][j] + "+UEevil"
#                         maintain.rules[fr_index][j] = new_name
#                 init_index += 1
#                 fr_index += 1
#                 if fr_index == 10:
#                     break
#                 if init_index == slot_num:
#                     break
#             if init_index == slot_num:
#                 break
#         # evil_substation.receive_rules(rule)
#         data_fpdu = evil_substation.source_data.data_FPDU(evil_substation.phy_mtu, evil_substation.data_length)
#         data_length = len(data_fpdu)
#         # 载波频率固定为10，range参数需要修改
#         length_index = 0
#         for i in range(10):
#             if i == 0:
#                 continue
#             else:
#                 j = evil_substation.new_find_position(evil_substation.rules[i], evil_substation.name)
#                 if j != "":
#                     evil_substation.data_frequency = i
#                     evil_substation.data_slot = j
#                     content = "batch" + ":" + str(batch) + "|status:" + str(evil_substation.status) + "|" + str(
#                         evil_substation.data_frequency) + "|" + str(evil_substation.data_slot) + "|" + data_fpdu[length_index] + "\n"
#                     # length_index += 1
#                     evil_substation.write_file(content)
#                 else:
#                     continue
#         evil_substation.status = 1
#     else:
#         return
#

# 判断计数，按照10s一帧为一个周期
batch = 0
# 定义一帧的时长
flame_time = 2
# 计算一个时隙的时长，20个用户时隙 和 1个主站下发信令时隙
slot_time = flame_time / 21
print("init:")
print("子站申请：")
for i in range(len(maintain.rules)):
    print(maintain.rules[i])
print("主站回执：")
for i in range(len(maintain.receipt)):
    print(maintain.receipt[i])
print('\n')
while True:
    batch += 1
    # ******************定义一帧内的操作交互操作********************
    maintain.clear_rules()
    maintain.clear_receipt()
    # 定义子站在每帧中登陆的个数，其满足泊松分布，其均值为5
    # num = np.random.poisson(10, 1)[0]
    num = 6
    # num = 4
    print("The flame sequence is :" + str(batch) + " and the active num is:" + str(num))
    # 根据num数初始化入网子站的状态 状态从0变为1,并且解析规则
    new_init_action(maintain, num, batch)

    c1 = ""
    c2 = ""
    c3 = ""
    for i in range(len(maintain.substations)):
        if maintain.substations[i].status == 1:
            c1 += " " + maintain.substations[i].name
        elif maintain.substations[i].status == 2:
            c2 += " " + maintain.substations[i].name
        elif maintain.substations[i].status == 3:
            c3 += " " + maintain.substations[i].name
    print(c1 + "：发送登陆突发")
    print(c2 + "：发送控制突发")
    print(c3 + "：发送数据突发")

    # 子站解析规则
    profile_r(maintain, batch)
    ncc_content = str(batch) + ":" + str(maintain.rules) + "\n"
    maintain.write_file(ncc_content)
    # 时隙计划表
    print("子站申请：")
    for i in range(len(maintain.rules)):
        print(maintain.rules[i])
    one_frame(batch)
    modulate(batch)
    # 主站解析数据
    print("主站解析：")
    demodulate(batch)
    receive_info = maintain.analysis(batch)
    maintain.write_receive_info(receive_info, batch)
    # 主站回执
    maintain.send_receipt(batch)
    print("主站回执：")
    for i in range(len(maintain.receipt)):
        print(maintain.receipt[i])
    # print('\n')
    receipt_content = str(batch) + ":" + str(maintain.receipt) + "\n"
    maintain.write_receipt_file(receipt_content)
    one_response_frame(batch)
    response_modulate(batch)
    # 子站接收
    response_demodulate(batch)
    print("子站接收：")
    response_analysis(batch)
    for i in range(10):
        for j in range(20):
            if not isinstance(maintain.receipt[i][j], int):
                id = int(maintain.receipt[i][j].strip("UE")) - 1
                # print(maintain.receipt[i][j] + " " + str(maintain.substations[id].status))
                if maintain.substations[id].status == 3 or maintain.substations[id].status == 0:
                    maintain.substations[id].status = 0
                else:
                    permission = maintain.substations[id].receive_receipt(batch)
                    if permission == 1:
                        maintain.substations[id].status += 1
                    else:
                        print(maintain.substations[id].name + " retry...")
    print('\n')


    # ***********************************************************
    # 延时一帧时长
    time.sleep(flame_time)
    if batch == 20:
        break

# 生成二进制数据的帧汇总文件
new_generate_frame()

# response_frame()





