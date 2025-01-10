"""
autor: Jiaqiang Zhao

time: 24/9/20 20:26
主要编写与环境交互相关的函数，数据传输部分因为读取和传输的都是字符串，并不高效，可以修改为结构体

再加一个急停函数
加一个判断是否停止控制的判断
还未完成修改

"""
import serial
import torch
import time
from torch import Tensor


# from config import Config
# argparse = Config()


def reward_compute(state: Tensor):
    return -torch.sqrt((state ** 2).sum())


def link_serial(port, rate, times):
    """
    :param port: COM端口号
    :param rate: 波特率
    :param times: 延时时间
    :return: 串口对象Arduino,并打印"串口连接初始化完成!!!"
    """
    Arduino = serial.Serial(port, rate, timeout=times)
    time.sleep(times)
    print("串口连接初始化完成!!!")
    return Arduino


def read_state(arduino, state_size) -> Tensor:
    """
    :param state_size: 状态数据形状(state_dim,)
    :param arduino: arduino串口对象
    :return: 系统状态，(a,h)
    """
    while True:
        if arduino.in_waiting > 0:
            break

    sensor_data = arduino.readline().decode('utf-8').strip()  # 读取 Arduino 发来的数据,字符串类型
    data = sensor_data.split(',')  # 按逗号分隔成多个部分，返回一个字符串列表["a1","y1"……,"an","yn"]
    print(f"已接收数据:{data},类型{type(data)}")
    data_float = [float(i) for i in data]  # 将字符串转换为浮点数
    state = torch.tensor(data_float, dtype=torch.float).view(1, 2 * 10)  # 这儿是一个bug，无法调用config.py

    return state


def env_step(arduino, action):  # 返回值，还需要加上是否终止控制的信号
    """
    :param arduino: arduino串口对象
    :param action: actor网络输出的动作，类型Tensor(torch.float),形状(,)？？？？
    :return: 返回后立马作用于系统，得到的新状态
    """
    # 1.数据发送
    amplitude, phase, frequency = action[0]  # action[0]形状(1,action_dim)
    arduino.write(f"{amplitude},{phase},{frequency}\n".encode('utf-8'))  # 以UTF-8格式编码的字节流表示字符串
    print("已发送！！！")

    # 2.等待系统响应，要返回确定舵机执行了相关动作的反馈消息
    while True:
        if arduino.in_waiting > 0:
            break
    print("舵机已经转动相应的角度")

    # 3.返回状态观测值
    sensor_data = arduino.readline().decode('utf-8').strip()  # 读取 Arduino 发来的数据,字符串类型
    data = sensor_data.split(',')
    print(data)
    data_float = [float(i) for i in data]
    next_state = torch.tensor(data_float, dtype=torch.float).view(1, 2 * 10)

    return next_state


if __name__ == "__main__":
    arduino = link_serial("COM8", 9600, 1)
    state = read_state(arduino, 20)
    print(state)
    print(state.shape)
    action = torch.tensor([[10, 10, 10]], dtype=torch.float)
