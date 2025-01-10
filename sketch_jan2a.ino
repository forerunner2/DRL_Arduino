/**********************************************************************
项目名称/Project          : 强化学习的机械实现
程序名称/Program name     : Arduino-side implementation
作者/Author               : 赵佳强
日期/Date                 : 2024/09/26
程序目的/Purpose          : 此程序为“强化学习的机械实现”项目的Arduino端代码。
该项目应用强化学习算法实现对振动系统的振动控制。
Arduino UNO板完成系统状态的读取与舵机控制，python完成智能决策(根据系统状态，输出舵机转动的角度)的过程。
python端和Arduino端的数据传输通过串口通信的方式实现。

-----------------------------------------------------------------------
该程序主要参考自太极创客团队制作的Arduino开发教程：
http://www.taichi-maker.com/homepage/reference-index/arduino-code-reference/ 
除此之外，还借助了chatgpt等人工智能工具协助完成代码编写
***********************************************************************/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Servo.h>
#include <math.h>

/*-------------------------------------------参数设置---------------------------------------- */
Adafruit_MPU6050 mpu;   
sensors_event_t accel, gyro, temp;  
Servo myServo;

#define steps 10
float amplitude; 
float phase;      
float frequency;  

unsigned long startTime = millis();  // 系统开始运动的时间


/*--------------------------------------------初始化--------------------------------------- */
void setup() {
  Serial.begin(9600);
  while (Serial.read()>=0) {
    // 初始化，清理串口缓存
  }

  if (!mpu.begin()) { 
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);  // 加速度计的量程8g
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);  // 陀螺仪的量程500。/s
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);  // 传感器的滤波带宽

  myServo.attach(9);  // 连接舵机到Arduino的9号引脚
  myServo.write(0); // 初始化舵机的位置
}


void loop() {
  
  /*-----------------------------------------发送数据---------------------------------------- */
  String states = "";
  int n = 0;

  while (n < steps) {
    mpu.getEvent(&accel, &gyro, &temp);
    float pitch = atan2(accel.acceleration.y, sqrt(accel.acceleration.x * accel.acceleration.x + accel.acceleration.z * accel.acceleration.z)) * 180 / PI; 
    // float roll = atan2(accel.acceleration.x, sqrt(accel.acceleration.y * accel.acceleration.y + accel.acceleration.z * accel.acceleration.z)) * 180.0 / PI;

    states += String(accel.acceleration.x) + "," + String(pitch) + ",";   // 拼接
    // delay(100);  
    n += 1;
  }

  states.remove(states.length() - 1);
  Serial.println(states);
  Serial.flush();


  /*----------------------------------------读取“动作”------------------------------------ */
  while (Serial.available() == 0) {
    // 等待 Python 端读取发送数据
  }

  String data = Serial.readStringUntil('\n');  // 从串口读取数据
  int commaIndex1 = data.indexOf(',');  // 找到第一个逗号的位置
  int commaIndex2 = data.indexOf(',', commaIndex1 + 1);  // 找到第二个逗号的位置

  if (commaIndex1 > 0 && commaIndex2 > commaIndex1) {
    // 提取并转换为浮值
    amplitude = data.substring(0, commaIndex1).toFloat();
    phase = data.substring(commaIndex1 + 1, commaIndex2).toFloat();
    frequency = data.substring(commaIndex2 + 1).toFloat();
  }


  /*---------------------------------------env.step()------------------------------------ */
  int m = 0;
  String next_states = "";

  while (m < steps) {
    unsigned long currentTime = millis();
    float time = (currentTime - startTime) / 1000.0; // 转换为秒
    float angle = amplitude * sin(2 * PI * frequency * time + phase);  // 计算正弦角度
    myServo.write(angle);  

    mpu.getEvent(&accel, &gyro, &temp);
    float next_pitch = atan2(accel.acceleration.y, sqrt(accel.acceleration.x * accel.acceleration.x + accel.acceleration.z * accel.acceleration.z)) * 180 / PI;
    // float next_roll = atan2(accel.acceleration.x, sqrt(accel.acceleration.y * accel.acceleration.y + accel.acceleration.z * accel.acceleration.z)) * 180.0 / PI;
    next_states += String(accel.acceleration.x) + "," + String(next_pitch) + ",";   // 拼接状态
    // delay(100);  
    m += 1;
  }
  next_states.remove(next_states.length() - 1);
  Serial.println(next_states);
  Serial.flush();

}

