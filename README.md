# opp_with_lstm  
##### [ROS2 自作 Turtlebot3 による 草刈りロボット開発。#9 LSTM で経路計画をする。](http://www.netosa.com/blog/2024/11/ros2-turtlebot3-9-lstm.html)  

#### 1. 開発環境  
PC: Ubuntu Mate 22.04  
Python 3.10.12  
virtual_env  
tensorflow 2.16.2  
keras 3.6.0  

#### 2. download  
   $ git clone https://github.com/tosa-no-onchan/opp_with_lstm.git
   
   pre trained model extract  
   $ cd opp_with_lstm/Models/test_opp  
   $ unzip a.model.zip 

#### 4. 学習。
    
    $ python train.py

#### 5. predict
    
    $ python inferencModel.py

#### 6. refference  
   1) Trainning data set by mysel  
   [huggingface.co/datasets/tosa-no-onchan/opp](https://huggingface.co/datasets/tosa-no-onchan/opp)  
   
   2) Original speech to text LSTM model  
   [Introduction to speech recognition with TensorFlow](https://pylessons.com/speech-recognition)

#### 7. YouTube に動画をアップしました。  

[opp_with_lstm デモ](https://www.youtube.com/watch?v=PXCq2HicOwA)   
[![alt設定](http://img.youtube.com/vi/PXCq2HicOwA/0.jpg)](https://www.youtube.com/watch?v=PXCq2HicOwA)


