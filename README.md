# opp_with_lstm  
Obstacl Path Planner with Lstm - ROS2
##### [ROS2 自作 Turtlebot3 による 草刈りロボット開発。#9 LSTM で経路計画をする。](http://www.netosa.com/blog/2024/11/ros2-turtlebot3-9-lstm.html)  
![inferencModel result](https://github.com/tosa-no-onchan/opp_with_lstm/blob/main/work/v_245.jpg)  
![inferencModel result2](https://github.com/tosa-no-onchan/opp_with_lstm/blob/main/work/v_6.jpg)  

#### 1. PC environment  
PC: Ubuntu Mate 22.04  
Python 3.10.12  
virtual_env  
tensorflow 2.16.2  
keras 3.7.0  

or  
Ubuntu Mate 24.04  
virtual_env  
tensorflow 2.18.0  

#### 2. download  
   $ git clone https://github.com/tosa-no-onchan/opp_with_lstm.git
   
   pre trained frozen model extract  
   $ cd opp_with_lstm/Models/test_opp  
   $ unzip a.model_frozen.pb.zip     ---> a.model_frozen.pb   

#### 4. Trainning
    
    $ python train.py

#### 4.1 Trainning with Tensorflow 2.18.0
    
    $ export WRAPT_DISABLE_EXTENSIONS=true  
    $ python train.py

#### 5. Freeze  

    $ python test-model-freeze.py  
    Models/test_opp/a.model_frozen.pb  ---> freozen model  

#### 6. predict
    
    $ python inferencModel.py

#### 7. Original Trainning Data generate  
   [cource_classgen.cpp](https://github.com/tosa-no-onchan/opp_with_transformer_cpp)  
   
   test data of mine  
   
   [test data](https://huggingface.co/datasets/tosa-no-onchan/opp)


#### 8. refference  
   1) Trainning data set by mysel  
   [huggingface.co/datasets/tosa-no-onchan/opp](https://huggingface.co/datasets/tosa-no-onchan/opp)  
   
   2) Original speech to text LSTM model  
   [Introduction to speech recognition with TensorFlow](https://pylessons.com/speech-recognition)

#### 9. YouTube に動画をアップしました。  

[opp_with_lstm デモ](https://www.youtube.com/watch?v=PXCq2HicOwA)   
[![alt設定](http://img.youtube.com/vi/PXCq2HicOwA/0.jpg)](https://www.youtube.com/watch?v=PXCq2HicOwA)

#### 10. Run with Ros2 Robot  

[turtlebot3_navi_my for Ros2 humble r6](https://github.com/tosa-no-onchan/turtlebot3_navi_my)  
2025.1.3 Version r6 AutoMower で、Obstacle path planner with Lstm tflite を使えるようにしました。
