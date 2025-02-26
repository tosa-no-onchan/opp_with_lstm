"""
test-model-freeze.py

https://jimmy-shen.medium.com/how-to-freeze-graph-in-tensorflow-2-x-3a3238c70f19

https://www.tensorflow.org/guide/saved_model?hl=ja

"""

import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import keras

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from configs import ModelConfigs
from tensorflow.python.framework import convert_to_constants

test_date="test_opp"
model_dir="Models/"+test_date
configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

input_model = "Models/test_opp/a.model"
output_model = "Models/test_opp/a.model.freeze.pb"

#to tensorflow lite
#converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
#tflite_quant_model = converter.convert()

LOAD_3=True         # use Keras saved model --> OK
                    #  ,but trained model with model.py use model = Model(inputs=inputs, outputs=output)
                    # Frozen model inputs: [<tf.Tensor 'x:0' shape=(1, 600, 122) dtype=float32>]
                    # Frozen model outputs: [<tf.Tensor 'Identity:0' shape=(1, 300, 53) dtype=float32>]
LOAD_4=False        # use TF saved model 
                    #  ,but trained model with model.py use model = MyModel(inputs=inputs, outputs=output)

LOAD_5=False        # use TF saved model  -->  OK
                    #  ,but trained model with model.py use model = Model(inputs=inputs, outputs=output)
                    # and 
                    # Frozen model inputs: [<tf.Tensor 'inputs:0' shape=(None, 600, 122) dtype=float32>]
                    # Frozen model outputs: [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

base_dir='Models/test_opp'

if LOAD_3==True:
        # こちらは、 cudnnRNNV3 を取り除けます。
        # Keras saved model with cudnn
        # You can train model with cudnn.
        # This converter can remove cudnnRNNV3 from trained model.
        # a.model_frozen.pb and a.model.tflite are not include cudnnRNNV3
        # https://www.tensorflow.org/lite/convert/concrete_function?hl=ja
        print("LOAD_3")

        # disble cudnn
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        model = keras.models.load_model(configs.model_path+'/a.model.keras',safe_mode=False)

        # 具象関数を Keras モデルから取得
        run_model = tf.function(lambda x : model(x))

        # 具象関数を保存
        concrete_func = run_model.get_concrete_function(
                tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        #tf.saved_model.save(model, configs.model_path+'/a.model_frozen.pb', concrete_func)

        # Get frozen ConcreteFunction    
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # [<tf.Tensor 'x:0' shape=(None, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

        print(">>> saved a.model_frozen.pb")
        tf.io.write_graph(graph_or_graph_def=constantGraph.graph, logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 

        # Get frozen ConcreteFunction    
        #frozen_func = convert_variables_to_constants_v2(full_model)    
        #frozen_func.graph.as_graph_def()
       
        #concrete_func.inputs[0].set_shape([1, 600, 122])
        #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        #tflite_model = converter.convert()

        print(">>> start tflite convert")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([constantGraph])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        print(">>> saved a.model.tflite")
        with open('a.model.tflite', 'wb') as f:
            f.write(tflite_model)


if LOAD_4==True:
        print("LOAD_4")
        # /home/nishi/kivy_env/lib/python3.10/site-packages/tensorflow/python/saved_model/load.py
        model = tf.saved_model.load(configs.model_path+'/a.model')

        # https://github.com/leimao/Frozen-Graph-TensorFlow/tree/master/TensorFlow_v2
        #  example_1.py

        #@tf.function(input_signature=[tf.TensorSpec(shape=configs.input_shape, dtype=tf.float32)])
        #def to_save(x):
        #        return model(x)

        f = model.to_save.get_concrete_function()
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(f)

        constantGraph.graph.as_graph_def()
        layers = [op.name for op in constantGraph.graph.get_operations()]
        #print(layers)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # <tf.Tensor 'x:0' shape=(1, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(1, 300, 53) dtype=float32>]

        #tf.io.write_graph(constantGraph.graph.as_graph_def(), <output_dir>, <output_file>) 
        #tf.io.write_graph(graph_or_graph_def=constantGraph.graph.as_graph_def(), logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 
        tf.io.write_graph(graph_or_graph_def=constantGraph.graph, logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 


if LOAD_5==True:        
        # Tensorflow saved model ->  OK
        print("LOAD_5")
        # https://www.tensorflow.org/lite/convert/concrete_function?hl=ja

        # https://github.com/leimao/Frozen-Graph-TensorFlow/tree/master/TensorFlow_v2
        #  example_1.py

        model = tf.saved_model.load(configs.model_path+'/a.model')
        concrete_func = model.signatures[
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # Get frozen ConcreteFunction    
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # [<tf.Tensor 'inputs:0' shape=(None, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]
        tf.io.write_graph(graph_or_graph_def=constantGraph.graph, logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 

