"""
test-convert2tflite.py

https://medium.com/axinc/keras%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92tflite%E3%81%AB%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B-e8f5a1dd7ad5

https://www.tensorflow.org/lite/convert/python_api?hl=ja
RNN の場合
https://www.tensorflow.org/lite/convert/rnn?hl=ja


"""

import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

import keras


#to tensorflow lite
#converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
#tflite_quant_model = converter.convert()

from inferencModel import wrap_frozen_graph

tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":

    LOAD_1=False       # exporte model OK
    LOAD_2=False       # tensorflow saved model OK
    LOAD_3=True        # Keras seved model OK
    LOAD_4=False       # NG

    input_model = "Models/test_opp/a.model"
    export_model = "Models/test_opp/export.model"
    output_model = "Models/test_opp/a.model.tflite"

    base_dir='Models/test_opp'

    if LOAD_1 == True:
        # こちらは、下記で、 CudnnRNNV3 を取り除けない!!
        # use_cudnn=False で、 trainning しないといけない。
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "-1"

        # Convert the model.
        #converter = TFLiteConverter.from_saved_model(saved_model_dir)
        converter =  tf.lite.TFLiteConverter.from_saved_model(export_model)
        #converter.experimental_enable_resource_variables=True
        # For example, converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open("a.model.tflite", "wb") as f:
                f.write(tflite_model)

    if LOAD_2==True:
        print("LOAD_2")
        # こちらは、下記で、 CudnnRNNV3 を取り除けない!!
        # use_cudnn=False で、 trainning しないといけない。

        # /home/nishi/kivy_env/lib/python3.10/site-packages/tensorflow/python/saved_model/load.py
        # https://stackoverflow.com/questions/70250454/how-to-convert-frozen-graph-to-tensorflow-lite

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = tf.saved_model.load(base_dir+'/a.model')
        #converter = tf.lite.TFLiteConverter.from_saved_model(base_dir+'/a.model')
        #tflite_model = converter.convert()

        #model = tf.saved_model.load(export_dir)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concrete_func.inputs[0].set_shape([1, 600, 122])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()

        with open('a.model.tflite', 'wb') as f:
            f.write(tflite_model)

    if LOAD_3 == True:
        # こちらは、 cudnnRNNV3 を取り除けます。
        # Keras saved model with cudnn
        # You can train model with cudnn.
        # This converter can remove cudnnRNNV3 from trained model.
        # a.model_frozen.pb and a.model.tflite are not include cudnnRNNV3
        # https://www.tensorflow.org/lite/convert/concrete_function?hl=ja

        print("LOAD_3")
        from tensorflow.python.framework import convert_to_constants

        # disble cudnn
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        #os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "-1"

        model = keras.models.load_model('Models/test_opp/a.model.keras',safe_mode=False)

        # 具象関数を Keras モデルから取得
        run_model = tf.function(lambda x : model(x))

        # 具象関数を保存
        concrete_func = run_model.get_concrete_function(
                tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction    
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # [<tf.Tensor 'x:0' shape=(None, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

        print(">>> start tflite convert")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([constantGraph])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        print(">>> saved a.model.tflite")
        with open('a.model.tflite', 'wb') as f:
            f.write(tflite_model)

    if LOAD_4==True:
        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile(base_dir+"/a.model_frozen.pb", "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())


        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["x:0"],        # keras frozen model
                                        #inputs=["inputs:0"],    # tf frozen model
                                        outputs=["Identity:0"],
                                        print_ops=True)


        # Import the graph into a new TensorFlow session
        with tf.compat.v1.Session() as sess:
                tf.import_graph_def(graph_def, name='')

        input_s = tf.TensorSpec([None,600,122],tf.float32)
        # Convert the TensorFlow graph to TensorFlow Lite format
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_s], ["Identity:0"])
        tflite_model = converter.convert()
                
