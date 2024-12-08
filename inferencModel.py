# opp_with_lstm/inferencModel.py
import os
import sys
import typing
import numpy as np

# add by nishi 2024.11.30
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf

#from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder

#from train import WavReaderMel

from configs import ModelConfigs

import keras
#from tensorflow import keras 

#from keras.models import load_model
#from tensorflow.keras.models import load_model

#keras.saving.load_model
#from keras.saving import load_model
#from keras.config import enable_unsafe_deserialization
from mltu.tensorflow.metrics import CERMetric

from mltu.tensorflow.losses import CTCloss
#from losses import CTCloss

import mltu.tensorflow.losses

import pandas as pd
from tools_mltu import *

from scipy.special import softmax
import cv2

import onnxruntime as ort
#import onnx

# Set CPU as available physical device
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

#----------------
# https://github.com/leimao/Frozen-Graph-TensorFlow/blob/master/TensorFlow_v2/utils.py
#----------------
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False,print_ops=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    if print_ops ==True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        ops = import_graph.get_operations()
        print(ops[0])
        print("Input layer: ", layers[0])
        print("Output layer: ", layers[-1])
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    test_date="test_opp"
    model_dir="Models/"+test_date
    #configs = BaseModelConfigs.load(model_dir+"/configs.yaml")
    configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

    print("configs.model_path:",configs.model_path)
    if os.path.exists("./work")==False:
        os.mkdir("./work")

    #sys.exit(0)

    LOAD_1=False    # ONNX
    LOAD_2=False        # load keras saved model  ->  OK
    LOAD_3=False        # load tf saved model   -> OK
    LOAD_4=True        # load frozen model       -> OK

    if LOAD_1==True:
        print("LOAD_1")
        force_cpu = True
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" and not force_cpu else ["CPUExecutionProvider"]
        sess = ort.InferenceSession("./a.model.onnx", providers=providers)

        print("sess.get_inputs()[0]:",sess.get_inputs()[0])
        # sess.get_inputs()[0]: NodeArg(name='source:0', type='tensor(float)', shape=[1, 600, 122])
        print("sess.get_outputs()[0]:",sess.get_outputs()[0])
        # sess.get_outputs()[0]: NodeArg(name='Identity:0', type='tensor(int32)', shape=[1, 150])

    # use Keras saved model
    if LOAD_2==True:
        # load keras saved model
        print("LOAD_2")
        #loss_fn=CTCloss()
        #cer_m=CERMetric(configs.vocab)
        objs_x={
               "tf":tf,
                #"CTCloss":CTCloss(),
                #"CERMetric":CERMetric(configs.vocab),
                }
        #model = keras.models.load_model(configs.model_path+'/a.model.keras',custom_objects={'tf':tf},safe_mode=False)
        #model = keras.models.load_model(configs.model_path+'/a.model.keras',custom_objects=objs_x,safe_mode=False)
        # OK
        model = keras.models.load_model(configs.model_path+'/a.model.keras',safe_mode=False)

    # use TF saved model
    if LOAD_3==True:
        print("LOAD_3")
        # /home/nishi/kivy_env/lib/python3.10/site-packages/tensorflow/python/saved_model/load.py
        model = tf.saved_model.load(configs.model_path+'/a.model')
        model_f = model.signatures["serving_default"]

    # use Frozen model
    if LOAD_4==True:
        # frozen model を試す。
        print("LOAD_4")
        # https://github.com/leimao/Frozen-Graph-TensorFlow/blob/master/TensorFlow_v2/example_1.py

        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile(configs.model_path+"/a.model_frozen.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["x:0"],        # keras frozen model
                                        #inputs=["inputs:0"],    # tf frozen model
                                        outputs=["Identity:0"],
                                        print_ops=False)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        # [<tf.Tensor 'x:0' shape=(1, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
        # [<tf.Tensor 'Identity:0' shape=(1, 300, 53) dtype=float32>]

        #for s in frozen_func.__dict__:
        #    print(s)

    #print("model.__dict__:",model.__dict__)
    #sys.exit(0)

    #df = pd.read_csv("Models/05_sound_to_text/202306191412/val.csv").values.tolist()
    dataset_val = pd.read_csv(model_dir+"/val.csv").values.tolist()

    if False:
        val_data_provider = DataProviderAsr(
            dataset=dataset_val,
            skip_validation=True,
            #batch_size=configs.batch_size,
            # changed by nishi 2024.10.2
            batch_size=1,
            data_preprocessors=[
                #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape),
                ],
            transformers=[
                #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                #LabelIndexer(configs.vocab),
                LabelIndexer_my(configs.vocab,configs.char_to_idx),
                LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                ],
        )
        val_ds=val_data_provider

    op_reader=OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape)

    sp_pad=SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0)

    print("configs.max_spectrogram_length:",configs.max_spectrogram_length)
    idx_to_char = configs.vocab

    DISP_F=True

    for img_path, label in dataset_val:
        print("img_path:",img_path)
        dt = op_reader.get_opp_data(img_path)
        img=dt.T

        if DISP_F==True:
            cv2.imshow('Source image', img)
            cv2.waitKey(300)

        w=dt.shape[0]
        #print("dt.shape:",dt.shape)
        #print("type(dt):",type(dt))
        dt_in,label=sp_pad(dt,None)
        dt_in = np.expand_dims(dt_in,axis=0)
        #print("dt_in.shape:",dt_in.shape)
        #print("type(dt_in):",type(dt_in))
        #print("dt_in.dtype:",dt_in.dtype)
        print("go pred!!")

        if LOAD_1==True:
            text = sess.run(["Identity:0"], {'source:0': dt_in})[0]
        if LOAD_2==True:
            text = model.predict(dt_in)
        if LOAD_3==True:
            text = model_f(inputs=dt_in)
        if LOAD_4==True:
            # Get predictions for test images
            #frozen_graph_predictions = frozen_func(x=tf.constant(test_images))[0]
            text = frozen_func(x=tf.constant(dt_in))[0]        # kears frozen model
            #text = frozen_func(inputs=tf.constant(dt_in))[0]    # tf frozen model

        if LOAD_4==True or LOAD_2==True:
            preds=text
        else:
            print('text:',text)
            # {'output_0': <tf.Tensor: shape=(1, 300, 53), dtype=float32, numpy=
            preds=text['output_0']

        prob = softmax(preds[0], -1)
        scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

        #https://www.tech-teacher.jp/blog/python-opencv/
        prediction = ""
        #x=0
        x=2
        for idx in labels:
            y=12+idx*2
            #if idx < len(idx_to_char):
            if idx < 49:
                prediction += idx_to_char[idx]
                if x < w:
                    cv2.circle(img,(x,y),2,(128),-1)
            x+=2

        cv2.imwrite('./work/v_'+img_path, img)

        print("prediction:",prediction)
        if DISP_F==True:
            cv2.imshow('Detect image', img)
            cv2.waitKey(0)



    
    