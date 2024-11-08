import os
import sys
import typing
import numpy as np

import tensorflow as tf

from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder

#from train import WavReaderMel

from configs import ModelConfigs

import keras

#from keras.models import load_model
#from tensorflow.keras.models import load_model

#keras.saving.load_model
from keras.saving import load_model
from keras.config import enable_unsafe_deserialization
from mltu.tensorflow.metrics import CERMetric

from mltu.tensorflow.losses import CTCloss
#from losses import CTCloss

import mltu.tensorflow.losses

import pandas as pd
from tools_mltu import *

from scipy.special import softmax
import cv2


class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(None, {self.input_name: data_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


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

    LOAD_1=False
    LOAD_2=False
    LOAD_3=True

    if LOAD_1==True:
        print("LOAD_1")
        model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)

    if LOAD_2==True:
        print("LOAD_2")
        loss_fn=CTCloss()
        cer_m=CERMetric(configs.vocab)

        objs_x={
               "tf":tf,
                #"CTCloss":CTCloss(),
                #"CERMetric":CERMetric(configs.vocab),
                }

        #model = load_model(configs.model_path+'/a.model.keras',safe_mode=False)
        #model = load_model(configs.model_path+'/a.model.keras',custom_objects=objs_x,safe_mode=False)
        #model = load_model(configs.model_path+'/a.model.hdf5',custom_objects=objs_x,safe_mode=False)
        model = keras.models.load_model(configs.model_path+'/a.model.keras',custom_objects={'tf':tf},safe_mode=False)

        #loaded_model = keras.model.load_model('model_name.h5')

    if LOAD_3==True:
        print("LOAD_3")
        model = tf.saved_model.load(configs.model_path+'/a.model')
        model_f = model.signatures["serving_default"]

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

    for img_path, label in dataset_val:
        print("img_path:",img_path)
        dt = op_reader.get_opp_data(img_path)
        img=dt.T
        w=dt.shape[0]
        #print("dt.shape:",dt.shape)
        #print("type(dt):",type(dt))
        dt_in,label=sp_pad(dt,None)
        dt_in = np.expand_dims(dt_in,axis=0)
        #print("dt_in.shape:",dt_in.shape)
        #print("type(dt_in):",type(dt_in))
        #print("dt_in.dtype:",dt_in.dtype)
        if LOAD_2==True:
            text = model.predict(dt_in)
        if LOAD_3==True:
            text = model_f(inputs=dt_in)

        #print(text)
        # {'output_0': <tf.Tensor: shape=(1, 300, 53), dtype=float32, numpy=
        preds=text['output_0']
        prob = softmax(preds[0], -1)
        scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

        #https://www.tech-teacher.jp/blog/python-opencv/
        prediction = ""
        x=0
        for idx in labels:
            y=12+idx*2
            #if idx < len(idx_to_char):
            #if idx < 48:
            if idx < 49:
                prediction += idx_to_char[idx]
                if x < w:
                    cv2.circle(img,(x,y),2,(128),-1)
            x+=2

        cv2.imwrite('./work/v_'+img_path, img)

        print("prediction:",prediction)
        if False:
            cv2.imshow('Color image', img)
            cv2.waitKey(0)



    
    