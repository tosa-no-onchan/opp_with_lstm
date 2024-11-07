import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os
import sys
import tarfile
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

import glob


from mltu.preprocessors import WavReader

# mltu 1.0.12  -> 1.2.5
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
#from mltu.tensorflow.losses import CTCloss
from losses import CTCloss


#from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.callbacks import TrainLogger
from callbacks_my import Model2onnx
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs

from keras.models import load_model

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

import random


import keras

from tools_mltu import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
# /home/nishi/kivy_env/lib/python3.10/site-packages/keras/src/callbacks/model_checkpoint.py

import yaml
from pathlib import Path

from scipy.special import softmax


from itertools import groupby


"""
## Callbacks to display predictions
"""
class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28,d_provider=1):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token
        self.d_provider = d_provider    # add by nishi 0:original 1:mltu provider

    def on_epoch_end(self, epoch, logs=None):
        #if epoch % 5 != 0:
        if epoch % 10 != 0:
            return
        if self.d_provider==0:
            source = self.batch["source"]
            target = self.batch["target"].numpy()
        else:
            #print('type(self.batch)',type(self.batch))
            bt=self.batch.__getitem__(1)
            #print('type(bt)',type(bt))
            # type(bt) <class 'list'>
            #source = bt[0][0]
            # changed by nishi 2024.10.2
            source = bt[0]
            print('type(source)',type(source))
            # type(source) <class 'numpy.ndarray'>
            print('np.shape(source)',np.shape(source))
            # np.shape(source) (4, 600, 122)  -> opp
            # np.shape(source) (8, 1392, 118)  -> asr
            print('tf.shape(source)',tf.shape(source))
            # tf.shape(source) tf.Tensor([  4 600 122], shape=(3,), dtype=int32)  -> opp
            # tf.shape(source) tf.Tensor([   8 1392  118], shape=(3,), dtype=int32)  -> asr
            #target = bt[0][1]
            # changed by nishi 2024.10.2
            target = bt[1]
            #print('type(target)',type(target))
            # type(target) <class 'numpy.ndarray'>
            #print('np.shape(target)',np.shape(target))
            # np.shape(target) (8, 186)

        bs = tf.shape(source)[0]
        #preds = self.model.generate(source, self.target_start_token_idx)

        #preds = self.model.evaluate(source)
        preds = self.model.predict(source)

        #print('tf.shape(preds)',tf.shape(preds)) 
        # tf.shape(preds) tf.Tensor([  8 186], shape=(2,), dtype=int32)       
        #preds = preds.numpy()
        #print("preds:",preds)
        print("np.shape(preds):",np.shape(preds))
        # np.shape(preds): (4, 300, 53)  -> opp
        # np.shape(preds): (4, 696, 31)  -> asr
        for i in range(bs):
            #target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            target_text = ""
            #cls = softmax(preds[i])
            prob = softmax(preds[i], -1)
            scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)
            #print("labels:",labels)

            print("len(labels):",len(labels))

            for c in target[i, :]:
                if c < len(self.idx_to_char):
                    target_text += self.idx_to_char[c]
            prediction = ""

            if False:
                """
                /home/nishi/kivy_env/lib/python3.10/site-packages/mltu/utils/text_utils.py
                ctc_decoder() が参考になる
                """
                grouped_preds = [[k for k,_ in groupby(self.idx_to_char[preds])] for preds in labels]
                    # convert indexes to chars
                #prediction = ["".join([self.idx_to_char[k] for k in group if k < len(self.idx_to_char)]) for group in grouped_preds]
                prediction=grouped_preds
            else:
                #for idx in preds[i, :]:
                for idx in labels:
                    if idx < len(self.idx_to_char):
                        prediction += self.idx_to_char[idx]
                    #if idx == self.target_end_token_idx:
                    #    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")

"""
## Learning rate schedule
オリジナルは、バグがあります。
1) tensorflow.keras.callbacks.TensorBoard を併用すると、epoch が、 tf.int64 になって、tf.float64 にキャストできない旨のエラーがでる。
2) lr が、通常とは違って、一度上がって、また下がって行くみたい。--> これで、oK か
"""
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    # https://www.tensorflow.org/api_docs/python/tf/print
    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        #print('epoch:',epoch)
        #tf.print(" epoch:",epoch, output_stream=sys.stdout)
        epoch_f = tf.cast(epoch, tf.float32)
        #tf.print(" epoch_f:",epoch_f, output_stream=sys.stdout)
        lrx = self.calculate_lr(epoch_f)
        return lrx
        #return self.calculate_lr(epoch_f)

    # add by nishi 2024.6.2
    def get_config(self):
        #base_config = super().get_config()
        config = {
                "init_lr": self.init_lr,
                "lr_after_warmup": self.lr_after_warmup,
                "final_lr": self.final_lr,
                "warmup_epochs": self.warmup_epochs,
                "decay_epochs": self.decay_epochs,
                "steps_per_epoch": self.steps_per_epoch,
        }
        #return dict(list(base_config.items()) + list(config.items()))
        return config



# https://analytics-note.xyz/machine-learning/keras-learningratescheduler/
def lr_schedul(epoch):
    x = 0.0005
    #if epoch >= 15:
    #    x = 0.00015
    #elif epoch >= 9:
    #    x = 0.00025
    return x

lr_decay = LearningRateScheduler(
    lr_schedul,
    # verbose=1で、更新メッセージ表示。0の場合は表示しない
    verbose=1,
)

vectorizer = VectorizeChar()


if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    #CONT_F=True
    CONT_F=False
    
    #test_date="training-save0s"
    # 178 ?
    #epoch_num=10000
    #epoch_num=178
    epoch_num=1
    test_date="test_opp"

    USE_TEST_DATA_OPP = True    # Obstacle Path Planning

    USE_lr_MY=False
    initial_epoch=0             # start 0

    # 初期乱数作成
    set_seed(0)

    if CONT_F==False:
        # Create a ModelConfigs object to store model configurations
        configs = ModelConfigs()
        configs.vocab =  vectorizer.vocab
        configs.char_to_idx = vectorizer.char_to_idx     
        configs.model_path = os.path.join("Models/", test_date)   
    else:
        configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

    checkpoint_dir= configs.model_path+'/training'
    print('checkpoint_dir:',checkpoint_dir)

    if CONT_F==True:
        latest=latest_checkpoint(checkpoint_dir)
        initial_epoch =latest_checkno(latest)

    print('initial_epoch:',initial_epoch)
    checkpoint_path = checkpoint_dir+"/cp-{epoch:04d}.weights.h5"


    if CONT_F==False:

        if USE_TEST_DATA_OPP == True:
            imgs = glob("{}/*.yaml".format(configs.opp_path), recursive=False)
            datasetx=[]
            for s in imgs:
                #print("s:",s)
                with open(s, 'r') as yml:
                    config = yaml.safe_load(yml)
                    #print("image:",config['image'])
                    #print("data:",config['data'])
                    datasetx.append([config['image'],config['data']])

            dataset = random.sample(datasetx, len(datasetx))
            dt_len=len(dataset)
            train_l=int(dt_len*0.9)

            print("train_l:",train_l)

            #print("dataset[1]:",dataset[1])

            #max_spectrogram_length=800
            max_spectrogram_length=600

            input_shape_cols=max_spectrogram_length
            #input_shape_rows=120
            input_shape_rows=122

            configs.input_shape = [max_spectrogram_length, input_shape_rows]

            x_interval=4
            #x_interval=2
            max_text_length= int(max_spectrogram_length/x_interval)
            # 150 以下は、OK
            #max_text_length=148
            configs.max_spectrogram_length = max_spectrogram_length
            configs.max_text_length = max_text_length
            configs.save()

            train_data_provider = DataProviderAsr(
                dataset=dataset[:train_l],
                skip_validation=True,
                batch_size=configs.batch_size,
                data_preprocessors=[
                    #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=49),        # 'B'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=50),       # '.'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=24),        # center
                    ],
            )

            val_data_provider = DataProviderAsr(
                dataset=dataset[train_l:],
                skip_validation=True,
                batch_size=4,
                data_preprocessors=[
                    #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    OppReader(opp_path=configs.opp_path,input_shape=configs.input_shape),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=49),    # 'B'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=50),   # '.'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=24),    # center
                    ],
            )

            # Save training and validation datasets as csv files
            train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
            val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

            # Split the dataset into training and validation sets
            #train_data_provider, val_data_provider = data_provider.split(split = 0.9)
            print('train_data_provider.__len__():',train_data_provider.__len__())

            ds=train_data_provider
            val_ds=val_data_provider
    #else:


    max_target_len = configs.max_text_length
    #data = get_data(wavs, id_to_text, max_target_len)

    print('max_target_len:',max_target_len)
    #vectorizer = VectorizeChar(max_target_len)
    print("vocab size", len(vectorizer.get_vocabulary()))

    if USE_TEST_DATA_OPP==True:
        idx_to_char = configs.vocab
        display_cb = DisplayOutputs(
            val_ds,
            # changed by nishi 2024.10.2
            #batch, 
            idx_to_char, target_start_token_idx=0
        )  # set the arguments as per vocabulary index for '<' and '>'


    print("configs.model_path:",configs.model_path)


    # tensorboard 
    # https://teratail.com/questions/97nyrumr5iix6d
    tb_callback = TensorBoard(checkpoint_dir+"/logs", update_freq=1)


    print("passed:#3 ")

    #sys.exit(0)

    initial_lr=lr_schedul(initial_epoch)

    # Creating TensorFlow model architecture
    model = train_model(
        input_dim = configs.input_shape,
        output_dim = len(configs.vocab),
        dropout=0.5
    )

    # add by nishi 2024.12.2
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr)

    # https://keras.io/api/losses/

    #loss_fn = keras.losses.CategoricalCrossentropy(
    #    from_logits=True,
    #    label_smoothing=0.1,
    #)
    #loss_fn=keras.losses.CTC(reduction="sum_over_batch_size", name="ctc")
    loss_fn=CTCloss()
    #cer_m=CERMetric(vocabulary=configs.vocab)
    cer_m=CERMetric(configs.vocab)

    # Compile the model and print summary
    model.compile(
        #optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), 
        optimizer=optimizer,
        #loss=CTCloss(),
        loss=loss_fn,
        metrics=[
            #CERMetric(vocabulary=configs.vocab),
            #cer_m,
            #WERMetric(vocabulary=configs.vocab)
            ],
        run_eagerly=False
    )

    #model.summary(line_length=110)
    #sys.exit(0)


    # Define callbacks
    #earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
    #earlystopper = EarlyStopping(monitor="val_CER", patience=5, verbose=1, mode="min")
    #earlystopper = EarlyStopping(monitor="loss", patience=80, verbose=1, mode="min")
    earlystopper = EarlyStopping(monitor="loss", patience=40, verbose=1, mode="min")

    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto")
    model2onnx = Model2onnx(f"{configs.model_path}/a.model.h5")
    #model2onnx = Model2onnx(f"{configs.model_path}/a.model.keras",save_on_epoch_end=False)

    #batch_size = 32
    batch_size = configs.batch_size
    
    #checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        #f"{configs.model_path}/a.model.keras",
        #f"{configs.model_path}/a.model.h5",
        #monitor="val_CER", 
        monitor="loss", 
        verbose=1, 
        save_best_only=True,
        save_weights_only=True,
        save_freq=5*configs.batch_size,
        #save_freq=20*configs.batch_size,
        mode="min")

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="loss",
        #monitor="CER",
        #monitor="val_CER",
        verbose=1, 
        save_best_only=True,
        save_weights_only=True,
        #save_freq=20*batch_size, 
        mode="min")

    print('initial_epoch:',initial_epoch)

    #sys.exit(0)


    # Train the model
    history = model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=epoch_num+initial_epoch,
        initial_epoch=initial_epoch,
        #callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx, cp_callback,lr_decay],
        #callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, cp_callback,lr_decay],
        #callbacks=[earlystopper, lr_decay],
        callbacks=[display_cb, checkpoint,earlystopper],
        #callbacks=[display_cb, earlystopper],
        #workers=configs.train_workers
    )

    print('passed: #99')

    model.save(configs.model_path+'/a.model.keras',save_format='h5')
    #model.save(configs.model_path+'/a.model.h5',save_format='h5')
    #model.save(configs.model_path+'/a.model.keras')
    #model.save(configs.model_path+'/a.model.hdf5')

    print('passed: #100')

    import tf2onnx

    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]
    tf2onnx.convert.from_keras(model,input_signature, output_path=configs.model_path+'/model.onnx',opset=17)


    print('passed: #101')

    tf.saved_model.save(model,configs.model_path+'/a.model')
    model_tf = tf.saved_model.load(configs.model_path+'/a.model')

    # $ python -m tf2onnx.convert --saved-model Models/test_opp/a.model --output Models/test_opp/a.onnx



    if False:
        from keras.saving import load_model
        objs_x={#"CTCloss":CTCloss(),
                "CERMetric":CERMetric(configs.vocab),
                }


        model = load_model(configs.model_path+'/a.model.keras',custom_objects=objs_x,safe_mode=False)


    #from keras.saving import save_model

    #keras.saving.save_model(model, filepath, overwrite=True, zipped=None, **kwargs)
    #save_model(model, configs.model_path+'/a.model.keras', overwrite=True,output_shape=(None,600,122))
    #save_model(model, configs.model_path+'/a.model.keras', overwrite=True)


    conf_adam = model.optimizer.get_config()
    #print("conf_adam:",conf_adam)
    # save to yaml faile
    # https://stackoverflow.com/questions/12470665/how-can-i-write-data-in-yaml-format-in-a-file

    if True:
        with open(configs.model_path+'/conf_adam.yaml', 'w') as outfile:
            #yaml.dump(conf_adam, outfile, default_flow_style=False)
            yaml.dump(conf_adam, outfile, default_flow_style=True)

    if False:
        # read test
        yaml_dict = yaml.safe_load(Path(configs.model_path+'/conf_adam.yaml').read_text())
        print("yaml_dict:",yaml_dict)




    #print(result)

