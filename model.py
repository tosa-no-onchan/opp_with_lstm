import tensorflow as tf
from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block, activation_layer
# add by nishi 2024.11.13
from keras.ops import expand_dims

#from keras import backend as K

def activation_layer_my(layer, activation: str="relu", alpha: float=0.1) -> tf.Tensor:
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions
    Args:
        layer: tf.Tensor
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)
    Returns:
        tf.Tensor
    """
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        #layer = layers.LeakyReLU(alpha=alpha)(layer)
        layer = layers.LeakyReLU(negative_slope=alpha)(layer)
    return layer


'''
Now, this class becomes error when loaded form keras.models.load_model()
'''
class MyModel(Model):
    def __init__(self,inputs,outputs,**kwargs):
        self.inputs=inputs
        self.outputs=outputs
        super().__init__(inputs,outputs,**kwargs)

    def _get_save_spec(self,dynamic_batch,keep_original_batch_siz=False):
        print("keep_original_batch_siz:",keep_original_batch_siz)
        # keep_original_batch_siz: False
        print("dynamic_batch:",dynamic_batch)
        # dynamic_batch: True

        #print("self.inputs:",self.inputs)
        # self.inputs: [<KerasTensor shape=(None, 600, 122), dtype=float32, sparse=False, name=input>]
        #print("self.inputs[0]._shape:",self.inputs[0]._shape)
        # self.inputs[0]._shape: (None, 600, 122)

        #return [tf.TensorSpec(shape=[None,600,122], dtype=tf.float32)]
        return [tf.TensorSpec(shape=self.inputs[0]._shape, dtype=tf.float32)]

    # for model freeze
    # https://github.com/Unity-Technologies/barracuda-release/blob/release/0.3.2/Documentation~/Barracuda.md
    #@tf.function(input_signature=[tf.TensorSpec(shape=[None,600,122], dtype=tf.float32)])
    @tf.function(input_signature=[tf.TensorSpec(shape=[1,600,122], dtype=tf.float32)])
    def to_save(self,x):
        return self(x)

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2,use_cudnn='auto'):
    inputs = layers.Input(shape=input_dim, name="input")
    # expand dims to add channel dimension
    #input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
    # changed by nishi 2024.11.4
    #input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), output_shape=(input_dim[0], input_dim[1],1),dtype=tf.float32)(inputs)
    # changed by nishi 2024.11.30
    input = expand_dims(inputs,2)

    # Convolution layer 1
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    #x = activation_layer_my(x, activation="leaky_relu")

    # Convolution layer 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    #x = activation_layer_my(x, activation="leaky_relu")
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    #print(">>>  train_model(): passed #5")
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,use_cudnn=use_cudnn))(x)        #  keras.layers.LSTM -> use CudnnRNNV3 
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,use_cudnn=use_cudnn))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,use_cudnn=use_cudnn))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,use_cudnn=use_cudnn))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,use_cudnn=use_cudnn))(x)

    # Dense layer
    x = layers.Dense(256)(x)
    x = activation_layer(x, activation="leaky_relu")
    #x = activation_layer_my(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)
    # Classification layer
    output = layers.Dense(output_dim + 1, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=output)
    #model = MyModel(inputs=inputs, outputs=output)

    # https://github.com/Unity-Technologies/barracuda-release/blob/release/0.3.2/Documentation~/Barracuda.md
    #@tf.function(input_signature=[tf.TensorSpec(shape=[<input_shape>], dtype=tf.float32)])
    #@tf.function(input_signature=[tf.TensorSpec(shape=input_dim, dtype=tf.float32)])
    #def to_save(x):
    #    return model(x)

    return model
