import os
import tensorflow as tf
from keras.callbacks import Callback

import logging

class Model2onnx(Callback):
    """ Converts the model to onnx format after training is finished. """
    def __init__(
        self, 
        saved_model_path: str, 
        metadata: dict=None,
        save_on_epoch_end: bool=False,
        ) -> None:
        """ Converts the model to onnx format after training is finished.
        Args:
            saved_model_path (str): Path to the saved .h5 model.
            metadata (dict, optional): Dictionary containing metadata to be added to the onnx model. Defaults to None.
            save_on_epoch_end (bool, optional): Save the onnx model on every epoch end. Defaults to False.
        """
        super().__init__()
        self.saved_model_path = saved_model_path
        self.metadata = metadata
        self.save_on_epoch_end = save_on_epoch_end

        try:
            import tf2onnx
        except:
            raise ImportError("tf2onnx is not installed. Please install it using 'pip install tf2onnx'")
        
        try:
            import onnx
        except:
            raise ImportError("onnx is not installed. Please install it using 'pip install onnx'")

    @staticmethod
    def model2onnx(model: tf.keras.Model, onnx_model_path: str):
        try:
            import tf2onnx

            # convert the model to onnx format
            tf2onnx.convert.from_keras(model, output_path=onnx_model_path)

        except Exception as e:
            print(e)

    @staticmethod
    def include_metadata(onnx_model_path: str, metadata: dict=None):
        try:
            if metadata and isinstance(metadata, dict):

                import onnx
                # Load the ONNX model
                onnx_model = onnx.load(onnx_model_path)

                # Add the metadata dictionary to the model's metadata_props attribute
                for key, value in metadata.items():
                    meta = onnx_model.metadata_props.add()
                    meta.key = key
                    meta.value = str(value)

                # Save the modified ONNX model
                onnx.save(onnx_model, onnx_model_path)

        except Exception as e:
            print(e)  

    def on_epoch_end(self, epoch: int, logs: dict=None):
        """ Converts the model to onnx format on every epoch end. """
        if self.save_on_epoch_end:
            self.on_train_end(logs=logs)

    def on_train_end(self, logs=None):
        """ Converts the model to onnx format after training is finished. """
        self.model.load_weights(self.saved_model_path)
        onnx_model_path = self.saved_model_path.replace(".h5", ".onnx")
        self.model2onnx(self.model, onnx_model_path)
        self.include_metadata(onnx_model_path, self.metadata)

