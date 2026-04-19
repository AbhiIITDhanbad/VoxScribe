import os
import sys
import csv 

import tensorflow as tf 
from tensorflow import keras 

from SpeechToText.exceptions import STTException
from SpeechToText.logger import logging
from SpeechToText.entity.artifact_entity import DataPreprocessingArtifacts, ModelTrainerArtifacts
from SpeechToText.entity.config_entity import ModelTrainerConfig
from SpeechToText.entity.model_entity import CreateTensors
from SpeechToText.models.data_utils import VectorizeChar
from SpeechToText.constants import *
from SpeechToText.models.model import Transformer
from SpeechToText.models.model_utils import CustomSchedule, DisplayOutputs



class ModelTrainer():

    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_preprocessing_artifacts = data_preprocessing_artifacts
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise STTException(e,sys)
    
    def vectorizer(self) -> VectorizeChar:
        try:
            logging.info("Entering the vectorizer method of model_trainer")
            self.vectorizer = VectorizeChar()
            # self.vectorizer.adapt(self.data_preprocessing_artifacts.train_data_path)
            logging.info("Exiting the vectorizer method of model_trainer")
            return self.vectorizer
        except Exception as e:
            raise STTException(e,sys)
    def get_data(self):
        train_data = self.data_preprocessing_artifacts.train_data_path
        test_data = self.data_preprocessing_artifacts.test_data_path
        try:
            with open(train_data) as f:
                self.dt_train = [
                    {
                        k:v for k,v in row.items()
                    }
                    for row in csv.DictReader(f,skipinitialspace=True)
                    
                ]
            with open(test_data) as f:
                self.dt_test = [
                    {
                        k:v for k,v in row.items()
                    }
                    for row in csv.DictReader(f,skipinitialspace=True)
                ]

            logging.info("Exiting the get_data method of model_trainer")
            return self.dt_train, self.dt_test
        except Exception as e:
            raise STTException(e,sys)

    def get_tensors(self):
        try:
            self.ds = CreateTensors(data = self.dt_train, vectorizer = self.vectorizer).create_tf_dataset(bs=32)
            self.val_ds = CreateTensors(data = self.dt_test, vectorizer = self.vectorizer).create_tf_dataset(bs=8)
            logging.info("Exiting the get_tensors method of model_trainer")
            # return self.ds, self.val_ds
        except Exception as e:
            raise STTException(e,sys)


    def fit(self):
        try:
            logging.info("fitting the model")
            batch = next(iter(self.val_ds))

            idx_to_char = self.vectorizer.get_vocabulary()
            display_cb = DisplayOutputs(batch, idx_to_char, target_start_token_idx = 2, target_end_token_idx=3)
            self.model = Transformer(
                num_hid = 200,
                num_head = 2,
                num_feed_forward = 400,
                target_maxlen = MAX_TARGET_LENGTH,
                num_layers_enc = 4,
                num_layers_dec = 1,
                num_classes = 34
            )

            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits = True, label_smoothing = 0.1
            )

            learning_rate = CustomSchedule(
                init_lr = 0.00001,
                lr_after_warmup = 0.001,
                final_lr = 0.00001,
                warmup_epochs=15,
                decay_epochs = 40,
                steps_per_epoch = len(self.ds)
            )

            optimizer = keras.optimizers.Adam(learning_rate)
            self.model.compile(optimizer = optimizer, loss=loss_fn)

            self.model.fit(self.ds, validation_data = self.val_ds, callbacks = [display_cb], epochs = EPOCHS)

        except Exception as e:
            raise STTException(e,sys)

    def initiate_model_trainer(self) -> None:

        try:
            self.vectorizer()
            self.get_data()
            self.get_tensors()
            self.fit()

            model_loss = self.model.val_loss.numpy()

            os.makedirs(self.model_trainer_config.model_dir_path, exist_ok = True)
            weights_dir = os.path.join(self.model_trainer_config.model_dir_path, SAVED_MODEL_DIR)
            os.makedirs(weights_dir, exist_ok = True)
            weights_path = os.path.join(weights_dir, "model.weights.h5")
            self.model.save_weights(weights_path)

            model_trainer_artifact = ModelTrainerArtifacts(
                model_path = weights_path,
                model_loss = model_loss
            )

            return model_trainer_artifact
        except Exception as e:
            raise STTException(e,sys)