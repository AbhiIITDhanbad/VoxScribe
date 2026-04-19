import os
import sys
import csv
import numpy as np
import tensorflow as tf

from tensorflow import keras

from SpeechToText.exceptions import STTException
from SpeechToText.logger import logging
from SpeechToText.cloud_storage.s3_operations import S3Sync
from SpeechToText.entity.config_entity import ModelEvaluationConfig
from SpeechToText.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataPreprocessingArtifacts
from SpeechToText.constants import MAX_TARGET_LENGTH
from SpeechToText.models.model import Transformer
from SpeechToText.models.data_utils import VectorizeChar
from SpeechToText.entity.model_entity import CreateTensors


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifacts,
                 data_preprocessing_artifacts: DataPreprocessingArtifacts):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_preprocessing_artifacts = data_preprocessing_artifacts
        except Exception as e:
            raise STTException(e, sys)
    
    def get_best_model_path(self):
        try:
            os.makedirs(self.model_evaluation_config.model_evaluation_artifact_dir, exist_ok=True)
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(folder=best_model_dir, aws_bucket_uri=model_path)
            if len(os.listdir(best_model_dir)) != 0:
                weight_files = [f for f in os.listdir(best_model_dir) if f.endswith('.weights.h5')]
                if weight_files:
                    best_model_path = os.path.join(best_model_dir, weight_files[0])
                else:
                    best_model_path = best_model_dir
                logging.info(f"Best model found at {best_model_path}")
            else:
                logging.info("Model is not available in best_model_directory")
            
            return best_model_path
        except Exception as e:
            raise STTException(e,sys)

    def _get_val_dataset(self):
        """Load the test data and create a validation TF dataset for evaluation."""
        try:
            test_data_path = self.data_preprocessing_artifacts.test_data_path
            with open(test_data_path) as f:
                dt_test = [
                    {k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)
                ]
            vectorizer = VectorizeChar(MAX_TARGET_LENGTH)
            val_ds = CreateTensors(data=dt_test, vectorizer=vectorizer).create_tf_dataset(bs=8)
            return val_ds
        except Exception as e:
            raise STTException(e, sys)

    def evaluate_model(self):
        try:
            best_model_path = self.get_best_model_path() 
            if best_model_path is not None:
                s3_model = Transformer(
                    num_hid=200,
                    num_head=2,
                    num_feed_forward=400,
                    target_maxlen=MAX_TARGET_LENGTH,
                    num_layers_enc=4,
                    num_layers_dec=1,
                    num_classes=34,
                )
                dummy_source = tf.zeros((1, 2754, 129))
                dummy_target = tf.zeros((1, MAX_TARGET_LENGTH), dtype=tf.int32)
                s3_model([dummy_source, dummy_target])
                s3_model.load_weights(best_model_path)

                loss_fn = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True, label_smoothing=0.1
                )
                s3_model.compile(loss=loss_fn)

                val_ds = self._get_val_dataset()
                eval_results = s3_model.evaluate(val_ds)
                if isinstance(eval_results, list):
                    s3_model_loss = eval_results[0]
                elif isinstance(eval_results, dict):
                    s3_model_loss = eval_results.get("loss", list(eval_results.values())[0])
                else:
                    s3_model_loss = eval_results

                logging.info(f"S3 Model Validation loss is {s3_model_loss}")
                logging.info(f"Locally trained model loss is {self.model_trainer_artifact.model_loss}")
            else:
                logging.info("Model is not found on production server, So couldn't evaluate")
                s3_model_loss = None
            return s3_model_loss
        except Exception as e:
            raise STTException(e, sys)
    
    def initiate_model_evaluation(self):
        try:
            
            s3_model_loss = self.evaluate_model()
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss

            trained_model_loss = self.model_trainer_artifact.model_loss

            evaluation_response = tmp_best_model_loss > trained_model_loss

            model_evaluation_artifacts = ModelEvaluationArtifacts(s3_model_loss = s3_model_loss,
                                                                is_model_accepted = evaluation_response,
                                                                trained_model_path = self.model_trainer_artifact.model_path,
                                                                s3_model_path = self.get_best_model_path()
                                                                )
            logging.info(f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")

            return model_evaluation_artifacts
        except Exception as e:
            raise STTException(e, sys)