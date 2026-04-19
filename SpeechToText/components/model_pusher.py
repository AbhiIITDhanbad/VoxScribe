import os
import sys

from SpeechToText.exceptions import STTException
from SpeechToText.logger import logging

from SpeechToText.cloud_storage.s3_operations import S3Sync
from SpeechToText.entity.config_entity import ModelPusherConfig
from SpeechToText.entity.artifact_entity import ModelEvaluationArtifacts, ModelPusherArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,model_evaluation_artifacts: ModelEvaluationArtifacts):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifacts = model_evaluation_artifacts
        except Exception as e:
            raise STTException(e, sys)

    def initiate_model_pusher(self):
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path

                trained_model_dir = os.path.dirname(trained_model_path)

                s3_model_folder_path = self.model_pusher_config.s3_model_path
                s3_op = S3Sync()
                s3_op.sync_folder_to_s3(folder=trained_model_dir, aws_bucket_url=s3_model_folder_path)
                message = "Model Pusher pushed the current Trained model to Production server storage"
                response = {"is model pushed": True, "S3_model": s3_model_folder_path,"message" : message}
                logging.info(response) 
            else:
                s3_model_folder_path = self.model_pusher_config.s3_model_path
                message = "Current Trained Model is not accepted as model in Production has better accuracy"
                response = {"is model pushed": False, "S3_model":s3_model_folder_path,"message" : message}
                logging.info(response)
            
            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            return model_pusher_artifacts

        except Exception as e:
            raise STTException(e, sys)