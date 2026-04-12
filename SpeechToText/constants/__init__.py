ARTIFACT_DIR: str = "artifacts"
PIPELINE_NAME : str = "stt"
 
DATA_BUCKET_URI: str = "s3://speech-to-text-portfolio-project/LJSpeech-1.1.tar.bz2"
DATA_DIR_NAME : str = "data"
DOWNLOAD_DIR: str = "download_data"
ZIP_FILE_NAME: str = "LJSpeech-1.1.tar.bz2"
UNZIPPED_FOLDER_NAME: str = "LJSpeech-1.1"

DATA_PREPROCESSING_ARTIFACTS_DIR : str = "data_preprocessing_artifacts"
DATA_PREPROCESSING_TRAINING_DIR: str = "train"
DATA_PREPROCESSING_TEST_DIR: str = "test"
TRAIN_TEST_SPLIT : float = 0.99

METADATA_DIR: str = "metadata"
METADATA_FILE_NAME: str = "metadata.csv" 
WAVS_FILE_PATH: str = "wavs.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


MAX_TARGET_LENGTH : int = 200


MODEL_TRAINER_ARTIFACT_DIR: str = "model_trainer_artifact"
SAVED_MODEL_DIR: str = "saved_model/"
EPOCHS : int = 1 


MODEL_EVALUATION_ARTIFACT_DIR : str = "model_evaluation_artifact"
S3_BUCKET_URI = "s3://speech-to-text-portfolio-project/model/"
S3_MODEL_DIR_NAME: str = "s3_model/"

MODEL_PUSHER_DIR: str = "model_pusher"


PREDICTION_PIPELINE_DIR_NAME = "prediction artifacts"
PREDICTION_MODEL_DIR_NAME = "prediction model/"
APPLICATION_ARTIFACTS_DIR = "user_inputs"