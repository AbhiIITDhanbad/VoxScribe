import os
import sys
from glob import glob
import csv
from SpeechToText.entity.config_entity import DataPreprocessingConfig
from SpeechToText.entity.artifact_entity import DataPreprocessingArtifacts, DataIngestionArtifacts 
from SpeechToText.models.data_utils import VectorizeChar, get_data
from SpeechToText.logger import logging
from SpeechToText.exceptions import STTException
from SpeechToText.constants import * 


class DataPreprocessing():
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.data_ingestion_artifact = data_ingestion_artifacts
        
        except Exception as e:
            raise STTException(e,sys)

    
    def get_id_to_text(self) -> tuple:
        try:
            logging.info("Entering the get_id_to_index methos of DataPreprocessing")
            os.makedirs(self.data_preprocessing_config.data_preprocessing_artifacts_dir,exist_ok = True)
            
            metadata = os.path.join(self.data_ingestion_artifact.extracted_data_path, METADATA_FILE_NAME) ###
            waves_path = self.data_ingestion_artifact.extracted_data_path
            wavs = None
            logging.info("Writing the path to wavs")

            self.wavs = glob("{}/**/*.wav".format(waves_path), recursive = True)
            logging.info("Creating the dictionary to id_to_text")

            self.id_to_text = {}

            with open(metadata, encoding = "utf-8") as f:
                for line in f:
                    id = line.strip().split("|")[0]
                    text = line.strip().split("|")[2]
                    self.id_to_text[id] = text

            os.makedirs(self.data_preprocessing_config.metadata_dir_path, exist_ok=True)
            with open(self.data_preprocessing_config.wavs_file_path, "w") as f:
                write = csv.writer(f)
                
                write.writerows(self.wavs)

            logging.info("Exiting the get_id_to_index method of DataPreprocessing")
            return self.wavs, self.id_to_text

        except Exception as e:
            raise STTException(e,sys)

    def extract_data(self) -> None:

        try:
            logging.info("Entering the extracted_data method of preprocessing")
            self.data = get_data(self.wavs, self.id_to_text, maxlen = MAX_TARGET_LENGTH)
            logging.info("Exiting the extracted_data method of preprocessing")
        except Exception as e:
            raise STTException(e,sys)

    def train_test_split(self) -> tuple:
        try: 
            logging.info("Entered the train_test_split method of preprocessing")
            split = int(len(self.data)* TRAIN_TEST_SPLIT)
            train_data = self.data[:split]
            test_data = self.data[split:]
            logging.info("write train data")
            os.makedirs(self.data_preprocessing_config.train_dir_path, exist_ok = True)

            keys = train_data[0].keys()
            self.train_file_path = os.path.join(self.data_preprocessing_config.train_dir_path, TRAIN_FILE_NAME)

            self.test_file_path = os.path.join(self.data_preprocessing_config.test_dir_path, TEST_FILE_NAME)

            with open(self.train_file_path, "w", newline = "") as f:
                write = csv.DictWriter(f, fieldnames = keys)
                write.writeheader()
                write.writerows(train_data)
            logging.info("write test data")
            os.makedirs(self.data_preprocessing_config.test_dir_path, exist_ok = True)
            with open(self.test_file_path, "w", newline = "") as f:
                write = csv.DictWriter(f, fieldnames = keys)
                write.writeheader()
                write.writerows(test_data)
                f.close()
            logging.info("Exiting the train_test_split method of preprocessing")
            return train_data, test_data
        except Exception as e:
            raise STTException(e,sys)


    def initiate_data_preprocessing(self) -> DataPreprocessingArtifacts:
        try:
            logging.info("Entered the initiate_data_preprocessing method of preprocessing")
            self.get_id_to_text()
            self.extract_data()
            self.train_test_split()
            data_preprocessing_artifact = DataPreprocessingArtifacts(
                train_data_path = self.train_file_path,
                test_data_path = self.test_file_path 
            )
            logging.info("data_preprocessing is done successfully")
            return data_preprocessing_artifact
        except Exception as e:
            raise STTException(e,sys)


        

        