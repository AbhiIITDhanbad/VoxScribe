import sys
import os
import tarfile
from zipfile import ZipFile
from SpeechToText.exceptions import STTException
from SpeechToText.logger import logging
from SpeechToText.entity.config_entity import DataIngestionConfig
from SpeechToText.entity.artifact_entity import DataIngestionArtifacts
from SpeechToText.constants import *
from SpeechToText.cloud_storage.s3_operations import S3Sync



class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise STTException(e,sys)
        

    def get_data_from_cloud(self) -> None:
        try: 
            logging.info("Initiating data download from s3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            bucket_uri = self.data_ingestion_config.bucket_uri
            s3_zip_file_path = self.data_ingestion_config.s3_zip_file_path

            if os.path.isfile(s3_zip_file_path):
                logging.info(
                    f"Data is already present at {s3_zip_file_path}, So skipping downloading step"
                )
                return None
            
            else:
                os.makedirs(download_dir, exist_ok = True)
                self.s3_sync.download_file_from_s3(s3_zip_file_path, bucket_uri)
                if not os.path.isfile(s3_zip_file_path):
                    raise Exception(f"Download failed: file not found at {s3_zip_file_path}")
                logging.info(
                    f"Data is downloaded from s3 bucket to {s3_zip_file_path}"
                )
        except Exception as e:
            raise STTException(e,sys)
        

    def unzip_data(self)->None:
        try:
            logging.info("Extracting the downloaded tar.bz2 file from download directory")
            s3_zip_file_path = self.data_ingestion_config.s3_zip_file_path
            unzip_data_dir_path = self.data_ingestion_config.unzip_data_dir_path
            unzip_data_dir = os.path.join(unzip_data_dir_path, UNZIPPED_FOLDER_NAME)
            if os.path.isdir(unzip_data_dir):
                logging.info(
                    "Extracted file already exists in unzip directory"
                )
            else:
                os.makedirs(unzip_data_dir_path, exist_ok = True)
                # Handle .tar.bz2 files
                if s3_zip_file_path.endswith('.tar.bz2'):
                    with tarfile.open(s3_zip_file_path, "r:bz2") as tar_ref:
                        tar_ref.extractall(unzip_data_dir_path)
                # Handle .zip files (fallback)
                else:
                    with ZipFile(s3_zip_file_path, "r") as zip_file_ref:
                        zip_file_ref.extractall(unzip_data_dir_path)
                logging.info(f"Extracted file exists in unzip directory: {unzip_data_dir_path}")

        except Exception as e:
            raise STTException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("initiating the data ingestion component...")
            self.get_data_from_cloud()
            self.unzip_data()

            extracted_data_path = os.path.join(self.data_ingestion_config.unzip_data_dir_path, UNZIPPED_FOLDER_NAME)

            data_ingestion_artifact = DataIngestionArtifacts(
                downloaded_data_path = self.data_ingestion_config.download_dir,
                extracted_data_path = extracted_data_path

            )

            logging.info("data Ingestion completed")

            return data_ingestion_artifact
        
        except Exception as e:
            raise STTException(e,sys)
        