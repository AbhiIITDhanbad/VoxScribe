import os

class S3Sync:

    def sync_folder_to_s3(self, folder, aws_bucket_uri):
        command = f'aws s3 sync "{folder}" "{aws_bucket_uri}"'
        os.system(command)

    def sync_folder_from_s3(self, folder, aws_bucket_uri):
        command = f'aws s3 sync "{aws_bucket_uri}" "{folder}"'
        os.system(command)

    def download_file_from_s3(self, local_file_path, aws_s3_uri):
        command = f'aws s3 cp "{aws_s3_uri}" "{local_file_path}"'
        os.system(command)