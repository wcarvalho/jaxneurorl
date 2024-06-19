from google.cloud import storage
import os

from dotenv import load_dotenv

load_dotenv()


def initialize_storage_client():
    storage_client = storage.Client.from_service_account_json(
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    bucket_name = 'human-web-rl_cloudbuild'
    bucket = storage_client.bucket(bucket_name)
    return bucket


def list_files(bucket):
    blobs = bucket.list_blobs()
    print("Files in bucket:")
    for blob in blobs:
        print(blob.name)


def main():
    bucket = initialize_storage_client()

    # List files in the bucket
    list_files(bucket)

if __name__ == "__main__":
    main()
