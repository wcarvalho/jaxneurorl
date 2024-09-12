from google.cloud import storage
import os
import fnmatch

def initialize_storage_client():
    storage_client = storage.Client.from_service_account_json('projects/humansf/google_cloud_key.json')
    bucket_name = 'human-dyna'
    bucket = storage_client.bucket(bucket_name)
    return bucket

def download_matching_files(bucket_name, prefix, pattern, destination_folder):
    # Create a client
    bucket = initialize_storage_client()

    # List all blobs in the bucket with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download matching files
    for blob in blobs:
        if fnmatch.fnmatch(blob.name, pattern):
            destination_file = os.path.join(destination_folder, os.path.basename(blob.name))
            blob.download_to_filename(destination_file)
            print(f"Downloaded: {blob.name} to {destination_file}")
