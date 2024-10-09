import subprocess
import os
from google.cloud import storage
import fnmatch
from pathlib import Path

##############################
# User Data
##############################
def initialize_storage_client(bucket_name='human-dyna'):
    storage_client = storage.Client.from_service_account_json(
        'projects/humansf/google_cloud_key.json')
    bucket = storage_client.bucket(bucket_name)
    return bucket


def download_user_files(bucket_name, prefix, pattern, destination_folder):
    # Create a client
    bucket = initialize_storage_client(bucket_name)

    # List all blobs in the bucket with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download matching files
    for blob in blobs:
        if fnmatch.fnmatch(blob.name, pattern):
            destination_file = os.path.join(
                destination_folder, os.path.basename(blob.name))
            blob.download_to_filename(destination_file)
            print(f"Downloaded: {blob.name} to {destination_file}")


##############################
# Model Data
##############################
# download all .config and .safetensor files
def download_files(ssh, sftp, server_dir, local_dir, file_extensions):
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # List files in server directory
    stdin, stdout, stderr = ssh.exec_command(f'ls {server_dir}')
    files = stdout.read().decode().splitlines()

    # Download each file with matching extension
    for file in files:
        if any(file.endswith(ext) for ext in file_extensions):
            remote_path = f'{server_dir}/{file}'
            local_path = f'{local_dir}/{file}'
            print(f'Downloading {remote_path} to {local_path}')
            sftp.get(remote_path, local_path)


def run_command(command, dir_path):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred: {result.stderr}")
    else:
        print(f"Download {dir_path} successfully.")

def download_model_files(base_server_dir, base_local_dir, dir_list):
    # SSH connection details
    hostname = 'rcfas_login1'  # Using the SSH config alias
    
    # Common rsync options
    rsync_options = "-avz --prune-empty-dirs --exclude='*wandb*'"

    for dir_path in dir_list:
        server_dir = f'{base_server_dir}/{dir_path}'
        local_dir = f'{base_local_dir}/{dir_path}'

        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)

        # Download files
        run_command(f"rsync {rsync_options} {hostname}:{server_dir}/ {local_dir}/", local_dir)

##############################
# User Data
bucket_name = "human-dyna"
prefix = "data/"
#pattern = "data/data_user=*_name=r0-v2*debug=0.json"
human_data_pattern = "data/data_user=*r0-exp2-obj*-v0*debug=0.json"
destination_folder = "/Users/wilka/git/research/results/human_dyna/user_data/exp2"

##############################
# Model Data
server_dir = '/n/holylfs06/LABS/kempner_fellow_wcarvalho/results/jaxrl_result/housemaze_trainer'
local_dir = "/Users/wilka/git/research/results/human_dyna/model_data"
qlearning_dir = f'ql/save_data/ql-big-2/tota=40000000,exp=exp2'
sf_dir = f'usfa/save_data/usfa-big-10-search/sf_h=1024,num_=2,tota=40000000,exp=exp2'
dyna_dir = f'dynaq_shared/save_data/dynaq-big-4/alg=dynaq_shared,agen=256,tota=100000000,exp=exp2'

qlearning_server_dir = f'{server_dir}/{qlearning_dir}'
sf_server_dir = f'{server_dir}/{sf_dir}'
dyna_server_dir = f'{server_dir}/{dyna_dir}'

qlearning_local_dir = f'{local_dir}/{qlearning_dir}'
sf_local_dir = f'{local_dir}/{sf_dir}'
dyna_local_dir = f'{local_dir}/{dyna_dir}'

if __name__ == "__main__":
    pass
    #download_user_files(bucket_name, prefix, human_data_pattern, destination_folder)
    download_model_files(
        base_server_dir=server_dir,
        base_local_dir=local_dir,
        dir_list=[
            #qlearning_dir,
            sf_dir,
            #dyna_dir,
        ]
    )