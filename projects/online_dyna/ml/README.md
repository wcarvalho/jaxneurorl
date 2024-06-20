# AI experiments

```
```

# Web App
## Setup
Create a [Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts?) for accessing your Database. Select "CREATE SERVICE ACCOUNT", give it some name and the following permissions:
- Storage Object Creator (for uploading/saving data)
- Storage Object Viewer and Storage Object Admin (for viewing/downloading data)

Save the json as `./keys/datastore-key.json`.

## Running

#### web app
Note: for gcloud run, it will be installed via the dockerfile.

```
gcloud run deploy ${website-name} --source . --allow-unauthenticated
gcloud run deploy online-dyna --source . --allow-unauthenticated
```

#### local

**install**.
```
mamba create -n webrl python=3.10 pip wheel -y
pip install -r web_exp/requirements-online-dyna.txt
```

**Run**.
```
python web_app.py
```

