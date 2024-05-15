# Setup
Create a [Google Cloud key](https://console.cloud.google.com/iam-admin/serviceaccounts/details/111959560397464491265/keys?) for accessing your Database.
Store that cloud as `./keys/datastore-key.json`.

# Running

## web app
Note: for gcloud run, it will be installed via the dockerfile.

```
gcloud run deploy ${website-name} --source . --allow-unauthenticated
gcloud run deploy online-dyna --source . --allow-unauthenticated
```

## local

**install**.
```
mamba create -n webrl python=3.10 pip wheel -y
pip install -r requirements.txt
```

**Run**.
```
gunicorn -b 0.0.0.0:8080 --worker-class gevent --timeout 120 main:app
```

