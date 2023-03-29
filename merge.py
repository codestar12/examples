

import json
import oci
import os
from tqdm import trange


def download(bucket_name, object_name, file_name):
    config = oci.config.from_file()
    client = oci.object_storage.ObjectStorageClient(config=config)
    ret = client.get_object(namespace_name=client.get_namespace().data,
                            bucket_name=bucket_name,
                            object_name=object_name)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as out:
       for chunk in ret.data.raw.stream(1024 * 1024, decode_content=False):
           out.write(chunk)


def upload(file_path, bucket_name, object_name):
    config = oci.config.from_file()
    client = oci.object_storage.ObjectStorageClient(config=config,
                                                    retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
    upload_manager = oci.object_storage.UploadManager(client)

    upload_manager.upload_file(namespace_name=client.get_namespace().data,
                               bucket_name=bucket_name,
                               object_name=object_name,
                               file_path=file_path)


bucket = 'mosaicml-internal-dataset-mc4'
num_subsets = 100
shards = []
for subset in trange(num_subsets, leave=False):
    if subset == 15 or subset == 6 or subset == 10:
        continue
    path = f'mds/pretokenized/en/{subset:03}/index.json'
    download(bucket, path, path)
    obj = json.load(open(path))
    for i in range(len(obj['shards'])):
        for which in ['raw_data',]:
            basename = obj['shards'][i][which]['basename']
            obj['shards'][i][which]['basename'] = os.path.join(f'{subset:03}', basename)
    shards += obj['shards']
obj = {
    'version': 2,
    'shards': shards,
}
path = 'mds/pretokenized/en/index.json'
with open(path, 'w') as out:
    json.dump(obj, out)
upload(path, bucket, path)