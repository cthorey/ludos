import json
import os
import pickle
import shutil

from ludos.utils import s3

bucket = s3.S3Bucket('s3ludos')
ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


def init_folder(data_name, overwrite=False):
    '''
        Init the binary directory
    '''
    dest_folder = os.path.join(DATA_PATH, data_name)
    if os.path.isdir(dest_folder):
        if overwrite:
            shutil.rmtree(dest_folder)
        else:
            raise ValueError(
                'Dataset {} already exist but overwrite set to False'.format(
                    data_name))
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    if not os.path.isdir(os.path.join(dest_folder, 'data')):
        os.makedirs(os.path.join(dest_folder, 'data'))
    return dest_folder


def dump_payload(payload, fname, data_type='image'):
    if data_type == 'image':
        payload.save(fname)
    elif data_type == 'pkl':
        with open(fname, 'wb') as f:
            pickle.dump(payload, f)
    elif data_type == "annotations":
        json.dump(payload, open(fname, 'w+'))
    else:
        raise ValueError('data type {} not known'.format(data_type))
