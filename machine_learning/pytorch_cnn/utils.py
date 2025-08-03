import numpy as np
import torch
import keras
from PIL import Image
import cv2
import boto3
from tqdm import tqdm

from dotenv import load_dotenv
import os
import io
from typing import List

from image_processing_utils import transform_single_digit, process_image

def import_operator_data(x_train, y_train, x_test, y_test):
    try:
        operator_x_train = np.load("../data/processed_data/X_train.npy")
        operator_x_test = np.load("../data/processed_data/X_test.npy")
        operator_y_train = np.load("../data/processed_data/Y_train.npy")
        operator_y_test = np.load("../data/processed_data/Y_test.npy")
    except Exception as e:
        print(f"Failed to find the new operator data. Got error:\n{e}")

    x_train = np.concatenate((x_train, operator_x_train))
    x_test = np.concatenate((x_test, operator_x_test))

    y_train = np.concatenate((y_train, operator_y_train))
    y_test = np.concatenate((y_test, operator_y_test))

    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))

    x_train = x_train[train_perm]
    x_test = x_test[test_perm]

    y_train = y_train[train_perm]
    y_test = y_test[test_perm]

    return x_train, y_train, x_test, y_test

def preprocess_data(batch_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train, x_test, y_test = import_operator_data(x_train, y_train, x_test, y_test)

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32") / 255

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32") / 255

    x_train = torch.as_tensor(x_train)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    x_test = torch.as_tensor(x_test)
    y_test = torch.as_tensor(y_test, dtype=torch.long)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def load_images(s3_bucket_path: str = None, s3_bucket_name: str = None, local_path: str = None) -> List[Image.Image]:
    '''
    for s3_bucket_path, it expects pngs inside the bucket
    
    takes bucket and or local files, creates training and testing data loader for pytorch model
    
    note that all the files in the specified file path must have the same 
    '''

    if (s3_bucket_name is None) ^ (s3_bucket_path is None):
        raise Exception("Both the s3_bucket_name and s3_bucket_path parameters must be provided.")

    if s3_bucket_path:
        load_dotenv("../../backend/.env")

        images = []

        session = boto3.Session(
            aws_access_key_id=os.getenv('ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('SESSION_TOKEN')
        )

        s3 = session.resource('s3')
        path = s3_bucket_path
        bucket = s3.Bucket(s3_bucket_name)

        objs = bucket.objects.filter(Prefix=path)
        for obj in tqdm(objs, desc="Obtaining images..."):
            # Skip parent folder since it's included.
            if obj.key == path:
                continue

            curr_url = s3.Object("penman-lln", obj.key).get()["Body"].read()
            image = Image.open(io.BytesIO(curr_url))
            images.append(image)

    # TODO: Add option for local path once needed.
    return images

def batch_numpy(images: np.ndarray, batch_size: int, expected_output: int):
    x_data = torch.from_numpy(images).float()

    x_train, x_test = torch.utils.data.random_split(x_data, [0.8, 0.2])

    x_train = torch.stack([x for x in x_train])
    x_test  = torch.stack([x for x in x_test])

    y_train = torch.full((x_train.size(0),), fill_value=expected_output)
    y_test  = torch.full((x_test.size(0),), fill_value=expected_output)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data  = torch.utils.data.TensorDataset(x_test, y_test)

    if batch_size > x_train.size(0):
        raise Exception("Batch size too big for training data. Maximum batch size must be {}".format(x_train.size(0)))
    if batch_size > x_test.size(0):
        raise Exception("Batch size too big for testing data. Maximum batch size must be {}".format(x_test.size(0)))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def save_model(model, path: str):
    torch.save(model.state_dict(), path)

def create_dataloader(x_train: torch.tensor, y_train: torch.tensor, x_test: torch.tensor, y_test: torch.tensor, batch_size: int):
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader