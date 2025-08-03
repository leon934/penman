import mlflow
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from mlflow.models import infer_signature

import itertools
import logging

from image_processing_utils import resize_symbols, erode, harsh_gamma_enhance, normalize, process_image, resize_image
from utils import load_images, create_dataloader
from cnn import ConvNeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_accuracy(test_loader: torch.utils.data.DataLoader, model: ConvNeuralNetwork) -> float:
    with torch.no_grad():
        correct, total = 0, 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()

    return correct * 100 / total
    
def main():
    resizing_methods = [
        (transforms.Resize((28, 28)), "pytorch_resize"),
        (transforms.Lambda(lambda image: resize_symbols(image, mode="pad")), "default_pad"),
        (transforms.Lambda(lambda image: resize_symbols(image, mode="stretch")), "default_stretch")
    ]

    transformation_methods = [
        (transforms.Lambda(erode), "erode"),
        (transforms.Lambda(harsh_gamma_enhance), "gamma_enhance")
    ]

    # Creates each combination of the pipeline where the transformation methods go first and are combined with each resizing method.
    transformation_pipelines = []

    for i in range(2):
        for k in range(len(transformation_methods) + 1):
            for t_subset in itertools.permutations(transformation_methods, k):
                for resize, resize_name in resizing_methods:
                    pipeline = (
                        ([transforms.Lambda(resize_image)] if i == 0 else []) + 
                        list(map(lambda pair: pair[0], list(t_subset))) + 
                        [resize, transforms.Lambda(normalize), transforms.ToTensor()]
                    )
                    pipeline_name = "->".join(
                        (["resize"] if i == 0 else []) +
                        list(map(lambda pair: pair[1], list(t_subset))) +
                        [resize_name]
                    )

                    transformation_pipelines.append((transforms.Compose(pipeline), pipeline_name))

    image_data = load_images(s3_bucket_path="data/token/=/", s3_bucket_name="penman-lln")
    split = int(0.8 * len(image_data))

    model = ConvNeuralNetwork(num_classes=17)
    model.load_state_dict(torch.load("./model/model.pt"))

    # Decoder equiv. for = sign.
    expected = 14

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    mlflow.set_tracking_uri("http://10.0.0.21:5000")
    mlflow.set_experiment("'=' image transformation pipelines experiment")

    for curr_pipeline, pipeline_name in tqdm(transformation_pipelines, desc=f"Iterating through pipeline {pipeline_name}"):
        with mlflow.start_run(run_name=pipeline_name):
            mlflow.log_param("pipeline", pipeline_name)

            # Apply the image transformation pipeline onto each image in the list, obtaining a list of tensors.
            data = [process_image(img=image, pipeline=curr_pipeline) for image in image_data]
            
            train_inputs = torch.cat(data[:split])
            test_inputs = torch.cat(data[split:])

            # Creates the dataset to train the model.
            train_labels = torch.full((train_inputs.size(0),), expected)
            test_labels = torch.full((test_inputs.size(0),), expected)

            train_loader, test_loader = create_dataloader(train_inputs, train_labels, test_inputs, test_labels, batch_size=8)

            model.eval()

            # Tests baseline accuracy of model on testing data.
            initial_accuracy = get_accuracy(test_loader, model)
            mlflow.log_metric("initial_accuracy", initial_accuracy)
            mlflow.log_metric("accuracy", initial_accuracy, step=1)

            model.train()

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.0005,
                weight_decay=0.005
            )

            # Fine-tunes model with given num of epochs.
            NUM_EPOCH = 10
            for epoch in range(NUM_EPOCH):
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                mlflow.log_metric("train_loss", loss.item(), step=epoch + 1)

            model.eval()

            final_accuracy = get_accuracy(test_loader, model)
            mlflow.log_metric("final_accuracy", final_accuracy)
            mlflow.log_metric("accuracy", final_accuracy, step=2)

            model_input = torch.zeros((1, 1, 28, 28))
            signature = infer_signature(model_input.numpy(), model(model_input).detach().numpy())

            mlflow.pytorch.log_model(
                pytorch_model=model, 
                name=f"model {pipeline_name}", 
                signature=signature,
                input_example=model_input.numpy()
            )

if __name__ == "__main__":
    main()