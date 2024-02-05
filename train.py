import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image

# PROGRAMMER: PeiHsin Wang
# DATE CREATED: 1/09/2024 
# Purpose: Train a new network on a data set with train.py
# Action: Prints out training loss, validation loss, and validation accuracy as the network trains. In default, it will save a checkpoint.pth file in the current directory.
# Basic usage: python train.py data_directory
# Options: 
# * Choose architecture: python train.py data_dir --arch "vgg16" 
# * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 4 
# * Use GPU for training: python train.py data_dir --gpu
# * Check data images: python train.py data_dir --check_image
# * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 

def load_data(data_dir):
    '''
    Load the train, validation and test datasets with ImageFolder(inherit label from folder name), then transform them for training, validation and testing.
    '''
    # Define directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])


    val_test_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, val_test_transforms)
    test_data = datasets.ImageFolder(test_dir, val_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloaders, validloaders, testloaders

def imshow(image, ax=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def check_image(trainloaders, validloaders, testloaders):
    '''
    Check the first 4 loading images from train, validation, and test datasets. 
    '''

    # check the sample images
    train_iter = iter(trainloaders)
    valid_iter = iter(validloaders)
    test_iter = iter(testloaders)

    images_train, labels_train = next(train_iter)
    images_valid, labels_valid = next(valid_iter)
    images_test, labels_test = next(test_iter)

    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for idx in range(4):
        ax = axes[idx]
        imshow(images_train[idx], ax=ax)
    fig.suptitle('Training Examples', fontsize=14, y = 0.8)

    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for idx in range(4):
        ax = axes[idx]
        imshow(images_valid[idx], ax=ax)
    fig.suptitle('Validation Examples', fontsize=14, y = 0.8)

    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for idx in range(4):
        ax = axes[idx]
        imshow(images_test[idx], ax=ax)
    fig.suptitle('Testing Examples', fontsize=14, y = 0.8)

    return plt.show();

def build_model(device, arch = 'vgg16', hidden_units = 512, learning_rate = 0.01):
    '''
    Fine-tune a new network with the pretrained model and define the criterion and optimizer.
    '''
    # Load a pre-trained network and define the new fully-connected classifier
    if arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        print('Using vgg16 as a pre-trained model for training.')

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Sequential(nn.Linear(4096, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)

    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        print('Using resnet50 as a pre-trained model for training.')

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(nn.Linear(2048, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        print('Warning: Please choose a model from ("vgg16", "resnet50")')
        sys.exit(2)    

    # Define the loss
    criterion = nn.NLLLoss()

    model.to(device)

    return model, criterion, optimizer

def train_model(device, model, criterion, optimizer, trainloaders, validloaders, epochs = 3, print_every = 40):
    '''
    Train the model with given hyperparameters.
    '''
    start = time.time()
    steps = 0
    running_loss = 0
    print('Start training...')
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validate the model
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval() # turn off dropout
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Time: {(time.time() - start)/60:.2f} minutes, "
                      f"Train loss: {running_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(validloaders):.3f},  "
                      f"Valid accuracy: {accuracy/len(validloaders):.3f}")
                running_loss = 0
                model.train() # turn on dropout

    return model

def save_checkpoint(save_dir, model, arch, hidden_units, learning_rate, criterion, optimizer, epochs):
    '''
    Save the model checkpoint.
    '''

    # Attach the label mapping directory to the model.class_to_idx attribute
    with open('./cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    model.class_to_idx = cat_to_name


    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'criterion': criterion,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'model': model,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)

    return print(f'Checkpoint saved at {save_dir} !')

def main():

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f'Using GPU: {device}.')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f'Using GPU: {device}.')
        else:
            device = torch.device("cpu")
            print('GPU not avaiable, using CPU instead.')
    else:
        device = torch.device("cpu")
        print('Using CPU.')

    # Load the data
    trainloaders, validloaders, testloaders = load_data(args.data_directory)
    print('Data loaded!')
    
    # Check the images
    if args.check_image:
        check_image(trainloaders, validloaders, testloaders)
    
    # Build the model
    model_load, criterion, optimizer = build_model(device, args.arch, args.hidden_units, args.learning_rate)
    print(f'Model {args.arch} built!')
    
    # Train the model
    model_trained = train_model(device, model_load, criterion, optimizer, trainloaders, validloaders, args.epochs)
    print('Model trained!')
    
    # Save the model
    if args.save_dir:
        save_checkpoint(args.save_dir, model_trained, args.arch, args.hidden_units, args.learning_rate, criterion, optimizer, args.epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tuned a neural network on image data for a classifier")
    # Define command line arguments
    parser.add_argument('data_directory', type=str, help="Path to the image directory: './flowers'")
    parser.add_argument('--check_image', action = 'store_true', default = False,
                        help = 'show the first 4 images from train, validation, and test datasets')
    parser.add_argument('--arch', type = str, default = 'resnet50', 
                        help = 'choose one model from ("vgg16", "resnet50") to fine-tune')
    parser.add_argument('--learning_rate', type = float, default = 0.01,
                        help = 'set the learning rate for training')
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'set the number of hidden units for training. It should be < 2048 and > 102, default is 512.')
    parser.add_argument('--epochs', type = int, default = 4,
                        help = 'set the number of epochs for training')
    parser.add_argument('--gpu',  action='store_true', default = True,
                        help = 'use gpu for training')
    parser.add_argument('--save_dir', type = str, default = '',
                        help = 'set the directory to save the checkpoint')
    args = parser.parse_args()

    if not args.save_dir:
        # If save_dir is not provided, set it to os.getcwd() + args.arch + '_classifier.pth'
        args.save_dir = os.path.join(os.getcwd(), args.arch + '_classifier.pth')
    
    main()

# Sample of using config file instead of command line arguments
# if __name__ == "__main__":
    # with open('train_config.json', 'r') as config_file:
        # config = json.load(config_file)
    # main(config)

    # Then in main function:
    # data_directory = config.get('data_directory')
