from torchvision import models
import torch
from PIL import Image
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


# PROGRAMMER: PeiHsin Wang
# DATE CREATED: 1/11/2024 
# Purpose: Predict flower name from an image with predict.py along with the probability of that name.
# Action: Pass in a single image `/path/to/image` and return the flower name and class probability
# Basic usage: python predict.py /path/to/image checkpoint
# Options: 
# * Return top K most likely classes: python predict.py input checkpoint --top_k 3
# * Mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# * Use GPU for inference: python predict.py input checkpoint --gpu

def load_checkpoint(filepath, device, category_names):
    """
    Load pre-saved model, optimizer, and epochs.

    Parameters:
    - filepath: the filepath of the .ph file

    Returns:
    - Tuple: (model, optimizer, epochs, criterion)
    """
    checkpoint = torch.load(filepath, map_location = device)
    # load or rebuild the architecture of the saved model
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # In defualt, the model already has the class_to_name mapping
    # Here is for command line option to load the mapping
    if category_names != 'cat_to_name.json':
        with open(category_names, 'r') as f:
            category = json.load(f)
        model.class_to_name = category

    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    criterion = checkpoint['criterion']

    return model, optimizer, epochs, criterion

def process_image(image_path, debug=False):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    parameters: 
    - image_path: path of the image

    returns:
    - Numpy array of image
    '''
    # Load the image with PIL
    with Image.open(image_path) as im:
        if debug:
            print(im.format, im.size, im.mode)

        # Resize the image to have the shortest side is 256 pixels
        width, height = im.size

        if width >= height:
            new_height = 256
            new_width = int(new_height * (width / height))
        else:
            new_width = 256
            new_height = int(new_width * (height / width))

        im = im.resize((new_width, new_height))

        # Calculate the coordinates for cropping the center 224x224 portion
        width, height = im.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2

        im = im.crop((left, top, right, bottom))

        # Convert to floats between 0-1 and normalize them
        np_image = np.array(im)
        np_image = np_image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std

        # Reorder dimension for our model to predict
        np_image = np_image.transpose((2, 0, 1))
        t_image = torch.from_numpy(np_image).float()

        return t_image
    
def predict(image_path, model, device, topk=3):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.

    parameters:
    - image_path: path of the image
    - model: the model to be used for prediction
    - topk: return top K most likely classes, default = 3
    - gpu: use GPU for inference, default = True

    returns:
    - top probabilities of predicted classes
    - top classes of predicted image
    - top labels mapping to the classes
    - image tensor
    '''
    image = process_image(image_path)

    # Add a batch dimension to the single image tensor
    # The 'unsqueeze' function adds a dimension at the specified position (0 for batch)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    model.to(device)

    with torch.no_grad():
        model.eval()

        ps = torch.exp(model(image))
        probs, classes = ps.topk(topk, dim=1)

        # Convert into array
        probs, classes, model = probs.to('cpu'), classes.to('cpu'), model.to('cpu')
        probs = probs.data.numpy().squeeze()
        classes = classes.data.numpy().squeeze()

        labels = []
        for c in classes:
            # Check if the class index exists in the model's class_to_idx dictionary
            if str(c) in model.class_to_idx:
                label = model.class_to_idx[str(c)]  # Retrieve the label
                labels.append(label)
            else:
                # Handle cases where the class index is not found
                labels.append('Unknown')

    model.train()
    
    return probs, classes, labels

def imshow(image, ax=None):
    """
    Display the tensor image.
    
    parameters:
    - image: image tensor
    - ax: matplotlib axes

    returns:
    - display graph in the axes
    """
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
    ax.axis('off')

    return ax

def show_result(image, probs, classes, labels, category_names=None):
    '''
    Show the image and the barplot of the top K classes with probabilities.

    parameters:
    - image: image tensor
    - probs: top probabilities of predicted classes
    - classes: top classes of predicted image
    - labels: top labels mapping to the classes

    returns:
    - display image and barplot
    '''

    # Reorder the labels and probability
    top = list(zip(probs, classes, labels))
    sorted_top = sorted(top, key=lambda x: x[0], reverse=True)
    sorted_p, sorted_class, sorted_label = zip(*sorted_top)

    if category_names is not None:
        data = {'sorted_p': sorted_p, 'sorted_label': sorted_label}
        print(f'The probability of the top {args.top_k} classes are:')
        for i, l in enumerate(sorted_label):
            print(f'{l:<20} {sorted_p[i]:.4f}')
    else:
        data = {'sorted_p': sorted_p, 'sorted_class': sorted_class}
        print(f'The probability of the top {args.top_k} classes are:')
        for i, l in enumerate(sorted_classes):
            print(f'{l:<20} {sorted_p[i]:.4f}')
    df = pd.DataFrame(data)

    # Plot the image and the barplot
    fig, (ax1, ax2) = plt.subplots(figsize=(10,5), ncols=2)
    image = image.to('cpu')
    imshow(image, ax=ax1)

    sb.barplot(data=df, y='sorted_label', x='sorted_p', color='skyblue', ax=ax2)
    ax2.set_title('Class Probability')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Labels')

    plt.tight_layout()

    return plt.show();

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

    model, optimizer, epochs, criterion = load_checkpoint(args.checkpoint, device, args.category_names)
    probs, classes, labels = predict(args.image_path, model, device, args.top_k)
    image = process_image(args.image_path)
    show_result(image, probs, classes, labels, args.category_names)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load a pre-trained network to predict the class for an input image")
    # Define command line arguments
    parser.add_argument('image_path', type=str, help="Path to the image file to be predicted")
    parser.add_argument('checkpoint', type=str, help="Path to the checkpoint .ph file")
    parser.add_argument('--top_k', type=int, default=3, help="Return top K most likely classes")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Mapping of categories to real names")
    parser.add_argument('--gpu', action='store_true', default=True ,help="Use GPU for inference")
    args = parser.parse_args()
    main()

   