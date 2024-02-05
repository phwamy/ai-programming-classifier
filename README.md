<p align="center">#Built a Deep Neural Network Model for a Flower Image Classifier</p>
<h2 style="text-align: center;">AI Programming with Python Project</h2>

In this project, I developed code for an image classifier built with PyTorch and then converted it into a command-line application.

Project code for Udacity's AI Programming with Python Nanodegree program. 

## File Explaination
created: 01/11/2024

1. Image_Classifier_Project.ipynb: The code is developing in the `Image_Classifier_Project.ipynb`.
2. train.py: Train a new network on a data set with `train.py`. It prints out training loss, validation loss, and validation accuracy as the network trains. In default, it will save a checkpoint.pth file in the current directory.
* Basic usage: `python train.py data_directory`
* Options: 
    * Choose architecture: `python train.py data_dir --arch "resnet50" ` Choose one model from ("vgg16", "resnet50") to fine-tune for the classifier. 
    * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 4`  
    * Use GPU for training: `python train.py data_dir --gpu`
    * Check data images: `python train.py data_dir --check_image`
    * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory` 
3. predict.py: Predict flower name from an image with `predict.py` along with the probability of that name. It passes in a single image `/path/to/image` and returns image of the flower, and a barplot of the predicted name and class probability.
* Basic usage: `python predict.py /path/to/image checkpoint`
* Options: 
    * Return top K most likely classes: `python predict.py image checkpoint --top_k 3`
    * Mapping of categories to real names: `python predict.py image checkpoint --category_names cat_to_name.json`
    * Use GPU for inference: `python predict.py image checkpoint --gpu`
4. flowers: The folder includes tain, valid, and test flower images. It should be download via 
```
!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!mkdir -p flowers && tar -xzf flower_data.tar.gz -C flowers
```
5. cat_to_name.json: It is a directory mapping class lables with flower category names.
6. requirements.txt: the environment information for developing the project.

