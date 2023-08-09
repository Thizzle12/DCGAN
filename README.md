# DCGAN

## Examples of generated images
<div align="center">
    <img src="generated_images.svg" width="400" height="400" alt="css-in-readme">
</div>

## How the model works
The network is a 

## Install 
In order to run the training and the generation of images, you will need to install the requirements from the requirements file.  
Either install locally or create a new python virtual environment. 

Run: **pip install -r requirements.txt**

## Training
In order to train the Generator run: **python dcgan/training/train_model.py**  
in the *dcgan/model_params/params.yaml*, you will find the hyper parameters. Here you can change training parameters and add the path to the file containing your own training data. In the future I will add an argparser, so the parameters can be changed from the prompt.


