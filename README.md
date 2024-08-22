# Implementing Transformer (Attention Is All You Need) from Scratch with PyTorch

## Project's take-aways:
- Implementing Transformer from Scratch with PyTorch as a Python Package
- Notebook playground for all transformer's block-by-block
- Launching training process, saving trained model for each epoch
- Testing inference process with latest trained model


## Create Python environment using Conda

0. Navigate into the project directory: `cd project`
1. Create Python conda environment: `conda create -n ENV_NAME python=3.11`
2. Activate ENV_NAME environment: `conda activate ENV_NAME`
3. Install the required dependencies: 
    - `pip install -r requirements.txt`

## Notebook playground with transformer's building blocks and layers

- Using jupyterlab: `jupyter lab`
- Or, playing notebooks directly with VScode IDE

## Training
- It is recommended to train the transformer with GPU if possible. If GPU is available in your machine, installing appropriate Torch and Cuda for your system to activate GPU's usage. Follow the instructions in the 'requirements.txt' file.
- Launch the training process with the following command in the terminal: `python train.py`

## Inference
- Play with the notebook file: `nb_inference.ipynb`

# Towards GenAI Engineering
![GenAI](./image/GenAI-Course.png)
This project is a key part of my course on Udemy: [**Hands-on Generative AI Engineering with Large Language Models**](https://www.udemy.com/course/hands-on-generative-ai-engineering-with-large-language-model/?referralCode=0775DF5DDD432646AD97)