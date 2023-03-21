# MNIST classification

![](./images/mnist.jpg)

For MNIST model training:

For MNIST data classification:


To start detection:  
poetry run python main_detect.py --input_folder 'input_image_file' --output_folder 'output_image_folder' --saved_objects 'saved_objects.csv'

To see all parameters:
poetry run python main_detect.py --help

Config params are saved in config.py

### Before launching:
1. Before starting the program, you need to install poetry if it is not installed:  
curl -sSL https://install.python-poetry.org | python3 -
2. Install dependencies:
poetry install

Or install dependencies with pip:  
pip install -r requirements.txt


