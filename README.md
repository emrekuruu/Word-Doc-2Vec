# Word2Vec

This is a project for CS-449 / Natural Language Processing.

## Setup Instructions

### Prerequisites

Make sure you have Python 3.11 installed on your machine. You can download it from [python.org](https://www.python.org/).

### Install Poetry

Poetry is a dependency manager for Python. Follow the instructions below to install Poetry:

1. Open a terminal.
2. Run the following command to install Poetry: `pip install poetry`

### Set Up the Project

1. Install the project dependencies using Poetry: `poetry install`
2. Activate the virtual environment: `poetry shell`

### Running the Script

Once the virtual environment is activated, you can run the script to generate the figures: `python main.py`

### Outputs

All the resultant figures will be saved in the `figures` folder. The models are also saved in the `models` folder.

## Project Structure

├── models/ # Folder containing the pickled models

├── figures/ # Folder containing the generated figures 

├── main.py # Main script to run the project

├── embedders.py # Script for loading and managing word embedding models

├── data_processer.py # Script for processing data

├── visualizer.py # Script for visualization

├── README.md # This README file

├── pyproject.toml # Poetry configuration file