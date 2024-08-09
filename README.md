# Image Classification using Convolutional Neural Networks

This project involves building and training a Convolutional Neural Network (CNN) to classify images of dogs and cats using the PyTorch deep learning library. The dataset used is the popular Cats vs Dogs dataset from Kaggle.

## Project Overview

In this project, we perform the following steps:

1. **Install Dependencies**
   - Install PyTorch and other necessary libraries by running:
     ```bash
     pip install torch torchvision numpy matplotlib tqdm
     ```

2. **Download the Dataset**
   - Download the Cats vs Dogs dataset from Kaggle:
     ```bash
     kaggle competitions download -c dogs-vs-cats -p data/cats_vs_dogs/
     ```

3. **Preprocess the Data**
   - Preprocess the data, which includes:
     - Splitting the dataset into training and validation sets.
     - Normalizing the pixel values of the images.
     - Organizing the dataset into a suitable directory structure.

   This step is handled by the script `scripts/data_preprocessing.py`. To run the script, use:
   ```bash
   python scripts/data_preprocessing.py
   ```
4. **Define the CNN Model**
   - The CNN model architecture is defined in `models/cnn_model.py`. The model can be custom-built or you can fine-tune a pre-trained model like VGG16.

5. **Train the Model**
   - Train the model by running `scripts/train_model.py`. This script:
     - Defines an optimizer and a loss function.
     - Iterates over the dataset for a specified number of epochs.
     - Saves the trained model to disk.
  To run the script, use:
   ```bash
   python scripts/train_model.py
   ```
6. **Evaluate the Model**
   - Evaluate the model's performance on a test dataset by running `scripts/evaluate_model.py`. This script will load the trained model and calculate the accuracy on the test data.
   To run the script, use:
   ```bash
   python scripts/evaluate_model.py
   ```
