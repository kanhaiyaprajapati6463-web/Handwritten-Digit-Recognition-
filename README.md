# Handwritten Digit Recognition (Deep Learning)

A beginner-friendly deep-learning project that classifies handwritten digits from **0 to 9** using the MNIST dataset and a neural network built with TensorFlow/Keras.

## Features
- Loads the MNIST dataset automatically.
- Trains a small neural network.
- Reports test accuracy.
- Predicts a digit from a test image.
- Saves a sample prediction image and the trained model.

## Recommended environment
Python 3.10–3.11 is recommended for compatibility with many TensorFlow releases. Training is easiest in Google Colab.

## Run locally
```bash
pip install -r requirements.txt
python main.py
```

The first run downloads the MNIST dataset and can take a few minutes.

## Run in Google Colab
1. Open https://colab.research.google.com/
2. Create a new notebook.
3. Upload `main.py` or paste its contents into a code cell.
4. Run `!pip install tensorflow numpy matplotlib` if needed.
5. Run the code. A GPU is optional for this small demonstration.

## How it works
1. MNIST provides 28x28 grayscale images labelled 0–9.
2. Pixel values are normalized to the range 0–1.
3. A neural network learns patterns in the training images.
4. The model predicts the label of a test image.

## Concepts demonstrated
Deep Learning, neural networks, classification, image data, training/testing, loss, accuracy, and prediction.

## Limitation
This educational model is trained on MNIST and may not reliably recognize arbitrary photos or handwritten images outside that dataset.
