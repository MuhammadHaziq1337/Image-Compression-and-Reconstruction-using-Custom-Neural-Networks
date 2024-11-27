# Image Compression and Reconstruction using Neural Networks

## Objective
The aim of this project is to build a neural network model that compresses and reconstructs images. The model employs an autoencoder architecture, where the encoder compresses the input images into a compact latent space, and the decoder reconstructs them back to their original form. The project explores the impact of neural network design and hyperparameters on image reconstruction quality.

---

## Dataset
The dataset used for this project is the **LIVE Image Compression Dataset**. This dataset includes original and JPEG-compressed images, which are used to evaluate the performance of the image compression and reconstruction model.

- **Image Resolution**: All images are resized to a fixed resolution of 224x224 pixels.
- **Normalization**: The pixel values are normalized to a range between -1 and 1 to facilitate better convergence during training.

---

## Approach

### Model Architecture
The solution utilizes an **autoencoder architecture** consisting of two primary components: the encoder and the decoder.

- **Encoder**: 
  - The encoder processes the input image through a series of convolutional layers and pooling layers, reducing the spatial dimensions to a compact latent representation.
  - It aims to capture the most significant features of the image in a smaller, more efficient representation.

- **Decoder**: 
  - The decoder reconstructs the image from the compressed latent representation by using fully connected layers followed by upsampling layers and convolutional layers to restore the image's original resolution.

### Activation Functions
- **ReLU (Rectified Linear Unit)**: Used for the hidden layers in both the encoder and decoder to introduce non-linearity, allowing the model to learn complex patterns in the image data.
- **Sigmoid**: Applied at the final output layer to ensure that the reconstructed pixel values are between 0 and 1 (normalized).

### Loss Function
- **Mean Squared Error (MSE)**: The model's training objective is to minimize the mean squared error between the original input images and their reconstructed versions. This metric measures the pixel-wise difference between the two images and guides the model to produce more accurate reconstructions.

### Optimizer
- The **Adam optimizer** is used for training. This optimizer helps the model converge faster and performs well for the given task due to its adaptive learning rate capabilities.

---

## Training and Evaluation

### Data Preprocessing
- Images are resized to a uniform size of 224x224 pixels.
- Each image is normalized using the mean and standard deviation values calculated from the training dataset to ensure better model performance.

### Hyperparameters
- **Learning Rate**: A learning rate of 0.001 was used for the Adam optimizer.
- **Batch Size**: The model was trained with a batch size of 64 to balance training speed and memory usage.
- **Epochs**: The model was trained for a specified number of epochs, with performance evaluated after each epoch on a separate validation set.

---

## Results

After training, the reconstructed images are compared to the original images. The quality of the reconstruction is measured using the **Mean Squared Error (MSE)** between the original and the reconstructed images. The lower the MSE, the better the model's ability to accurately reconstruct the images from the compressed representation.

---

## Conclusion

This approach demonstrates the potential of using autoencoders for image compression and reconstruction. The model was able to compress the image data into a lower-dimensional latent space and then reconstruct it with a reasonable level of accuracy. Further optimization of the network architecture and training process could lead to better reconstruction quality and compression efficiency.

---


