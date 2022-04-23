MODEL NAME : CNN _ RESNET9

Resnet is a classic CNN model which is used for MNIST dataset and has high prediction rates.

Python Packages:
CV2
PyTorch
matplotlib
torchvision
numpy

Process:

1. Mnist dataset is loaded from torchvision.
2. Load custom image - Handwritten image captured from Baxter's head camera and perform below actions
    to transform the image compatible with MNIST dataset
      a. Crop the image
      b. Convert the image to binary (same as MNIST)
      c. Resize the image to 28 X 28 size
      d. convert the image to tensor so that it can be loaded in to the model
3. Split the dataset in to Train, Test and Validation
4. Define convultional block and use it in the resnet model
5. Define a forward function which will calculate the values of output layers based on inputs.
6. Set up the model by providing the input channel and number of classes (in MNIST it will be 10 classes from 0 to 9)
7. Train, test and validate the model
8. Pass the custom tensor to the model which will predict the class.


