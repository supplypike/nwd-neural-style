# Neural Style Transfer
## Nowhere Developers Conference
### Luke Godfrey and Stephen Ashmore




## The Code
- main.py: Contains the argument parser and the main entry point for the program. Look here for adding new arguments.
- image.py: This file contains helper methods for manipulating files, such as the load_image, and display_image methods.
- style_net.py: This file contains the StyleNet class, which handles all of the neural network and styling functionality. apply_style is the key method here, which handles setting up the loss functions and training the neural network.
- latent_value.py: Contains the LatentValue layer, which is used to store the stylized image as the neural network updates it.
