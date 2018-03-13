# Neural Style Transfer
## Nowhere Developers Conference
### Luke Godfrey and Stephen Ashmore

This respository contains the source code for the Deep Learning with Artistic Style workshop at the Nowhere Developers 2018 Conference. This workshop was intended to be an overview and introduction to using neural networks without focusing on the underlying theory. Essentially, this is geared towards developers and hackers to start using neural networks in their own projects. It builds off of Keras and Tensorflow, two fantastic libraries for machine learning.


## Instructions


## The Code
- main.py: Contains the argument parser and the main entry point for the program. Look here for adding new arguments.
- image.py: This file contains helper methods for manipulating files, such as the load_image, and display_image methods.
- style_net.py: This file contains the StyleNet class, which handles all of the neural network and styling functionality. apply_style is the key method here, which handles setting up the loss functions and training the neural network.
- latent_value.py: Contains the LatentValue layer, which is used to store the stylized image as the neural network updates it.


## Further Reading
Here are some further reading, tutorials, and courses for those who want to learn more about neural networks and how to use them.
- A Gentle Introduction to Neural Networks: https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc  This brief introduction should give you the absolute basics of neural networks including their inspiration, the human brain.
- Stanford's CS231n, Convolutional Neural Networks for Recognition: http://cs231n.github.io/ Stanford has generously put their full class notes online, and this is a fantastic resource for everything about convolutional neural networks, the type of neural networks we will be using in the workshop.
- Fast.AI's Practical Deep Learning For Coders: http://course.fast.ai/  This is a short course designed to teach you everything you need to know to *use* deep learning and neural networks for practical tasks.
- An article specifically on Neural Styling: https://shafeentejani.github.io/2016-12-27/style-transfer/ Neural styling is the technique we will be covering in our workshop, its a great demo of  neural networks and we'll show you how to take what you learned from it and create your own neural networks.

And for even more in-depth reading into the theory behind and inner-workings of machine learning and AI:
- AI: A Modern Approach by Stuart Russell and Peter Norvig.
- Deep Learning by Ian Goodfellow and Yoshua Bengio.