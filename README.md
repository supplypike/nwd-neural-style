# Neural Style Transfer
## Nowhere Developers Conference
### Luke Godfrey and Stephen Ashmore

This respository contains the source code for the Deep Learning with Artistic Style workshop at the Nowhere Developers 2018 Conference. This workshop was intended to be an overview and introduction to using neural networks without focusing on the underlying theory. Essentially, this is geared towards developers and hackers to start using neural networks in their own projects. It builds off of Keras and Tensorflow, two fantastic libraries for machine learning.


## Instructions

To style an image, run the following code:

    python main.py content_image style_image -d -s 480

The full set of options are...

- `content_image`: The picture to use as a base image.
- `style_image`: The picture from which to draw style.
- `-c, --cpu_cores`: How many cores to use if running on a CPU.
- `-d, --display`: Whether to display progress in a new window.
- `-i, --iterations`: How many iterations of style to apply.
- `-o, --output_image`: The path of where to save the final product. By default, the output is not saved.
- `-s, --size`: The maximum length of either width or height for the input images. If omitted, the full image size is used, but the code will probably run slowly.
- `--content_weight`: The weight on the content loss.
- `--style_weight`: The weight on the style loss.
- `--variation_weight`: The weight on the variation loss.

For a better understanding of how the VGG16 network is built, look at vgg16.py.
To see how the network is used to stylize images, look at style_net.py.

There is also code to test classification using VGG16 trained on ImageNet.
To do that, run the following code:

    python test_vgg16.py image

The output is a list of the top 5 classes that the network guesses the image belongs to.

### Installing
If you wish to install the dependencies you can follow the following instructions.

**CPU Only**

For CPU use only, it is quite easy to install. You will need `python 2.7.13` (there is a way to get it working with Python 3, but we won't detail that here) as well as `pip`. The only command you should need to run is:

`pip install -r cpu_requirements.txt`

Then you can follow the above instructions to run the neural styling tool.

**GPU Support**

For use with a GPU this can be more complex. You'll need a few extra libraries including CUDA, CUDADNN, and some others. We recommend following the instructions at tensorflow.org, or picking up a pre-installed GPU instance from a cloud provider such as Paperspace, AWS, Google Cloud, etc. 

## The Code
- main.py: Contains the argument parser and the main entry point for the program. Look here for adding new arguments.
- image.py: This file contains helper methods for manipulating files, such as the load_image, and display_image methods.
- style_net.py: This file contains the StyleNet class, which handles all of the neural network and styling functionality. apply_style is the key method here, which handles setting up the loss functions and training the neural network.
- test_vgg16.py: This file is used for testing classification using the VGG16 network.

## Further Reading
Here are some further reading, tutorials, and courses for those who want to learn more about neural networks and how to use them.
- A Gentle Introduction to Neural Networks: https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc  This brief introduction should give you the absolute basics of neural networks including their inspiration, the human brain.
- Stanford's CS231n, Convolutional Neural Networks for Recognition: http://cs231n.github.io/ Stanford has generously put their full class notes online, and this is a fantastic resource for everything about convolutional neural networks, the type of neural networks we will be using in the workshop.
- Fast.AI's Practical Deep Learning For Coders: http://course.fast.ai/  This is a short course designed to teach you everything you need to know to *use* deep learning and neural networks for practical tasks.
- An article specifically on Neural Styling: https://shafeentejani.github.io/2016-12-27/style-transfer/ Neural styling is the technique we will be covering in our workshop, its a great demo of  neural networks and we'll show you how to take what you learned from it and create your own neural networks.

And for even more in-depth reading into the theory behind and inner-workings of machine learning and AI:
- AI: A Modern Approach by Stuart Russell and Peter Norvig.
- Deep Learning by Ian Goodfellow and Yoshua Bengio.
