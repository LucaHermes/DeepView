# DeepView - Classification visualization

This is an implementation of the DeepView framework that was presented in this paper: https://arxiv.org/abs/1909.09154

## Requirements

All requirements to run DeepView are in ```requirements.txt```. 
To run the notebook, ```torch==1.3.1``` and ```torchvision==0.4.2``` are required as well.

## Usage Instructions

 1. Create a wrapper funktion like ```pred_wrapper``` which receives a numpy array of samples and returns according class probabilities from the classifier as numpy arrays
 2. Initialize DeepView-object and pass the created method to the constructor
 3. Run your code and call ```add_samples(samples, labels)``` at any time to add samples to the visualization together with the ground truth labels.
    * The ground truth labels will be visualized along with the predicted labels
    * The object will keep track of a maximum number of samples specified by ```max_samples``` and it will throw away the oldest samples first
 4. Call the ```show``` method to render the plot

The following parameters must be specified on initialization:


| Variable    | Meaning             |
|----------------------|-------------------|
| ```pred_wrapper```    | To enable DeepView to call the classifier |
| ```max_samples```      | The maximum amount of samples that DeepView will keep track of |
| ```img_size```         | Currently only images are supported as inputs, img size specifies width and height of the input samples |
| ```img_channels```     | Number of image channels |
| ```resolution```       | x- and y- Resolution of the decision boundary plot |
| ```cmap```             | Name of the colormap that should be used in the plots. |

## Sample visualization

![visualization](https://user-images.githubusercontent.com/30961397/71091913-628e4480-21a6-11ea-8a26-d94f13907548.png)
