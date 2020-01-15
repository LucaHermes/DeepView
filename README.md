# DeepView - Classification visualization

This is an implementation of the DeepView framework that was presented in this paper: https://arxiv.org/abs/1909.09154

## Requirements

All requirements to run DeepView are in ```requirements.txt```. 
To run the notebook, ```torch==1.3.1``` and ```torchvision==0.4.2``` are required as well.

## Usage Instructions

 1. Create a wrapper funktion (like ```pred_wrapper``` in the Demo notebook) which receives a numpy array of samples and returns according class probabilities from the classifier as numpy arrays
 2. Initialize DeepView-object and pass the created method to the constructor
 3. Run your code and call ```add_samples(samples, labels)``` at any time to add samples to the visualization together with the ground truth labels.
    * The ground truth labels will be visualized along with the predicted labels
    * The object will keep track of a maximum number of samples specified by ```max_samples``` and it will throw away the oldest samples first
 4. Call the ```show``` method to render the plot

The following parameters must be specified on initialization:


| Variable    | Meaning             |
|----------------------|-------------------|
| ```pred_wrapper```    | To enable DeepView to call the classifier |
| ```classes```          | All possible classes in the data |
| ```max_samples```      | The maximum amount of samples that DeepView will keep track of |
| ```batch_size```       | The maximal size of batches that are passed to the classifier (the ```pred_wrapper```-function) |
| ```data_shape```       | Shape of the input data (complete shape; for images, include the channel dimension) |
| ```n```                | Number of interpolations for distance calculation of two images. |
| ```lam```              | Weighting factor for the euclidian component of the distance calculation. |
| ```resolution```       | x- and y- Resolution of the decision boundary plot |
| ```cmap```             | Name of the colormap that should be used in the plots. |

## The λ-Parameter

The λ-Hyperparameter weights the euclidian distance component. When the visualization doesn't show class-clusters, **try a smaller lambda** to put more emphasis on the discriminative distance component, which considers the classes. A smaller λ will normally pull the datapoints further into their class-clusters. Therefore, **if λ is too small**, this can lead to collapsed clusters that don't represent any structural properties of the datapoints. Of course this behaviour also depends on the data and how well the label corresponds to certain structural properties.


## Sample visualization

![visualization](https://user-images.githubusercontent.com/30961397/71091913-628e4480-21a6-11ea-8a26-d94f13907548.png)
