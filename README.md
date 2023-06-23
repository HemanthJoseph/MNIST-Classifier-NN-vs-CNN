# MNIST data classifier using Deep Learning

In this project, I used the MNIST dataset, which is a dataset of handwritten digits
which is mostly use for learning about classification algorithms in both Machine
Learning and Deep Learning applications. My framework on choice for this project is
PyTorch, an open source Deep Learning library. 

Here, I compare two simple Deep Learning approaches for classification, one being a
straightforward **Neural Network** architecture and another being a **Convolutional
Neural Network** architecture which we know is better when dealing with images as its 
filters can detect features which makes it an easier job for classification.

## Architectures of the models
### Neural Network Architecture
![Neural Network](/images/Model_NN.jpg)

### Convolutional Neural Network Architecture
![Convolutional Neural Network](/images/Model_CNN.jpg)

## Results Comparision
### Neural Network Results
![Results_NN](/images/Results_NN.jpg)

### Convolutional Neural Network Results
![Results_CNN](/images/Results_CNN.jpg)

As we can see with both of these models, the CNN architecture performs better for
the same data.

## Observations
Since the accuracy on the testing data is high too, we can say that our model has
generalized very well. The bias vs variance tradeoff is balanced in both the 
architectures. We haven't over-trained the data and avoided the common pitfalls 
that arise from such practices.
