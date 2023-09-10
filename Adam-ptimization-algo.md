# URL
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# Adam optimization algorithm :
- is an extension to **stochastic gradient descent** that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.
- Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to **update network weights** iterative based in training data.

# How Does Adam Work?
- Stochastic gradient descent maintains a single **learning rate** (termed alpha) for all weight updates and the learning rate does not change during training.
- A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

- Adam as combining the advantages of two other extensions of stochastic gradient descent. Specifically:
    - **Adaptive Gradient Algorithm (AdaGrad)** that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
    - **Root Mean Square Propagation (RMSProp)** that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

# Adam Configuration Parameters
- **alpha** Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is - updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
- **beta1** The exponential decay rate for the first moment estimates (e.g. 0.9).
- **beta2** The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
- **epsilon** Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).


```
TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
```


# What Is the Learning Rate?
Deep learning neural networks are trained using the stochastic gradient descent algorithm.

Stochastic gradient descent is an **optimization algorithm** that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the **back-propagation** of errors algorithm, referred to as simply backpropagation.

**The amount that the weights are updated during training is referred to as the step size or the “learning rate.”**

Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.

During training, the backpropagation of error estimates the amount of error for which the weights of a node in the network are responsible. Instead of updating the weight with the full amount, it is scaled by the learning rate.

This means that a learning rate of 0.1, a traditionally common default value, would mean that weights in the network are updated 0.1 * (estimated weight error) or 10% of the estimated weight error each time the weights are updated.