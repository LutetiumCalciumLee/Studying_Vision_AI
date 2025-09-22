# Computer Vision
***
## Chapter 1. Fundamentals of Image Classification
### The Problem: Semantic Gap
- **Computer vs. Human Vision**: A computer perceives an image as a large grid of numerical pixel values (e.g., values from 0-255 across RGB channels), while a human interprets it semantically (e.g., identifying an object as a "cat"). The core challenge is bridging this "semantic gap".
- **Image Representation**: An image is treated as a numerical array. For instance, a 32x32 pixel image with 3 color channels (RGB) is represented as a 3072-dimensional vector of numbers for the classifier.

### Core Challenges in Classification
Object recognition is made difficult by several factors of variation :
- **Illumination**: Changes in lighting can drastically alter pixel values.
- **Deformation**: Objects can appear in various non-rigid shapes and poses.
- **Occlusion**: Objects may be partially hidden from view.
- **Background Clutter**: The object of interest may blend in with a complex background.
- **Intraclass Variation**: Objects within the same class can look very different from one another (e.g., different breeds of cats).
***
## Chapter 2. Classification Models
### k-Nearest Neighbors (k-NN)
- **Method**: A simple algorithm that classifies an image based on the majority class of its "k" nearest neighbors in the training data, typically measured by L1 (Manhattan) or L2 (Euclidean) distance between pixel values.
- **Limitations**: The k-NN approach is rarely used for image classification because it is very slow during testing, pixel-level distance metrics are not robust to appearance changes, and it performs poorly in high-dimensional spaces like images (an issue known as the "curse of dimensionality").

### Parametric Approach: Linear Classifiers
- **Core Idea**: Instead of comparing to all training images, this approach defines a score function that maps the raw pixel data to class scores using a set of parameters (weights).
- **Score Function**: A simple linear score function is defined as **f(x, W) = Wx + b**, where 'x' is the input image data (stretched into a column vector), 'W' is the weight matrix, and 'b' is a bias vector.
- **Process**: The weight matrix 'W' acts as a set of templates, one for each class. The input image is compared against each class template via a matrix multiplication to generate a score for each class. The class with the highest score is the predicted label.
***
## Chapter 3. Model Training and Optimization
### Loss Function
- **Purpose**: A loss function (or cost function) quantifies how well the model's predicted scores align with the ground-truth labels in the training data. The goal of training is to find the set of weights 'W' that minimizes this loss.
- **Multiclass SVM Loss**: Also known as Hinge Loss, it aims to ensure that the score for the correct class is higher than the scores for all incorrect classes by at least a fixed margin.
- **Softmax Classifier (Cross-Entropy Loss)**: This function interprets the raw class scores as unnormalized log probabilities. It then computes the probability for each class and aims to maximize the probability of the correct class. The loss is the negative log-likelihood of the correct class.

### Regularization
- **Goal**: Regularization is a technique used to prevent the model from becoming overly complex and overfitting to the training data. It encourages the model to use simpler weights, aligning with the principle of Occam's Razor.
- **Method**: A regularization penalty term is added to the loss function. Common types include **L2 regularization** (penalizes the squared magnitude of weights), **L1 regularization** (penalizes the absolute value of weights), and **Elastic Net** (a combination of L1 and L2).

### Optimization
- **Objective**: The ultimate goal is to find the optimal weight matrix 'W' that minimizes the total loss, which is the sum of the data loss (from the SVM or Softmax function) and the regularization loss. This process is called optimization.

---

## **Chapter 4. Optimization Algorithms**
- **Gradient Descent**: The fundamental optimization strategy is to iteratively update the weights by taking steps in the direction opposite to the gradient of the loss function. The goal is to "follow the slope" downhill to find the minimum loss.
    - **Numerical vs. Analytic Gradient**: The gradient can be computed numerically (slow, approximate) or analytically using calculus (fast, exact, but error-prone). In practice, analytic gradients are used and verified with numerical checks.
- **Stochastic Gradient Descent (SGD)**: Instead of computing the full loss over the entire dataset, SGD estimates the gradient using a small batch of data. This is much faster but can be noisy.
- **Challenges with SGD**:
    - It can get stuck in **local minima or saddle points** where the gradient is zero.
    - It struggles with loss landscapes that are shaped differently in various directions (poor conditioning), leading to slow convergence.
- **Advanced Optimizers**:
    - **SGD+Momentum**: This method introduces a "velocity" term that accumulates a running mean of past gradients. It helps the optimizer move past local minima and accelerates convergence, especially in ravines. Nesterov Momentum is a slight variant that often performs better.
    - **AdaGrad**: It adapts the learning rate for each parameter, using smaller updates for frequently occurring features and larger updates for infrequent ones. Its main weakness is that the learning rate can shrink to become infinitesimally small over time.
    - **RMSProp**: This optimizer resolves AdaGrad's diminishing learning rate issue by using a moving average of the squared gradients, preventing it from growing too large.
    - **Adam**: This is one of the most popular optimizers, combining the ideas of Momentum and RMSProp. It uses both the first moment (the mean, like momentum) and the second moment (the uncentered variance, like RMSProp) of the gradients.
---
## **Chapter 5. Neural Networks and Backpropagation**
- **Backpropagation**: This is the core algorithm for training neural networks. It uses the **chain rule** from calculus to efficiently compute the gradient of the loss function with respect to every weight in the network. The process involves a forward pass (computing the output and loss) followed by a backward pass (propagating the gradient from the output layer back to the input layer).
- **Computational Graphs**: Neural networks can be represented as computational graphs, where nodes are operations (e.g., multiplication, addition) and edges are the data flow. Backpropagation is essentially the process of computing gradients on this graph.
- **Activation Functions**: These functions introduce non-linearity into the network, allowing it to learn complex patterns.
    - **Sigmoid**: Historically popular but has issues like "killing" gradients when neurons saturate (output is close to 0 or 1) and its output not being zero-centered.
    - **Tanh**: Solves the zero-centered problem of Sigmoid but still suffers from gradient saturation.
    - **ReLU (Rectified Linear Unit)**: The most common activation function. It computes `f(x) = max(0, x)`. It is computationally efficient and avoids saturation in the positive region but can "die" if its output is always zero.
    - **Leaky ReLU**: A variant of ReLU that allows a small, non-zero gradient when the unit is not active (`f(x) = max(0.01x, x)`), preventing the "dying ReLU" problem.
---
## **Chapter 6. Training Neural Networks: Practical Aspects**
- **Data Preprocessing**: It's common practice to preprocess input data by making it **zero-centered** (subtracting the mean) and **normalized** (dividing by the standard deviation). This helps the model train more effectively.
- **Weight Initialization**:
    - Initializing all weights to zero is a mistake because all neurons will compute the same thing.
    - Initializing with small random numbers can lead to vanishing gradients in deep networks, where activations shrink to zero.
    - Initializing with large random numbers can cause saturating activations and exploding gradients.
    - **Xavier Initialization** is a method that keeps the variance of activations and gradients consistent across layers, which is crucial for training deep networks.
- **Batch Normalization**: A technique that normalizes the activations of a layer for each mini-batch. It stabilizes and accelerates training by solving the "internal covariate shift" problem, where the distribution of layer inputs changes during training. It also acts as a form of regularization.
- **Regularization (Dropout)**: During training, **Dropout** randomly sets a fraction of neurons to zero at each forward pass. This prevents neurons from co-adapting too much and forces the network to learn more robust, redundant representations, reducing overfitting.
- **Regularization (Data Augmentation)**: This technique artificially expands the training dataset by creating modified copies of images (e.g., random flips, rotations, color jitter). This helps the model generalize better to unseen data.
---
## **Chapter 7. Convolutional Neural Networks (CNNs)**
- **Convolution Layer**: The core building block of a CNN. It preserves the spatial structure of the input image by convolving a set of learnable filters (kernels) across it. Each filter is specialized to detect a specific feature (e.g., an edge, a color blob). The output is a set of **activation maps** representing the detected features.
- **Pooling Layer**: This layer is used to downsample the spatial dimensions of the activation maps, making the representation smaller and more manageable. **Max Pooling** is the most common form, where a small window is slid over the map, and only the maximum value in the window is kept. This provides a degree of translation invariance.
- **Fully Connected (FC) Layer**: Typically found at the end of a CNN architecture, this layer takes the high-level features from the convolutional and pooling layers and maps them to the final class scores. Each neuron in an FC layer is connected to all activations in the previous layer.


