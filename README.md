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
