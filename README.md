# Emotional Recognition Via Image Data

## Background

Understanding of emotional expression is a crucial component of human communication. Emotions can be expressed in a number of forms, including direct verbal communication, variations in tone, body language, and facial expressions. 

Effective communication with others requires an understanding of and adaptation to the changes in emotional state of the other party. However, for those who do not intuitively understand emotional expression, such as those with mental disorders (e.g., autism) or robots, multi-stage classifiers can provide emotional intuition so that these people can more effectively communicate with others according to their current emotional state. 

## How to Use

Clone the repository into an empty folder.

```git clone https://github.com/803christian/emotion_recognition.git```

Inside this folder, you can run `emotion_classifier.py` as-is to train (~12 hrs) and test the model. To use the pre-trained model, comment out the line `clf.fit(fer_path, fer_plus_path)` and run the code. 

To run custom images, place the image within the directory and add the name of the image (with extension) to the list `custom_images`. 

| Dataset      | Samples  | Accuracy |
|--------------|---------:|---------:|
| Training     | 172,254  | 57.31%   |
| PublicTest   | 21,534   | 56.87%   |
| PrivateTest  | 21,534   | 55.78%   |

## Data

To train our model, we use the FER-2013+ facial expressions dataset provided by [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and [Microsoft](https://github.com/microsoft/FERPlus). This dataset provides over $30,000$ examples of facial expressions tied to one of the seven emotional classes, split into training and testing subsets. 

Compared to the original FER-2013 dataset, the FER-2013+ dataset provides more robust emotional classification by providing a crowd-sourced list of labels that were deemed accurate emotional interpretations of each image according to humans. By considering images as valid for evaluation of multiple emotions, we expand the size of our dataset to $172,254$ training samples and $21,534$ testing samples. 

In addition to testing our model against these training and testing sets, we provide our model with image and video data and evaluate the accuracy of the model's predictions given our known intended emotional expression. 

## Pre-Processing

Consider an image $I_{raw}$ with height and width $(h, w)$ used to train or test our model. In order to improve model generalizability, we perturb our data with a series of transformations: horizontal flipping, a randomly selected rotation and translation, additive Gaussian noise, and simulated lighting changes via gamma correction. The application of each of these processes follows the order in which they are given, and the explicit application of these perturbations are given, in order, below. 

```math
I_{flip}(x,y) = I_{raw}(w-x-1, y)
```

```math
\begin{bmatrix}
        x \\ y
    \end{bmatrix}_{rot}
    = \begin{bmatrix}
        cos(\theta)&-sin(\theta) \\ sin(\theta)&cos(\theta)
    \end{bmatrix}
    \begin{bmatrix}
        x\\y
    \end{bmatrix}_{flip}, \qquad \theta \in [-15^\circ, 15^\circ]
```

```math
I_{trans}(x, y) = I_{rot}(x+\epsilon_x, y+\epsilon_y), \qquad \epsilon_x,\epsilon_y \in [-3, 3]
```

```math
I_{noise} = I_{trans} + \eta, \qquad \eta \in \mathcal{N}(0, \sigma^2), \sigma=2
```

```math
I_{pp} = I_{max}(\frac{I_{noise}}{I_{max}})^\gamma, \qquad I_{max}=255, \gamma\in[0.8, 1.2]
```

Once perturbed, we convert our images to grayscale and resize them to $48 \times48$ due to this resolution's validation for preserving critical facial structures in our selected dataset.

## Feature Encoding

Given a pre-processed image $I_{pp}$, we compute a histogram of oriented gradients (HOG) for our image to encode the gradients of the contours of the face in the image as our primary feature (see [this paper](https://ieeexplore.ieee.org/document/1467360) for details). We then average this HOG with another HOG computed using $I_{pp}$ with enhanced contrast via Contrast Limited Adaptive Histogram Equalization (CLAHE) (see [this paper](https://ieeexplore.ieee.org/document/10420184) for details). The resulting averaged histogram is a set of extracted features from our image.

Using the dataset listed in this paper, the set of extracted features $\mathcal{X}_{HOG}$ has a dimensionality of $2,352$. In order to speed up our training process, we scale our data according to:

```math
    \mathcal{X}_{scale} = \frac{\mathcal{X}_{HOG} - \mu_{\mathcal{X}_{HOG}}}{\sigma_{\mathcal{X}_{HOG}}},
```

 where $\mu$ and $\sigma$ are the mean and standard deviation of the dataset, respectively, and then perform principal component analysis (PCA) to reduce our dataset dimensionality to $900$. Given our scaled data, we construct a covariance matrix $\Sigma$ according to:

```math
     \Sigma = \frac{1}{N-1} \mathcal{X}_{scale}^T \mathcal{X}_{scale}.
```

We then compute the eigenvalues and corresponding eigenvectors for the feature dataset's current dimensionality d. In order to reduce dimensionality to 900, we select the first $900$ eigenvectors from the ordered set to construct a transformation matrix W such that:

```math
    \mathcal{X}_{PCA} = \mathcal{X}_{scale} W.
```

In this work, we find that the reduced dimensionality from $2,352$ to $900$ still maintains $99.55$\% of the original feature dataset's variance. 

After data processing, we perform our classification according to the method in Method and test our method's accuracy on the training and test datasets. 

## Classification

Given our organized dataset for N facial images, where $\mathbf{x}_i$ is our feature vector and $y_i$ is our list of possible emotions, we predict the emotional label $\hat{y}$ according to the optimization:

```math
\hat{y} = \arg\max_{k \in \{1,\dots,7\}} \mathbf{w}_k^\top \mathbf{x} + b_k,
```
where $\mathbf{w}_k$ and $b_k$ are the weights and biases, respectively. 

In order to maximize the margin of our separating hyperplane, we incorporate a hinge loss function:

```math
\ell(\mathbf{w},b,\mathbf{x}_i,y_i) = \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i+b))
```

such that the loss is 0 for a correct classification and increases linearly depending on the magnitude of the misclassification. Adding an $\ell_2$ regularization term, the resulting optimization becomes:

```math
\min_{\mathbf{w}_k, b_k} \frac{1}{2} \|\mathbf{w}_k\|^2 + C \sum_{i=1}^N \max\left(0, 1 - y_i^{(k)}(\mathbf{w}_k^\top \mathbf{x}_i + b_k)\right)
```

for a tunable regularization hyperparameter $C=0.01$. This optimization is solved using coordinate descent for maximum $10,000$ iterations. Each emotion is classified separately to produce our converged weights and biases. This method yields a computational complexity of $O(n_{samples}n_{features}n_{classes})$. 

Finally, at test time, each test image is augmented $5$ times, producing $6$ total versions of each image. We make predictions for each version of each image, and the final prediction is determined according to a majority decision of each version of the image.

For additional specifics on the classifier, please see sci-kit's [LinearSVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). 
