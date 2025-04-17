# Emotional Recognition Via Image Data

## Context

Understanding of emotional expression is a crucial component of human communication. Emotions can be expressed in a number of forms, including direct verbal communication, variations in tone, body language, and facial expressions. 

Effective communication with others requires an understanding of and adaptation to the changes in emotional state of the other party. However, for those who do not intuitively understand emotional expression, such as those with mental disorders (e.g., autism) or robots, multi-stage classifiers can provide emotional intuition so that these people can more effectively communicate with others according to their current emotional state. 

In this paper, we propose a multi-stage classification approach that, given image data, classifies facial expressions according to the six basic emotions (anger, fear, disgust, happiness, sadness, and surprise), along with a 'neutral' class for a lack of emotional expression. We find that this approach enables real-time emotional recognition via live-feed image data, which can be used to provide large-language models (LLMs) with emotional context when conversing with both humans and robots with facial expressibility. 

## Data

To train our model, we use the FER-2013+ facial expressions dataset provided by \cite{fer2013} and \cite{barsoum2016}. This dataset provides over $30,000$ examples of facial expressions tied to one of the seven emotional classes, split into training and testing subsets. 

Compared to the original FER-2013 dataset, the FER-2013+ dataset provides more robust emotional classification by providing a crowd-sourced list of labels that were deemed accurate emotional interpretations of each image according to humans. By considering images as valid for evaluation of multiple emotions, we expand the size of our dataset to $172,254$ training samples and $21,534$ testing samples. 

In addition to testing our model against these training and testing sets, we provide our model with image and video data and evaluate the accuracy of the model's predictions given our known intended emotional expression. 

## Pre-Processing

Consider an image $I_{raw}$ with height and width $(h, w)$ used to train or test our model. In order to improve model generalizability, we perturb our data with a series of transformations: horizontal flipping, a randomly selected rotation and translation, additive Gaussian noise, and simulated lighting changes via gamma correction. The application of each of these processes follows the order in which they are given, and the explicit application of these perturbations are given, in order, below. 

(https://latex.codecogs.com/svg.image?\begin{align}I_{flip}(x,y)=I_{raw}(w-x-1,y)\end{align}\begin{align}\begin{bmatrix}x\\y\end{bmatrix}_{rot}=\begin{bmatrix}cos(\theta)&-sin(\theta)\\sin(\theta)&cos(\theta)\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}_{flip},\qquad\theta\in[-15^\circ,15^\circ]\end{align}\begin{align}I_{trans}(x,y)=I_{rot}(x&plus;\epsilon_x,y&plus;\epsilon_y),\qquad\epsilon_x,\epsilon_y\in[-3,3]\end{align}\begin{align}I_{noise}=I_{trans}&plus;\eta,\qquad\eta\in\mathcal{N}(0,\sigma^2),\sigma=2\end{align}\begin{align}I_{pp}=I_{max}(\frac{I_{noise}}{I_{max}})^\gamma,\qquad&space;I_{max}=255,\gamma\in[0.8,1.2]\end{align})

Once perturbed, we convert our images to grayscale and resize them to $48 \times48$ due to this resolution's validation for preserving critical facial structures in our selected dataset \cite{goodfellow2013}.
