# Emotional Recognition Via Image Data

## Context

Understanding of emotional expression is a crucial component of human communication. Emotions can be expressed in a number of forms, including direct verbal communication, variations in tone, body language, and facial expressions. 

Effective communication with others requires an understanding of and adaptation to the changes in emotional state of the other party. However, for those who do not intuitively understand emotional expression, such as those with mental disorders (e.g., autism) or robots, multi-stage classifiers can provide emotional intuition so that these people can more effectively communicate with others according to their current emotional state. 

In this paper, we propose a multi-stage classification approach that, given image data, classifies facial expressions according to the six basic emotions (anger, fear, disgust, happiness, sadness, and surprise), along with a 'neutral' class for a lack of emotional expression. We find that this approach enables real-time emotional recognition via live-feed image data, which can be used to provide large-language models (LLMs) with emotional context when conversing with both humans and robots with facial expressibility. 

%===============================================================================
## Data

To train our model, we use the FER-2013+ facial expressions dataset provided by \cite{fer2013} and \cite{barsoum2016}. This dataset provides over $30,000$ examples of facial expressions tied to one of the seven emotional classes, split into training and testing subsets. 

Compared to the original FER-2013 dataset, the FER-2013+ dataset provides more robust emotional classification by providing a crowd-sourced list of labels that were deemed accurate emotional interpretations of each image according to humans. By considering images as valid for evaluation of multiple emotions, we expand the size of our dataset to $172,254$ training samples and $21,534$ testing samples. 

In addition to testing our model against these training and testing sets, we provide our model with image and video data and evaluate the accuracy of the model's predictions given our known intended emotional expression. 
