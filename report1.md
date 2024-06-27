### Overview of the model
#### Architecture
The architecture is borrowed from ResNet-18 for its efficiency in image classification, and keeping in mind the limited amount of resources available to keep the learnable parameters to a minimum.

#### Loss Functions
The standard Categorical Cross Entropy Loss cannot be used because of the noisy nature of the labels. To combat this Generalized Cross Entropy Loss from [here](https://arxiv.org/pdf/1805.07836.pdf) is used.

The Generalized Cross Entropy Loss encompasses the features of both MAE and CCE. MAE is inherently more robust to noisy training data, because it treats every sample in the same manner, and the error for a wrongly labelled sample will be treated the same as that for a correctly labelled sample. However, CCE would give higher bias to the former, and push the network towards categorizing the image according to the train label, rather than the actual label.


#### Speciality
The specialty of this model is that it can simultaneously test performance of various Deep Learning Architectures, and several different loss functions, just by tweaking a couple of hyperparameters.

The reason I chose to go with this architecture and loss function was because the reasoning presented in the paper behind not using either CCE or MAE was very intuitive in nature, and it made sense to therefore use a loss which encompassed the desired features from both losses. 

Thus, it would form a good baseline for the task, rather than trying models randomly.

#### References
[Resnet Architecture](https://arxiv.org/abs/1512.03385)
[Generalized Cross Entropy Loss](https://arxiv.org/pdf/1805.07836v4.pdf)