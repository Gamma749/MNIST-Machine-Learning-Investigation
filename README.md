# MNIST-Machine-Learning-Investigation

## Table of Contents
1. [Introduction](#Introduction)
2. [Model Architectures and Names](#Model-Architectures-and-Names)
3. [Findings](#Findings)
    * [Convolutional Layers](#Convolutional-Layers)
    * [Dense Layers](#Dense-Layers)
    * [Unit Number](#Unit-Number)
    * [Dropout Rate](#Dropout-Rate)

# Introduction
Starting with Tensorflow and neural networks for the first time can be confusing and challenging. In this project, we are training on the MNIST dataset, as this is well known and has a huge amount of data (also the images are small and greyscale so I can process them quickly on my low-power machine). I have started looking at models and how hyperparameters change the model accuracy. Specifically, I have looked at changing the number of layers (convolutional and dense) number of units in each layer, and regularization techniques (dropout rate). To track these changes in hyperparameters, the model architecture is encoded into the model name.

If you're after the findings from this project, or to just look at some pretty graphs, feel free to skip ahead to the [findings](#Findings)

# Model Architectures and Names
<details>
<summary>Read about the model architectures</summary>

Models are created using the `create_model` function in the `main.py` file. This function takes a whole bunch of parameters so we can customise our models as much as we want. However, this means that the model architecture is locked in by this function (of course we can change it later, but the models should be consistent across any one test). For this project, the architecture has been set up as follows:
1. Convolutional, Max Pooling, and Batch Normalisation Layers
2. A Flatten layer
3. Dense and Dropout layers
4. A Softmax layer
If max pooling, batch normalisation, or dropout layers are present there will always be one for each convolutional or dense layer respectively. Also, there are some layers which are not optional and make up the basis of all models. These layers exist mainly just to massage the data into the right input/output shape or to manage data as it passes through the model.

The model's most basic architecture consists of
* A convolutional layer followed by a max pooling layer (to get image data into the model)
* A flattening layer (to vectorise the data)
* A dense layer followed by a softmax layer (to get the output of the model, digits 0-9 and express this as a probability)

Any extra convolutional (or related) layers go between the initial convolutional layer and the flatten layer, and any extra dense layers go between the flatten layer and the final dense layer.

</details>

<details>
<summary>Read about the model naming conventions</summary>
Model names are intended to quickly show the architecture of the model without having to load it and view the summary. This is important as we are creating a large number of similar models, so being able to differentiate them at a glance is useful. However, some layers are excluded from the title as they are present in every model and are more to just massage data.

The model's most basic architecture consists of
* A convolutional layer followed by a max pooling layer
* A flattening layer
* A dense layer followed by a softmax layer

These are present in every model, are *not* tracked in the model names, as they are present in every model and do not change across the tests.


Each section of the model's name tells us
1. The layer type is present
2. How many of that layer type there is
3. Any extra information attached to that layer type (filter size, dropout rate, etc...)

Each type of layer is separated from one another in the title by an underscore and the extra information is denoted by a hyphen. This is janky, but fits with Windows naming conventions.

For example, this model:
> MODEL_0CONVD-2.2_MAXPOOL-2.2_0DENSE_UNITS-32

Has no extra convolutional layers (which each have a filter shape of (2,2)), max pooling layers which also have a filter shape of (2,2), no extra dense layers, and each layer having 32 units each. This would initially seem to be a degenerate, empty model, but remember the model name does NOT account for the basic non-optional layers. So actually, this model has an architecture of 
```
Model: "MODEL_0CONVD-2.2_MAXPOOL-2.2_0DENSE_UNITS-32"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 27, 27, 32)        160
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 32)        0
_________________________________________________________________
flatten_2 (Flatten)          (None, 5408)              0
_________________________________________________________________
dense_6 (Dense)              (None, 10)                54090
_________________________________________________________________
softmax_2 (Softmax)          (None, 10)                0
=================================================================
Total params: 54,250
Trainable params: 54,250
Non-trainable params: 0
_________________________________________________________________
```
This is the most basic model that can be created in this project.
</details>


# Findings
* [Convolutional Layers](#Convolutional-Layers)
* [Dense Layers](#Dense-Layers)
* [Unit Number](#Unit-Number)
* [Dropout Rate](#Dropout-Rate)


This project has been looking at how changing the architecture of the model affects the accuracy. In order to test both accuracy and overfitting, we will also check validation accuracy on a validation set. For this dataset, we will use sparse categorical accuracy as we have definite labels that we are aiming for (classification not regression). For the following sections, we are changing only one part of the architecture at a time, as referenced by the section header. All graphs are made using `matplotlib` and `numpy`. All tests were done to 10 epochs as this is when the accuracy increases seem to plateau.

## Convolutional Layers
![Sparse Categorical Accuracy Graph for Convolutional Layers](/graphs/conv_layers.png)
![Validation Sparse Categorical Accuracy Graph for Convolutional Layers](/graphs/conv_layers_validation.png)

Increasing the number of layers can make a neural network as powerful as we want, at the cost of increased training time and much higher likelihood of overfitting. This can be seen in our convolutional layer tests. Adding more convolutional layers with no extra regularisation caused the model to severely under perform. More convolutional layers appears to create a lacking model, at least for the chosen architecture, which for this part of the test has no form of regularisation. If we add batch normalisation to try and alleviate the poor performance we see a huge increase in performance.

![Sparse Categorical Accuracy Graph for Convolutional Layers and Batch Normalisation](/graphs/conv_layers_batchnorm.png)
![Validation Sparse Categorical Accuracy Graph for Convolutional Layers and Batch Normalisation](/graphs/conv_layers_batchnorm_validation.png)

Now we are seeing a model that we could be proud of. Rather than trying to minimize the number of extra layers, as they were doing more harm than good, with regularisation techniques we see that the model can start to perform at a very high level. Even though the validation accuracy starts very low, batch normalisation quickly pulls this metric up to an acceptable standard, which means our model can classify images accurately even for data it hasn't seen before. Overall, this test seems to show that more layers is not always better, if we are not also somehow applying regularisation, and of course we still have that trade off of longer training times.

[Back to Top](#Findings)

## Dense Layers

![Sparse Categorical Accuracy Graph for Dense Layers](/graphs/dense_layers.png)
![Validation Sparse Categorical Accuracy Graph for Dense Layers](/graphs/dense_layers_validation.png)

Again, adding more dense layers without additional regularisation doesn't do much to increase accuracy. It looks like our model gets caught in some sort of local minima and without dropout or batch normalisation it cannot make its way out of this minima to a better one, hence why our accuracy plateaus at an abysmal amount. Let's try adding dense layers *with* dropout now and see how this affects accuracy.

![Sparse Categorical Accuracy Graph for Dense Layers with Dropout](/graphs/dense_layers_dropout.png)
![Validation Sparse Categorical Accuracy Graph for Dense Layers with Dropout](/graphs/dense_layers_dropout_validation.png)

Again, we see that adding regularisation in the form of dropout has made our model perform far better. Now, instead of being trapped in a local minima with poor accuracy we are seeing improvement right up until our tenth epoch. Even the validation accuracy is good, meaning our model performs very well with dropout added. As expected, having 0 extra Dense layers means dropout cannot have an effect, so this model is unaffected between the two runs. Also, having too many extra layers appears to perform worse than having just a few more (1 vs 4 in the dropout graphs). I think that with better tuning of the dropout rates, and with longer training times (more epochs) we would probably see the models with more dense layers meet or exceed those with lower dense layers, but for these limited tests that didn't have the chance to occur. For these tests, we didn't add anything complex like rotating the images, but I think that if we ahd then the extra dense layers (or maybe convolutional layers) would have been able to truly shine as those models could make better use of the added complexity in the architecture. 

[Back to Findings](#Findings)

## Unit Number

![Sparse Categorical Accuracy Graph for Unit Number](/graphs/unit_number.png)
![Validation Sparse Categorical Accuracy Graph for Dense Layers](/graphs/unit_number_validation.png)

Increasing unit numbers in each layer will make our model more complex and better able to classify, but will also take much longer to train and is more prone to overfitting or getting stuck in local minima. As seen here, the low unit number models are terrible as they do not have the complexity that neural networks rely on to be so powerful. If we have only a handful of units in each layer, they cannot hope to capture the entire image and all the data within to classify it well. With larger unit numbers, we are probably seeing some redundancy between units which explains the diminshing returns, but if we were to include more complexity in our data (e.g. rotation) I think that there would be a clear increase in accuracy as the larger number of units could help to capture some of the extra information. The use of dropout in these models also helped to exclude any issues discussed above also influencing this test. 

[Back to Findings](#Findings)

## Dropout Rate

![Sparse Categorical Accuracy Graph for Unit Number](/graphs/dropout_rates.png)
![Validation Sparse Categorical Accuracy Graph for Dense Layers](/graphs/dropout_rates_validation.png)

Looking at dropout rates we see pretty much what we expect. A dropout rate of 0 does as well as the models in previous tests that had no dropout, as that is exactly what is happening (but probably still wasting CPU cycles). Increasing dropout rates tends to get better accuracies up to a point. Around a rate of 0.2 or 0.3 we see the accuracy start to drop again (although validation accuracy remains high until a rate of around 0.5). This is because the nodes that are being "switched off" and changing the model accuracy are *not* switched off during validation, so we are still getting the full power of the model so to speak. 

[Back to Findings](#Findings)