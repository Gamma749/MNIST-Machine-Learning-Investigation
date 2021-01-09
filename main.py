# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def create_model(input_shape, output_units, hidden_units, filter_shape,
                convolution_layers, dense_layers, 
                max_pooling=(2,2), batch_normalisation=False, dropout=False, 
                activation_type='relu',)->tf.keras.Sequential:
    """
    Create and return a new sequential model with the specified architecture
    This model will be suited for the task of image classification, or anything using a Convolutional network
    This model will have the form of

    convolution_layers number of Convd2D layers with hidden_units number of units(the first having input_shape)
    optional maxPool2D layers and batchNormalisation layers
    A flatten layer
    dense_layers number of Dense layers
    optional Dropout layers with dropout rate
    A final Dense layer with output_units number of units (non-optional)
    A Softmax layer to normalise the outputs

    Params:
    input_shape : tuple of integers
        tuple of integers, does not include the sample axis, the data shape given to the model
    
    output_units : integer
        The shape of the output ie how many classes we are classifying
    
    hidden_units : integer
        How many units each hidden layer will have

    filter_shape : tuple of integers
        shape of filters to use in Conv2D layers

    convolution_layers : integer
        How many Conv2D layers to have (not including first input layer)

    dense_layers : integer
        How many Dense layers to include (not including final output Dense layer)
    
    max_pooling : tuple of integers, optional
        Dimension of pooling to use, default is (2,2)

    batch_normalisation : Boolean, optional
        Whether to have a batch_normalisation layer after each Conv2d layer

    dropout : float, optional
        False if no dropout layer, or a float between 0.0 and 1.0 for dropout rate after each Dense layer

    activation_type : string, optional
        The type of activation to use for each layer, default ReLu
    """

    #Create the model
    model = tf.keras.Sequential()
    model._name = f'MODEL_{convolution_layers}CONVD-{filter_shape[0]}.{filter_shape[1]}_MAXPOOL-{max_pooling[0]}.{max_pooling[1]}_{"BATCHNORM_" if batch_normalisation else ""}{dense_layers}DENSE{"_DROPOUT-"+str(dropout) if dropout else ""}_UNITS-{hidden_units}'
    #add the first convd layer
    model.add(tf.keras.layers.Conv2D(hidden_units, filter_shape, activation=activation_type, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(max_pooling))
    if batch_normalisation:
        model.add(tf.keras.layers.BatchNormalization())

    #add a convolutional layer for each required
    for i in range(convolution_layers):
        if batch_normalisation:
            model.add(tf.keras.layers.BatchNormalization())
        try:
            model.add(tf.keras.layers.Conv2D(hidden_units, filter_shape, activation=activation_type))
            model.add(tf.keras.layers.MaxPool2D(max_pooling))
        except ValueError:
            #if MaxPool breaks the model, just don't add it
            pass
        
    #Add the flatten layer
    model.add(tf.keras.layers.Flatten())

    #Add the dense layers
    for i in range(dense_layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation=activation_type))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))

    #add the final dense layer and softmax
    model.add(tf.keras.layers.Dense(output_units, activation=activation_type))
    model.add(tf.keras.layers.Softmax())

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model



#import our data
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

#normalise the data to [0,1]
ds_train = ds_train.map(lambda image,label: (tf.cast(image, tf.float16) / 255., label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
# cache dataset for better performance
ds_train = ds_train.cache()
#Shuffle the training data for true randomness at each epoch
ds_train = ds_train.shuffle(512)
#Batches after shuffling
ds_train = ds_train.batch(256)
#Finally, prefetch for performance
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(lambda image,label: (tf.cast(image, tf.float16) / 255., label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(512)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

#Define what we are going to be training
log_path = "logs/"
model_path = "models/unit_number/"

histories = list()
csv_logger = tf.keras.callbacks.CSVLogger(log_path+'unit_number.csv', append=True, separator=';')
number_variants = 10
for current_variant in range(number_variants):
    print("-"*80)
    model = create_model(input_shape=ds_info.features["image"].shape,
    output_units=10, hidden_units=2**current_variant, filter_shape=(2,2), convolution_layers=2,
    dense_layers=2, batch_normalisation=False, dropout=0.2)
    print(f'{current_variant+1}/{number_variants}: {model.name}')
    model.summary()
    
    history = model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
        callbacks=[csv_logger]
    )

    model.save(f'{model_path}{model.name}', save_format='h5')
    histories.append(history)

for i in range(len(histories)):
    plt.plot(histories[i].history["sparse_categorical_accuracy"], marker='x')

plt.legend([f'{2**i} Units' for i in range(len(histories))], title="Legend", loc="upper left", bbox_to_anchor=(1.05,0.75), fontsize="xx-small")
plt.title("Sparse Categorical Accuracy vs Epochs\nfor fixed 2 CONV2D Layers, 2 Dense layers, Dropout Rate 0.2")
plt.xlabel("Epoch number")
plt.ylabel("Sparse Categorical Accuracy")
plt.ylim(0.0, 1.0)
plt.show()

for i in range(len(histories)):
    plt.plot(histories[i].history["val_sparse_categorical_accuracy"], marker='x')

plt.legend([f'{2**i} Units' for i in range(len(histories))], title="Legend", loc="upper left", bbox_to_anchor=(1.05,0.75), fontsize="xx-small")
plt.title("Validation Sparse Categorical Accuracy vs Epochs\nfor fixed 2 CONV2D Layers, 2 Dense layers, Dropout Rate 0.2")
plt.xlabel("Epoch number")
plt.ylabel("Validation Sparse Categorical Accuracy")
plt.ylim(0.0, 1.0)
plt.show()