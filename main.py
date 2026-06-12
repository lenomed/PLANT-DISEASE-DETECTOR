import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.metrics import classification_report, confusion_matrix


training_set = keras.utils.image_dataset_from_directory(
    'PlantVillage/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
)

validation_set = keras.utils.image_dataset_from_directory(
    'PlantVillage/val',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
)
for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break
    # Building model


# To avoid overshooting 

#i chose a small learning rate because the model is deep and i want to make sure that the model converges and also to avoid overshooting and divergence of the model
#i increased the number of neurons in the dense layer because i want to increase the capacity of the model and also to avoid underfitting and also to capture more complex patterns in the data
#i added more convolutional layers to capture more complex features in the data and also to increase the capacity of the model and also to avoid underfitting and confusion between classes and also to capture more complex patterns in the data
model = Sequential()

#Convolution layer
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128,128,3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu')) # i didn't add padding here because i want to reduce the size of the feature map and also to avoid overfitting and number of parameters

#max pooling
model.add(MaxPool2D(pool_size=2, strides=2))    


#Convolution layer
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3,  activation='relu'))

#max pooling
model.add(MaxPool2D(pool_size=2, strides=2))   

#Convolution layer
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))

#max pooling
model.add(MaxPool2D(pool_size=2, strides=2))    

#Convolution layer
model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))

#max pooling
model.add(MaxPool2D(pool_size=2, strides=2))    

#Convolution layer
model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3,activation='relu'))

#max pooling
model.add(MaxPool2D(pool_size=2, strides=2))    
#Flattening our ferures before feeding them to the NN

model.add(Dropout(0.25))# to avoid overfitting
model.add(Flatten())

# adding dense layers

model.add(Dense(units=1500, activation='relu'))
model.add(Dropout(0.4))

#output later

model.add(Dense(units=38, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

#Model training
train_hist = model.fit(x=training_set, validation_data=validation_set, epochs=10)
print(train_hist)
print(train_hist.history)

#Model evaluation

training_loss, training_accuracy = model.evaluate(training_set)
print(f'Training Loss: {training_loss}')
print(f'Training Accuracy: {training_accuracy}')


validation_loss, validation_accuracy = model.evaluate(validation_set)
print(f'Validation Loss: {validation_loss}')
print(f'Validation Accuracy: {validation_accuracy}')

#Saving the model
#model.save('plant_disease_model.h5')# you can also save the model as .keras or .tf format
#.keras uses less space and is faster to save and load

#Saving the training history to a json file because i might plot it out later
with open('training_history.json', 'w') as f:
    json.dump(train_hist.history, f)

epochs = [i for i in range(1,11 )]
plt.plot(epochs, train_hist.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, train_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Testing the model
# i am going to use the validation set as a test set because i don't have a separate test set and also to evaluate the model on unseen data and also to avoid overfitting and also to get a better estimate of the model's performance on unseen data i will change the shuffle to False.

test_set = keras.utils.image_dataset_from_directory(
    'PlantVillage/val',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False, # change to False to evaluate on unseen data
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
)

#Predicting the classes of the test set
y_pred = model.predict(test_set)
print(y_pred, y_pred.shape)

#The output of the model is a probability distribution over the 38 classes for each image in the test set. To get the predicted class for each image, we can use the argmax function to get the index of the class with the highest probability.
predicted_classes = tf.argmax(y_pred, axis=1)
print(predicted_classes, predicted_classes.shape)

true_categories = [y for x,y in test_set    ]
true_categories = tf.concat(true_categories, axis=0)
print(true_categories, true_categories.shape)

y_true = tf.argmax(true_categories, axis=1)
print(y_true, y_true.shape)

# calculate precision recall and f1 score
print("Classification Report:-------------------------------------------")
print(classification_report(y_true, predicted_classes, target_names=test_set.class_names))
print('\n')
print("Confusion Matrix:-------------------------------------------")
con_matrix = confusion_matrix(y_true, predicted_classes, target_names=test_set.class_names)
# i added class_names to both the classification report and confusion matrix to get the class names instead of the class indices in the output. This will make it easier to interpret the results and also to identify which classes are being misclassified.
plt.figure(figsize=(50,54))
plt.xlabel('Predicted Classes', fontsize=42)
plt.ylabel('Actual Classes', fontsize=42)
plt.title('Plant DiseaseConfusion Matrix', fontsize=48)
sns.heatmap(con_matrix, annot = True, cmap='coolwarm', annot_kws={'size': 24})
plt.show()