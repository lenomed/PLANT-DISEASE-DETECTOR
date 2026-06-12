import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from main import test_set
model= tf.keras.models.load_model('PLANT-DISEASE-DETECTOR/plant_disease_model.h5')

#DATA VISUALIZATION.............

image_path = 'PLANT-DISEASE-DETECTOR\PlantVillage\val'

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB format
# display the image
plt.imshow(img)
plt.title('Sample Image')
plt.show()

#Testing the model
#preprocess the image to the exact characteristics that the training sets had
image = keras.preprocessing.image.load_img(image_path, target_size=(128,128))
image_arr = keras.preprocessing.image.img_to_array(image)
image_arr =np.array(image_arr)# convert the shape of the image so that it will be in the form of batch, just as the training shapes were.
print(image_arr)

# perform predictionsssssssss
prediction = model.predict(image_arr)
print(prediction.shape)# you would see something like: 1,38
print(prediction.prediction.shape)# here you'd see all the 38 class that we are getting

result_indx = np.argmax(prediction) # out of the 38 possible answers it will pick the one with the highest probability
print(result_indx)

# let's find the actual classs name
model_predicted_class = test_set.class_names[result_indx]
print(model_predicted_class)
plt.imshow(image)
plt.title(f'Predicted Class: {model_predicted_class}')
plt.xticks([])  # Hide x-axis ticks
plt.yticks([])  # Hide y-axis ticks
plt.show()
