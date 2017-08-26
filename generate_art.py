#Objective: take a basic image and style it

import numpy as np
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
#import image_processing as img # import image_processing as img #helper class to convert image to an array
from PIL import Image as img
from IPython.display import Image #image display


#Step 1 - prepare dataset
base_image = img.open('./base_image.jpg')
base_image = np.asarray(base_image)
print base_image.ndim
base_image = K.variable(base_image) #transform image to tensor

style_reference_image = img.open('./style_image.jpg')
style_reference_image = np.asarray(style_reference_image)
style_reference_image = K.variable(style_reference_image)


combination_image = K.placeholder((254,198,3)) #placeholder to initialize height and width
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0) #combine the 3 images into a single Keras tensor

#Step 2 - create the model
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False) #build our prebuilt model VGG16 with newly created tensor as input
model.add(Flatten(3))
#it already knows how to separate the image. 
#It has a loss function, error value to minimize. Content loss and style loss.
#content loss and style loss function which we want to minimize.
#shape is given by content classification. Then we calculate the loss on the content.
#then, we calculate style loss, is also minimized.
#eucledian to calculate loss (rankings, recommendations, similarities)

#Step 3 - training or activation of the features of the image
#we add a step to measure correlation of the activations
#runs a gram matrix to measure which features tend to activate together
#once we have this we can define the style loss using the eucledian distance between the images.
#we compute only one layer for content
#we compute a weighted "style" loss with several layers of the image.

#runs a function to update our output image to minimize the lost.
loss = img.combination_loss(model, combination_image)

#get the gradients of the generated image write the loss
grads = K.gradients(loss, combination_image)

#run optimization over pixels to minimize loss
combination_image = img.minimize_loss(grads, loss, combination_image)

#optimization technique to enhance paterns
image()