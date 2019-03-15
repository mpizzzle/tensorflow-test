import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from plot import plot_image
from plot import plot_value_array
from PIL import Image
from PIL import ImageOps  

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

#predictions = model.predict(test_images)
#num_rows = 5
#num_cols = 5
#num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#  plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  plot_image(i, predictions, test_labels, test_images, class_names)
#  plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  plot_value_array(i, predictions, test_labels)
#plt.show()

my_image = Image.open("clothes/IMG_20190314_153500.jpg").convert('L')
#my_image = Image.open("clothes/IMG_20190314_154002.jpg").convert('L')
resized = my_image.resize((28, 28), Image.ANTIALIAS)
np_array = np.array(ImageOps.invert(resized))
np_array.resize(28, 28)

#attempting to filter out some jpeg noise
for x in np.nditer(np_array, op_flags = ['readwrite']):
  if x[...] < 128:
    x[...] = 0
 
img = (np.expand_dims(np_array, 0))
prediction = model.predict(img)
print(prediction)

#plt.figure()
#plot_value_array(0, prediction, test_labels)
#_ = plt.xticks(range(10), class_names, rotation=45)

print(img)
print("\nPrediction: ", class_names[np.argmax(prediction[0])])
plt.imshow(np_array, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(True)
plt.show()
