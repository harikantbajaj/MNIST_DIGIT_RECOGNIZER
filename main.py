# import os 
# import pandas as pd 
# import cv2  as cv
# import numpy as np
# import matplotlib.pyplot as plt 
# import tensorflow as tf

# data= tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test)= mnist.load_data()

# x_train =tf.keras.utils.normalize(x_train, axis=1)
# x_test =tf.keras.utils.normalize(x_test, axis=1)

# model =tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation ='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.keras')



# model =tf.keras.models.load_model('handwritten.keras')

# while os.path.isfile(f"python\Screenshot 2024-06-16 010156.png"):
#     img = cv2.imread(f"python\Screenshot 2024-06-16 010156.png")[:,:,0]
#     img = np.invert(np.array([img]))
#     prediction= model.predict(img)
#     print(f"this digit is pro a{np.argmax(prediction)}")
#     plt.imshow(img[0], cmap=plt.cm.binary)
#     plt.show()
    
        
        
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data()# split the data in training set as tuple

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data

loss , accuracy  =model.evaluate(x_test,y_test)
print(accuracy)
print(loss)



for x in range(0,10):
    # now we are going to read images it with open cv

    img=cv.imread(f'{x}.png')[:,:,0]#all of it and 1st and last one
    img=np.invert(np.array([img]))#invert black to white in images so that model wont get confues
    prediction=model.predict(img)
    print("----------------")
    print("The predicted value is : ",np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)#change the color in black and white
    plt.show()

