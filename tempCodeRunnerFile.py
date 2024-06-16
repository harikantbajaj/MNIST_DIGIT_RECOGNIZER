model =tf.keras.models.load_model('handwritten.keras')

while os.path.isfile(f"python\Screenshot 2024-06-16 010156.png"):
    img = cv2.imread(f"python\Screenshot 2024-06-16 010156.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction= model.predict(img)
    print(f"this digit is pro a{np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    
        