# %%
import tensorflow as tf
import cv2
import time
import numpy as np

new_model = tf.keras.models.load_model('best.h5')

# Initialize the camera
cap = cv2.VideoCapture(0) # 0 is the index of the built-in camera, change if you have multiple cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture frames from the camera
while True:
    ret, frame = cap.read() # read a frame from the camera
    cv2.imwrite('captured_image.jpg', frame)

    time.sleep(5)

    img = tf.keras.utils.load_img(
    'captured_image.jpg', target_size=(160, 160)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array)
    #print(predictions)
    score = tf.nn.relu(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    class_names=['No', 'yes']

    message = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    #print(predictions)
    #print(message)
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow(message, frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


