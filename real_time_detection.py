#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import cv2
from model import *

def main():
    # Initialization:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, 'model/model.ckpt')
    cam = cv2.VideoCapture(0)

    while True:
        # Read image frame from camera:
        ret, frame = cam.read()
        assert ret == True

        # Image pre-processing:
        frame = frame[:, 80:560] # corp the center
        frame_display = frame.copy() # make a copy for displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale image
        frame = imresize(frame, [28, 28]) # resize to 28x28
        img = frame.astype(np.float32) # convert image to float32
        img = 255.0 - img # invert image
        img = img / 255.0 * 2.0 - 1.0 # normalize image
        img = img[np.newaxis, :, :, np.newaxis] # reshape from [28,28] to [1,28,28,1]

        # Feed image to neural network, get prediction:
        model_prediction = prediction.eval(feed_dict={x: img})
        cv2.putText(frame_display, str(model_prediction[0]), (240, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))
        cv2.imshow('frame', frame_display) # display for viewing

        # If user press 'q' then exit the program:
        if cv2.waitKey(10) == ord('q'):
            break

    # Clean up:
    cam.release()
    sess.close()

if __name__ == '__main__':
   main() 
