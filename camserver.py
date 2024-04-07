import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import sys
import os.path
import random
import os
import glob
import operator
import time

import requests

'''
INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)
'''

# ESP32 URL
URL = "http://192.168.31.49"
AWB = True
image_size = 128
num_channels = 3

# Create a list to store the class labels
class_labels = ['Asphalt', 'Paved', 'Unpaved']
# Face recognition and opencv setup
cap = cv2.VideoCapture(URL + ":81/stream")
#face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem
def run_inference(frame, sess, graph, x, y_pred):
    # Preprocess the frame
    frame = frame[newHeight - 5 : height - 50, 0 : width]
    frame = cv2.resize(frame, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    frame = np.array(frame, dtype=np.uint8)
    frame = frame.astype('float32')
    frame = np.multiply(frame, 1.0 / 255.0)
    
    # Prepare the frame for inference
    x_batch = frame.reshape(1, image_size, image_size, num_channels)
    
    # Run inference
    feed_dict_testing = {x: x_batch}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    
    # Get the predicted class and probability
    class_index = np.argmax(result)
    class_label = class_labels[class_index]
    probability = result[0, class_index]
    
    return class_label, probability

outputFile ="./eric_output.mp4"
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Get the frame dimensions
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

newHeight = int(round(height / 2))

# Load the TensorFlow graphs and restore the models
graph = tf.Graph()
sess = tf.Session(graph=graph)
with graph.as_default():
    saver = tf.train.import_meta_graph('roadsurfaceType-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('typeCheckpoint/'))
    x = graph.get_tensor_by_name("x:0")
    y_pred = graph.get_tensor_by_name("y_pred:0")

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

if __name__ == '__main__':
    set_resolution(URL, index=8)

    while cv2.waitKey(1) < 0:
        
        ret, frame = cap.read()
        class_label, probability = run_inference(frame, sess, graph, x, y_pred)

    # Display the classification results on the frame
            
        cv2.rectangle(frame, (0, 0), (200, 100), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, 'Class: ', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, class_label, (100, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, 'Probability: ', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, str(probability), (100, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
    
    # Display the frame
        cv2.imshow('Live Classification', frame)

    # Save the frame to the video output
        vid_writer.write(frame)
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray = cv2.equalizeHist(gray)

                #faces = face_classifier.detectMultiScale(gray)
                #for (x, y, w, h) in faces:
                 #   center = (x + w//2, y + h//2)
                    #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)
        if not ret:
            print("Classification done!")
            print("Results saved as: ", outputFile)
            cv2.waitKey(3000)
            break
            #cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    cap.release()
    vid_writer.release()

# Close the TensorFlow session
    sess.close()
            
            
               
            #if key == ord('r'):
             #   idx = int(input("Select resolution index: "))
              #  set_resolution(URL, index=idx, verbose=True)

            #elif key == ord('q'):
             #   val = int(input("Set quality (10 - 63): "))
              #  set_quality(URL, value=val)

            #elif key == ord('a'):
             #   AWB = set_awb(URL, AWB)

            #elif key == 27:
             #   break

    cv2.destroyAllWindows()
    