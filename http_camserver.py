from http.server import BaseHTTPRequestHandler, HTTPServer
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

image_size = 128
num_channels = 3
newHeight = 0
height = 0
width = 0
# Create a list to store the class labels
class_labels = ['Asphalt', 'Paved', 'Unpaved']

def run_inference(frame, sess, graph, x, y_pred):
    # Preprocess the frame
    #frame = frame[newHeight - 5 : height - 50, 0 : width]
    frame = cv2.resize(frame, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    #frame = np.array(frame, dtype=np.uint8)
    #frame = frame.astype('float32')
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




# Load the TensorFlow graphs and restore the models
graph = tf.Graph()
sess = tf.Session(graph=graph)
with graph.as_default():
    saver = tf.train.import_meta_graph('roadsurfaceType-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('typeCheckpoint/'))
    x = graph.get_tensor_by_name("x:0")
    y_pred = graph.get_tensor_by_name("y_pred:0")

# HTTPRequestHandler class
class HTTPRequestHandler(BaseHTTPRequestHandler):
    # GET
    # POST
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
    
    # Display the frame
        #cv2.imshow('Live Classification', post_data)
    # Save the image data to a file
        try:
            with open('received_image.jpg', 'wb') as f:
                f.write(post_data)
                print('Image saved successfully')
                

        except Exception as e:
            print('Error saving image:', e)
        
        frame = cv2.imread("./received_image.jpg")
        print("frame", frame)
        cv2.imshow("frame", frame)
        width = frame.shape[1]
        height = frame.shape[0]

        newHeight = int(round(height / 2))
        class_label, probability = run_inference(frame, sess, graph, x, y_pred)
        print("class", class_label)
    # Display the classification results on the frame
            
        cv2.rectangle(frame, (0, 0), (200, 100), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, 'Class: ', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, class_label, (100, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, 'Probability: ', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
        cv2.putText(frame, str(probability), (100, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))

    # Send response
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Image received')

# Server settings
host = '0.0.0.0'
port = 80

# Create server object with custom HTTPRequestHandler
server = HTTPServer((host, port), HTTPRequestHandler)

# Start the server
print('Starting server on {}:{}'.format(host, port))
server.serve_forever()
