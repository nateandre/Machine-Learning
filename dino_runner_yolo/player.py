"""
This is the main program which plays the chrome dino game
"""

import tensorflow as tf 
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from model import conv,forward_pass
from keras import backend as K
import cv2
import sys
import time

def main():
    driver = webdriver.Chrome("/Users/natethegreat/Desktop/chromedriver")
    run_model(driver)
    driver.quit()

# Initiates the dinosaur game
# Returns element associated with the game
def initiate(driver):
    print("Starting game.")
    driver.get('https://www.google.com')
    time.sleep(1) # wait for load
    game = driver.find_element_by_id("t") # start game
    game.send_keys(Keys.ARROW_UP)
    return game

# Runs the tensorflow model
def run_model(driver):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32,shape=(None,448,448,3))
    Z = forward_pass(X)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("../../data/dinorunner/models_stochastic/yolo_model.ckpt.meta")
        saver.restore(sess, "../../data/dinorunner/models_stochastic/yolo_model.ckpt")
        print("Loaded tensorflow model.")
        cam = cv2.VideoCapture(0) # initializing camera
        game = initiate(driver) # initiate game after model loaded
        while(True):
            img = get_image(cam)
            encoding = sess.run(Z,feed_dict={X:img}) # get image encoding
            key = get_key(encoding)
            send_key(game,key)

# Get which key should be pressed based on image encoding
# Finds the cell with the highest probability of an object being there to get key value
def get_key(encoding):
    encoding.shape = (14,14,14)
    sec_1 = encoding[:,:,0:7]
    sec_2 = encoding[:,:,7:14]
    indic1 = np.argmax(sec_1[:,:,0])
    indic2 =  np.argmax(sec_2[:,:,0])
    conf_score_1 = sec_1[indic1//14,indic1-(indic1//14*14),0]
    conf_score_2 = sec_2[indic2//14,indic2-(indic2//14*14),0]

    if conf_score_1 > conf_score_2: # Use sec_1 encoding 
        key = get_key_value(sec_1[indic1//14,indic1-(indic1//14*14),5:7])
    else: # Use sec_2 encoding
        key = get_key_value(sec_2[indic2//14,indic2-(indic2//14*14),5:7])

    return key

# Gets the highest prob classification from encoding
# enc shape (2,), where enc[0] is prob score for open palm and enc[1] is for closed palm
def get_key_value(enc):
    if enc[0] > enc[1]:
        return "up"
    else:
        return "down"

# Send the correct key to the game, up or down arrow/do nothing
def send_key(game,key):
    if key == "up": # send up arrow
        game.send_keys(Keys.ARROW_UP)
        print("UP")
    else: # either don't send an arrow at all or send down arrow
        # currently I turn off sending a down arrow
        # game.send_keys(Keys.ARROW_DOWN)
        print("DOWN")


# Gets images from the webcam and resizes them to (448,448,3)
def get_image(cam):
    noerror, img = cam.read()
    if noerror:
        img = cv2.resize(img,(448,448),interpolation=cv2.INTER_AREA)
        img.shape = (1,448,448,3)
        return img
    else:
        print("Could not get an image.")
        sys.exit

if __name__ == "__main__":
    main()