""" This program is meant to represent the simplest way to make a trigger word detector. The approach is 
    simple and involves a loop of listening to audio and then processing it. There is an obvious fault with
    this approach - namely, the predictive model takes some time to ultimately we might be missing chunks of
    audio. The other issue is this writes to disk and reopens because there was an issue with wave opening 
    a file not from disk.

    Since the model needs a continuous stream of 10 second audio clips, the initial 10 seconds will be a randomly
    selected background clip. Every 0.5 seconds, audio will be collected and then added to the end of this 10 
    second input. The first 5 seconds of the clip will be removed in order to keep the clip 10 seconds long.
"""

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pyaudio
import time
import wave
import io

rate = 16100
record_time = 0.5
chunk = 1024
size = 61
f_name = "./out.wav"

def main():
    background = start_up() # getting the background clip

    tf.reset_default_graph() # creating the tensorflow graph
    x = tf.placeholder(tf.float32, shape=[None,1402,118], name='X')
    out = rnn_cell(x)
    pred = prediction(out)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver = tf.train.import_meta_graph("../../data/trigger_word/model/trigger_model.ckpt.meta")
    saver.restore(sess, "../../data/trigger_word/model/trigger_model.ckpt")

    print("-----")
    print("-----")
    print("Program is ready, you can begin to talk.")
    print("-----")
    while(True): # continue to stream audio data
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,channels=2,
                rate=rate,input=True,frames_per_buffer=chunk)

        get_audio(stream,p)
        spec_data = get_specgram() # about 0.1sec to get specgram, shape (61,118)

        background.shape = (1402,118)
        background = np.concatenate((background[61:,:],spec_data)) # updating input to model
        background.shape = (1,1402, 118)

        _,desc_pred = sess.run(pred,feed_dict={x:background})
        check_if_trigger(desc_pred)
        
        stream.stop_stream()
        stream.close()

# checks if the model detected a trigger word and prints 
def check_if_trigger(desc_pred):
    check = np.sum(desc_pred[0,desc_pred.shape[1]-68:]) # counts the number of 1s
    if check > 5:
        print("Tigger word detected.")
    else:
        print("No trigger word.")

# load the model and the initial background noise file
def start_up():
    _,data = wavfile.read("./background.wav")
    pxx,_,_,_ = plt.specgram(data[:,0],234,8000,noverlap=120)
    lis = [pxx[:,i] for i in range(pxx.shape[1])]
    pxx = np.array(lis[:1402])
    return pxx

# gets a chunk of audio from the recording device and saves it to disk
def get_audio(stream,p):
    frames = []
    for i in range(0, int(rate / chunk * record_time)):
        data = stream.read(chunk)
        frames.append(data)
        get_wav_data(frames,stream,p)

# converts the pyaudio data into wav format and saves to disk
def get_wav_data(frames,steam,p):
    wf = wave.open(f_name, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# returns the spectogram for given wav data
def get_specgram():
    _,data = wavfile.read(f_name) #io.BytesIO
    pxx,_,_,_ = plt.specgram(data[:,0],234,8000,noverlap=120)
    lis = [pxx[:,i] for i in range(pxx.shape[1])] # get pxx to right shape
    pxx = np.array(lis)
    return pxx

# Forward prop step consisting of conv 1d, multilayered rnn, and dropout usage
def rnn_cell(the_input):
    # Conv 1D step:
    Z = tf.layers.conv1d(the_input,filters=196,kernel_size=28,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Bn = tf.layers.batch_normalization(Z)
    A = tf.nn.relu(Bn)
    D = tf.nn.dropout(A,keep_prob=0.8)
    # Multilayered GRU units with dropout:
    cell1 = tf.nn.rnn_cell.GRUCell(128,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1,output_keep_prob=0.8)
    cell2 = tf.nn.rnn_cell.GRUCell(128,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2,output_keep_prob=0.8)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2]) # multilayered cell
    outputs,curr_state = tf.nn.dynamic_rnn(multi_cell,inputs=D,dtype=tf.float32)
    flats = tf.map_fn(lambda x: tf.layers.flatten(x),outputs)
    out = tf.map_fn(lambda x: tf.layers.dense(x,1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),reuse=tf.AUTO_REUSE),flats)
    out = tf.reshape(out,[1,1375])
    return out

# Sigmoid prediction for a given vector of logits, returns both a sigmoid activation output and discrete classes - 0,1
def prediction(logits):
    sigmoid_out = tf.nn.sigmoid(logits)
    desc_out = tf.to_int32(sigmoid_out > 0.5)
    return sigmoid_out,desc_out

if __name__ == "__main__":
    main()