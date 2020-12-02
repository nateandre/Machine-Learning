""" Pointer-generation model implementation

Author: Nathaniel Andre
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Bidirectional,LSTM,Input,RepeatVector,Activation,Softmax,Embedding,Dot,Lambda
from tensorflow.keras.layers import Softmax,Concatenate,Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adagrad
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np
from sklearn.utils import shuffle # does not shuffle in place
import sys
from datetime import datetime
import time
import pytz

import warnings
warnings.filterwarnings('ignore')


def main():
    """ main function
    """
    prefix="_500"
    data_dir= "../data/"
    x = np.load(data_dir+"x{}.npy".format(prefix))
    x_indices = np.load(data_dir+"x_indices{}.npy".format(prefix)) # shape:(5900, 500)
    att_mask = np.load(data_dir+"att_mask{}.npy".format(prefix))
    loss_mask = np.load(data_dir+"loss_mask{}.npy".format(prefix)) # shape:(5900, 101)
    decoder_x = np.load(data_dir+"decoder_x{}.npy".format(prefix))
    y_indices = np.load(data_dir+"y_indices{}.npy".format(prefix))
    embedding_matrix = np.load(data_dir+"word_embeddings.npy".format(prefix)) # (30000,100)
    
    np_int = "int32" # uploaded data should already be of these types
    np_float = "float32"

    train_end = 5500
    x_train = x[0:train_end] # shape:(5500, 500)
    x_indices_train = x_indices[0:train_end] 
    att_mask_train = att_mask[0:train_end]
    loss_mask_train = loss_mask[0:train_end]
    decoder_x_train = decoder_x[0:train_end]
    y_indices_train = y_indices[0:train_end]

    test_start = 5500
    test_val_size= 200
    x_val = x[test_start:test_start+test_val_size] # shape:(200, 500)
    x_indices_val = x_indices[test_start:test_start+test_val_size]
    att_mask_val = att_mask[test_start:test_start+test_val_size]
    loss_mask_val = loss_mask[test_start:test_start+test_val_size]
    decoder_x_val = decoder_x[test_start:test_start+test_val_size]
    y_indices_val = y_indices[test_start:test_start+test_val_size]

    x_test = x[test_start+test_val_size:test_start+test_val_size*2] # shape:(200, 500)
    x_indices_test = x_indices[test_start+test_val_size:test_start+test_val_size*2]
    att_mask_test = att_mask[test_start+test_val_size:test_start+test_val_size*2]
    loss_mask_test = loss_mask[test_start+test_val_size:test_start+test_val_size*2]
    decoder_x_test = decoder_x[test_start+test_val_size:test_start+test_val_size*2]
    y_indices_test = y_indices[test_start+test_val_size:test_start+test_val_size*2]

    tf_float = tf.float32
    tf_int = tf.int32
    continue_training=False # if should continue training from the previous trained parameters; note the optimizer will be starting from scratch
    use_dropout=False
    batch_size= 10
    optimizer = Adagrad(learning_rate=0.05,initial_accumulator_value=0.1,clipnorm=2.0) # Adam(lr=0.01)
    epochs=20
    use_coverage_loss=False
    coverage_lam=0.0
    model_save_path="../model_params/" # for loading model parameters
    model_checkpoints_path="../model_params/" # for storing model information
    model_checkpoints_name="model_checkpoints.txt"

    print(x.shape,x_train.shape,x_val.shape,x_test.shape,embedding_matrix.shape)
    print("\nData processing done.\n")
    start = time.time()
    encoder,decoder = get_pointer_gen_network(embedding_matrix=embedding_matrix,embedding_dim=100,input_len=500,tf_float=tf_float,tf_int=tf_int,use_dropout=use_dropout)
    print("\nModel initialized. Took {} min.\n".format(round((time.time()-start)/60,2)))

    if continue_training is True:
        # loading model weights:
        encoder.load_weights(model_save_path+"encoder")
        decoder.load_weights(model_save_path+"decoder")
        # loading optimizer state:
        grad_vars = encoder.trainable_variables+decoder.trainable_variables
        optimizer.apply_gradients(zip([tf.zeros_like(w) for w in grad_vars],grad_vars)) # giving optimizer information about the trainable weights, so the old values can be loaded
        optimizer_weights = np.load(model_save_path+"optimizer_weights.npy",allow_pickle=True)
        optimizer.set_weights(optimizer_weights)
        print("\nModel parameters loaded.\n")

    with tf.device('/device:GPU:0'): #tf.device('/device:GPU:0'): # ensure the GPU is being used during training
        train_model(x_train,x_indices_train,att_mask_train,loss_mask_train,decoder_x_train,y_indices_train,x_val,x_indices_val,att_mask_val,loss_mask_val,decoder_x_val,y_indices_val,x_test,x_indices_test,att_mask_test,loss_mask_test,decoder_x_test,y_indices_test,encoder,decoder,batch_size,optimizer,epochs=epochs,coverage_lam=coverage_lam,use_coverage_loss=use_coverage_loss,model_save_path=model_save_path,model_checkpoints_path=model_checkpoints_path,model_checkpoints_name=model_checkpoints_name)


def apply_scatter_nd(updates,indices,tf_int,tf_float):
    """ applies scatter_nd over the batch dimension
    """
    out = Lambda(lambda entry: K.map_fn(lambda entry: tf.scatter_nd(entry[0],entry[1],tf.constant([30100],dtype=tf_int)),entry,dtype=tf_float))([indices,updates]) # assuming a max vocab_size+unique_words_in_input of 30000+100
    return out


def apply_scatter_nd_add(tensor,updates,indices,tf_int,tf_float):
    """ applies the tensor_scatter_nd_add over the batch dimension
    """
    out = Lambda(lambda entry: K.map_fn(lambda entry: tf.tensor_scatter_nd_add(entry[0],entry[1],entry[2]),entry,dtype=tf_float))([tensor,indices,updates])
    return out


def pointer_gen_encoder(embedding_layer,encoder_h=128,input_len=500,tf_int=tf.int32,use_dropout=False):
    """ Returns the encoder portion of the pointer-gen network
    """
    x = Input(shape=(input_len),dtype=tf_int) # input to the encoder
    input_e = embedding_layer(x) # embeddings for the input
    if use_dropout:
        input_e = Dropout(0.25)(input_e)
    h = Bidirectional(LSTM(encoder_h,activation="tanh",return_sequences=True),merge_mode="concat")(input_e) # encoder
    
    model = Model(inputs=[x],outputs=[h])
    return model


def pointer_gen_decoder(embedding_layer,decoder_lstm,att_w1,att_w2,att_w3,att_v,vocab_d,vocab_d_pre,pgen_w1,pgen_w2,pgen_w3,encoder_h=128,input_len=500,output_len=101,tf_float=tf.float32,tf_int=tf.int32):
    """ Returns the decoder portion of the pointer-gen network
    args:
        input_len: the length of the input sequence (to the encoder)
        output_len: the length of the output sequence (from the decoder)
        tf_float,tf_int: defining datatypes for use in this model
    """
    h = Input(shape=(input_len,encoder_h*2),dtype=tf_float) # the input embedding from the encoder model
    x_indices_ = Input(shape=(input_len),dtype=tf_int) # represents where each input word prob. should be added in joint prob. vector
    x_indices = tf.expand_dims(x_indices_,axis=-1)
    fixed_vocab_indices_ = Input(shape=(30000),dtype=tf_int) # the size of the input vocabulary
    fixed_vocab_indices = tf.expand_dims(fixed_vocab_indices_,axis=-1)
    att_mask = Input(shape=(input_len),dtype=tf_float) # mask used with the attention distribution to mask out padding
    decoder_x = Input(shape=(output_len),dtype=tf_int) # delayed y_data for input to the decoder (for teacher-forcing)
    y_indices = Input(shape=(output_len),dtype=tf_int) # indices of the correct word in the joint_probabilities vector
    s_ = Input(shape=(256),dtype=tf_float) # decoder_h
    c_ = Input(shape=(256),dtype=tf_float)
    coverage_vector_ = Input(shape=(input_len),dtype=tf_float)
    s,c,coverage_vector = s_,c_,coverage_vector_
    
    decoder_e = embedding_layer(decoder_x) # embeddings for delayed input to the decoder
    outputs = [] # stores probability of correct ground-truth predictions at each decoder output step
    coverage_loss_contributions = [] # stores coverage loss contribution for each decoder output step
    
    for i in range(output_len): # loop through each step of the decoder
        decoder_input = decoder_e[:,i,:]  # input to the decoder at this timestep
        s,_,c = decoder_lstm(tf.expand_dims(decoder_input,axis=1),initial_state=[s,c])
        
        # calculating attention (probabilities over input):
        s_rep = RepeatVector(input_len)(s) # copying the decoder hidden state
        e = att_v(Activation("tanh")(att_w1(h)+att_w2(s_rep)+att_w3(tf.expand_dims(coverage_vector,axis=-1)))) # unscaled attention
        e = tf.squeeze(e,axis=-1)+att_mask # using attention mask (masks out padding in the input sequence)
        a = Activation("softmax")(e) # scaled attention (represents prob. over input)
        
        # handling coverage vector computations:
        step_coverage_loss = tf.reduce_sum(tf.minimum(coverage_vector,a),axis=-1) # cov loss at this decoder step
        coverage_loss_contributions.append(step_coverage_loss)
        coverage_vector+=a
        
        # calculating probabilities over fixed vocabulary:
        context = Dot(axes=1)([a,h]) # calculating the context vector
        pre_vocab_prob = Concatenate()([s,context])
        pre_vocab_prob = vocab_d_pre(pre_vocab_prob) # extra Dense layer
        pre_vocab_prob = vocab_d(pre_vocab_prob)
        vocab_prob = Activation("softmax")(pre_vocab_prob)
        
        # calculation probabilty for text generation:
        pre_gen_prob = pgen_w1(context)+pgen_w2(s)+pgen_w3(decoder_input)
        gen_prob = Activation("sigmoid")(pre_gen_prob)
    
        # calculating joint-probability for generation/copying:
        vocab_prob *= gen_prob # probability of generating a word from the fixed vocabulary
        copy_prob = a*(1-gen_prob) # probability of copying a word from the input
        
        # creating the joint-probability vector:
        vocab_prob_projected = apply_scatter_nd(vocab_prob,fixed_vocab_indices,tf_int,tf_float)
        joint_prob = apply_scatter_nd_add(vocab_prob_projected,copy_prob,x_indices,tf_int,tf_float)
        
        # gathering predictions from joint-probability vector - doing it here will reduce memory consumption
        y_indices_i = tf.expand_dims(y_indices[:,i],axis=-1) # getting predictions at time i for whole batch
        predictions_i = tf.squeeze(tf.gather(joint_prob,y_indices_i,batch_dims=1,axis=-1),axis=-1)
        outputs.append(predictions_i)
    
    prediction_probabilities = K.permute_dimensions(tf.convert_to_tensor(outputs),(1,0))
    coverage_loss_contributions = K.permute_dimensions(tf.convert_to_tensor(coverage_loss_contributions),(1,0))
    
    model = Model(inputs=[h,x_indices_,decoder_x,att_mask,y_indices,s_,c_,coverage_vector_,fixed_vocab_indices_],outputs=[prediction_probabilities,coverage_loss_contributions])
    return model


def get_pointer_gen_network(embedding_matrix,embedding_dim=100,input_len=500,tf_float=tf.float32,tf_int=tf.int32,use_dropout=False,output_len=101):
    """ initializes re-used model layers and creates the pointer-gen keras model object
    args:
        embedding_matrix: the matrix of pretrained weights
        embedding_dim: the dimensionality of the word embeddings
    """
    embedding_layer = Embedding(input_dim=30000,output_dim=embedding_dim,weights=[embedding_matrix],trainable=True,mask_zero=True) # re-used for both the encoder and decoder
    decoder_h=256
    encoder_h=128
    decoder_lstm = LSTM(decoder_h,activation="tanh",return_state=True)
    att_w1 = Dense(256,use_bias=True,activation=None)
    att_w2 = Dense(256,use_bias=True,activation=None)
    att_w3 = Dense(256,use_bias=True,activation=None) # should be 256x1 weight matrix
    att_v = Dense(1,use_bias=False,activation=None)
    vocab_d_pre = Dense(512,use_bias=True,activation="relu") # an additional hidden layer before prediction vocab probs.
    vocab_d = Dense(30000,use_bias=True,activation=None) # 30000 is fixed_vocabulary size
    pgen_w1 = Dense(1,use_bias=True,activation=None)
    pgen_w2 = Dense(1,use_bias=True,activation=None)
    pgen_w3 = Dense(1,use_bias=True,activation=None)

    if use_dropout:
        print("\nUsing Dropout.\n")
    
    encoder = pointer_gen_encoder(embedding_layer,encoder_h=encoder_h,input_len=input_len,tf_int=tf_int,use_dropout=use_dropout)
    decoder = pointer_gen_decoder(embedding_layer,decoder_lstm,att_w1,att_w2,att_w3,att_v,vocab_d,vocab_d_pre,pgen_w1,pgen_w2,pgen_w3,encoder_h=encoder_h,input_len=input_len,output_len=output_len,tf_float=tf_float,tf_int=tf_int)
    return encoder,decoder


def loss_function(prediction_probabilities,loss_mask,coverage_loss,lam,use_coverage_loss,return_indiv_loss=False):
    """ Returns the loss for this batch - also allows for the returning of the loss value for the given input
    args:
        prediction_probabilities: model-assigned probabilities for ground-truth predictions
        loss_mask: vector of 1s,0s specifying whether an input should contribute to the loss
        coverage_loss: coverage loss for this batch of examples
        lam: hyperparameter determining the contribution of coverage_loss to overall loss
        use_coverage_loss: whether coverage loss should be used
    """
    p_words = -tf.math.log(prediction_probabilities)
    p_words *= loss_mask # applying the loss mask
    p_words = tf.reduce_sum(p_words,axis=-1)
    general_loss_component = tf.reduce_mean(p_words)
    
    # incorporating the coverage loss:
    coverage_loss_component = 0
    if use_coverage_loss:
        coverage_loss *= loss_mask # applying the loss mask
        coverage_loss = tf.reduce_sum(coverage_loss,axis=-1)
        coverage_loss_component = lam*tf.reduce_mean(coverage_loss)
        
    total_loss = general_loss_component+coverage_loss_component
    if return_indiv_loss:
        indiv_losses = p_words
        if use_coverage_loss:
            indiv_losses+=coverage_loss
        return total_loss,indiv_losses
    else:
        return total_loss


def get_validation_set_loss(x_val,x_indices_val,att_mask_val,loss_mask_val,decoder_x_val,y_indices_val,encoder,decoder,batch_size,coverage_lam,use_coverage_loss,epoch,checkpoints_path,s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset):
    """ Get the average loss for the validation set
        -also saves the validation and test losses for each example to a file
    """
    losses = []
    checkpoints_file = open(checkpoints_path,"a+")
    checkpoints_file.write("-----epoch "+str(epoch)+":\n")
    for i in range(0,len(x_val),batch_size):
        x_subset = x_val[i:i+batch_size]
        x_indices_subset = x_indices_val[i:i+batch_size]
        decoder_x_subset = decoder_x_val[i:i+batch_size]
        att_mask_subset = att_mask_val[i:i+batch_size]
        y_indices_subset = y_indices_val[i:i+batch_size]
        loss_mask_subset = loss_mask_val[i:i+batch_size]

        h = encoder(x_subset)
        joint_probabilities,coverage_loss = decoder([h,x_indices_subset,decoder_x_subset,att_mask_subset,y_indices_subset,s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset])        
        loss,indiv_losses = loss_function(joint_probabilities,loss_mask_subset,coverage_loss,lam=coverage_lam,use_coverage_loss=use_coverage_loss,return_indiv_loss=True)
        losses.append(float(loss))

        for j in range(batch_size): # saving loss value for each validation set example on individual line
            indiv_loss = indiv_losses[j]
            checkpoints_file.write(str(i+j)+": "+str(round(float(indiv_loss),6))+"\n")

    return round(sum(losses)/max(len(losses),1),6)


@tf.function
def training_step(encoder,decoder,optimizer,x_subset,x_indices_subset,decoder_x_subset,att_mask_subset,y_indices_subset,loss_mask_subset,s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset,coverage_lam,use_coverage_loss):
    """ training step - calculates the gradient w/ respect to encoder & decoder parameters
        - improves runtime by about 2x
    """
    with tf.GradientTape() as tape:
        h = encoder(x_subset)
        joint_probabilities,coverage_loss = decoder([h,x_indices_subset,decoder_x_subset,att_mask_subset,y_indices_subset,s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset])
        loss = loss_function(joint_probabilities,loss_mask_subset,coverage_loss,lam=coverage_lam,use_coverage_loss=use_coverage_loss,return_indiv_loss=False)
    
    gradients = tape.gradient(loss, encoder.trainable_variables+decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables+decoder.trainable_variables))
    return loss


def train_model(x,x_indices,att_mask,loss_mask,decoder_x,y_indices,x_val,x_indices_val,att_mask_val,loss_mask_val,decoder_x_val,y_indices_val,x_test,x_indices_test,att_mask_test,loss_mask_test,decoder_x_test,y_indices_test,encoder,decoder,batch_size,optimizer,epochs,coverage_lam,use_coverage_loss,model_save_path,model_checkpoints_path,model_checkpoints_name):
    """ training the model
    args:
        x,x_indices,...: training data
        x_val,x_indices_val,...: validation data
        x_test,x_indices_test,...: test data
        model_checkpoints_path: saves checkpoint data to a file after each epoch
    """
    print_epoch_value = int((100//batch_size)*batch_size)
    save_epoch_value = int((1000//batch_size)*batch_size)

    x,x_indices,att_mask,loss_mask,decoder_x,y_indices = shuffle(x,x_indices,att_mask,loss_mask,decoder_x,y_indices) # shuffling data
    # initializing the fixed input to the decoder model:
    s_subset = np.zeros((batch_size,256)).astype("float32")
    c_subset = np.zeros((batch_size,256)).astype("float32")
    coverage_vector_subset = np.zeros((batch_size,500)).astype("float32")
    fixed_vocab_indices_subset = np.vstack([[i for i in range(30000)] for _ in range(batch_size)]).astype("int32")

    for epoch_i in range(epochs): # epochs
        checkpoints_file = open(model_checkpoints_path+model_checkpoints_name,"a+")
        date_time = datetime.now(tz=pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime("%m/%d/%y %H:%M:%S")
        checkpoints_file.write(date_time+"\n"+"--------------------------------"+"\n")
        print("training start time:",date_time)

        losses = []
        for i in range(0,len(x),batch_size): # looping through each batch
            x_subset = x[i:i+batch_size]
            x_indices_subset = x_indices[i:i+batch_size]
            decoder_x_subset = decoder_x[i:i+batch_size]
            att_mask_subset = att_mask[i:i+batch_size]
            y_indices_subset = y_indices[i:i+batch_size]
            loss_mask_subset = loss_mask[i:i+batch_size]
            batch_loss = training_step(encoder,decoder,optimizer,x_subset,x_indices_subset,decoder_x_subset,att_mask_subset,y_indices_subset,loss_mask_subset,s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset,coverage_lam,use_coverage_loss)
            
            float_loss = round(float(batch_loss),6)
            losses.append(float_loss)
            if i % print_epoch_value == 0:
                date_time = datetime.now(tz=pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime("%m/%d/%y %H:%M:%S")
                print("i:",i,":",float_loss,";",date_time)
            
            if i % save_epoch_value == 0: # adding another model checkpoint
                encoder.save_weights(model_save_path+"encoder")
                decoder.save_weights(model_save_path+"decoder")
                np.save(model_save_path+"optimizer_weights.npy",optimizer.get_weights()) # saving optimizer state
        
        # writing out information to screen and saving to checkpoints file
        print_train_loss = "epoch {}; training loss: {}".format(epoch_i+1,round(sum(losses)/max(len(losses),1),6))
        print(print_train_loss)
        val_loss = get_validation_set_loss(x_val,x_indices_val,att_mask_val,loss_mask_val,decoder_x_val,y_indices_val,encoder,decoder,batch_size,coverage_lam,use_coverage_loss,epoch_i,model_checkpoints_path+"val_losses.txt",s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset)
        print_val_loss = "epoch {}; validation loss: {}".format(epoch_i+1,val_loss)
        print(print_val_loss)
        test_loss = get_validation_set_loss(x_test,x_indices_test,att_mask_test,loss_mask_test,decoder_x_test,y_indices_test,encoder,decoder,batch_size,coverage_lam,use_coverage_loss,epoch_i,model_checkpoints_path+"test_losses.txt",s_subset,c_subset,coverage_vector_subset,fixed_vocab_indices_subset)
        print_test_loss = "epoch {}; test_set loss: {}".format(epoch_i+1,test_loss)
        print(print_test_loss)
        last_line = "--------------------------------"
        print(last_line)
        date_time = datetime.now(tz=pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime("%m/%d/%y %H:%M:%S") # time at which epoch completed
        checkpoints_file.write(date_time+"\n")
        checkpoints_file.write(print_train_loss+"\n")
        checkpoints_file.write(print_val_loss+"\n")
        checkpoints_file.write(print_test_loss+"\n")
        checkpoints_file.write(last_line+"\n")
        checkpoints_file.close()
        encoder.save_weights(model_save_path+"encoder")
        decoder.save_weights(model_save_path+"decoder")
        np.save(model_save_path+"optimizer_weights.npy",optimizer.get_weights())


if __name__=="__main__":
    main()
