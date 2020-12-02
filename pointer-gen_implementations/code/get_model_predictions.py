""" all the code in order to run the model predictions at test time

Author: Nathaniel Andre
"""

import spacy
import numpy as np
import json
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Bidirectional,LSTM,Input,RepeatVector,Activation,Softmax,Embedding,Dot,Lambda
from tensorflow.keras.layers import Softmax,Concatenate
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from collections import defaultdict


def main():
    """
    """
    prefix="_500"
    data_dir = "../data/len_500_data-100_dim/"
    predictions_outfile = open("./predictions.txt","w+")
    summaries_outfile = open("./summaries.txt","w+")
    bids = np.load(data_dir+"bids{}.npy".format(prefix))
    bids = bids[5500:] # getting only val and test set bids
    with open("{}bill_information.json".format(data_dir)) as in_file:
        data_dict = json.load(in_file)

    for bid in bids:
        predictions_outfile = open("./predictions.txt","a")
        summaries_outfile = open("./summaries.txt","a")
        data_point = data_dict[bid]
        utterances = data_point['utterances']
        summary = data_point['summary']
        summaries_outfile.write(summary+"\n")
        #predicted_tokens = get_runtime_prediction(utterances,max_tokens=100,data_dir=data_dir)
        #prediction = " ".join(predicted_tokens)
        #predictions_outfile.write(prediction+"\n")
        
        """ printing the input to the model
        nlp = spacy.load("en_core_web_sm")
        all_token_lists = [[token.text.lower() for token in nlp(utterance)] for utterance in utterances]
        all_tokens = []
        for token_list in all_token_lists:
            all_tokens += token_list
        utterance_tokens = all_tokens[:500] # contiguous sequence of tokens for this input
        """
        predictions_outfile.close()
        summaries_outfile.close()


def get_model_inputs(nested_utterances,data_dir="../data/len_500_data/",token_cutoff=500):
    """ Gets the input representations for running with the model - based on the input of utterances for a given BID
    
    args:
        nested_utterances: nested list of utterances (each nested list is a different DID)(and for each DID there utterances)(these utterances are only those by the author) - @@@ actually current implementation assumes all BID subarrays are already appended together: e.g.: [string1, string2, ...]
        data_dir: path to the directory which holds word_2_id and id_2_word, etc.
        token_cutoff: number of tokens in the input which should be send to model encoder

    Returns: x,x_indices,att_mask, x_indices_dict, index_to_word
    """
    nlp = spacy.load("en_core_web_sm")

    with open(data_dir+"word_to_index.json") as in_file:
        word_to_index = json.load(in_file)
    with open(data_dir+"index_to_word.json") as in_file: # the key is a string
        index_to_word = json.load(in_file)
    num_fixed_words = len(word_to_index)

    all_token_lists = [[token.text.lower() for token in nlp(utterance)] for utterance in nested_utterances]
    all_tokens = []
    for token_list in all_token_lists:
        all_tokens += token_list
    utterance_tokens = all_tokens[:token_cutoff] # contiguous sequence of tokens for this input
    
    x = [] # stores the integer/index representation for the input
    for token in utterance_tokens:
        if token in word_to_index:
            x.append(word_to_index[token])
        else:
            x.append(word_to_index['<UNK>'])

    att_mask = [0 for _ in range(len(x))] # stores the attention mask (for the encoder)
    amount_to_pad = token_cutoff-len(x)
    att_mask += [-np.inf for _ in range(amount_to_pad)]

    x_indices = [] # indices of each of the input tokens in the joint probability vector (used to assign the probabilities to the correct index in this vector)
    x_indices_dict = {} # stores index-to-word for tokens which aren't in the fixed vocabulary(used in decoding)
    non_vocab_dict = {} # stores word_to_index which is used to create the encoding(temporary)
    index=num_fixed_words

    for token in utterance_tokens:
        if token in word_to_index:
            x_indices.append(word_to_index[token])
        else:
            if token in non_vocab_dict: # this word is OOV but has been seen before in this input
                x_indices.append(non_vocab_dict[token])
            else: # this word is OOV and has never been seen before
                non_vocab_dict[token]=index
                x_indices_dict[index]=token
                x_indices.append(index)
                index += 1

    x += [0 for _ in range(amount_to_pad)]
    x_indices += [0 for _ in range(amount_to_pad)]
    assert(len(x)==len(x_indices)==len(att_mask)==token_cutoff)
    x = np.expand_dims(np.asarray(x,dtype="int32"),axis=0)
    x_indices = np.expand_dims(np.asarray(x_indices,dtype="int32"),axis=0)
    att_mask = np.expand_dims(np.asarray(att_mask,dtype="float32"),axis=0)
    return x,x_indices,att_mask,x_indices_dict,index_to_word,utterance_tokens


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


def pointer_gen_encoder(embedding_layer,encoder_h=128,input_len=500,tf_int=tf.int32):
    """ Returns the encoder portion of the pointer-gen network
    """
    x = Input(shape=(input_len),dtype=tf_int) # input to the encoder
    input_e = embedding_layer(x) # embeddings for the input
    h = Bidirectional(LSTM(encoder_h,activation="tanh",return_sequences=True),merge_mode="concat")(input_e) # encoder
    
    model = Model(inputs=[x],outputs=[h])
    return model


def pointer_gen_decoder(embedding_layer,decoder_lstm,att_w1,att_w2,att_w3,att_v,vocab_d,vocab_d_pre,pgen_w1,pgen_w2,pgen_w3,encoder_h=128,input_len=500,tf_float=tf.float32,tf_int=tf.int32):
    """ Returns the decoder portion of the pointer-gen network 
        -implemented so that it does only a single step
    """
    h = Input(shape=(input_len,encoder_h*2),dtype=tf_float) # the input embedding from the encoder model
    x_indices_ = Input(shape=(input_len),dtype=tf_int) # represents where each input word prob. should be added in joint prob. vector
    x_indices = tf.expand_dims(x_indices_,axis=-1)
    fixed_vocab_indices_ = Input(shape=(30000),dtype=tf_int) # the size of the input vocabulary
    fixed_vocab_indices = tf.expand_dims(fixed_vocab_indices_,axis=-1)
    att_mask = Input(shape=(input_len),dtype=tf_float) # mask used with the attention distribution to mask out padding
    
    decoder_x = Input(shape=(1),dtype=tf_int) # delayed y_data for input to the decoder (last prediction)
    s_ = Input(shape=(256),dtype=tf_float) # decoder_h
    c_ = Input(shape=(256),dtype=tf_float)
    coverage_vector_ = Input(shape=(input_len),dtype=tf_float) # loaded at each step
    s,c,coverage_vector = s_,c_,coverage_vector_
    
    decoder_e = embedding_layer(decoder_x) # embeddings for delayed input to the decoder
    decoder_input = decoder_e[:,0,:]  # input to the decoder at this timestep
    s,_,c = decoder_lstm(tf.expand_dims(decoder_input,axis=1),initial_state=[s,c])

    # calculating attention (probabilities over input):
    s_rep = RepeatVector(input_len)(s) # copying the decoder hidden state
    e = att_v(Activation("tanh")(att_w1(h)+att_w2(s_rep)+att_w3(tf.expand_dims(coverage_vector,axis=-1)))) # unscaled attention
    e = tf.squeeze(e,axis=-1)+att_mask # using attention mask (masks out padding in the input sequence)
    a = Activation("softmax")(e) # scaled attention (represents prob. over input)

    # handling coverage vector computations - note that coverage loss is not collected:
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

    model = Model(inputs=[h,x_indices_,decoder_x,att_mask,s_,c_,coverage_vector_,fixed_vocab_indices_],outputs=[joint_prob,s,c,coverage_vector])
    return model


def get_pointer_gen_network(embedding_dim=100,input_len=500,tf_float=tf.float32,tf_int=tf.int32,model_save_path="../model_params/"):
    """ loads the encoder and decoder models from memory
    args:
        embedding_dim: the dimensionality of the word embeddings
        model_save_path: directory which stores the saved model parameters
    """
    embedding_layer = Embedding(input_dim=30000,output_dim=embedding_dim,mask_zero=True) # re-used for both the encoder and decoder
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

    encoder = pointer_gen_encoder(embedding_layer,encoder_h=encoder_h,input_len=input_len,tf_int=tf_int)
    encoder.load_weights(model_save_path+"encoder")
    decoder = pointer_gen_decoder(embedding_layer,decoder_lstm,att_w1,att_w2,att_w3,att_v,vocab_d,vocab_d_pre,pgen_w1,pgen_w2,pgen_w3,encoder_h=encoder_h,input_len=input_len,tf_float=tf_float,tf_int=tf_int)
    decoder.load_weights(model_save_path+"decoder")
    return encoder,decoder


def run_beam_search(x,x_indices,att_mask,x_indices_dict,index_to_word,encoder,decoder,max_tokens,beam_width,alpha,c=1e-18):
    """ Gets the top-prob. predictions based on beam search
    args:
        max_tokens: set maximum number of tokens for generated summary
        beam_width: the number of channels to use for beam search
        alpha: controls the length normalization for beam search
    """
    vocab_size=len(index_to_word)

    models = defaultdict(dict) ## starting the decoding process
    s = np.zeros((1,256)).astype("float32")
    c = np.zeros((1,256)).astype("float32")
    coverage_vector = np.zeros((1,500)).astype("float32")
    fixed_vocab_indices = np.array([[i for i in range(30000)]]).astype("int32")
    decoder_x = np.ones((1,1)).astype("int32") # represents first input of "<SENT>"

    h = encoder([x])
    joint_prob,s,c,coverage_vector = decoder([h,x_indices,decoder_x,att_mask,s,c,coverage_vector,fixed_vocab_indices])
    joint_prob = joint_prob.numpy()

    # getting the initial top n=beam_width models:
    for i in range(beam_width):
        arg_max = np.argmax(joint_prob)
        models[i]['prob']=np.log(joint_prob[0,arg_max]) # using log-prob.
        if arg_max < vocab_size: # predicted word is in the fixed-vocabulary
            models[i]['tokens']=[index_to_word[str(arg_max)]]
            models[i]['next_input']=np.array([[arg_max]]).astype("int32") # effectively the decoder_x for the next step
        else: # predicting a word which is OOV but in the input
            models[i]['tokens']=[x_indices_dict[arg_max]]
            models[i]['next_input']=np.array([[2]]).astype("int32") # represents the <UNK> token
        
        models[i]['s'],models[i]['c'],models[i]['coverage_vector']=s,c,coverage_vector
        models[i]['done'] = (arg_max==1 or len(models[i]['tokens'])==max_tokens) # conditions for the end state
        joint_prob[0,arg_max]=-np.inf
        
    ## run until the end condition is met for all n=beam_width models/outputs
    while sum([models[i]['done'] for i in range(beam_width)]) != beam_width:
        
        # first calculating all the new joint_probabilities for the n=beam_width models:
        all_joint_probs = []
        for i in range(beam_width):
            if models[i]['done'] is False: # this model has not reached its end state; adding a new token at this step
                s,c,coverage_vector,decoder_x = models[i]['s'],models[i]['c'],models[i]['coverage_vector'],models[i]['next_input']
                joint_prob,s,c,coverage_vector = decoder([h,x_indices,decoder_x,att_mask,s,c,coverage_vector,fixed_vocab_indices])
                joint_prob = (models[i]['prob']+np.log(joint_prob.numpy()))*(1/((len(models[i]['tokens'])+1)**alpha)) # normalization/scaling
                models[i]['s'],models[i]['c'],models[i]['coverage_vector']=s,c,coverage_vector
            else: # this model has already reached its end state; NOT adding a token at this state
                joint_prob = np.full(joint_prob.shape,-np.inf).astype("float32")
                joint_prob[0,0]=models[i]['prob']*(1/(len(models[i]['tokens'])**alpha)) # only one cell will contain probability for this model (preventing the same "done" model from being selected multiple times); this simplifies the logic
            all_joint_probs.append(joint_prob)

        all_joint_probs = np.hstack(all_joint_probs)
        
        # based on the potential predicted sequences, getting the next n=beam_width best models:
        new_models = defaultdict(dict) # dict to store the next best models
        for i in range(beam_width): # getting the n=beam_width best paths
            arg_max = np.argmax(all_joint_probs) # arg_max for the concatenation of all joint_prob arrays
            model_no = arg_max // joint_prob.shape[1] # model associated with this argmax
            
            if models[model_no]['done'] is True: # highest prob. model is the finished model; simply copy eveything from the existing model
                new_models[i]['s'],new_models[i]['c'],new_models[i]['coverage_vector']=models[model_no]['s'],models[model_no]['c'],models[model_no]['coverage_vector']
                new_models[i]['prob'],new_models[i]['tokens'],new_models[i]['next_input'],new_models[i]['done']=models[model_no]['prob'],models[model_no]['tokens'],models[model_no]['next_input'],models[model_no]['done']
                
            else: # highest prob. model is not finished adding words/tokens
                new_models[i]['prob']=all_joint_probs[0,arg_max]/(1/((len(models[model_no]['tokens'])+1)**alpha)) # getting rid of the scaling
                model_arg_max = arg_max-(joint_prob.shape[1]*model_no) # arg_max for the joint_prob for this model
                if model_arg_max < vocab_size: # predicted word is in the fixed-vocabulary
                    new_models[i]['tokens'] = models[model_no]['tokens']+[index_to_word[str(model_arg_max)]]
                    new_models[i]['next_input']=np.array([[model_arg_max]]).astype("int32")
                else: # predicting a word which is OOV but in the input
                    new_models[i]['tokens'] = models[model_no]['tokens']+[x_indices_dict[model_arg_max]]
                    new_models[i]['next_input']=np.array([[2]]).astype("int32") # represents the <UNK> token
                    
                new_models[i]['s'],new_models[i]['c'],new_models[i]['coverage_vector']=models[model_no]['s'],models[model_no]['c'],models[model_no]['coverage_vector']
                new_models[i]['done'] = (model_arg_max==1 or len(new_models[i]['tokens'])==max_tokens)
            
            all_joint_probs[0,arg_max]=-np.inf
        models = new_models
        
    predicted_tokens = models[0]['tokens'] # get the model w/ the highest prob.
    return predicted_tokens


def get_runtime_prediction(utterances,max_tokens=200,beam_width=3,alpha=1,embedding_dim=100,input_len=500,data_dir="../data/len_500_data/",model_save_path="../model_params/"):
    """ Gets runtime predictions using beam search
    args:
        utterances: 1D list of utterances (the second dimension of discussion id has been collapsed into the 1D)
        embedding_dim: the dimensionality of the word embeddings
        model_save_path: path to directory which holds pretrained model params
        data_dir: path to directory which holds preprocessed data
        max_tokens: set maximum number of tokens for generated summary
        beam_width: the number of channels to use for beam search
        alpha: controls the length normalization for beam search
    """
    x,x_indices,att_mask,x_indices_dict,index_to_word,_ = get_model_inputs(utterances,data_dir=data_dir,token_cutoff=input_len)
    encoder,decoder = get_pointer_gen_network(embedding_dim=embedding_dim,input_len=input_len,model_save_path=model_save_path)
    predicted_tokens = run_beam_search(x,x_indices,att_mask,x_indices_dict,index_to_word,encoder,decoder,max_tokens,beam_width,alpha)
    return predicted_tokens


if __name__=="__main__":
    main()
