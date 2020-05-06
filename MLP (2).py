
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers as initializations
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Add, Reshape, Flatten, Dropout, Concatenate
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='tp.',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                  embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate(axis=-1)([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    print(layers, type(layers[0]))
    reg_layers = eval(args.reg_layers)
    print(reg_layers, type(reg_layers[0]))
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time.time()
    (mae, hit, ndcg) = evaluate_model(model, train, testRatings, testNegatives, topK, evaluation_threads)
    # average value
    print(
        "mae: %.6f , hit: %.6f , ndcg : %.6f" % (np.array(mae).mean(), np.array(hit).mean(), np.array(ndcg).mean()))
    hr = np.array(mae).mean()
    print('Init: MAE = %.4f,\t [%.1f s]' % (hr, time.time() - t1))

    # Train model
    best_hr, best_iter = hr, -1
    for epoch in range(epochs):
        t1 = time.time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels), batch_size=batch_size,
                         epochs=1, verbose=0, shuffle=True)
        t2 = time.time()

        # Evaluation
        if epoch % verbose == 0:

            (mae, hit, ndcg) = evaluate_model(model, train, testRatings, testNegatives, topK, evaluation_threads)
            hr, loss = np.array(mae).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: MAE = %.4f, loss = %.4f [%.1f s]' % (
                epoch, t2 - t1, hr, loss, time.time() - t2))
            if hr > best_hr: best_hr, best_iter = hr, epoch
            if args.out > 0: model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  MAE = %.4f. " % (best_iter, best_hr))
    if args.out > 0: print("The best GMF model is saved to %s" % (model_out_file))