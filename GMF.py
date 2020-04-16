import numpy as np
#import theano.tensor as T
import keras
from keras import backend as K
#from keras import initializers
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
#from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from keras.layers import Multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from dataset import Dataset
from evaluate import evaluate_model
import time
import multiprocessing as mp
import sys
import math
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='',help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users, num_items, latent_dim, regs=[0,0]):

    #构建模型结构，首先定义Input为一维多列的数据
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    #Embedding层，Embedding主要是为了降维，就是起到了lookup的作用
    #MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
    #MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',init = init_normal, W_regularizer = l2(regs[1]), input_length=1)
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',embeddings_initializer ='random_normal',embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',embeddings_initializer='random_normal', embeddings_regularizer = l2(regs[1]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    #Embedding层的物品的latent_dim和用户的latent_dim是一致的，在实际中未必两者的维度是一致的。
    #这里受限于keras的merge函数的参数要求，输入的数据的shape必须是一致的，所以必须是一致的。
    #以及Merge中的mode参数，至于什么时候选择contact，什么时候选择mul，依赖于模型效果，在实际工程中选择使得最优的方式。
    #然后是Merge层，将用户和物品的张量进行了内积相乘(latent_dim表示两者的潜在降维的维度是相同的，因此可以做内积)
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    predict_vector = Multiply()([user_latent, item_latent])

    #全连接层，激活函数为sigmoid。
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    prediction = Dense(1, activation="sigmoid", name="prediction", kernel_initializer="lecun_uniform")(predict_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)

    return model
#每个正样本，随机选择 4 个负样本 样本数量=train中的item数量(5825-100)*5.
#该函数是获取用户和物品的训练数据，其中num_negatives控制着正负样本的比例，负样本的获取方法也简单粗暴，直接随机选取用户没有选择的其余的物品。
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1 #mp.cpu_count()

    print("GMF arguments: %s" %(args))
    t1 = time.localtime(time.time())
    t1 = str(t1.tm_mon) + "/" +str(t1.tm_mday) + "_" + \
         str(t1.tm_hour) + ":" + str(t1.tm_min)
    print(t1)
    model_out_file = 'Pretrain/GMF_%s_%d_%s.h5' %(args.dataset, num_factors,t1 )
    #print(model_out_file)

    # Loading data
    t1 = time.time()
    # print(t1)
    dataset = Dataset(args.path+args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data： [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"%(time.time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    #Build model
    model = get_model(num_users, num_items, num_factors, regs)
    # print(learner.lower())

    #优化器
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    # print(model.summary())
    # print("trainmatrix:")
    # print(train[0,1])

#   Init performance
    t1 = time.time()
    (mae,hit,ndcg) = evaluate_model(model, train,testRatings, testNegatives, topK, evaluation_threads)
    #average value
    print("mae: %.6f , hit: %.6f , ndcg : %.6f" % (np.array(mae).mean(),np.array(hit).mean(),np.array(ndcg).mean()))
    hr = np.array(mae).mean()
    print('Init: MAE = %.4f,\t [%.1f s]' % (hr,  time.time()-t1))

    # Train model
    best_hr,  best_iter = hr,  -1
    for epoch in range(epochs):
        t1 = time.time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],np.array(labels), batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time.time()

        # Evaluation
        if epoch % verbose == 0:

            (mae,hit,ndcg) = evaluate_model(model, train,testRatings, testNegatives, topK, evaluation_threads)
            hr,loss = np.array(mae).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: MAE = %.4f, loss = %.4f [%.1f s]'% (epoch,t2-t1,hr,loss,time.time()-t2))
            if hr > best_hr:best_hr,best_iter = hr,epoch
            if args.out > 0:model.save_weights(model_out_file, overwrite=True)


    print("End. Best Iteration %d:  MAE = %.4f. " %(best_iter, best_hr))
    if args.out > 0:print("The best GMF model is saved to %s" %(model_out_file))