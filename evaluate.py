
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes

_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model,trainRatings,testRatings,testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _trainRatings
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _trainRatings=trainRatings
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    
    Mae=[]
    for idx in range(len(_testRatings)):
        e = eval_one_rating(idx)
        Mae.append(e)
            
    return (Mae)

def eval_one_rating(idx):

    rating = _testRatings[idx]      #testRatings: [userid, itemid, rating]
    items = _testNegatives[idx]     #[...],共99个,负样本的itemID. testNegatives[1]:编号为1的test的负样本item
    
    u = rating[0]
    gtItem = rating[1]
    r = rating[2]                     # r is the real rating
    items.append(gtItem)

    map_item_score = {}             # Get prediction scores
    users = np.full(len(items),u,dtype = 'int32')
    predictions = _model.predict([users, np.array(items)],batch_size = 100, verbose = 0)
    print(predictions)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
#        print("map-item" )
#        print(item)
#        print( map_item_score[item])
    items.pop()
    #print
    # Evaluate top rank list
    #ranklist里面存放的是什么？
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    #MAE
    r1=0
    #w=0
    w=1
    #print("_trainRatings.shape:")
    #print(_trainRatings.shape)
    #print(_trainRatings[u,item])
#    print("ranklist:")
#    print(ranklist)
    for i in range(len(ranklist)):
       neighbor_item= ranklist[i]
#       print("u")
#       print(u)
#       print("titem")
#       print(gtItem)
#       print(" neighbor_item" )
#       print( neighbor_item)
       neighbor_w= map_item_score[neighbor_item]
       w=w+neighbor_w
#       print("_trainRatings[u,neighbor_item]")
#       print(_trainRatings[u,neighbor_item])
       r1=r1+_trainRatings[u,neighbor_item-1]*neighbor_w
    r1=r1/w
    e=abs(r1-r)
    return (e)
#    print('u')
#    print(idx)
#    print(u)
#    print(gtItem)
#    print('ranklist:')
#    for i in range(len(ranklist)):
#        print(ranklist[i])
    
    #hr = getHitRatio(ranklist, gtItem)
    #ndcg = getNDCG(ranklist, gtItem)
    #return (hr, ndcg)

#def getHitRatio(ranklist, gtItem):
#    for item in ranklist:
#        if item == gtItem:
#            return 1
#    return 0
#
#def getNDCG(ranklist, gtItem):
#    for i in range(len(ranklist)):
#        item = ranklist[i]
#        if item == gtItem:
#            return math.log(2) / math.log(i+2)
#    return 0
