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
    Evaluate the performance (Mae,Hit_Ratio, NDCG) of top-K recommendation
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
    
    Maes,hits, ndcgs = [],[],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (mae,hr,ndcg) = eval_one_rating(idx)
        Maes.append(mae)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (Maes,hits,ndcgs)

def eval_one_rating(idx):

    # testRatings: [userid, itemid, rating]
    rating = _testRatings[idx]
    # [...],共99个,负样本的itemID. testNegatives[1]:编号为1的test的负样本item
    items = _testNegatives[idx]
    
    u = rating[0]
    gtItem = rating[1]
    r = rating[2]                     # r is the real rating

    items.append(gtItem)

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    # 保存了训练的模型,输入只要保证和训练的时候的格式一样即可
    # 事先构建了negative的数据，即对negative的物品和测试集合中的某一个物品进行了预测
    # 最终选取topK的，来评测是否在其中(注getHitRatio函数不是最终结果，只是0/1)
    # eval_one_rating 函数只是对测试集合中的某个用户的某个物品，以及和事先划分好的负样本组合在一起进行预测，最终输出该测试物品是否在topK中。
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    mae = getMae(ranklist,gtItem,u,r,map_item_score)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (mae,hr, ndcg)


def getMae(ranklist,gtItem,u,r,map_item_score):
    r1,w= 0,1
    for i in range(len(ranklist)):
        neighbor_item = ranklist[i]
        neighbor_w = map_item_score[neighbor_item]
        w = w + neighbor_w
        r1 = r1 + _trainRatings[u, neighbor_item - 1] * neighbor_w
    r1 = r1 / w
    return abs(r1 - r)

def getHitRatio(ranklist, gtItem):
   for item in ranklist:
       if item == gtItem:
           return 1
   return 0

def getNDCG(ranklist, gtItem):
   for i in range(len(ranklist)):
       item = ranklist[i]
       if item == gtItem:
           return math.log(2) / math.log(i+2)
   return 0
