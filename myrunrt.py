import dataload
import os
import logging
import logging
from time import time
#import evaluator

# parameter config area
para = {'dataPath': 'data/',
		'dataName': 'dataset1',
		'dataType': 'rt', # set the dataType as 'rt' or 'tp'#timerank
		'outPath': 'result/',
		#'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # delete where appropriate
		#'density': np.arange(0.05, 0.31, 0.05), # matrix density
		#'rounds': 20, # how many runs are performed at each matrix density
		#'dimension': 10, # dimenisionality of the latent factors
		#'lambda': 30, # regularization parameter
		#'maxIter': 300, # the max iterations
		#'saveTimeInfo': False, # whether to keep track of the running time
		#'saveLog': True, # whether to save log into file
		#'debugMode': False, # whether to record the debug info
       	#'parallelMode': True # whether to leverage multiprocessing for speedup
		}


# start timing
startTime = time()

# 形成三种文件

dataMatrix = dataload.load(para)

# evaluate QoS prediction algorithm
#evaluator.execute(dataMatrix, para)

print(time()-startTime)
