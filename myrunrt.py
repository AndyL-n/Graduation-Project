import dataload
import os
import logging
import logging
from time import time

# parameter config area
para = {'dataPath': 'data/',
		'dataName': 'dataset1',
		'dataType': 'rt',
		'outPath': 'result/',
		}


# start timing
startTime = time()

# 形成三种文件

dataMatrix = dataload.load(para)

print("数据载入用时：%.2f s" % (time()-startTime))
