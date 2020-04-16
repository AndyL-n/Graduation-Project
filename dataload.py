import os
import sys
import time
import logging
import numpy as np


# 数据加载
def load(para):
    if para['dataName'] == 'dataset1':
        datafile = para['dataPath'] + para['dataName'] + '/' + para['dataType'] + 'Matrix.txt'
        print("数据源" + datafile)
        dataMatrix = np.loadtxt(datafile)
        numuser = dataMatrix.shape[0]
        numitem = dataMatrix.shape[1]
        print('Data size: %d users * %d services' % (numuser, numitem))
        os.remove('data/train.rating')
        os.remove('data/test.rating')
        os.remove('data/test.negative')
        trainfile = open('data/train.rating', 'a+')
        testfile = open('data/test.rating', 'a+')
        negativefile = open('data/test.negative', 'a+')

#        for userid in range(numuser):
        for userid in range(0,10):
            # 当前userid对所有item的rating
            itemlist = np.arange(0,len(dataMatrix[userid])) #按次序生成itemlist的编号0,1，....,共5825个
            userlist = np.full(len(dataMatrix[userid]), userid, dtype = 'int32')#长度为5825
            ratitemlist = []
            for i in range(numitem):
                rating, item = dataMatrix[userid][i], i
                ratitemlist.append([rating, item])
            ratitemlist.sort()
            # print("ratitemlist")
            # print(ratitemlist)

            # 获取最后99个作为负样本
            negativelist = []
            numnegative = 99
            negativelist.extend(ratitemlist[numitem - numnegative:numitem])
            # print(negativelist)

            # 训练集
            trainlist = [i for i in ratitemlist if i not in negativelist]
            # print(trainlist)

            #
            index = np.random.randint(numitem - numnegative)
            item = trainlist[index][1]
            rating = trainlist[index][0]
            trainlist.remove(trainlist[index])
            # print(userid,item,rating)
            testfile.write(str(userid) + '\t' + str(item) + '\t' + str(rating) + '\t' + '\n')

            # 生成负样本数据文件
            negativefile.write('(' + str(userid) + ',' + str(item) + ')' + '\t')
            for i in range(len(negativelist)):
            #    item = negativelist[i][1],rating = negativelist[i][0]
                negativefile.write(str(negativelist[i][1]) + '\t')
            negativefile.write('\n')

            #生成训练数据文件
            for i in range(len(trainlist)):
                item = trainlist[i][1]
                rating = trainlist[i][0]
                trainfile.write(str(userid) + '\t' + str(item) + '\t' +str(rating) + '\t' +'\n')

        trainfile.close()
        testfile.close()
        negativefile.close()
        return dataMatrix
#     elif para['dataName'] == 'dataset#2':
#         datafile = para['dataPath'] + para['dataName'] + '/' + para['dataType'] + 'data.txt'
#         logger.info('Loading data: %s'%os.path.abspath(datafile))
#         #dataMatrix = -1 * np.ones((142, 4500, 64))
#         trainfile=open('data/Train','a+')
#         testfile=open('data/test','a+')
#         negativefile=open('data/negative','a+')
#         fid = open(datafile, 'r')
# #        curID = 0
# #        datalist=[]
# #        for line in fid:
# #            data = line.split(' ')
# #            #rt = float(data[3])
# #            #if rt > 0:
# #            userID=int(data[0])
# #            if userID == curID:
# #                datalist.append([data[3]*1000,userID,data[1]]) #把rating放在第一个列，为了排序方便
# #                #dataMatrix[int(data[0]), int(data[1]), int(data[2])] = rt
# #            else:
# #               result=sorted(datalist) #datalist.sort()
# #               np.savetxt('data/Train', result)
# #               datalist=[]
# #        trainfile.close()
# #        testfile.close()
# #        negativefile.close()
#         fid.close()
#         logger.info('Data size: %d users * %d services * %d timeslices'\
#             %(dataMatrix.shape[0], dataMatrix.shape[1], dataMatrix.shape[2]))
#     #处理datamatrix，然后分出负样本，并且写入文件
#
#     #dataMatrix = preprocess(dataMatrix, para)
#     logger.info('Loading data done.')
#     logger.info('----------------------------------------------')
#     #return dataMatrix
#        return [itemlist,userlist]

#def load_rating_file_as_list(self, filename):
#        ratingList = []
#        with open(filename, "r") as f:
#            line = f.readline()
#            while line != None and line != "":
#                arr = line.split("\t")
#                user, item = int(arr[0]), int(arr[1])
#                ratingList.append([user, item])
#                line = f.readline()
#        return ratingList