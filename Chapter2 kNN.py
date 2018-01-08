from numpy import *
import numpy as np
import operator
from os import listdir
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
'''
加载约会数据集
'''
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0) +1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
'''
将数据转化为numpy数组
'''
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()#逐行读取
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFormLine=line.split('\t')
        returnMat[index,:]=listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index+=1
    return returnMat,classLabelVector
'''
数据归一化
'''
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
'''
测试代码
'''    
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('D:\Anaconda3\workspace\machinelearninginaction\Ch02\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classfierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d,the real answer is: %d" % (classfierResult,datingLabels[i]))
        if (classfierResult !=datingLabels[i]):
            errorCount+=1.0
        print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
'''
约会网站预测函数
'''   
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video game?"))#获取输入
    ffMiles=float(input("frequent fliter miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('D:\Anaconda3\workspace\machinelearninginaction\Ch02\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=np.array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will propably like this person:",resultList[classifierResult-1])
'''
手写识别
数据说明：
trainingDigits：为训练集，目录下所有文档名（比如 2_126.txt）中_前面的数字表示其分类结果，即里面显示的是什么数字
'''
def img2vector(filename):
    #将32*32的二进制图像转化为1*1024的一维向量
    returnVect=zeros((1,1024))#创建一个一维数组
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
        
def handwritingClassTest():
    hwlabels=[]
    trainingFileList=listdir('D:/Anaconda3/workspace/machinelearninginaction/Ch02/trainingDigits')#获取训练集目录集下所有文件名
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))#构造一个m行1024列的矩阵，每一行代表一个训练集
    for i in range(0,m):
        fileNameStr=trainingFileList[i]
        filestr=fileNameStr.split('.')[0]#获取.txt前的名称
        classNum=int(filestr.split('_')[0])#获取分类类别号
        hwlabels.append(classNum)
        path='D:/Anaconda3/workspace/machinelearninginaction/Ch02/trainingDigits/'+fileNameStr
        trainingMat[i,:]=img2vector(path)
    testFileList=listdir('D:/Anaconda3/workspace/machinelearninginaction/Ch02/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])#分类标签
        path='D:/Anaconda3/workspace/machinelearninginaction/Ch02/trainingDigits/'+fileNameStr
        vectorUnderTest=img2vector(path)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwlabels,3)#
        print("the classifier came back with %d the real answer is %d" %(classifierResult,classNumStr))
        if (classifierResult!=classNumStr):
            errorCount+=1.0
    print ("\nthe total number of errors is:%d" % (errorCount))
    print ("\nthe total error rate is: %f" %(errorCount/float(mTest)))
    
                
    
