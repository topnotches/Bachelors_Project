
import sys
import os
import random as rd
import math
from imgaug.augmentables.batches import UnnormalizedBatch
from xml.etree import ElementTree, ElementInclude
import xml.etree.ElementTree as ET
import multiprocessing as mp
def getIOU(box1, box2):

    if len(box1) == 4:
        box1 = [box1[2]-box1[0], box1[3]-box1[1]]
    else:
        assert(len(box1) == 2)

    if len(box2) == 4:
        box2 = [box2[2]-box2[0], box2[3]-box2[1]]
    else:
        assert(len(box2) == 2)

    area1 = box1[0]*box1[1]
    area2 = box2[0]*box2[1]
    xy1 = [-box1[0]/2, -box1[1]/2, box1[0]/2, box1[1]/2]
    xy2 = [-box2[0]/2, -box2[1]/2, box2[0]/2, box2[1]/2]
    x = [0.0, 0.0]
    y = [0.0, 0.0]
    if(xy1[0] > xy2[0]):
        x[0] = xy1[0]
    else:
        x[0] = xy2[0]
    
    if(xy1[1] > xy2[1]):
        y[0] = xy1[1]
    else:
        y[0] = xy2[1]

    if(xy1[2] < xy2[2]):
        x[1] = xy1[2]
    else:
        x[1] = xy2[2]
    
    if(xy1[3] < xy2[3]):
        y[1] = xy1[3]
    else:
        y[1] = xy2[3]

    intersection = (x[1]-x[0])*(y[1]-y[0])
    denominator = (area1+area2-intersection)
    if denominator == 0:
        return 0
    else:
        return intersection/(area1+area2-intersection)

def getBinCrossLossAndDeviation(anchBox, trueBoxes):
    listIOU = []
    sumLoss = 0.0
    sumIOU  = 0.0
    meanIOU = 0.0
    sumL2  = 0.0
    for trueBox in trueBoxes:
        IOU = getIOU(trueBox, anchBox)
        if IOU < 0.005:
            sumLoss += (-math.log10(IOU+0.001))
        else:

            sumLoss += (-math.log10(IOU))
        listIOU.append(IOU)
    for IOU in listIOU:
        sumIOU += IOU
    meanIOU = sumIOU/len(trueBoxes)

    for IOU in listIOU:
        sumL2 += (IOU-meanIOU)**2
    return sumLoss/len(trueBoxes), math.sqrt(sumL2/len(trueBoxes)), meanIOU


def getVariance(data, mean):
    sumIOU = 0.0
    for trueBox in trueBoxes:
        sumIOU+=-math.log10(getIOU(trueBox, anchBox))
    
    return sumIOU/len(trueBoxes)

def meanBox(boxes):
    sumWidth = 0.0
    sumHeight = 0.0
    
    for box in boxes:
        sumWidth += box[0]
        sumHeight += box[1]
    return [sumWidth/len(boxes), sumHeight/len(boxes)]   

def getAnchors(initializers, iterations, bboxes, dictIndex, resDict):
    anchors = initializers
    K = len(initializers)
    prev_anchors = []
    for iter in range(iterations):
        clusters = [[] for i in range(len(initializers))]
        cluster_wins = []
        for _ in range(len(initializers)):
            cluster_wins.append(0)
        #print(iter)
        for box in bboxes:
            intersection = 0.0
            index = 0
            for i, anchor in enumerate(anchors):
                tmptersec = getIOU(box, anchor)
                if tmptersec > intersection:
                    index = i
                    intersection = tmptersec
            clusters[index].append(box)
            cluster_wins[index] += 1
        prev_prev_anchors = prev_anchors[:]
        prev_anchors = anchors[:]
        for i in range(K):
            if len(clusters[i]) != 0:
                anchors[i] = meanBox(clusters[i])
        if prev_anchors == anchors or anchors == prev_prev_anchors:
            print("Stopping K-means at iteration {}".format(iter))
            break
    prop_anchors = []
    prop_wins = []
    prop_loss = []
    prop_dev = []
    prop_mean = []
    for i in range(K):
        if cluster_wins[i] > 1:
            prop_anchors.append(anchors[i])
            prop_wins.append(cluster_wins[i])
            loss, deviation, mean = getBinCrossLossAndDeviation(anchors[i], clusters[i])
            prop_loss.append(loss)
            prop_dev.append(deviation)
            prop_mean.append(mean)
    resDict[dictIndex] = [prop_anchors, prop_wins, prop_loss, prop_dev, prop_mean]
def getBestCluster(box, anchors, processNumber, dictOfIndices):
    intersection = 0.0
    index = 0
    for i, anchor in enumerate(anchors):
        tmptersec = getIOU(box, anchor)
        if tmptersec > intersection:
            index = i
            intersection = tmptersec
    dictOfIndices[processNumber] = index
def getIOUs(boxes, anchor, processNumber, dictOfIOU):
    for i, box in enumerate(boxes):
        dictOfIOU[processNumber][i] = getIOU(box, anchor)
def meanBoxos(clusters, i, dictos):
    sumWidth = 0.0
    sumHeight = 0.0
    for box in clusters[i]:
        sumWidth += box[0]
        sumHeight += box[1]
    dictos[i] = [sumWidth/len(clusters[i]), sumHeight/len(clusters[i])]   

def getAnchorsMultiprocess(initializers, iterations, bboxes):
    anchors = initializers
    K = len(initializers)
    prev_anchors = []
    for iter in range(iterations):
        clusters = [[] for i in range(len(initializers))]
        cluster_wins = []

        for _ in range(len(initializers)):
            cluster_wins.append(0)

        #print(iter)
        manager = mp.Manager()
        dictos = manager.dict()
        listOfDicts = []
        for i in range(K):
            listOfDicts.append(manager.dict())

        jobs = []
        for i, anchor in enumerate(anchors):
            p = mp.Process(target = getIOUs, args=(bboxes, anchor, i, listOfDicts))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        for i, box in enumerate(bboxes):
            tmp = 0.0
            index = 0
            for ii in range(len(anchors)):
                if tmp < listOfDicts[ii][i]:
                    tmp = listOfDicts[ii][i]
                    index = ii
            clusters[index].append(box)
            cluster_wins[index] += 1

        prev_prev_anchors = prev_anchors[:]
        prev_anchors = anchors[:]

        jobs = []
        for i in range(K):
            if len(clusters[i]) != 0:
                p = mp.Process(target = meanBoxos, args=(clusters, i, dictos))
                jobs.append(p)
                p.start()

        for proc in jobs:
            proc.join()
        
        for i in range(K):
            if len(clusters[i]) != 0:
                anchors[i] = dictos[i]
                
        if prev_anchors == anchors or anchors == prev_prev_anchors:
            print("Stopping K-means at iteration {}".format(iter))
            break
    prop_anchors = []
    prop_wins = []
    prop_loss = []
    prop_dev = []
    prop_mean = []
    for i in range(K):
        if cluster_wins[i] > 1:
            prop_anchors.append(anchors[i])
            prop_wins.append(cluster_wins[i])
            loss, deviation, mean = getBinCrossLossAndDeviation(anchors[i], clusters[i])
            prop_loss.append(loss)
            prop_dev.append(deviation)
            prop_mean.append(mean)
    return prop_anchors, prop_wins, prop_loss, prop_dev, prop_mean

def loadBoxes(dirLabels):
    bboxes = []
    for filename in os.listdir(dirLabels):
        tree = ElementTree.parse(dirLabels+"/"+filename)
        root = tree.getroot()
        bboxesImage = []
        for obj in root.findall("object"):
            
            size = root.find("size")
            xMax = float(size.find("width").text)
            yMax = float(size.find("height").text)
            for bbox in obj.findall("bndbox"):
            
                
                bboxes.append([float(bbox.find("xmax").text)/xMax-float(bbox.find("xmin").text)/xMax, float(bbox.find("ymax").text)/yMax-float(bbox.find("ymin").text)/yMax])
    
    return bboxes
def getClusterMean(means, count, total):
    sumOfMeans = 0.0
    for i, mean in enumerate(means):
        sumOfMeans += (mean*count[i])/total
    return sumOfMeans#/len(count)

def getBestClusters(results, boxCount):
    bestMean = 0.0
    index = 0
    for i, result in enumerate(results):
        tmpMean = getClusterMean(result[4], result[1], boxCount)

        #print(result[4])
        if tmpMean > bestMean:
            index = i
            bestMean = tmpMean
    #print(bestMean)
    return results[index]
            
def main():
    if (len(sys.argv)) != 4: 
        print("Argument count should actually be 2... dumbass")
        for arg in sys.argv:
            print(arg)
        return 1
    boxes = loadBoxes(sys.argv[1])
    results = []
    jobs = []
    manager = mp.Manager()
    resDict = manager.dict()
    for i in range(int(sys.argv[3])):
        init = []


        for _ in range(1, int(sys.argv[2])+1):
            init.append(rd.choice(boxes))

        p = mp.Process(target = getAnchors, args=(init, 1000, boxes, i, resDict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    results = []
    for i in range(int(sys.argv[3])):
        results.append(resDict[i])
    res_anchors, res_wins, res_loss, res_dev, res_mean = getBestClusters(results, len(boxes))
    for i, anchor in enumerate(res_anchors):
        print("\nANCHOR BOX #{}:".format(i+1))
        print("box: {}".format(anchor))
        print("wins: {}".format(res_wins[i]))
        print("mean: {}".format(res_mean[i]))
        print("deviation: {}".format(res_dev[i]))
        print("loss: {}".format(res_loss[i]))
if __name__ == "__main__":
    main()
