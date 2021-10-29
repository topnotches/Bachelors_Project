
import sys
import os
import math
from imgaug.augmentables.batches import UnnormalizedBatch
from xml.etree import ElementTree, ElementInclude
import xml.etree.ElementTree as ET
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
    
    return intersection/(area1+area2-intersection)

def getBinCrossLossAndDeviation(anchBox, trueBoxes):
    listIOU = []
    sumLoss = 0.0
    sumIOU  = 0.0
    meanIOU = 0.0
    sumL2  = 0.0
    for trueBox in trueBoxes:
        IOU = getIOU(trueBox, anchBox)
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

def getAnchors(initializers, iterations, bboxes):
    anchors = initializers
    K = len(initializers)
    prev_anchors = []
    for iter in range(iterations):
        clusters = [[] for i in range(len(initializers))]
        cluster_wins = []
        for _ in range(len(initializers)):
            cluster_wins.append(0)
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
            
                
                bboxes.append([float(bbox.find("xmin").text)/xMax, float(bbox.find("ymin").text)/yMax, float(bbox.find("xmax").text)/xMax, float(bbox.find("ymax").text)/yMax])
    
    return bboxes

def main():
    if (len(sys.argv)) != 4: 
        print("Argument count should actually be 2... dumbass")
        for arg in sys.argv:
            print(arg)
        return 1
    boxes = loadBoxes(sys.argv[1])
    init = []
    for i in range(1, int(sys.argv[3])+1):
        init.append([i/(int(sys.argv[3])+1), i/(int(sys.argv[3])+1)])
    print(init)
    
    get, geet, gurd, gyt, gat = (getAnchors(init, int(sys.argv[2]), boxes))
    for git, gut in enumerate(get):
        print("\nANCHOR BOX #{}:".format(git+1))
        print("box: {}".format(gut))
        print("wins: {}".format(geet[git]))
        print("mean: {}".format(gat[git]))
        print("deviation: {}".format(gyt[git]))
        print("loss: {}".format(gurd[git]))
if __name__ == "__main__":
    main()
