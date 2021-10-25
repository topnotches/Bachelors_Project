from typing import List
import cv2
import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from random import randrange
from imgaug.augmentables.batches import UnnormalizedBatch
from xml.etree import ElementTree, ElementInclude
def loadData(dirLabels, dirImages):
    images = []
    bboxes = []
    for filename in os.listdir(dirLabels):
        img = cv2.imread(os.path.join(folder,filename[:-4]))
        if img is not None:
            images.append(img)
        tree = ElementTree.parse(dirLabels+"/"+filename)
        root = tree.getroot()
        bboxesImage = []
        for obj in root.findall("object"):
            for bbox in obj.findall("bndbox"):
            

                bboxesImage.append(ia.BoundingBox(x1=float(bbox.find("xmin").text),y1=float(bbox.find("ymin").text),x2=float(bbox.find("xmax").text),y2=float(bbox.find("ymax").text)))
        bboxes.append(bboxesImage)
    return images, bboxes

seq = iaa.Sequential([
    
    iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 3.0))),
    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))),
    iaa.Sometimes(0.4, iaa.Canny(alpha=(0.0, 0.5))),
    iaa.Sometimes(0.4, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
    iaa.Sometimes(0.4, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.Fog(severity=randrange(1,4))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.ZoomBlur(severity=randrange(1,4))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.DefocusBlur(severity=randrange(1,4))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.MotionBlur(severity=randrange(1,4))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.Contrast(severity=randrange(1,4))),
    iaa.Sometimes(0.4, iaa.imgcorruptlike.Brightness(severity=randrange(1,4)))
    
])

def writeImages(imLists):

    for listNum, losots in enumerate(imLists):
        for imNum, image in enumerate(losots):
            cv2.imwrite('./data/augment/scene{0:05d}_{1:02d}.PNG'.format(listNum,imNum), image)
            
    return

def main():
    
    print('STEP 1: Loading images')
    images, bboxes = loadData('./data/preAug/Annotations', './data/preAug/JPEGImages')
    augmentedImages = []
    augmentedBboxes = []
    batchsize = 7
    batchcount = int(len(images)/batchsize)


    print('STEP 2: Augmenting batches')
    for ii in range(7):

        print('Creating augmented dataset {} of 7:'.format(ii+1))
        batches = [UnnormalizedBatch(images=images[i:i+batchsize], bounding_boxes=bboxes[i:i+batchsize]) for i in range(batchcount)]
        batches_aug = list(seq.augment_batches(batches, background=True))
        imAug = []
        print('Appending images...')
        ListOfImages[j].extend(im)
    print('STEP 3: Writing images')
    writeImages(ListOfImages)

 




if __name__ == '__main__':
    main()
