from typing import List
import cv2
import os
import numpy as np
import imgaug.augmenters as iaa
from random import randrange

def loadFrames(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

seq = iaa.Sequential([
    
    iaa.GaussianBlur(sigma=(0, 3.0))
    
])

def writeImages(imLists):

    for listNum, list in enumerate(imLists):
        for imNum, image in enumerate(list):
            cv2.imwrite(image, './data/augmented/scene{}_{}.PNG'.format("%05d",listNum,imNum))
            
    return

def main():
    
    images = loadFrames('/home/topnotces/frames')
    ListOfImages = [ [] for _ in range(len(images)) ]

    for _ in range(10):
        imAug = seq(images=images) 
        for j, im in enumerate(imAug):
            print(j)
            ListOfImages[j].append(im)

    writeImages(ListOfImages)

 




if __name__ == '__main__':
    main()
