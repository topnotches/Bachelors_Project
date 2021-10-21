from typing import List
import cv2
import os
import numpy as np
import imgaug.augmenters as iaa
from random import randrange
from imgaug.augmentables.batches import UnnormalizedBatch

def loadFrames(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

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
    images = loadFrames('/home/topnotches/frames')
    ListOfImages = [ [] for _ in range(len(images)) ]
    batchsize = 7
    batchcount = int(len(images)/batchsize)

    print('STEP 2: Appending images')
    for j, im in enumerate(images):
        ListOfImages[j].append(im)
        

    print('STEP 3: Augmenting batches')
    for ii in range(7):

        print('Creating augmented dataset {} of 7:'.format(ii+1))
        batches = [UnnormalizedBatch(images=images[i:i+batchsize]) for i in range(batchcount)]
        batches_aug = list(seq.augment_batches(batches, background=True))
        imAug = []
        print('Appending images...')
        for j in range(batchcount):
            for i in range(batchsize):
            
                imAug.append(batches_aug[j].images_aug[i])
        for j, im in enumerate(imAug):
            ListOfImages[j].append(im)
    print('STEP 5: Writing images')
    writeImages(ListOfImages)

 




if __name__ == '__main__':
    main()
