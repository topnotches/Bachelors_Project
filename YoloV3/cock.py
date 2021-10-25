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
        img = cv2.imread(os.path.join(dirImages,filename[:-4]+".png"))
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
    
    iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 3.0)))
    #iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))),
    #iaa.Sometimes(0.4, iaa.Canny(alpha=(0.0, 0.5))),
    #iaa.Sometimes(0.4, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
    #iaa.Sometimes(0.4, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.Fog(severity=randrange(1,4))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.ZoomBlur(severity=randrange(1,4))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.DefocusBlur(severity=randrange(1,4))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.MotionBlur(severity=randrange(1,4))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.Contrast(severity=randrange(1,4))),
    #iaa.Sometimes(0.4, iaa.imgcorruptlike.Brightness(severity=randrange(1,4)))
    
])

def saveLabels(bboxes, dirLabels, dirImages, name, size):
    
    root = ET.Element("annotations")
    ET.SubElement(root, "folder").text = dirImages[dirImages.rfind('/'):]
    ET.SubElement(root, "filename").text = name + '.png'
    ET.SubElement(root, "path").text = dirImages + '/' + name + '.png'
    src = ET.SubElement(root, "source").text = name + '.png'
    ET.SubElement(src, "width").text = str(size[0])
    ET.SubElement(src, "height").text = str(size[1])
    ET.SubElement(src, "depth").text = str(size[2])
    ET.SubElement(root, "segmented").text = '.vscode/0'
    
    for bbox in bboxes:
        bbox = ET.SubElement(root, "object")
        ET.SubElement(bbox, "name").text = 'polyp'
        ET.SubElement(bbox, "pose").text = 'Unspecified'
        ET.SubElement(bbox, "truncated").text = '0'
        ET.SubElement(bbox, "difficult").text = '0'
        bndbox = ET.SubElement(bbox, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox[0][0])
        ET.SubElement(bndbox, "ymin").text = str(bbox[0][1])
        ET.SubElement(bndbox, "xmax").text = str(bbox[1][0])
        ET.SubElement(bndbox, "ymax").text = str(bbox[1][1])
        

def saveData(images, bboxes, dirLabels, dirImages):

    for imNum, image in enumerate(images):
        name = '/scene{0:05d}'.format(imNum)
        cv2.imwrite(dirImages + name + '.png', image)
        saveLabels(bboxes[imNum], dirLabels, dirImages, name, [len(image[0]), len(image), len(image[0][0])])

def main():
    
    print('STEP 1: Loading images')
    images, bboxes = loadData("./data/preAug/Annotations", "./data/preAug/JPEGImages")
    augmentedImages = []
    augmentedBboxes = []
    batchsize = 7
    batchcount = int(len(images)/batchsize)


    print('STEP 2: Augmenting batches')
    for ii in range(2):

        print('Creating augmented dataset {} of 7:'.format(ii+1))
        batches = [UnnormalizedBatch(images=images[i:i+batchsize], bounding_boxes=bboxes[i:i+batchsize]) for i in range(batchcount)]
        batches_aug = list(seq.augment_batches(batches, background=True))
        print('Appending images...')
        if ii == 0:
            for batch in batches:
                augmentedBboxes.extend(batch.bounding_boxes_unaug)
                augmentedImages.extend(batch.images_unaug)
        

        for batch in batches:
            augmentedBboxes.extend(batch.bounding_boxes_aug)
            augmentedImages.extend(batch.images_aug)
                
    print('STEP 3: Writing images')
    saveData(augmentedImages, augmentedBboxes, "./data/postAug/Annotations", "./data/postAug/JPEGImages")

 




if __name__ == '__main__':
    main()
