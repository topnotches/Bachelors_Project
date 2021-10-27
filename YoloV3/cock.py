from typing import List
import cv2
import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from random import randrange
from imgaug.augmentables.batches import UnnormalizedBatch
from xml.etree import ElementTree, ElementInclude
import xml.etree.ElementTree as ET
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
    
    iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 3.0))),
    iaa.Sometimes(0.4, iaa.TranslateX(px=(-20, 20))),
    iaa.Sometimes(0.4, iaa.Rotate((-45, 45))),
    iaa.Sometimes(0.4, iaa.AddToBrightness((-30, 30))),
    iaa.Sometimes(0.4, iaa.Sequential([ iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"), iaa.WithChannels(0, iaa.Add((50, 100))), iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])),
    iaa.Sometimes(0.4, iaa.TranslateY(px=(-20, 20)))

    
])

def saveLabels(bboxes, dirLabels, dirImages, name, size):
    
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = dirImages[dirImages.rfind('/')+1:]
    ET.SubElement(root, "filename").text = name + '.png'
    ET.SubElement(root, "path").text = dirImages + '/' + name + '.png'

    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = 'Unknown'
    siz = ET.SubElement(root, "size")
    ET.SubElement(siz, "width").text = str(size[0])
    ET.SubElement(siz, "height").text = str(size[1])
    ET.SubElement(siz, "depth").text = str(size[2])
    ET.SubElement(root, "segmented").text = '0'
    
    for box in bboxes:
        bbox = ET.SubElement(root, "object")
        ET.SubElement(bbox, "name").text = 'polyp'
        ET.SubElement(bbox, "pose").text = 'Unspecified'
        ET.SubElement(bbox, "truncated").text = '0'
        ET.SubElement(bbox, "difficult").text = '0'
        bndbox = ET.SubElement(bbox, "bndbox")
        if float(box[0][0]) < 0.0:
            ET.SubElement(bndbox, "xmin").text = str(0)
        else:
            ET.SubElement(bndbox, "xmin").text = str(box[0][0])
        if float(box[0][1]) < 0.0:
            ET.SubElement(bndbox, "ymin").text = str(0)
        else:
            ET.SubElement(bndbox, "ymin").text = str(box[0][1])

            
        if float(box[1][0]) > float(size[0]):
            ET.SubElement(bndbox, "xmax").text = str(size[0])
        else:
            ET.SubElement(bndbox, "xmax").text = str(box[1][0])
        if float(box[1][1]) > float(size[1]):
            ET.SubElement(bndbox, "ymax").text = str(size[1])
        else:
            ET.SubElement(bndbox, "ymax").text = str(box[1][1])
    
    tree = ElementTree.ElementTree()
    tree._setroot(root)
    tree.write(dirLabels+'/'+name+'.xml')
        

def saveData(images, bboxes, dirLabels, dirImages):

    for imNum, image in enumerate(images):
        name = 'scene{0:05d}'.format(imNum)
        cv2.imwrite(dirImages + '/' + name + '.png', image)
        saveLabels(bboxes[imNum], dirLabels, dirImages, name, [len(image[0]), len(image), len(image[0][0])])

def main():
    
    print('STEP 1: Loading images')
    images, bboxes = loadData("./data/preAug/Annotations", "./data/preAug/JPEGImages")
    augmentedImages = []
    augmentedBboxes = []
    batchsize = 7
    batchcount = int(len(images)/batchsize)
    runners = 30

    batches = [UnnormalizedBatch(images=images[i:i+batchsize], bounding_boxes=bboxes[i:i+batchsize]) for i in range(batchcount)]
    print('STEP 2: Augmenting batches')
    for ii in range(runners):

        print('Creating augmented dataset {} of {}:'.format(ii+1, runners))
        batches_aug = list(seq.augment_batches(batches, background=True))
        print('Appending images...')
        if ii == 0:
            for batch in batches_aug:
                augmentedBboxes.extend(batch.bounding_boxes_unaug)
                augmentedImages.extend(batch.images_unaug)
        

        for batch in batches_aug:
            augmentedBboxes.extend(batch.bounding_boxes_aug)
            augmentedImages.extend(batch.images_aug)
                
    print('STEP 3: Writing images')
    saveData(augmentedImages, augmentedBboxes, "./data/postAug/Annotations", "./data/postAug/JPEGImages")

 




if __name__ == '__main__':
    main()
