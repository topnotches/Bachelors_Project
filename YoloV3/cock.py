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
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
deq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)
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
        if not(float(box[0][0]) > float(size[0]) or float(box[0][1]) > float(size[1]) or float(box[1][0]) < 0.0 or float(box[1][1]) < 0.0):
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
    runners = 60

    batches = [UnnormalizedBatch(images=images[i:i+batchsize], bounding_boxes=bboxes[i:i+batchsize]) for i in range(batchcount)]
    print('STEP 2: Augmenting batches')
    for ii in range(runners):

        print('Creating augmented dataset {} of {}:'.format(ii+1, runners))
        batches_aug = list(deq.augment_batches(batches, background=True))
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
