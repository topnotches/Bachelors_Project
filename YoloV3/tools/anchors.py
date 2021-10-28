
import sys



def getAnchors(K, initializers, iterations, bboxes):


def loadData(dirLabels):
    bboxes = []
    for filename in os.listdir(dirLabels):
        tree = ElementTree.parse(dirLabels+"/"+filename)
        root = tree.getroot()
        bboxesImage = []
        for obj in root.findall("object"):
            for bbox in obj.findall("bndbox"):
            

                bboxesImage.append(x1=float(bbox.find("xmin").text),y1=float(bbox.find("ymin").text),x2=float(bbox.find("xmax").text),y2=float(bbox.find("ymax").text)
        bboxes.append(bboxesImage)
    return bboxes

def main():
    if (len(sys.argv)) != 2: 
        print("Argument count should actually be 2... dumbass")
        return 1
    
if __name__ == "__main__":
    main()
