from pycocotools.coco import COCO
from pycocotools.mask import decode, encode, merge
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

import zipfile

dataDir = 'E:\extracted_data\COCO'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)

path = r'E:\extracted_data\COCO\annotations\instances_train2017.json'

coco_ann = COCO(annFile)

coco_ann.info()

# print(coco_train.dataset['annotations'][0])

# get all images containing given categories, select one at random
catIds = coco_ann.getCatIds(catNms=['car', 'bus', 'motorcycle', 'truck'])
imgIds = coco_ann.getImgIds(catIds=catIds)
# imgIds = coco_ann.getImgIds()
# img = coco_train.loadImgs(imgIds[9])[0]
#
img = coco_ann.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco_ann.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_ann.loadAnns(annIds)
print(anns)
print(len(anns))
coco_ann.showAnns(anns, draw_bbox=False)
plt.show()

merge()

if len(anns) != 0:
    for i, ann in enumerate(anns):
        mask = coco_ann.annToMask(ann)
        img = encode(mask)
        print(img)
        # io.imsave('/1.png', img)
        plt.imshow(mask)
        plt.savefig('1.jpg')
        plt.show()
