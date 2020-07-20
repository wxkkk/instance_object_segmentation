from pycocotools.coco import COCO
from pycocotools.mask import decode, encode, merge

import numpy as np
import skimage.io as io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return 'None'


if __name__ == '__main__':

    dataDir = 'E:\extracted_data\COCO'
    dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    # print(cats)

    # print('class name is', get_class_name(77, cats))

    filter_classes = ['car', 'truck', 'bus', 'motorcycle']

    # fetch class ids
    filter_cat_ids = coco.getCatIds(catNms=filter_classes)
    img_ids = coco.getImgIds(catIds=filter_cat_ids)
    print('Number of images containing the classes in list:', len(img_ids))

    # load and display a random image
    img = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
    I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0

    plt.axis('off')
    plt.imshow(I)
    # plt.show()

    # load and display annotations
    plt.imshow(I)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns)
    # plt.show()

    #
    mask = np.zeros((img['height'], img['width']))

    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    for i in range(len(anns)):
        class_name = get_class_name(anns[i]['category_id'], cats)
        pixel_value = filter_classes.index(class_name) + 1
        # annToMask return a binary image
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

    plt.imshow(mask)
    plt.show()