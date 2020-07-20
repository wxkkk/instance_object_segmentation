from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt


from tutorial import tutorial_1, tutorial_2

'''
implement process segmentation mask into one(semantic segmentation)
'''


def process_to_mask(img):
    mask = np.zeros((img['height'], img['width']))

    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    for i in range(len(anns)):
        class_name = tutorial_1.get_class_name(anns[i]['category_id'], cats)

        if class_name in filter_classes:
            pixel_value = filter_classes.index(class_name) + 1
            # annToMask return a binary image
            mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

    img_name = str(img['id']).zfill(12)
    plt.figure(img_name)
    # plt.imshow(mask)
    plt.subplot(1, 2, 1)
    plt.imshow(mask)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)

    plt.show()


if __name__ == '__main__':
    dataDir = 'E:\extracted_data\COCO'
    dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    filter_classes = ['car', 'truck', 'bus', 'motorcycle']

    dataset = tutorial_2.filter_classes_combined(filter_classes, coco)

    for _, image in enumerate(dataset):
        process_to_mask(image)
