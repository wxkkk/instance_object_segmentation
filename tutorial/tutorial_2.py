from pycocotools.coco import COCO

'''
implement the combination classes filter
'''


def filter_classes_combined(filter_classes, coco):

    images = []
    if filter_classes != None:
        # iterate for each individual class in the list
        for class_name in filter_classes:
            # get all images containing given classes
            cat_ids = coco.getCatIds(catNms=class_name)
            img_ids = coco.getImgIds(catIds=cat_ids)
            images += coco.loadImgs(img_ids)
    else:
        img_ids = coco.getImgIds()
        images = coco.loadImgs(img_ids)

    # filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    return unique_images


if __name__ == '__main__':
    filter_classes = ['car', 'truck', 'bus', 'motorcycle']

    dataDir = 'E:\extracted_data\COCO'
    dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    dataset = filter_classes_combined(filter_classes[3:4], coco)

    print('Number of images containing the filter classes:', len(dataset))

    # print(dataset[0]['file_name'])
