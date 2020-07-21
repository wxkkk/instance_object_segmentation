from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

'''
procedure:
1. read the original data
2. filter classes we need
3. process masks into one
4-. show the original image and the mask
4-. save data into h5
'''


def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return 'None'


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


def process_to_mask(coco, img):
    mask = np.zeros((img['height'], img['width']))

    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    for i in range(len(anns)):
        class_name = get_class_name(anns[i]['category_id'], coco.loadCats(cat_ids))

        if class_name in filter_classes:
            pixel_value = filter_classes.index(class_name) + 1
            # annToMask return a binary image
            mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

    return mask


def show_one_by_one(coco):

    dataset = filter_classes_combined(filter_classes, coco)

    for _, image in enumerate(dataset):
        mask = process_to_mask(coco, image)
        img_name = str(image['id']).zfill(12)
        img_name = '{}/images/{}/{}.jpg'.format(data_dir, data_type, img_name)
        original_img = cv2.imread(img_name)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        plt.figure(img_name)

        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        print(type(original_img), original_img.shape)

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        print(type(mask), mask.shape)

        plt.show()


def save_to_h5(coco, path):
    dataset = filter_classes_combined(filter_classes, coco)
    dataset = dataset[0: 100]
    data_arr = np.zeros((len(dataset), 256, 256, 3), dtype=np.float64)
    mask_arr = np.zeros((len(dataset), 256, 256), dtype=np.uint8)

    for i, image in enumerate(dataset):
        img_name = str(image['id']).zfill(12)

        img_name = '{}/images/{}/{}.jpg'.format(data_dir, data_type, img_name)
        original_img = cv2.imread(img_name)

        mask = process_to_mask(coco, image)

        print(original_img.shape)
        data_arr[i] = cv2.resize(original_img, (256, 256))
        mask_arr[i] = cv2.resize(mask, (256, 256))

    with h5py.File(path, 'w') as f:
        f['data'] = data_arr
        f['mask'] = mask_arr

    print(len(dataset))


def read_h5(path):
    with h5py.File(path, 'r') as f:
        data = np.array(f['data'])
        masks = np.array(f['mask'])

    return data, masks


if __name__ == '__main__':
    data_dir = 'E:\extracted_data\COCO'
    data_type = 'train2017'
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)

    filter_classes = ['car', 'truck', 'bus', 'motorcycle']

    h5_path = '../data/train_100.h5'

    # load COCO dataset
    coco = COCO(ann_file)

    # show original images and masks
    # show_one_by_one(coco)

    # save the original dataset into h5
    save_to_h5(coco, h5_path)

    # read h5 file
    # data, masks = read_h5(h5_path)
    # print(data.shape, masks.shape)