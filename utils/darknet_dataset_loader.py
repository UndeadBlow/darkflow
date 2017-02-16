
import os
import sys
import cv2

def GetAllFilesList(path):
    files = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
    return filter(lambda file: 'directory' not in file, files)

def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all

def darknet_data_loader(labels_path, images_path, classes):
    txt_files = []
    imgs_files = []
    if os.path.isfile(labels_path):
        # then load files from txt list file
        with open(labels_path, 'r') as file:
            txt_files = file.read().split('\n')

        for f in txt_files:
            f = f.strip()

        with open(images_path, 'r') as file:
            imgs_files = file.read().split('\n')

        for f in imgs_files:
            f = f.strip()

    elif os.path.isdir(labels_path):
        possible_img_extensions = ['.png', '.PNG', '.jpg', '.jpeg']
        files = GetAllFilesList(labels_path)

        for file in files:
            if '.txt' in file:
                pure_filename = file[file.rfind('/') + 1 : file.rfind('.')]
                #print('pure filename:', pure_filename)
                img_filename = images_path + '/' + pure_filename
                for ext in possible_img_extensions:
                    if os.path.isfile(img_filename + ext):
                        img_filename += ext

                if not os.path.isfile(file) or not os.path.isfile(img_filename):
                    print('Not possible to load ', file, ' or ', img_filename,
                    ' check that txt file has ' +
                    'img pair and is correct! That file will be not loaded')
                else:
                    txt_files.append(file)
                    imgs_files.append(img_filename)

        if len(txt_files) != len(imgs_files):
            print('Lenghts of images and labels is not equal. Please check your path and try again!')
            quit()

        dumps = []
        for txt_file, img_file in zip(txt_files, imgs_files):
            with open(txt_file) as tfile:
                data = tfile.read()

            img = cv2.imread(img_file)
            objects = []
            for line in data.split('\n'):
                line = line.strip()
                line = line.strip('\n')
                obj_data = line.split(' ')
                if (obj_data == ['']):
                    continue
                if (len(obj_data) != 5): # class x y w h
                    print('File ', txt_file, ' seems to be corrupted and will not be loaded!')
                    print(obj_data)
                    break
                cl = obj_data[0]
                x = int(float(obj_data[1]) * img.shape[1])
                y = int(float(obj_data[2]) * img.shape[0])
                w = int(float(obj_data[3]) * img.shape[1])
                h = int(float(obj_data[4]) * img.shape[0])
                objects.append([classes[int(cl)], x, y, x + w, y + h])

            if objects == []:
                continue
            #print('New file append: ', [img_file[img_file.rfind('/') + 1: ], [img.shape[1], img.shape[0], objects]])
            dumps.append([img_file[img_file.rfind('/') + 1: ], [img.shape[1], img.shape[0], objects]])

        return dumps
