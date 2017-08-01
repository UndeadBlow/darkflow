import os
import sys
import cv2
import utils.slicer as slicer
import random

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

def navmii_data_loader(path, classes_names, FLAGS, classes_to_load = []):

    files = GetAllFilesListRecusive(path, ['_o.xml'])
    random.shuffle(files)
    video_exts = ['.mp4', '.mpeg', '.avi', '.mov']
    all_videofiles = GetAllFilesListRecusive(path, video_exts)
    dumps = []

    if FLAGS.slice_frame:
        slice_step = FLAGS.slice_step
        slice_win_size = FLAGS.slice_win_size
        slice_down_rate = FLAGS.slice_down_rate
        dumps = []

        for file in files:
            frame, objects = slicer.getObjectsAndFrame(file)
            subframes, sliced_objects, positions, sizes = slicer.SliceFrame(frame, objects, slice_step, slice_win_size, slice_down_rate)

            for index in range(len(subframes)):
                subframe_sliced_objects = sliced_objects[index]
                position = positions[index]
                size = sizes[index]
                subframe = subframes[index]

                video_filename = file[: file.rfind('/')]

                frame_number = file[file.rfind('/'): ]
                frame_number = frame_number[frame_number.find('f') + 1: frame_number.find('_o')]

                video_filename = ([f for f in all_videofiles if video_filename in f])[0]
                if not video_filename or not os.path.exists(video_filename):
                    continue

                video_filename = video_filename + ':' + frame_number
                video_filename += '@' + str(position[0]) + '_' + str(position[1]) + '_' +\
                str(size[0]) + '_' + str(size[1]) + '_' + str(slice_win_size)

                dump_objects = []
                for obj in subframe_sliced_objects:
                    cl = obj.type - 1

                    if cl not in classes_to_load and classes_to_load != []:
                        continue

                    x = obj.rect.x

                    if x < 0:
                        x = 0
                    if x > frame.shape[1]:
                        x = frame.shape[1]

                    y = obj.rect.y

                    if y < 0:
                        y = 0
                    if y > frame.shape[0]:
                        y = frame.shape[0]

                    w = obj.rect.w

                    if w < FLAGS.min_width:
                        continue
                    if w + x > (frame.shape[1]):
                        w -= (w + x - frame.shape[1])
                    if w < FLAGS.min_width:
                        continue
                    if w > (frame.shape[0] * 0.3):
                        continue

                    h = obj.rect.h

                    if h < FLAGS.min_height:
                        continue
                    if h + y > (frame.shape[0]):
                        h -= (h + y - frame.shape[0])
                    if h < FLAGS.min_height:
                        continue
                    if h > (frame.shape[1] * 0.3):
                        continue

                    if h < (w / 2.0) and w < (h / 2.0):
                        continue

                    print([classes_names[int(cl)], x, y, x + w, y + h])

                    dump_objects.append([classes_names[int(cl)], x, y, x + w, y + h])

                if dump_objects == []:
                    continue

                dumps.append([video_filename, [subframe.shape[1], subframe.shape[0], dump_objects]])

    else:
        for file in files:

            video_filename = file[: file.rfind('/')]
            t = list([f for f in all_videofiles if video_filename in f])
            video_filename = t[0] if len(t) > 0 else file

            if slicer.isVideofile(video_filename):
                frame_number = file[file.rfind('/'): ]
                frame_number = frame_number[frame_number.find('f') + 1: frame_number.find('_o')]

            frame, objects = slicer.getObjectsAndFrame(file)

            if not frame.any():
                print(file, "Null frame")
                continue

            dump_objects = []
            for obj in objects:

                cl = obj.type - 1

                if cl not in classes_to_load and classes_to_load != []:
                    continue

                x = obj.rect.x

                if x < 0:
                    x = 0
                if x > frame.shape[1]:
                    x = frame.shape[1]

                y = obj.rect.y

                if y < 0:
                    y = 0
                if y > frame.shape[0]:
                    y = frame.shape[0]

                w = obj.rect.w

                if w < FLAGS.min_width:
                    continue
                if w + x > (frame.shape[1]):
                    w -= (w + x - frame.shape[1])
                if w < FLAGS.min_width:
                    continue
                if w > (frame.shape[0] * 0.3):
                    continue

                h = obj.rect.h

                if h < FLAGS.min_height:
                    continue
                if h + y > (frame.shape[0]):
                    h -= (h + y - frame.shape[0])
                if h < FLAGS.min_height:
                    continue
                if h > (frame.shape[1] * 0.3):
                    continue

                if h < (w / 2.0) and w < (h / 2.0):
                    continue

                dump_objects.append([classes_names[int(cl)], x, y, x + w, y + h])

            if dump_objects == []:
                continue

            if slicer.isVideofile(video_filename):
                dumps.append([video_filename + ':' + frame_number, [frame.shape[1], frame.shape[0], dump_objects]])
            else:
                dumps.append([file.replace('_o.xml', '.png'), [frame.shape[1], frame.shape[0], dump_objects]])

    return dumps

def darknet_data_loader(path, classes_names, classes_to_load = []):
    txt_files = []
    imgs_files = []
    if os.path.isfile(path):
        # then load files from txt list file
        with open(labels_path, 'r') as file:
            txt_files = file.read().split('\n')

        for f in txt_files:
            f = f.strip()

        with open(images_path, 'r') as file:
            imgs_files = file.read().split('\n')

        for f in imgs_files:
            f = f.strip()

    elif os.path.isdir(path):
        possible_img_extensions = ['.png', '.PNG', '.jpg', '.jpeg']
        files = GetAllFilesListRecusive(path, ['.txt'])

        for file in files:
            if '.txt' in file:
                pure_filename = file[file.rfind('/') + 1 : file.rfind('.')]
                #print('pure filename:', pure_filename)
                img_filename = path + '/' + pure_filename
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
                if cl in classes_to_load or classes_to_load == []:
                    x = int(float(obj_data[1]) * img.shape[1])
                    y = int(float(obj_data[2]) * img.shape[0])
                    w = int(float(obj_data[3]) * img.shape[1])
                    h = int(float(obj_data[4]) * img.shape[0])
                    obj = [classes_names[int(cl)], x, y, x + w, y + h]
                    objects.append(obj)

            if objects == []:
                continue
            #print('New file append: ', [img_file[img_file.rfind('/') + 1: ], [img.shape[1], img.shape[0], objects]])
            dumps.append([img_file[img_file.rfind('/') + 1: ], [img.shape[1], img.shape[0], objects]])

        return dumps
