import cv2
import xml.etree.ElementTree as ET
import numpy as np
import copy

def SliceFrameOnFourFixed(frame, objects):
    # only x nedeed
    ver_line = frame.shape[1] / 2
    # only y nedeed
    hor_line = frame.shape[0] / 2

    line = ver_line
    line_left_shift = 0
    line_right_shift = 0

    intersect_objects_count = sum([obj.isVerLineIntersectObject(line) for obj in objects])

    while (intersect_objects_count != 0):
        line_left_shift = line_left_shift + 1
        line = ver_line - line_left_shift
        intersect_objects_count = sum([obj.isVerLineIntersectObject(line) for obj in objects])

    line = ver_line
    intersect_objects_count = sum([obj.isVerLineIntersectObject(line) for obj in objects])
    while (intersect_objects_count != 0):
        line_right_shift = line_right_shift + 1
        line = ver_line + line_right_shift
        intersect_objects_count = sum([obj.isVerLineIntersectObject(line) for obj in objects])

    if (line_left_shift > line_right_shift):
        ver_line = ver_line + line_right_shift
    else:
        ver_line = ver_line - line_left_shift

    line = hor_line
    line_left_shift = 0
    line_right_shift = 0

    intersect_objects_count = sum([obj.isHorLineIntersectObject(line) for obj in objects])
    while (intersect_objects_count != 0):
        line_left_shift = line_left_shift + 1
        line = hor_line - line_left_shift
        intersect_objects_count = sum([obj.isHorLineIntersectObject(line) for obj in objects])

    line = hor_line
    intersect_objects_count = sum([obj.isHorLineIntersectObject(line) for obj in objects])
    while (intersect_objects_count != 0):
        line_right_shift = line_right_shift + 1
        line = hor_line + line_right_shift
        intersect_objects_count = sum([obj.isHorLineIntersectObject(line) for obj in objects])

    if (line_left_shift > line_right_shift):
        hor_line = hor_line + line_right_shift
    else:
        hor_line = hor_line - line_left_shift

    rois = []
    # y1 : y2, x1 : x2
    rois.append(frame[0 : hor_line, 0 : ver_line])
    rois.append(frame[0 : hor_line, ver_line : frame.shape[1]])
    rois.append(frame[hor_line : frame.shape[0], 0 : ver_line])
    rois.append(frame[hor_line : frame.shape[0], ver_line : frame.shape[1]])

    objs = []

    objs.append([obj for obj in objects if isObjInROI(0, ver_line, 0, hor_line, obj)])
    for obj in objs[len(objs) - 1]:
        obj.rect.calcRelativies(rois[0].shape[1], rois[2].shape[0])

    objs.append([obj for obj in objects if isObjInROI(ver_line, frame.shape[1], 0, hor_line, obj)])
    for obj in objs[len(objs) - 1]:
        obj.rect.x -= ver_line
        obj.rect.calcRelativies(rois[1].shape[1], rois[1].shape[0])

    objs.append([obj for obj in objects if isObjInROI(0, ver_line, hor_line, frame.shape[0], obj)])
    for obj in objs[len(objs) - 1]:
        obj.rect.y -= hor_line
        obj.rect.calcRelativies(rois[2].shape[1], rois[2].shape[0])

    objs.append([obj for obj in objects if isObjInROI(ver_line, frame.shape[1], hor_line, frame.shape[0], obj)])
    for obj in objs[len(objs) - 1]:
        obj.rect.x -= ver_line
        obj.rect.y -= hor_line

        obj.rect.calcRelativies(rois[3].shape[1], rois[3].shape[0])

    return rois, objs

def isVideofile(filename):
    if isinstance(filename, np.ndarray):
        return False
    filename = filename.lower()
    return ('.mp4' in filename or '.avi' in filename or '.mpeg' in filename or
    '.mov' in filename)

def isImagefile(filename):
    if isinstance(filename, np.ndarray):
        return False
    filename = filename.lower()
    return ('.jpg' in filename or '.jpeg' in filename or '.png' in filename)

import os
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

def RecalcObject(input_object, position, window_size):
    obj = copy.deepcopy(input_object)

    obj.rect.x -= position[1]
    if (obj.rect.x < 0):
        obj.rect.x = 0
    if (obj.rect.x > window_size):
        obj.rect.x = window_size

    obj.rect.y -= position[0]
    if (obj.rect.y < 0):
        obj.rect.y = 0
    if (obj.rect.y > window_size):
        obj.rect.y = window_size

    if (obj.rect.w > window_size):
        obj.rect.w = window_size
    if (obj.rect.h > window_size):
        obj.rect.h = window_size

    obj.rect.calcRelativies(window_size, window_size)

###########################################
# Remove for all types !!!!!!!!!!!!!!!!!!!!
###########################################
    if obj.type > 5:
        return []

    # Emperic values of image size
    if obj.rect.relative_w < 0.15 or obj.rect.relative_h < 0.15:
        return []

    if obj.rect.relative_w > 0.97 or obj.rect.relative_h > 0.97:
        return []

    return obj

def CreateSubframes(frame, objects, step, window_size):
    subframes = []
    positions = []

    objs = []
    temp_objects = copy.deepcopy(objects)
    # y, x
    position = [0, 0] # initial position

    # Walking through Y
    while (position[0] + window_size) < frame.shape[0]:

        # Walking through X
        while (position[1] + window_size) < frame.shape[1]:
            subframes.append(frame[position[0] : (position[0] + window_size),
                              position[1] : (position[1] + window_size)])
            # Adding objects that belongs to ROI
            roi_objcts_list = [RecalcObject(obj, position, window_size) for obj in temp_objects if isObjInROI(position[1], (position[1] + window_size),
            position[0],  (position[0] + window_size), obj)]
            # Fix new coords
            roi_objcts_list = filter(lambda x: x != [], roi_objcts_list)

            objs.append(roi_objcts_list)
            positions.append(copy.deepcopy(position))

            position = [position[0], position[1] + int(frame.shape[1] * step)]

        position = [position[0] + int(frame.shape[0] * step), 0]


    return subframes, objs, positions

def SliceFrame(frame, objects, step, win_size, step_downsample):

    subframes = []
    positions = []
    objs = []
    sizes = []

    while frame.shape[0] >= win_size and frame.shape[1] >= win_size:
        f, o, p = CreateSubframes(frame, objects, step, win_size)

        size = [frame.shape[0], frame.shape[1]]
        for i in range(len(f)):
            sizes.append(size)

        subframes += f
        objs += o
        positions += p

        frame = cv2.resize(frame, (int(frame.shape[1] * step_downsample),
                                   int(frame.shape[0] * step_downsample)))

        for obj in objects:
            obj.rect.x *= step_downsample
            obj.rect.y *= step_downsample
            obj.rect.w *= step_downsample
            obj.rect.h *= step_downsample

    return subframes, objs, positions, sizes

def isObjInROI(rx1, rx2, ry1, ry2, obj):
    ox1 = obj.rect.x
    oy1 = obj.rect.y

    ox2 = obj.rect.x + obj.rect.w
    oy2 = obj.rect.y + obj.rect.h

    if (ox1 < 0):
        ox1 = 0

    if (oy1 < 0):
        oy1 = 0

    # if (ox2 > frame.shape[1]):
    #     ox2 = frame.shape[1]
    #
    # if (oy2 > frame.shape[0]):
    #     oy2 = frame.shape[0]

    max_diff = obj.rect.w * 0.25

    if (ox1 < rx1):
        if (rx1 - ox1) > max_diff:
            return False

    if (ox1 > rx2):
        if (ox1 - rx2) > max_diff:
            return False

    if (oy1 < ry1):
        if (ry1 - oy1) > max_diff:
            return False

    if (oy1 > ry2):
        if (oy1 - ry2) > max_diff:
            return False

    if (ox2 < rx1):
        if (rx1 - ox2) > max_diff:
            return False

    if (ox2 > rx2):
        if (ox2 - rx2) > max_diff:
            return False

    if (oy2 < ry1):
        if (ry1 - oy2) > max_diff:
            return False

    if (oy2 > ry2):
        if (oy2 - ry2) > max_diff:
            return False

    ox_center = obj.rect.x + obj.rect.w / 2
    oy_center = obj.rect.y + obj.rect.h / 2

    # print('ox_center', ox_center)
    # print('oy_center', oy_center)
    # print('rx1', rx1)
    # print('rx2', rx2)
    # print('ry1', ry1)
    # print('ry2', ry2)

    return (ox_center >= rx1 and ox_center <= rx2 and oy_center >= ry1 and oy_center <= ry2)

class Rect:
    def __init__(self, x, y, w, h, img_w, img_h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

        self.calcRelativies(img_w, img_h)

    def calcRelativies(self, img_w, img_h):
        self.relative_x = float(self.x) / float(img_w)
        self.relative_y = float(self.y) / float(img_h)

        self.relative_w = float(self.w) / float(img_w)
        self.relative_h = float(self.h) / float(img_h)

    def containsPoint(self, point):
        y1 = self.y
        y2 = self.y + self.h
        x1 = self.x
        x2 = self.x + self.w

        # assume point[0] is X and point[1] is Y
        return (point[0] > x1 and point[0] < x2 and point[1] > y1 and point[1] < y2)

class Object:

    def __init__(self, t, subtype, rect_str, frame_shape):
        self.type = int(t)
        self.subtype = int(subtype)
        rect_strs = rect_str.strip().split(' ')
        self.rect = Rect(rect_strs[0], rect_strs[1], rect_strs[2], rect_strs[3],
        frame_shape[1], frame_shape[0])

    def print_obj(self):
        # print("-------/n")

        print("Type: ", self.type)
        print("Sub Type: ", self.subtype)
        print("X: ", self.rect.x)
        print("Y: ", self.rect.y)
        print("W: ", self.rect.w)
        print("H: ", self.rect.h)

        print("Relative X: ", self.rect.relative_x)
        print("Relative Y: ", self.rect.relative_y)
        print("Relative W: ", self.rect.relative_w)
        print("Relative H: ", self.rect.relative_h)

        # print("/n-------")

    def isVerLineIntersectObject(self, x):
        obj_x1 = self.rect.x
        obj_x2 = self.rect.x + self.rect.w

        return (x > obj_x1 and x < obj_x2)

    def isHorLineIntersectObject(self, y):
        obj_y1 = self.rect.y
        obj_y2 = self.rect.y + self.rect.h

        return (y > obj_y1 and y < obj_y2)

def getFrameFromVideo(video_filename, frame_number):
    cap = cv2.VideoCapture()

    cap.open(video_filename)
    if not (cap.isOpened()):
        print('Cant open file ', video_filename)
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    return frame

def getVideoLength(video_filename):
    cap = cv2.VideoCapture()
    print(video_filename)
    print(cap.open(video_filename))
    if not (cap.isOpened()):
        print('Can\'t open file ', video_filename)
        return 0
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)

def getObjectsAndFrame(xml_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    cap = cv2.VideoCapture()

    possible_video_filename = xml_filename[: xml_filename.rfind('/')]

    files = GetAllFilesList(possible_video_filename[: possible_video_filename.rfind('/')])
    # If possible to find correlated video file - then read frame from it
    video_filenames = list([file for file in files if isVideofile(file) and possible_video_filename in file])[0]
    video_filename = video_filenames if len(video_filenames) else None

    if not video_filename == None:
        frame_number = xml_filename[xml_filename.rfind('/'): ]
        frame_number = frame_number[frame_number.find('f') + 1: frame_number.find('_o')]
        try:
            frame_number = int(frame_number)
        except:
            return None, objects

        print('opening filename: ', video_filename)
        cap.open(video_filename)
        if not (cap.isOpened()):
            print('Can\'t open file', video_filename)
            frame = None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
    else:
        if os.path.exists(xml_filename.replace('_o.xml', '.png')):
            #print('Reading: ', xml_filename.replace('_o.xml', '.png'))
            frame = cv2.imread(xml_filename.replace('_o.xml', '.png'))
        else:
            #print(xml_filename.replace('_o.xml', '.png'), ' not exists')
            frame = None

    objects = []

    for obj in root.findall('Objects')[0].findall('_'):
        objects.append(Object(obj.find('type').text,
        obj.find('subtype').text,
        obj.find('rect').text, (frame.shape if not frame == None else [1, 1, 1])))
        # objects[len(objects) - 1].print_obj()

    return frame, objects

def GetLastFrameNumberInDir(dir):
    files = GetAllFilesList(dir)
    stripped_filenames = []

    if len(files) == 0:
        return 0

    for file in files:
        if isImagefile(file):
            stripped_filename = file[file.rfind('/') + 1: file.rfind('.png')].strip()
            stripped_filenames.append(int(stripped_filename))

    return sorted(stripped_filenames)[len(stripped_filenames) - 1]


if __name__ == "__main__":
    output_dir = "/home/wildchlamydia/CarsVideo/train_sliced_subframes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #filename = '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160902/2016_09_02_09_16_35/f00003376_o.xml'
    dirs = ['/home/wildchlamydia/CarsVideo/LV8817/Cuba','/home/wildchlamydia/CarsVideo/LV8817/Germany/20160405/2016_04_05_12_50_54',
            '/home/wildchlamydia/CarsVideo/LV8817/Germany/20160405/2016_04_05_13_32_24',
            '/home/wildchlamydia/CarsVideo/LV8817/Germany/20160405/2016_04_05_13_47_59',
            '/home/wildchlamydia/CarsVideo/LV8817/Germany/20160405/2016_04_05_13_58_23', "/home/wildchlamydia/CarsVideo/LV8817/SPb/20160821",
            "/home/wildchlamydia/CarsVideo/LV8817/SPb/20160902", '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160904/20160905/2016_09_04_14_09_40',
            '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160904/20160905/2016_09_05_07_53_02', '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160910/2016_09_10_15_34_51']
    # dirs = ['/home/wildchlamydia/CarsVideo/LV8817/SPb/20160904/20160905/2016_09_04_13_28_01', '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160910/2016_09_10_10_56_31',
    # '/home/wildchlamydia/CarsVideo/LV8817/SPb/20160823/2016_08_22_17_26_15', '/home/wildchlamydia/CarsVideo/LV8817/Germany/20160405/2016_04_05_14_20_53']

    for dir in dirs:
        for filename in GetAllFilesListRecusive(dir, ['_o.xml']):
            if '_o.xml' in filename:
                print('Processing file ', filename)
                frame_name = filename[filename.rfind('/'): filename.rfind('.xml')]
                frame, objects = getObjectsAndFrame(filename)
                rois, objs, _, _ = SliceFrame(frame, objects, 0.07, 120, 0.8)

                if not output_dir:
                    new_path = filename[: filename.rfind('/')] + '_sliced/'
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                        print(new_path)
                if output_dir:
                    last_index = GetLastFrameNumberInDir(output_dir)
                for i in xrange(len(rois)):
                    if len(objs[i]) > 0:
                        if not output_dir:
                            new_img_filename = new_path + frame_name +'_slice_' + str(i) +'.png'
                            new_xml_filename = new_path + frame_name +'_slice_' + str(i) +'.txt'
                        else:
                            new_img_filename = output_dir + '/' + str(last_index) + '.png'
                            new_xml_filename = output_dir + '/' + str(last_index) + '.txt'
                            last_index += 1
                        cv2.imwrite(new_img_filename, rois[i])
                        data_text = ''
                        for obj in objs[i]:
                            data_text = data_text + str(obj.type - 1) + ' ' + str(obj.rect.relative_x) + ' ' + \
                            str(obj.rect.relative_y) + ' ' + str(obj.rect.relative_w) + ' ' + str(obj.rect.relative_h) + '\n'
                        file = open(new_xml_filename, 'w')
                        # print(new_img_filename, data_text)
                        file.write(data_text)
                        file.close()
