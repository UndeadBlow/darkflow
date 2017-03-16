import os
import time
import numpy as np
import tensorflow as tf
import pickle
import cv2
import utils.box as BoundBox
import utils.serialize as serial

import utils.slicer as slicer

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)

def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    ckpt = 0
    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key]
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)


def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all

# Returns True if all is processed and False otherwise.
# Use it in while loop
def processBoxes(self, boxes, max_iou = 0.25):
    self.say('processBoxes, got {} boxes'.format(len(boxes)))

    for box_one in boxes:
        for box_two in boxes:
            # Classes must be different, otherwise it can be two intersected correct boxes
            if box_one != box_two and box_one.class_num == box_two.class_num:
                iou = BoundBox.box_iou(box_one, box_two)
                if (iou > max_iou):
                    # Two boxes conflict. Which one will survive?
                    max_indx_one = np.argmax(box_one.probs)
                    max_prob_one = box_one.probs[max_indx_one]

                    max_indx_two = np.argmax(box_two.probs)
                    max_prob_two = box_two.probs[max_indx_two]

                    if box_one.w + box_one.h > box_two.w + box_two.h:
                        boxes.remove(box_two)
                    else:
                        boxes.remove(box_one)

                    return False

            if (box_one.x > box_two.x) and\
             (box_one.x + box_one.w < box_two.x + box_two.w) and\
              (box_one.y > box_two.y) and\
               (box_one.y + box_one.h < box_two.y + box_two.h):
                # One box inside other while them both the same class? It's definitely not ok
                boxes.remove(box_one)
                return False

            if (box_two.x > box_one.x) and\
             (box_two.x + box_two.w < box_one.x + box_one.w) and\
              (box_two.y > box_one.y) and\
               (box_two.y + box_two.h < box_one.y + box_one.h):
                # One box inside other while them both the same class? It's definitely not ok
                boxes.remove(box_two)
                return False

    return True

def saveFrameToXML(boxes, classes, xml_filename):
    xml = serial.points_to_xml(boxes, classes)
    if xml == '':
        return
    file = open(xml_filename, 'w')
    file.write(xml)

def convBoxesCoordsToAbsolute(subframe_boxes, threshold, orig_frame_shape, frame_shape, subframe_shape, position):
    boxes = []
    # Boxes has relative x, y, w and h and subframes was taken
    # from picture of different sizes. We need to cast that all
    for box in subframe_boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        # label = 'object' * int(C < 2)
        # label += labels[max_indx] * int(C>1)

        if max_prob <= threshold:
            # Probability below treshold
            continue

        # Original frame shape - it's shape of original big frame
        # Frame shape - it's shape of resized frame during slicing
        # Subframe shape - it's shape of sliced from Frame subframe
        # So we have Original Frame > Frames > Subframes
        # Position is Subframe position on Frame
        orig_h = float(orig_frame_shape[0])
        orig_w = float(orig_frame_shape[1])
        scale_h = orig_h / float(frame_shape[0])
        scale_w = orig_w / float(frame_shape[1])

        # self.say("Size {}".format(size))
        # self.say("Scale W {}, scale H {}".format(scale_w, scale_h))

        # YOLO returns center coords, surprise
        box.x = box.x - (box.w / 2.)
        box.y = box.y - (box.h / 2.)

        #### FROM RELATIVE TO ABSOLUTE SUBFRAME COORDS
        box.x = subframe_shape[1] * box.x
        box.y = subframe_shape[0] * box.y

        box.w = subframe_shape[1] * box.w
        box.h = subframe_shape[0] * box.h
        # self.say('Box absolute subframe x {} y {} w {} h {}'.format(box.x, box.y, box.w, box.h))

        #### FROM ABSOLUTE SUBFRAME TO ABSOLUTE FRAME
        box.x = box.x + position[1]
        box.y = box.y + position[0]

        #### SCALE UP IF FRAME WAS RESIZED DURING SLICE
        box.x = int(box.x * scale_w)
        box.y = int(box.y * scale_h)

        box.w = int(box.w * scale_w)
        box.h = int(box.h * scale_h)

        # self.say('Box absolute frame x {} y {} w {} h {}'.format(box.x, box.y, box.w, box.h))
        # self.say('\n--------------\n')
        boxes.append(box)

    return boxes

def predictTestVideos(self):
    inp_path = self.FLAGS.test
    step = self.FLAGS.video_step
    files = GetAllFilesListRecusive(inp_path, ['.mp4', '.avi', '.mpeg'])
    for filename in files:
        filenames = []
        out_names = []
        position = 1
        if slicer.isVideofile(filename):
            self.say('Processing video {}'.format(filename))
            duration_frames = int(slicer.getVideoLength(filename))
            self.say('Duration in frames {}'.format(duration_frames))

            while position < duration_frames:
                filenames.append(filename + ':' + str(position))
                framename_path = filename

                for ext in [".mp4", ".avi", ".mov", ".mpeg"]:
                    if ext in filename:
                        framename_path = filename.replace(ext, '')

                if not os.path.exists(framename_path): os.makedirs(framename_path)
                fr_num = "%08d" % int(position)
                frame_filename = framename_path + "/f{}.png".format(fr_num)
                out_names.append(frame_filename)
                position += step

        self.predictList(filenames, out_names)

def predictTestVideosWithSlicing(self):
    inp_path = self.FLAGS.test
    step = self.FLAGS.video_step
    files = GetAllFilesListRecusive(inp_path, ['.mp4', '.avi', '.mpeg'])
    for filename in files:
        position = 1
        if slicer.isVideofile(filename):
            self.say('Processing video {}'.format(filename))
            duration_frames = int(slicer.getVideoLength(filename))
            self.say('Duration in frames {}'.format(duration_frames))
            while position < duration_frames:
                frame = slicer.getFrameFromVideo(filename, position)
                framename_path = filename

                for ext in [".mp4", ".avi", ".mov"]:
                    if ext in filename:
                        framename_path = filename.replace(ext, '')

                if not os.path.exists(framename_path): os.makedirs(framename_path)
                fr_num = "%08d" % int(position)
                frame_filename = framename_path + "/f{}.png".format(fr_num)
                self.processFrameBySlicing(frame, frame_filename, filename + ':' + fr_num)
                position += step

def predictTestImagesWithSlicing(self):

    inp_path = self.FLAGS.test
    imgs = GetAllFilesListRecusive(inp_path, ['.png', '.jpg', '.jpeg'])
    if not imgs:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    # For every image
    for frame_filename in imgs:
        frame = cv2.imread(frame_filename)
        self.processFrameBySlicing(frame, frame_filename)

def processFrameBySlicing(self, frame, frame_filename, video_filename = ''):
    meta = self.meta
    threshold = self.FLAGS.threshold
    classes = meta['labels']
    inp_path = self.FLAGS.test

    boxes = []
    self.say('Processing {} frame'.format(frame_filename))
    subframes, _, positions, sizes = slicer.SliceFrame(frame, [],
    self.FLAGS.slice_step, self.FLAGS.slice_win_size,
    self.FLAGS.slice_down_rate)
    self.say('Got {} subframes'.format(len(subframes)))

    # For every subframe of image
    for i in range(len(subframes)):
        subframe = subframes[i]
        position = positions[i]
        size = sizes[i]

        input = self.framework.preprocess(subframe)
        expanded = np.expand_dims(input, 0)
        feed_dict = {self.inp : expanded}

        # Forward network
        #self.say('Forwarding subframe on position {}'.format(position))
        prediction = self.sess.run(self.out, feed_dict)

        # Process output
        subframe_boxes = self.framework.getProcessedBoxes(prediction)

        # Convert coordinates to absolute coords of original frame
        # Dont include boxes below treshold
        boxes += convBoxesCoordsToAbsolute(subframe_boxes, threshold,
        frame.shape, size, subframe.shape, position)

    # Process to remove conflicted and bad boxes
    while(not self.processBoxes(boxes, self.FLAGS.slice_max_iou)): continue

    # Save results
    if self.FLAGS.save_xml:
        xml_filename = frame_filename.replace('.png', '_o.xml')
        saveFrameToXML(boxes, classes, xml_filename)
    if self.FLAGS.save_image:
        name = os.path.join(inp_path, frame_filename)
        if video_filename != '': # For videos because we need to know frame number
            name = video_filename
        self.framework.drawAndSaveResults(boxes, name,
        raw_yolo_coords = False, out_name = frame_filename)

def predictList(self, images_names, output_names = []):
    inp_path = self.FLAGS.test
    batch = min(self.FLAGS.batch, len(images_names))
    classes = self.meta['labels']

    index_for_names = 0
    for j in range(len(images_names) // batch):
        inp_feed = list(); new_all = list()
        all_inp = images_names[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}

        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            boxes = self.framework.getProcessedBoxes(prediction)

            if not output_names:
                if self.FLAGS.save_xml:
                    xml_filename = os.path.join(inp_path, all_inp[i]).replace('.png', '_o.xml')
                    saveFrameToXML(boxes, classes, xml_filename)
                self.framework.drawAndSaveResults(boxes, os.path.join(inp_path, all_inp[i]))
            else:
                if self.FLAGS.save_xml:
                    xml_filename = output_names[i].replace('.png', '_o.xml')
                    saveFrameToXML(boxes, classes, xml_filename)
                self.framework.drawAndSaveResults(boxes, im = all_inp[i], out_name = output_names[index_for_names])
                index_for_names += 1

        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

def predict(self):
    inp_path = self.FLAGS.test
    images_names = GetAllFilesListRecusive(inp_path, ['.png', '.jpg', '.jpeg'])
    if not images_names:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    self.predictList(images_names)
