#! /usr/bin/env python

from net.build import TFNet
from tensorflow import flags
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string("test", "./test/", "path to testing directory")
flags.DEFINE_boolean("slice_frame", False, "Test by slicing big frame on smaller, because YOLO can't see small objects")
flags.DEFINE_boolean("from_videos", False, "Predict from video (that will be found in test dir) frames with video_step")
flags.DEFINE_integer("video_step", 120, "Step for next frames in video")
flags.DEFINE_float("slice_step", 0.07, "Step of slicing window")
flags.DEFINE_float("slice_max_iou", 0.1, "Max IOU above which rectangles will be considered as conflicting and will be choosed most probable one")
flags.DEFINE_integer("slice_win_size", 100, "Size of window that will slice frame in pixels")
flags.DEFINE_boolean("save_xml", False, "Saves objects in custom XML file")
flags.DEFINE_boolean("save_image", False, "Saves image with painted objects")
flags.DEFINE_float("slice_down_rate", 0.8, "Downsample rate for frame on each iteration ([0.0 - 1.0], new size will be win_size * rate)")
flags.DEFINE_string("binary", "./bin/", "path to .weights directory")
flags.DEFINE_string("config", "./cfg/", "path to .cfg directory")
flags.DEFINE_string("backup", "./ckpt/", "path to backup folder")
flags.DEFINE_string("darkflow_dataset", "", "path to darkflow annotation directory (VOC style img-asnwer, where answer VOC xml)")
flags.DEFINE_string("darknet_dataset", "", "path to darknet dataset directory or .txt file (Darknet style img-asnwer, where answer is TXT with relative positions)")
flags.DEFINE_string("navmii_dataset", "", "path to navmii dataset directory or .txt file (Navmii style img-asnwer, where answer is XML with objects array inside tags <_>)")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "", "configuration of choice")
flags.DEFINE_string("trainer", "rmsprop", "training algorithm")
flags.DEFINE_float("momentum", 0.0, "applicable for rmsprop and momentum optimizers")
flags.DEFINE_boolean("verbalise", True, "say out loud while building graph")
flags.DEFINE_boolean("train", False, "train the whole net")
flags.DEFINE_string("load", "", "how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "how much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "learning rate")
flags.DEFINE_integer("keep",20,"Number of most recent training results to save")
flags.DEFINE_integer("batch", 16, "batch size")
flags.DEFINE_integer("epoch", 1000, "number of epoch")
flags.DEFINE_integer("save", 2000, "save checkpoint every ? training examples")
flags.DEFINE_string("demo", '', "demo on webcam")
flags.DEFINE_boolean("profile", False, "profile")
FLAGS = flags.FLAGS

# make sure all necessary dirs exist
def get_dir(dirs):
	for d in dirs:
		this = os.path.abspath(os.path.join(os.path.curdir, d))
		if not os.path.exists(this): os.makedirs(this)
get_dir([FLAGS.binary, FLAGS.backup])

# fix FLAGS.load to appropriate type
try: FLAGS.load = int(FLAGS.load)
except: pass

tfnet = TFNet(FLAGS)

if FLAGS.profile:
	tfnet.framework.profile(tfnet)
	exit()

if FLAGS.demo:
	tfnet.camera(FLAGS.demo)
	exit()

if FLAGS.train:
	print('Enter training ...'); tfnet.train()
	if not FLAGS.savepb: exit('Training finished')

if FLAGS.savepb:
	print('Rebuild a constant version ...')
	tfnet.savepb(); exit('Done')

if (FLAGS.from_videos and FLAGS.slice_frame):
	tfnet.predictTestVideosWithSlicing()
elif (FLAGS.slice_frame):
	tfnet.predictTestImagesWithSlicing()
elif FLAGS.from_videos:
	tfnet.predictTestVideos()
else:
	tfnet.predict()
