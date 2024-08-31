# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = [0]
_C.MODEL.META_ARCHITECTURE = "MaskRCNN3D"
_C.MODEL.BACKBONE = "unet"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the input patch, in the former of (x, y, z)
_C.INPUT.PATCH_SIZE = (128, 128, 32)

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Folder of the dataset for training, arranged in the format of simulated dataset
_C.DATASETS.TRAIN = ""
# Folder of the dataset for training, arranged in the format of simulated dataset
_C.DATASETS.TRAIN_SUFFIX = ""
# Folder of the dataset for finetuning, arranged in the format of simulated dataset
_C.DATASETS.FINETUNE = ""
# Folder of the dataset for finetuning, arranged in the format of simulated dataset
_C.DATASETS.FINETUNE_SUFFIX = ""
# Path of the data for validation
_C.DATASETS.VALID = ""
# Path of the data for testing
_C.DATASETS.TEST = ""
# Path for saving temp training data
_C.DATASETS.TRAIN_SAVE = ""
# Path for saving temp validation data
_C.DATASETS.VALID_SAVE = ""
# Path for saving temp testing data
_C.DATASETS.TEST_SAVE = ""
# Whether to use valitation data or split from training data
_C.DATASETS.IS_VALID = True
# Whether to use data augmentation in validation
_C.DATASETS.VALID_AUG = False
# Split ratio for train and validation data
_C.DATASETS.SPLIT = 0.2
# Whether the validation or testing data has label or not 
_C.DATASETS.IS_LABEL = False
# Whether to reverse training image
_C.DATASETS.REVERSE = False
# Crop area for validation dataset
_C.DATASETS.VAL_CROP = None

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 16
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# -----------------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()

# for precrop transform
_C.TRANSFORM.PRECROP_SIZE = (148, 148, 48)

# for spatial transform
_C.TRANSFORM.RANDOM_CROP = True

_C.TRANSFORM.DO_ROTATION = True
_C.TRANSFORM.P_ROTATION = 0.2
_C.TRANSFORM.ROTATION_X = (0.0, 0.0)
_C.TRANSFORM.ROTATION_Y = (0.0, 0.0)
_C.TRANSFORM.ROTATION_Z = (0.0, 2*3.1416)

_C.TRANSFORM.DO_ELASTIC = True
_C.TRANSFORM.P_ELASTIC = True
_C.TRANSFORM.ELASTIC_ALPHA = (0., 1500.)
_C.TRANSFORM.ELASTIC_SIGMA = (30., 50.)

_C.TRANSFORM.DO_SCALE = True
_C.TRANSFORM.P_SCALE = 0.2
_C.TRANSFORM.SCALE = (0.8, 1.1)

_C.TRANSFORM.IMG_BORDER_MODE = 'nearest'
_C.TRANSFORM.IMG_BORDER_CVAL = 0
_C.TRANSFORM.IMG_BORDER_ORDER = 3
_C.TRANSFORM.SEG_BORDER_MODE = 'constant'
_C.TRANSFORM.SEG_BORDER_CVAL = 0
_C.TRANSFORM.SEG_BORDER_ORDER = 0

# for gaussion noise transform
_C.TRANSFORM.NOISE_VAR = (0.0,0.1)
_C.TRANSFORM.P_NOISE = 0.1

# for brightness transform
_C.TRANSFORM.MULTIPLIER_RANGE = (0.4, 1.2)
_C.TRANSFORM.P_BRIGHTNES = 0.5

# for contrast augmentation transform
_C.TRANSFORM.CONTRAST_RANGE = (0.4, 1.2)
_C.TRANSFORM.PRESERVE_RANGE = True
_C.TRANSFORM.P_CONTRAST = 0.5

# for low resolution transform
_C.TRANSFORM.ZOOM_RANGE = (0.5, 1)
_C.TRANSFORM.ORDER_UPSAMPLE = 3
_C.TRANSFORM.ORDER_DOWNSAMPLE = 0
_C.TRANSFORM.P_LOWRESOLUTION = 0.2

# for random erase transform
_C.TRANSFORM.AREA_RANGE = (4, 4, 2, 8, 8, 4)
_C.TRANSFORM.P_ERASE = 0.0


# for finetune
# _C.TRANSFORM.NOISE_VAR = (0.5,2.0)
# _C.TRANSFORM.P_NOISE = 0.5

# _C.TRANSFORM.MULTIPLIER_RANGE = (0.4, 1.6)
# _C.TRANSFORM.P_BRIGHTNES = 0.5

# _C.TRANSFORM.CONTRAST_RANGE = (0.4, 1.6)
# _C.TRANSFORM.PRESERVE_RANGE = True
# _C.TRANSFORM.P_CONTRAST = 0.5

# _C.TRANSFORM.ZOOM_RANGE = (0.2, 0.6)
# _C.TRANSFORM.ORDER_UPSAMPLE = 3
# _C.TRANSFORM.ORDER_DOWNSAMPLE = 0
# _C.TRANSFORM.P_LOWRESOLUTION = 0.5

# _C.TRANSFORM.AREA_RANGE = (24, 24, 8, 48, 48, 16)
# _C.TRANSFORM.P_ERASE = 0.5

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.OUTPUT = CN()
_C.MODEL.OUTPUT.SEG_CLASS = 2 

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Normalization strategy, one of None, "batch_norm" or "instance_norm" 
_C.MODEL.FPN.USE_NORM = None
# Activation strategy, one of None, "relu" or "leakyrelu"
_C.MODEL.FPN.USE_RELU = "relu"
    # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
_C.MODEL.FPN.PYRAMID_LEVELS = [0, 1, 2, 3]
# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
# _C.MODEL.RPN = CN()

# _C.MODEL.RPN.USE_NORM = None
# _C.MODEL.RPN.USE_RELU = "relu"
# # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input, in the order of x, y, z
# # _C.MODEL.RPN.ANCHOR_SCALES = [[8, 16, 32, 64], [8, 16, 32, 64], [2, 4, 8, 16]]
# _C.MODEL.RPN.ANCHOR_SCALES = [[8, 16, 32, 64], [8, 16, 32, 64], [8, 16, 32, 64]]
# # feature map strides per pyramid level are inferred from architecture.
# _C.MODEL.RPN.FEATURE_STRIDES  = [[4, 8, 16, 32], [4, 8, 16, 32], [1, 2, 4, 8]]
# # For FPN, number of strides should match number of scales
# _C.MODEL.RPN.ANCHOR_STRIDE = 1
# # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
# _C.MODEL.RPN.N_FEATURES = 128
# # RPN anchor aspect ratios
# _C.MODEL.RPN.ASPECT_RATIOS = [0.5, 1.0, 2.0]
# # _C.MODEL.RPN.ASPECT_RATIOS = [1.0]
# # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
# _C.MODEL.RPN.STRADDLE_THRESH = 0
# # Minimum overlap required between an anchor and ground-truth box for the
# # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# # ==> positive RPN example)
# _C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# # Maximum overlap allowed between an anchor and ground-truth box for the
# # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# # ==> negative RPN example)
# _C.MODEL.RPN.BG_IOU_THRESHOLD = 0.01
# # Total number of RPN examples per image
# _C.MODEL.RPN.TRAIN_ANCHRO_PER_IMAGE = 1000
# _C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 8
# # Target fraction of foreground (positive) examples per RPN minibatch
# _C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# # Number of top scoring RPN proposals to keep before applying NMS
# # When FPN is used, this is *per FPN level* (not total)
# _C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 6000
# _C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 3000
# # Number of top scoring RPN proposals to keep after applying NMS
# _C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 200
# _C.MODEL.RPN.POST_NMS_TOP_N_TEST = 500
# # NMS threshold used on RPN proposals
# _C.MODEL.RPN.NMS_THRESH = 0.7
# # Proposal height and width both need to be greater than RPN_MIN_SIZE
# # (a the scale used during training or inference)
# _C.MODEL.RPN.MIN_SIZE = 1
# # Number of top scoring RPN proposals to keep after combining proposals from
# # all FPN levels
# _C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 200
# _C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 500

# _C.MODEL.RPN.SHEM_POOLSIZE = 10

# _C.MODEL.RPN.RPN_BOX_STD_DEV = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2)


# # ---------------------------------------------------------------------------- #
# # ROI HEADS options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_HEADS = CN()
# # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
# _C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.3
# # Overlap threshold for an RoI to be considered background
# # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
# _C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.01
# # RoI minibatch size *per image* (number of regions of interest [ROIs])
# # Total number of RoIs per training minibatch =
# #   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# _C.MODEL.ROI_HEADS.DETECTION_NMS_SCORE = 0.1
# _C.MODEL.ROI_HEADS.MAX_BATCH_INSTANCE = 100
# # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
# _C.MODEL.ROI_HEADS.ROI_POSITIVE_RATIO = 0.5

# _C.MODEL.ROI_HEADS.ROI_BOX_STD_DEV = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2)
# _C.MODEL.ROI_HEADS.TRAIN_BOX_PER_IMAGE = 200


# _C.MODEL.ROI_HEADS.MIN_SIZE = 1
# _C.MODEL.ROI_HEADS.MIN_CONFIDENCE = 0.1

# _C.MODEL.ROI_HEADS.SHEM_POOLSIZE = 10

# # Only used on test mode

# # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# # balance obtaining high recall with not having too many low precision
# # detections that will slow down inference post processing steps (like NMS)
# _C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# # Overlap threshold used for non-maximum suppression (suppress boxes with
# # IoU >= this threshold)
# _C.MODEL.ROI_HEADS.NMS = 0.5
# # Maximum number of detections to return per image (100 is based on the limit
# # established for the COCO dataset)
# _C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 500


# _C.MODEL.ROI_BOX_HEAD = CN()
# _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = (7, 7, 3)
# _C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
# # Hidden layer dimension when using an MLP for the RoI box head
# _C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# _C.MODEL.ROI_BOX_HEAD.ROI_BATCH = 200
# # GN
# _C.MODEL.ROI_BOX_HEAD.USE_NORM = None
# _C.MODEL.ROI_BOX_HEAD.USE_RELU = "relu"
# # Dilation
# _C.MODEL.ROI_BOX_HEAD.DILATION = 1
# _C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
# _C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


# _C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.IN_CHANNELS = 18
_C.MODEL.RESNETS.OUT_CHANNELS = 36
# for probabilistic detection
_C.MODEL.RESNETS.N_LATENT_DIMS = 0
_C.MODEL.RESNETS.ARCHITECTURE = "resnet50"
_C.MODEL.RESNETS.BLOCK_EXPANSION = 4
_C.MODEL.RESNETS.SIXTH_POOLING = False


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.ITER_PER_EPOCH = 100
_C.SOLVER.MAX_EPOCH = 120

# _C.SOLVER.BASE_LR = [3e-4, 1e-4, 1e-5, 1e-6]
_C.SOLVER.BASE_LR = [3e-5, 1e-5, 5e-6, 1e-6]
_C.SOLVER.BIAS_LR_FACTOR = 1


_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.CHECKPOINT_PERIOD = 200
_C.SOLVER.TEST_PERIOD = 10

_C.SOLVER.FINETUNE_ITER = 8
_C.SOLVER.FINETUNE_THRESHOLD = 0.3
_C.SOLVER.FINETUNE_LOWPASS = 0.1
_C.SOLVER.FINETUNE_CUTOFF = 400
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100
_C.TEST.MIN_OVERLAP = [64, 64, 16]




# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.OUTPUT_TEST_DIR = ""
_C.OUTPUT_VAL_DIR = ""

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
