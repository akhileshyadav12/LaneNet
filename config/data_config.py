#!/usr/bin/env python3
"""
Set global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import data_cfg

data_cfg = __C

# TRAIN options
__C.DATA = edict()
__C.DATA.BASE="tusimple_dataset"
__C.DATA.TRAIN="train_set"
__C.DATA.TEST="test_set"
__C.DATA.TRAIN_JSON=["label_data_0313.json","label_data_0531.json","label_data_0601.json"]
__C.DATA.OUTPUT="PROCESSED"

