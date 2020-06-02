from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- System Libs --- #
import argparse
import os
import sys
import random

# --- 3rd Packages -- #
# import tensorflow as tf
# if tf.__version__.startswith('1.'):
#     import tensorflow as tf1
# else:  # tf v2.x
#     import tensorflow.compat.v1 as tf1
#     tf1.disable_v2_behavior()

# --- Custom Packages --- #
from modules import *
from helpers import *

# --- Custom Models --- #
from config import *


# --- Main --- #
def main(FLAGS):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default=Config.ExperimentName,
        help="relative path of the target experiment"
    )
    FLAGS, _undefined_ = parser.parse_known_args()

    Config.ExperimentName = FLAGS.__dict__.pop('experiment')

    main(FLAGS)

