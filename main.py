from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- System Libs --- #
import argparse
import os
import sys
import random
from typing import Iterable

# --- 3rd Packages -- #
# import tensorflow as tf
# if tf.__version__.startswith('1.'):
#     import tensorflow as tf1
# else:  # tf v2.x
#     import tensorflow.compat.v1 as tf1
#     tf1.disable_v2_behavior()

# --- Custom Packages --- #
from config import *
from helpers.util import *
import modules


# --- Main --- #
def main(*args, **kwargs):
    INFO('--- Experiment begins: {}---------------'.format(Config.ExperimentName))
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)

    from helpers.tf_helper import async_preload_gpu_devices
    async_preload_gpu_devices()

    # TODO: explain terms or key concepts in comments, ref: overlook.vsd
    from modules.data.data_manager import DataManager
    x_train, y_train = None, None
    x_test, y_test = None, None

    from modules.models.model_manager import ModelManager
    model = None

    if config_experiment.train.enabled:
        INFO('--- Training begins ---------')
    else:
        INFO('--- Training was disabled ---------')

    if config_experiment.predict.enabled:
        INFO('--- Prediction begins ---------')

        if model is None:  # not config_experiment.train.enabled
            config_model: Params = config_experiment.model_set.model_trained
            if config_model.is_blank():
                raise ValueError('Config error: `model_trained` node is not defined')
            model = ModelManager.load_model(config_model.signature, **config_model)

        predictions = ModelManager.model_predict(model, x_test, y_test, **config_experiment.predict)

        INFO(f"predictions: {', '.join([str(_) for _ in predictions])}")
    else:
        INFO('--- Prediction was disabled ---------')

    INFO('--- Experiment ends: {}---------------'.format(Config.ExperimentName))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default=Config.ExperimentName,
        help=f"relative path of the target experiment, e.g.'{Config.ExperimentName}'"
    )
    FLAGS, _undefined_ = parser.parse_known_args()

    Config.ExperimentName = FLAGS.__dict__.pop('experiment')

    main(FLAGS)

