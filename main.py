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
        config_data: Params = config_experiment.data_set.data
        data = DataManager.load_data(config_data.signature, **config_data)
        INFO(f"data loaded: {dump_iterable_data(data)}")
        if not isinstance(data, tuple) or any(not isinstance(_, tuple) for _ in data):
            raise ValueError("data loaded must be in type of ((x_train, y_train), (x_test, y_test))")
        (x_train, y_train), (x_test, y_test) = data  # unpack: tuple of 4 np.ndarrays
        # e.g.((60000,28,28), (60000,)), ((10000,28,28), (10000,))

        config_model: Params = config_experiment.model_set.model_base
        model = ModelManager.load_model(config_model.signature, **config_model)
        model = ModelManager.model_train(model, data=(x_train, y_train), **config_experiment.train)
        eval_metrics = ModelManager.model_evaluate(model, data=(x_test, y_test))

    else:
        INFO('--- Training was disabled ---------')

    if config_experiment.predict.enabled:
        INFO('--- Prediction begins ---------')
        if 'meta_info' not in vars():
            meta_info = {}  # retrieve meta info from DataManager
        if x_test is None or y_test is None:  # not config_experiment.train.enabled
            data_key = config_experiment.predict.data_inputs.__str__()
            config_data_test: Params = config_experiment.data_set[data_key]
            # test signature "ui_web_files", need to keep compatibility with other type of data
            data = DataManager.load_data(config_data_test.signature,
                                         meta_info=meta_info, **config_data_test)
            INFO(f"data loaded: {dump_iterable_data(data)}")
            import tensorflow as tf
            from helpers.tf_helper import is_tfdataset
            if isinstance(data, tf.data.Dataset):
                if type(data.element_spec) is tuple:
                    # x_test = data.map(lambda x, y: x)
                    # y_test = data.map(lambda x, y: y)
                    from helpers.tf_helper import tf_obj_to_np_array
                    # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API for unzip?
                    x_test = tf_obj_to_np_array(data.map(lambda x, y: x))
                    y_test = tf_obj_to_np_array(data.map(lambda x, y: y))
                else:
                    data = data.batch(1)  # TODO: read config `batch_size` in model_train()
                    data = data.prefetch(1)
                    x_test, y_test = data, None

        if model is None:  # not config_experiment.train.enabled
            config_model: Params = config_experiment.model_set.model_trained
            if config_model.is_defined():
                raise ValueError('Config error: `model_trained` node is not defined')
            model = ModelManager.load_model(config_model.signature, **config_model)

        predictions = ModelManager.model_predict(model, (x_test, y_test), **config_experiment.predict)
        if not isinstance(predictions, Iterable):
            predictions = [predictions]

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

