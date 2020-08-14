from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- System Libs --- #
import argparse
import os.path as osp
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

# --- Main --- #
def main(*args, **kwargs):
    INFO('--- Experiment begins: {}---------------'.format(Config.ExperimentName))
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)

    init_logging(VerbosityLevel.DEBUG)

    from helpers.tf_helper import async_preload_gpu_devices
    async_preload_gpu_devices()

    # TODO: explain terms or key concepts in comments, ref: overlook.vsd
    from modules.data.data_manager import DataManager
    x_train, y_train = None, None
    x_test, y_test = None, None
    from modules.models.model_manager import ModelManager
    model = None

    # NOTE: for `style_transfer/vgg(tensorlayer)+vgg_rev(adain)`, only demo predict, train is disabled.
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
        config_data_test = vars().get('config_data', None)
        enc, dec = None, None
        predictions = None
        while True:
            meta_info = vars().get('meta_info', {})  # retrieve meta info from DataManager
            if x_test is None or y_test is None:  # not config_experiment.train.enabled
                data_key = config_experiment.predict.data_inputs.__str__()
                config_data_test: Params = config_experiment.data_set[data_key]
                # test signature "ui_web_files", need to keep compatibility with other type of data
                # NOTE: when use `ui_web_files`(d&d), the first file selected will be taken as 1st data item,
                #   whereas when `ui_copy_files`(clipboard), the *second file selected* will be the 1st data item.
                data = DataManager.load_data(config_data_test.signature,
                                             meta_info=meta_info, **config_data_test)
                INFO(f"data loaded: {dump_iterable_data(data)}")
                import tensorflow as tf
                from helpers.tf_helper import is_tfdataset
                if isinstance(data, tf.data.Dataset):
                    from helpers.tf_helper import tf_obj_to_np_array
                    if type(data.element_spec) is tuple:
                        # IMPROVE: any efficient unzip API for ZipDataset?
                        # x_test = data.map(lambda x, y: x)
                        # y_test = data.map(lambda x, y: y)
                        x_test = tf_obj_to_np_array(data.map(lambda x, y: x))
                        y_test = tf_obj_to_np_array(data.map(lambda x, y: y))
                    else:
                        # x_test, y_test = tf_obj_to_np_array(data), None
                        x_test, y_test = [data.take(1), data.skip(1).take(1)], None

            if enc is None or dec is None:  # not config_experiment.train.enabled
                config_model_enc: Params = config_experiment.model_set.model_enc
                config_model_dec: Params = config_experiment.model_set.model_dec
                if not (config_model_enc.is_defined() and config_model_dec.is_defined()):
                    raise ValueError('Config error: `model_enc` or `model_dec` node is not defined')
                enc = ModelManager.load_model(config_model_enc.signature, **config_model_enc)
                dec = ModelManager.load_model(config_model_dec.signature, **config_model_dec)

            input_handled = False
            output_saved_path = ""

            if config_data_test.signature == "ui_web_files":
                webapp = ensure_web_app()
                @webapp.on_task_query(namespace="main::model_predict", onetime=False)
                def handle_task_query(task_id):
                    nonlocal input_handled, output_saved_path
                    handler_result = {}
                    if not input_handled:
                        handler_result.update({'status': 'processing'})
                    else:
                        # return abspath to webapp
                        handler_result.update({'status': 'finished', 'result': output_saved_path})
                        input_handled = False
                    return handler_result

            # IMPROVE: iterate through dataset, fetch 2 inputs per step to the model
            c_feature = ModelManager.model_predict(enc, x_test[0], **config_experiment.predict_enc)
            s_feature = ModelManager.model_predict(enc, x_test[1], **config_experiment.predict_enc)
            from modules import features
            target_features = features.AdaIN(c_feature, s_feature, alpha=1)
            generated = ModelManager.model_predict(dec, target_features, **config_experiment.predict_dec)
            # IMPROVE: denormalize can be handled by decode_prediction config
            import tensorflow as tf
            if isinstance(generated, tf.Tensor):
                if generated.dtype == tf.float32:
                    generated = tf.cast(generated, tf.uint8)
                generated = generated.numpy()
            # show_image_mat(generated[0])
            output_saved_path = osp.join(Path.ExperimentFolderAbs, tmp_filename_by_time('jpg'))
            save_image_mat(generated[0], output_saved_path)
            input_handled = True
            INFO(f"generated: {output_saved_path}")
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
