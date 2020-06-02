from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --- System Libs --- #
import sys
from typing import Iterable

# --- 3rd Packages -- #
# NOTE: import heavy libs here will slow down all test methods even if do not need these libs
# import tensorflow as tf
# if tf.__version__.startswith('1.'):
#     import tensorflow as tf1
# else:  # tf v2.x
#     import tensorflow.compat.v1 as tf1
#     tf1.disable_v2_behavior()

# --- Custom Packages --- #
from helpers.util import *

# --- Custom Models --- #
from config import *


# --- Test Methods --- #
def test_util_logging():
    init_logging(log_level=VerbosityLevel.INFO, logger_name="UnKnown")
    INFO("info message test...", "tag_test")
    ERROR("error message test...", "tag_test")
    INFO(f"whereami (filepath, lineno, funcname, stack_info): \n  {whereami(filename_only=False)}")
    data_set = [((1,1),(1,1)), [[1,1],[1,1]], ([1,1],[1,1])]
    dump = "\n".join([f"{_} as {dump_iterable_data(_)}" for _ in data_set])
    INFO("dump_iterable_data test: \n" + dump)
    # assert False  # uncomment this line to check realtime logging output
    print("print message here")  # print() should be replaced by logging func, it cannot be captured by pytest

def test_util_miscs():
    image_path = Config.QuickTest.GrayscaleImagePath  # InputImagePath
    image_mat = load_image_mat(image_path)
    show_image_mat(image_mat)
    save_image_mat(image_mat, get_new_name_if_exists(image_path))
    save_image_mat(image_mat, get_new_name_if_exists(image_path))
    save_image_mat(image_mat, get_new_name_if_exists(image_path))
    import os.path as osp
    cache_object(image_mat, osp.split(image_path)[0], src_type="Image_mat")
    image_mat_list = [image_mat]*10
    save_image_mats(image_mat_list, None, save_dir=osp.dirname(image_path))


def test_config():
    Config.ExperimentName = "retrain/inceptionresnetv2+tlearn(33class)"  # modifiable
    INFO(f"Experiment: {Config.ExperimentName}")
    INFO(f"Path.ExperimentsFolderAbs = {Path.ExperimentsFolderAbs}")
    INFO(f"Path.ExperimentFolderAbs = {Path.ExperimentFolderAbs}")
    # path = Path.ExperimentMainConfigAbs  # class property -> str value
    config_main = ConfigSerializer.load(Path.ExperimentMainConfigAbs)
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
    INFO("Dynamic config loaded ...")
    params_data_local = Params(name=None, signature=None, decode_x=dict(colormode='rgba')) \
        .update_to(config_experiment.data)
    # deprecated: .update_to(config_experiment.get('data', {}))
    INFO(f"  params_data_local={params_data_local}")
    params_partial = params_data_local.fromkeys(['name', 'decode_x'])
    INFO(f"  params_partial after fromkeys={params_partial}")
    params_partial = Params(decode_x=dict(colormode='rgba')).left_join(params_partial)
    INFO(f"  params_partial after left_join={params_partial}")
    params_partial = Params(decode_x_new_name=dict(colormode='rgba'))\
        .left_join(params_partial, key_map={"decode_x_new_name": "decode_x"})
    INFO(f"  params_partial after left_join with key mapping={params_partial}")
    params_data_local_need_map = Params(name=None, signature=None, decode_x_new_name=dict(colormode='rgba'))\
        .update_to(config_experiment.data, key_map={"decode_x_new_name": "decode_x"})
    INFO(f"  params_data_local with key mapping={params_data_local_need_map}")
    INFO(f"Params: if missing will return {Params().na} with length={len(Params().na)}. == {{}}:{Params().na == {}}")
    INFO(f"  support get(key, None) for convenience: {Params().get('na', None)}")
    INFO(f"  while if using `[]` or get(key) returns: {Params()['na']} and {Params().get('na')}")
    INFO(f"  notice hasattr(Params,name) always returns True..: {hasattr(Params(), 'na')}")
    try:
        # IMPROVE: wrap basic config class as boilerplate
        INFO("  config_experiment.data.decode_x.colormode = " + config_experiment.data.decode_x.colormode)
    except Exception as e:
        ERROR(f"Exception: method_name={sys._getframe().f_code.co_name}, reason={e}")
        # traceback.print_exc()


def test_main():
    INFO('--- Experiment begins: {}---------------'.format(Config.ExperimentName))
    # config_main = ConfigSerializer.load(Path.ExperimentMainConfigAbs)
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
    experiment_type = str(Config.ExperimentName).split('/')[0]
    INFO('Experiment Type: ' + experiment_type)
    if experiment_type == 'retrain':
        pass
    elif experiment_type == '_test_':
        if Config.ExperimentName == '_test_/tf_1x_to_2x_3':
            from modules.data.data_manager import DataManager
            (x_train, y_train), (x_test, y_test) = (None, None), (None, None)
            if config_experiment.train.enabled:
                config_data: Params = list(config_experiment.data_set.values())[0]  # FIXME: ordered dict?
                data = DataManager.load_data(config_data.signature, category='all', **config_data)
                INFO(f"data loaded: {dump_iterable_data(data)}")
                if not isinstance(data, tuple) or any(not isinstance(_, tuple) for _ in data):
                    raise ValueError("data loaded must be in type of ((x_train, y_train), (x_test, y_test))")
                (x_train, y_train), (x_test, y_test) = data  # unpack: tuple of 4 np.ndarrays
                                                             # e.g.((60000,28,28), (60000,)), ((10000,28,28), (10000,))
            __test_flow__ = ('before_wrapping', 'wrapped')[1]
            INFO(f"__test_flow__: {__test_flow__}")  # IMPROVE: wrap as helper tracing class
            if __test_flow__ == 'before_wrapping':
                # UPDATE: reshape should be configured as data preprocessing param
                # params_model = dict(input_shape=[None, 28, 28, 1], epochs=5)
                # input_shape = [-1 if _ is None else _ for _ in params_model['input_shape']]
                # x_train = x_train.reshape(input_shape)
                # x_test = x_test.reshape(input_shape)
                import tensorflow as tf
                model = tf.keras.Sequential([
                    # NOTE: 1.TF2.x已无需限定Input层的维度，甚至各层间都能自动衔接
                    #      2.Conv层中无需设定上一层的(h,w)，只需设定filter数、kernel维度、padding(使h,w保持)等
                    tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    tf.keras.layers.Flatten(),  # 下面的神经网络需要1维的数据
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=params_model['epochs'])
                test_loss, test_acc = model.evaluate(x_test, y_test)
                INFO(f"test_loss={test_loss:8.4}, test_acc={test_acc*100:4.2}%")
                predictions = model.predict(x_test)
                for i in range(24):
                    INFO(f"predict:{tf.argmax(predictions[i])} label:{y_test[i]}")
            else:
                from modules.models.model_manager import ModelManager
                model = None
                if config_experiment.train.enabled:
                    config_model: Params = list(config_experiment.model_set.values())[0]
                    model = ModelManager.load_model(config_model.signature, **config_model)
                    model = ModelManager.model_train(model, data=(x_train, y_train), **config_experiment.train)
                    eval_metrics = ModelManager.model_evaluate(model, data=(x_test, y_test))
                if config_experiment.predict.enabled:
                    if model is None:  # not config_experiment.train.enabled
                        config_model: Params = config_experiment.model_set.model_trained
                        model = ModelManager.load_model(config_model.signature, **config_model)
                        __test_flow__ = ('use_mnist', 'use_labeled_folders')[0]
                        INFO(f"__test_flow__: {__test_flow__}")  # IMPROVE: wrap as helper tracing class
                        if __test_flow__ == 'use_mnist':
                            config_data_test: Params = list(config_experiment.data_set.values())[0]  # FIXME: ordered dict?
                            data = DataManager.load_data(config_data_test.signature, category='test', **config_data_test)
                            INFO(f"data loaded: {dump_iterable_data(data)}")
                            if not isinstance(data, tuple):
                                raise ValueError("data loaded must be in type of (x_test, y_test)")
                            x_test, y_test = data  # unpack: tuple of 4 np.ndarrays
                            test_size = min(10000, len(x_test))
                            x_test, y_test = x_test[:test_size], y_test[:test_size]
                        else:
                            config_data_test: Params = config_experiment.data_set.data_simple_test
                            meta_info = {}
                            data = DataManager.load_data(config_data_test.signature, category="test",
                                                         meta_info=meta_info, **config_data_test)
                            INFO(f"data loaded: {dump_iterable_data(data)}")
                            # trained model accepts ndarray(ndim=4) only, need to be converted
                            import tensorflow as tf
                            if isinstance(data, tf.data.Dataset):
                                from helpers.tf_helper import tf_data_to_np_array
                                # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API for unzip?
                                x_test = tf_data_to_np_array(data.map(lambda x, y: x))
                                y_test = tf_data_to_np_array(data.map(lambda x, y: y))
                            vocabulary_list = meta_info.get('vocabulary', [])
                            filenames = meta_info.get('filenames', [])
                    predictions = ModelManager.model_predict(model, x_test, y_test, **config_experiment.predict)
                    if not isinstance(predictions, Iterable):
                        predictions = [predictions]
                    if 'vocabulary_list' in vars() and 'filenames' in vars():
                        INFO(f"truths: {', '.join(['/'.join([safe_get(vocabulary_list, _), filenames[idx]]) for idx, _ in enumerate(y_test)])}")
                        INFO(f"predictions: {', '.join([safe_get(vocabulary_list, _) for _ in predictions])}")
                    else:
                        INFO(f"truths: {', '.join([str(_) for idx, _ in enumerate(y_test)])}")
                        INFO(f"predictions: {', '.join([str(_) for _ in predictions])}")
                pass
        pass
    elif experiment_type == 'tripletloss':
        # FD: tripletloss demo = load tflearn model + get features + load features + calc distance + sort
        from modules.models.model_manager import ModelManager
        model = None
        if config_experiment.train.enabled:
            raise NotImplementedError("tripletloss training for TF2.x not implemented.")
            pass
        if config_experiment.predict.enabled:
            if model is None:  # not config_experiment.train.enabled
                config_model: Params = config_experiment.model_set.model_trained
                model = ModelManager.load_model(config_model.signature, **config_model)
            from modules.data.data_manager import DataManager
            config_data: Params = config_experiment.data_set.data_simple_test  # IMPROVE: retrieve through a specified key
            data = DataManager.load_data(config_data.signature, **config_data)
            INFO(f"data loaded: {dump_iterable_data(data)}")
            x_test, y_test = data, None
            predictions = ModelManager.model_predict(model, x_test, y_test, **config_experiment.predict)
        pass
    else:
        raise ValueError('Unhandled experiment type: ' + experiment_type)

    INFO('--- Experiment ends: {}---------------'.format(Config.ExperimentName))


def test_data_data_manager():
    __test_flow__ = ('before_wrapping', 'wrapped')[1]  # NOTE: control test flow
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == 'before_wrapping':
        Config.ExperimentName = "retrain/inceptionresnetv2+tlearn(33class)"
        config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
        import modules.data.dataset_labeled_folders as dataset_labeled_folders
        ds = dataset_labeled_folders.dataset(config_experiment.data.path, category='train',
                                             **config_experiment.data)
        INFO(f"loaded dataset: {ds}")
    else:
        Config.ExperimentName = "_test_/tf_1x_to_2x_3"
        config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
        from modules.data.data_manager import DataManager
        config_data_test: Params = config_experiment.data_set.data_simple_test
        meta_info = {}
        data = DataManager.load_data(config_data_test.signature, category="test",
                                     meta_info=meta_info, **config_data_test)
        INFO(f"data loaded: {dump_iterable_data(data)}")
        vocabulary_list = meta_info.get('vocabulary', None)
        if vocabulary_list is not None:
            # try to convert tf.data to list
            import tensorflow as tf
            import numpy as np
            if isinstance(data, tf.data.Dataset):
                data_ds = data.map(lambda x, y: x)
                labels_ds = data.map(lambda x, y: y)
                # iterate through the generator
                # Method 1: OK
                # labels = []
                # for label in labels_ds:
                #     labels.append(label)
                # Method 2: OK
                # labels = list(labels_ds.as_numpy_iterator())
                # Method 3: OK
                labels = np.array(list(labels_ds.as_numpy_iterator()))
                INFO(f"labels (from dataset to list): {', '.join([safe_get(vocabulary_list,_) for _ in labels])}")
                # data = np.array(list(data_ds.as_numpy_iterator()))
                from helpers.tf_helper import tf_data_to_np_array
                data = tf_data_to_np_array(data_ds)
                INFO(f"data (from dataset to list): {dump_iterable_data(data)}")
            pass
    pass


def test_models_model_manager():
    Config.ExperimentName = "retrain/inceptionresnetv2+tlearn(33class)"
    INFO('--- Experiment begins: {}---------------'.format(Config.ExperimentName))
    config_main = ConfigSerializer.load(Path.ExperimentMainConfigAbs)
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
    experiment_type = str(Config.ExperimentName).split('/')[0]
    if experiment_type == 'retrain':
        INFO('Experiment Type: ' + experiment_type)
        __test_flow__ = ('before_wrapping', 'wrapped')[1]
        INFO(f"__test_flow__: {__test_flow__}")
        if __test_flow__ == 'before_wrapping':
            import tensorflow as tf
            model = tf.saved_model.load(config_experiment.base_model.path, tags='train')  # _tag_set=[[''],['train']]
            INFO(list(model.signatures.keys()))
            # ['image_feature_vector', 'default', 'image_feature_vector_with_bn_hparams']
            infer = model.signatures["image_feature_vector"]
            INFO(infer.structured_outputs)
            # {dict: 14}: ... 输出=1536位fv
            # 'default' = <tf.Tensor 'hub_output/feature_vector/SpatialSqueeze:0' shape=(None, 1536) dtype=float32>
        else:
            from modules.models.model_manager import ModelManager
            model = ModelManager.load_model(config_experiment.base_model.signature,
                                            **config_experiment.base_model)
    else:
        raise ValueError('Unhandled experiment type: ' + experiment_type)

    INFO('--- Experiment ends: {}---------------'.format(Config.ExperimentName))


def test_tf_helper():
    from helpers.tf_helper import preload_gpu_devices
    preload_gpu_devices()


def test_web_flask_app():
    from web import get_webapp
    webapp = get_webapp()
    webapp.config['UPLOAD_FOLDER'] = Path.UploadFolderAbs
    webapp.run(port=2020)
    print("listening to port 2020")


def test_tf_misc(show_done=False):
    import tensorflow as tf
    # -------------------------------------------------------------------------------------------
    # [2020-03-30] integer-string label -> integer
    label = "12345"
    __test_flow__ = ("tf.io.decode_raw", "tf.strings.to_number")[1]
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == "tf.io.decode_raw":
        label = tf.io.decode_raw(label, tf.dtypes.uint8)  # label将变成5个单位的Tensor
        label = tf.reshape(label, [])  # label should be a scalar. 这一步将失败，因为不匹配
    else:
        label = tf.strings.to_number(label, tf.dtypes.int32)  # NOTE: uint8 is not acceptable
    tf.get_logger().info(label)
    INFO(f"label={label}")  # both logging can be seen in pytest console
    pass

    # --- Done ----------------------------------------------------------------------------------
    if show_done:
        # [2020-03-27] 任意tf.function代码（含逻辑控制）保存进SavedModel
        @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
        def control_flow(x):
            if x < 0:
                tf.print("Invalid!")
            else:
                tf.print(x % 3)

        to_export = tf.Module()
        to_export.control_flow123 = control_flow  # 可直接赋值给尚不存在的属性..参见基类tf.training.tracking.AutoTrackable
        save_model_path = "/tmp/saved_models/simple_test"
        ensure_dir_exists(save_model_path)  # must exists!
        tf.saved_model.save(to_export, save_model_path)
        imported = tf.saved_model.load(save_model_path)
        imported.control_flow123(tf.constant(-1))
        imported.control_flow123(tf.constant(2))
        imported.control_flow123(tf.constant(3))
