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
    init_logging(log_level=VerbosityLevel.DEBUG, logger_name="UnKnown")
    DEBUG("debug message test...", "tag_test")
    INFO("info message test...", "tag_test")
    WARN("warning message test...", "tag_test")
    ERROR("error message test...", "tag_test")
    INFO(f"whereami (filepath, lineno, funcname, stack_info): \n  {whereami(filename_only=False)}")
    data_set = [((1,1),(1,1)), [[1,1],[1,1]], ([1,1],[1,1])]
    dump = "\n".join([f"{_} as {dump_iterable_data(_)}" for _ in data_set])
    INFO("dump_iterable_data test: \n" + dump)
    # assert False  # uncomment this line to check realtime logging output
    print("print message here")  # print() should be replaced by logging func, it cannot be captured by pytest

def test_util_miscs():
    __test_flow__ = ('Params', 'data_process', 'image_mat')[2]
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == 'Params':
        config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
        # test value of missed (not existing) key
        value = config_experiment.predict.key_miss
        INFO(f"(string) value of missed key: {value}, is_defined: {value.is_defined()}, is `None`: {value is None}")
        config_experiment.data_set.data.format = "not_existed_value"
        INFO(f"set a (string) attr and get from attr: {config_experiment.data_set.data.format}")
        INFO(f"set a (string) attr and get from key: {config_experiment.data_set.data['format']}")
        params_none = Params(not_existed=None)
        INFO(f"params_none before update_to: {params_none}")
        params_none.update_to(Params(existed='existed'))
        INFO(f"params_none after update_to: {params_none}")
        params_data = Params(
            shuffle=Params(not_existed=None),
            decode_x=Params(colormode=None, normalize=False, reshape=[], not_existed=None))
        # NOTE: (updated) now can check leaf node of param by `is None` after `update_to()`.
        INFO(f"params_data (with `None` value) before update_to: {params_data}")
        params_data.update_to(config_experiment.data_set.data)
        INFO(f"params_data (with `None` value) after update_to: {params_data}")
        # INFO(f"  is_defined: {params_data.shuffle.not_existed.is_defined()}")  # <-cannot call is_defined on leaf node
        INFO(f"  is `None`: {params_data.shuffle.not_existed is None}")
        params_data.left_join(config_experiment.data_set.data)
        INFO(f"params_data.decode_x after left_join: {params_data.decode_x}")
    elif __test_flow__ == 'data_process':
        dict1 = {'a': 1, 'b': 1, 'c': 1}
        dict2 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new, changed = dict_compare(dict1, dict2)
        DEBUG(f'compare_dict: new={new}, changed={changed}')
    elif __test_flow__ == 'image_mat':
        image_path = Config.QuickTest.GrayscaleImagePath  # InputImagePath
        image_mat = load_image_mat(image_path)
        __test_flow__ = ('blocking_show_and_save', 'nonblocking_show')[0]
        INFO(f"__test_flow__: {__test_flow__}")
        if __test_flow__ == 'blocking_show_and_save':
            # import matplotlib.pyplot as plt
            # plt.ioff()
            fig = show_image_mat(image_mat)
            save_image_mat(image_mat, get_new_name_if_exists(image_path))
            save_image_mat(image_mat, get_new_name_if_exists(image_path))
            save_image_mat(image_mat, get_new_name_if_exists(image_path))
            import os.path as osp
            cache_object(image_mat, osp.split(image_path)[0], src_type="Image_mat")
            image_mat_list = [image_mat]*10
            save_image_mats(image_mat_list, None, save_dir=osp.dirname(image_path))
        elif __test_flow__ == 'nonblocking_show':
            from async_ import AsyncManager, amend_blank_cbs, AsyncLoop

            __test_flow__ = ('not_use_asyncio', 'use_asyncio')[1]
            INFO(f"__test_flow__: {__test_flow__}")
            if __test_flow__ == 'not_use_asyncio':
                import matplotlib.pyplot as plt
                import time
                fig = show_image_mat(image_mat, block=False)
                for i in range(10):
                    fig.show()
                    # INFO(f'fig state before pause= {fig.__dict__}')
                    plt.pause(1)
                    # INFO(f'fig state after pause= {fig.__dict__}')
                    # time.sleep(1)
            elif __test_flow__ == 'use_asyncio':
                import asyncio

                @track_entry_and_exit.coro()
                async def coro_show_image_mat(data, cbs, task_id=None):
                    # NOTE: matplot backend is single-threaded, and can only be accessed from its host thread.
                    import matplotlib.pyplot as plt

                    image_mat, image_name, block = data
                    DEBUG(f'[coro_show_image_mat]: {locals()}')
                    on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
                    fig = show_image_mat(image_mat, block=block)
                    DEBUG(f'[coro_show_image_mat] after show_image_mat')
                    fig_state_origin = fig.__dict__
                    if not block:
                        # TODO: plt.pause(1) or fig.ginput(timeout), fig.waitforbuttonpress(timeout) or yield?
                        # TODO: how to get figure's on_close event?
                        try:
                            __test_flow__ = ('use_coro_pause', 'use_async_generator')[0]
                            INFO(f"__test_flow__: {__test_flow__}")
                            if __test_flow__ == 'use_coro_pause':
                                async def coro_pause(delay_s):
                                    plt.pause(delay_s)
                                    await asyncio.sleep(0.1)
                                while True:
                                    fig.show()
                                    await coro_pause(1)
                            elif __test_flow__ == 'use_async_generator':
                                async def asyncgen_pause(delay_s):
                                    while True:
                                        # await asyncio.sleep(delay_s)
                                        plt.pause(delay_s)
                                        yield   # NOTE: yield will change coro to async generator, and cannot passed to asyncio.wait()
                                async for _ in asyncgen_pause(0.1):
                                    fig.show()
                                    await asyncio.sleep(0.1)
                                    # done, pending = await asyncio.wait({coro_pause(1)})
                        except Exception as e:  # for TKinter backend, a _tkinter.TclError
                            DEBUG(f'[coro_show_image_mat] show loop ended: {e}')
                    new, changed = dict_compare(fig_state_origin, fig.__dict__)
                    DEBUG(f'[coro_show_image_mat] figure closed: {image_name}. state={{new:{new}, changed:{changed}}}')
                    plt.close(fig)
                    on_done(image_name)

                def on_done_show_image_mat(image_name): INFO(f'figure {image_name} closed.')

                __test_flow__ = ('use_main_thread', 'use_ui_thread')[1]
                INFO(f"__test_flow__: {__test_flow__}")
                if __test_flow__ == 'use_main_thread':
                    ui_loop = AsyncManager.get_loop(AsyncLoop.Main)
                    task1 = AsyncManager.create_task(coro_show_image_mat((image_mat, '1', False), (on_done_show_image_mat, )), loop=ui_loop)
                    task2 = AsyncManager.create_task(coro_show_image_mat((image_mat, '2', False), (on_done_show_image_mat, )), loop=ui_loop)
                    AsyncManager.run_task(AsyncManager.gather_task(task1, task2, loop=ui_loop), loop=ui_loop)
                elif __test_flow__ == 'use_ui_thread':
                    # NOTE: ensure to initialize matplot in the specified loop (ui_loop or others) than it works.
                    ui_loop = AsyncManager.get_loop(AsyncLoop.UIThread)
                    __test_flow__ = ('use_gather_for_parallel', 'try_parallel_between_batches')[1]
                    INFO(f'__test_flow__: {__test_flow__}')
                    if __test_flow__ == 'use_gather_for_parallel':
                        task1 = AsyncManager.create_task(
                            coro_show_image_mat((image_mat, '1', False), (on_done_show_image_mat,)), loop=ui_loop)
                        task2 = AsyncManager.create_task(
                            coro_show_image_mat((image_mat, '2', False), (on_done_show_image_mat,)), loop=ui_loop)
                        AsyncManager.run_task(AsyncManager.gather_task(task1, task2, loop=ui_loop), loop=ui_loop)
                    elif __test_flow__ == 'try_parallel_between_batches':
                        task1 = AsyncManager.create_task(
                            coro_show_image_mat((image_mat, '1', False), (on_done_show_image_mat,)), loop=ui_loop)
                        AsyncManager.run_task(task1, loop=ui_loop)
                        task2 = AsyncManager.create_task(
                            coro_show_image_mat((image_mat, '2', False), (on_done_show_image_mat,)), loop=ui_loop)
                        AsyncManager.run_task(task2, loop=ui_loop)
                # Test Result:
                # 0.run from CLI or PyCharm will have different result.
                # 1.if use main loop, AsyncManager.run_task must call loop.run_until_completed(). but
                #   show_image_mat(block=False) will still show a freezing window. Unless use gather_task! then,
                #   2 freezing windows appeared ;)  Further, if use asyncgen_pause iteration, 2 active windows appeared!
                #   but must pause 1 seconds, if too short (0.1s) when 1 window closed another will failed and exit!
                #   Furthermore, if use coro_pause+await async.sleep(), also succeeded!
                #   Finally, main loop will be blocked. So try to show in a ui thread.
                # 2.if use ui_thread loop (=initialize matplot in a ui thread), ..succeed immediately!
                import time
                interval = 1
                i = 0
                while True:
                    INFO(f"Waiting...{i*interval} seconds")
                    time.sleep(interval)
                    i += 1


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
        # TODO: practice after tutorial
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
                    if 'meta_info' not in vars():
                        meta_info = {}  # retrieve meta info from DataManager
                    if x_test is None or y_test is None:  # not config_experiment.train.enabled
                        __test_flow__ = ('use_mnist', 'use_labeled_folders', 'use_config')[2]
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
                        elif __test_flow__ == 'use_labeled_folders':
                            config_data_test: Params = config_experiment.data_set.data_simple_test
                            # meta_info = {}  # retrieve meta info from DataManager
                            data = DataManager.load_data(config_data_test.signature,  # category="test", in config
                                                         meta_info=meta_info, **config_data_test)
                            INFO(f"data loaded: {dump_iterable_data(data)}")
                            # trained model accepts ndarray(ndim=4) only, need to be converted
                            import tensorflow as tf
                            if isinstance(data, tf.data.Dataset):
                                from helpers.tf_helper import tf_obj_to_np_array
                                # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API for unzip?
                                x_test = tf_obj_to_np_array(data.map(lambda x, y: x))
                                y_test = tf_obj_to_np_array(data.map(lambda x, y: y))
                        else:
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
                        model = ModelManager.load_model(config_model.signature, **config_model)
                    predictions = ModelManager.model_predict(model, (x_test, y_test), **config_experiment.predict)
                    if not isinstance(predictions, Iterable):
                        predictions = [predictions]
                    if 'vocabulary' in meta_info and 'filenames' in meta_info:
                        vocabulary_list = meta_info.get('vocabulary', [])
                        filenames = meta_info.get('filenames', [])
                        if y_test is not None:
                            INFO(f"truths: {', '.join(['/'.join([safe_get(vocabulary_list, _), filenames[idx]]) for idx, _ in enumerate(y_test)])}")
                        INFO(f"predictions: {', '.join([safe_get(vocabulary_list, _) for _ in predictions])}")
                    else:
                        if y_test is not None:
                            INFO(f"truths: {', '.join([str(_) for idx, _ in enumerate(y_test)])}")
                        INFO(f"predictions: {', '.join([str(_) for _ in predictions])}")
                pass
        pass
    elif experiment_type == 'tripletloss':
        # FD: tripletloss demo = load tflearn model + get features + load features + calc distance + sort
        from helpers.tf_helper import async_preload_gpu_devices
        async_preload_gpu_devices()
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
            data_key = config_experiment.predict.data_inputs.__str__()
            config_data_test: Params = config_experiment.data_set[data_key]
            # NOTE: for consistency use the same method for user-interactive data load, but it will be blocking.
            data = DataManager.load_data(config_data_test.signature, **config_data_test)
            INFO(f"data loaded: {dump_iterable_data(data)}")
            x_test, y_test = data, None
            predictions = ModelManager.model_predict(model, (x_test, y_test), **config_experiment.predict)
        pass
    else:
        raise ValueError('Unhandled experiment type: ' + experiment_type)

    INFO('--- Experiment ends: {}---------------'.format(Config.ExperimentName))


def test_data_data_manager():
    __test_flow__ = ('before_wrapping', 'wrapped')[1]  # NOTE: control test flow
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == 'before_wrapping':
        Config.ExperimentName = "retrain/inceptresv2+zipper(33class)"
        config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
        import modules.data.dataset_labeled_folders as dataset_labeled_folders
        ds = dataset_labeled_folders.dataset(config_experiment.data.path, category='train',
                                             **config_experiment.data)
        INFO(f"loaded dataset: {ds}")
    else:
        __test_flow__ = ('labeled_folders', 'mnist_tf_data_to_np_array', 'ui_web_files', 'ui_copy_files')[3]
        INFO(f'__test_flow__: {__test_flow__}')
        if __test_flow__ == 'labeled_folders':
            Config.ExperimentName = "retrain/inceptresv2+scansnap(6class)"
            config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
            from modules.data.data_manager import DataManager
            config_data_train: Params = config_experiment.data_set.data
            meta_info = {}  # retrieve meta info from DataManager
            data, _ = DataManager.load_data(config_data_train.signature,
                                         meta_info=meta_info, **config_data_train)
            INFO(f"data loaded: {dump_iterable_data(data)}")
            import tensorflow as tf
            if isinstance(data, tf.data.Dataset):
                from helpers.tf_helper import tf_obj_to_np_array
                # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API for unzip?
                x_train = tf_obj_to_np_array(data.map(lambda x, y: x))
                y_train = tf_obj_to_np_array(data.map(lambda x, y: y))
            vocabulary_list = meta_info.get('vocabulary', [])
            filenames = meta_info.get('filenames', [])
            pass
        elif __test_flow__ == 'mnist_tf_data_to_np_array':
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
                    INFO(f"labels (from dataset to list): {', '.join([safe_get(vocabulary_list, _) for _ in labels])}")
                    # data = np.array(list(data_ds.as_numpy_iterator()))
                    from helpers.tf_helper import tf_obj_to_np_array
                    data = tf_obj_to_np_array(data_ds)
                    INFO(f"data (from dataset to list): {dump_iterable_data(data)}")
                pass
        elif __test_flow__ == 'ui_web_files':
            Config.ExperimentName = "tripletloss/inceptresv2_tlearn33c+tripletloss+ykk(5c,251)"
            config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
            from helpers.tf_helper import async_preload_gpu_devices
            async_preload_gpu_devices()
            from modules.data.data_manager import DataManager
            config_data_test: Params = config_experiment.data_set.data_ui_test
            # NOTE: for consistency use the same method for user-interactive data load, but it will be blocking.
            data = DataManager.load_data(config_data_test.signature, **config_data_test)
            INFO(f"data loaded: {dump_iterable_data(data)}")
        elif __test_flow__ == 'ui_copy_files':
            Config.ExperimentName = "_test_/tf_1x_to_2x_3"
            config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
            from modules.data.data_manager import DataManager
            config_data_test: Params = config_experiment.data_set.data_ui_test
            # NOTE: for consistency use the same method for user-interactive data load, but it will be blocking.
            data = DataManager.load_data(config_data_test.signature, **config_data_test)
            INFO(f"data loaded: {dump_iterable_data(data)}")
    pass


def test_data_decode_tf():
    # NOTE: may comment out @tf.function for easier debug, remember update back
    from modules.data.decode_tf import decode_image_file
    image_path = Config.QuickTest.InputImagePath  # GrayscaleImagePath
    image_t = decode_image_file(image_path, normalize=False)
    fig = show_image_mat(image_t)
    print('test end.')

def test_models_model_manager():
    # Config.ExperimentName = "retrain/inceptionresnetv2+tlearn(33class)"
    INFO('--- Experiment begins: {}---------------'.format(Config.ExperimentName))
    # config_main = ConfigSerializer.load(Path.ExperimentMainConfigAbs)
    config_experiment = ConfigSerializer.load(Path.ExperimentConfigAbs)
    experiment_type = str(Config.ExperimentName).split('/')[0]
    if experiment_type == 'retrain':
        INFO('Experiment Type: ' + experiment_type)
        __test_flow__ = ('before_wrapping', 'wrapped')[0]
        INFO(f"__test_flow__: {__test_flow__}")
        if __test_flow__ == 'before_wrapping':
            import tensorflow as tf
            from config import __abspath__
            import os.path as osp
            path = path_possibly_formatted(config_experiment.model_set.model_base.path)
            path = __abspath__(path) if not osp.isabs(path) else path

            __test_flow__ = ('use_tf_saved_model', 'use_tfhub')[1]
            INFO(f"__test_flow__: {__test_flow__}")
            if __test_flow__ == 'use_tf_saved_model':
                model = tf.saved_model.load(path, tags=None)  # for feature_vector: _tag_set={set(), {'train'}}
                INFO(list(model.signatures.keys()))
                # ['image_feature_vector', 'default', 'image_feature_vector_with_bn_hparams']
                if len(model.signatures) > 0:
                    func_fv = model.signatures["image_feature_vector"]
                    INFO(f'func_fv.structured_outputs: {func_fv.structured_outputs}')
                    # {dict: 14}: ... 输出=1536位fv
                    # 'default' = <tf.Tensor 'hub_output/feature_vector/SpatialSqueeze:0' shape=(None, 1536) dtype=float32>
                    func_fv = model.signatures["default"]
                    INFO(f'func_predict.structured_outputs: {func_fv.structured_outputs}')
                    # {'default': <tf.Tensor 'hub_output/feature_vector/SpatialSqueeze:0' shape=(None, 1536) dtype=float32>}
            else:
                import tensorflow_hub as tfhub
                model = tf.keras.Sequential([
                    # NOTE: cannot specify tags={..} (to load model with multi-tags) in TF2.1..solved in 2.2
                    tfhub.KerasLayer(path)
                ])
                # model = tfhub.load(path, tags={})
                # model.build([None, 299, 299, 3])  # since incept_resnet_v2/4, input shape can be arbitary

            import numpy as np
            image_path = Config.QuickTest.InputImagePath
            image_mat = load_image_mat(image_path)
            if any([str(image_mat.dtype).startswith(_) for _ in ('int', 'uint')]):
                image_mat = image_mat.astype(np.float32) / 255.0  # normalize = true
            if image_mat.shape.__len__() == 3:
                image_mat = np.expand_dims(image_mat, axis=0)
            # predictions = model.predict(image_mat)  # TODO: modify ModelManager::model_predict() logic
            import tensorflow as tf
            image_t = tf.convert_to_tensor(image_mat)

            if 'func_fv' in vars() and callable(func_fv):
                result = func_fv(image_t)
                INFO(f"result of model.signatures['default'] function={result}")
            else:
                assert callable(model)
                result = model(image_t)  # if feature_vector shape=(1,1536), if classification shape=(1,1001)
                if tuple(result.shape) == (1, 1001):
                    probs = np.exp(result[0]) / np.sum(np.exp(result[0]), axis=0)
                    labels = quick_load_imagenet_labels()
                    probs, idxs = tf.math.top_k(probs, k=3)
                    INFO(f"result of model()={dump_iterable_data(result)}, label={labels[idxs.numpy()]}, probs={probs.numpy()}")
                else:
                    INFO(f"result of model()={dump_iterable_data(result)}")
        else:
            from modules.models.model_manager import ModelManager
            model = ModelManager.load_model(config_experiment.model_set.model_base.signature,
                                            **config_experiment.base_model)
    else:
        raise ValueError('Unhandled experiment type: ' + experiment_type)

    INFO('--- Experiment ends: {}---------------'.format(Config.ExperimentName))


def test_tf_helper():
    from helpers.tf_helper import preload_gpu_devices
    preload_gpu_devices()


def test_web_flask_app():
    # 0. initialize a web app
    from web import get_webapp
    webapp = get_webapp()
    config_deploy = ConfigSerializer.load(Path.DeployConfigAbs)
    import os.path as osp
    webapp.config['UPLOAD_FOLDER'] = config_deploy.web.upload_folder if osp.isabs(config_deploy.web.upload_folder)\
        else __abspath__(config_deploy.web.upload_folder)
    params_webapp = Params(host="127.0.0.1", port="2020", ssl_context=None)\
        .left_join(config_deploy.web, {"host": "local_ip", "port": "local_port"})
    if config_deploy.web.use_https:
        params_webapp.ssl_context = (config_deploy.web.certfile_path, config_deploy.web.keyfile_path)

    # 1. on uploads, create and run an async_task
    from async_ import AsyncManager, amend_blank_cbs
    init_logging(VerbosityLevel.DEBUG)

    @webapp.on_uploads(namespace="test_web_flask_app")
    def ui_web_inputs_accepted(filepath_or_list):
        # AsyncTask.1: 正式处理数据，但需声明为async(=coroutine func)
        @track_entry_and_exit.coro()
        async def coro_consume_inputs(filepath_or_list, cbs, task_id=None):
            INFO(f'[coro_consume_inputs]: data={filepath_or_list}, task_id={task_id}')
            on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
            filepaths = filepath_or_list if isinstance(filepath_or_list, list) else [filepath_or_list]
            result = {}  # filenames: str[], error: optional(str)

            import numpy as np
            image_mats = []
            result['filenames'] = []
            import os.path as osp
            for idx, filepath in enumerate(filepaths):
                image_mats.append(load_image_mat(filepath))
                result['filenames'].append(osp.basename(filepath))
            image_mats = np.array(image_mats)

            try:
                # 1.block tasks in the same batch, the same loop
                # 2.do not block tasks in the same batch, the same loop
                # 3.even do not block tasks in the next batch!
                __test_flow__ = ('blocking_task_batch', 'no_blocking_current_and_new_task_batches')[1]
                INFO(f'__test_flow__: {__test_flow__}')
                if __test_flow__ == 'blocking_task_batch':
                    show_image_mats(image_mats)
                elif __test_flow__ == 'no_blocking_current_and_new_task_batches':
                    asynctask = async_show_image_mats(image_mats)
                    result.update({'asynctask_id': asynctask.id})
                # save_image_mat(image_mat, get_new_name_if_exists(image_path))
            except Exception as e:
                result.update({'error': e.__repr__()})
            on_done(result)

        def on_done_consume_inputs(result):
            INFO(f'on_done_consume_inputs: {result}')

        # IMPROVE: needn't to hack task_id into a coro. coro can asyncio.current_task(get_running_loop()) get task.
        # task_id = AsyncManager.new_id(AsyncManager.current_loop().id)
        # task = AsyncManager.create_task(coro_consume_inputs(filepath_or_list, (on_done_consume_inputs,), task_id),
        #                                 given_id=task_id)
        # AsyncManager.run_task(task)
        task = AsyncManager.run_task(coro_consume_inputs(filepath_or_list, (on_done_consume_inputs, ), None))

        handler_result = {'asynctask_id': task.id}
        return handler_result

    # 3. launch the web app
    __test_flow__ = ('webapp_main_thread', 'webapp_new_thread')[1]
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == 'webapp_main_thread':
        webapp.run(**params_webapp)
    else:
        from async_ import AsyncLoop, AsyncManager

        async def coro_webapp_run(**params): webapp.run(**params)
        webapp_loop = AsyncManager.get_loop(AsyncLoop.WebApp)
        task_dumb = AsyncManager.run_task(coro_webapp_run(**params_webapp), loop=webapp_loop)
        INFO(f"listening to port {params_webapp.port}")

        import time
        from helpers.util import adjust_interrupt_handlers
        adjust_interrupt_handlers()
        try:
            interval = 1
            i = 0
            while True:
                DEBUG(f"[master thread] Waiting...{i * interval} seconds")
                time.sleep(interval)
                i += 1
        except KeyboardInterrupt:
            INFO("----- Keyboard interruption -----")
            # loop.stop()
            webapp_loop.call_soon_threadsafe(webapp_loop.stop)


def test_asyncio_misc():
    import asyncio
    import time
    import random
    now = lambda: time.time()
    # task_id: args(=time_cost), future
    all_tasks = []

    def cb_on_done(*args):
        INFO(f'Done: {args}')

    def cb_on_succeeded(*args):
        INFO(f'Succeeded: {args}')

    def cb_on_failed(*args):
        INFO(f'Failed: {args}')

    def cb_on_progress(*args):
        INFO(f'Progress: {args}')

    def _amend_cbs_arg(cbs):
        amend_cbs = [lambda:{}, lambda:{}, lambda:{}, lambda:{}]
        if hasattr(cbs, '__len__'):
            for i, cb in enumerate(cbs):
                amend_cbs[i] = cb
        return amend_cbs

    async def coro_task(data, cbs=None, id=None):
        time_cost, *_ = data  # example of unpacking
        # TODO: if None, assign a dumb func(...)
        on_done, on_succeeded, on_failed, on_progress = _amend_cbs_arg(cbs)
        result = {}  # id, error, result
        progress = {'id': id, 'progress': 0}
        INFO(f"task id: {id}, time cost assumed: {time_cost}")
        if time_cost == 0 or time_cost == 3:
            error = f'time cost cannot be 0 or 3! get {time_cost}'
            result.update({'error': error})
            on_failed(result)
            on_done(result)
            # NOTE: for simplicity avoid throw exception in coro_work
            # raise ValueError(error)
            return
        for i in range(time_cost):
            await asyncio.sleep(1)  # 模拟耗时的I/O操作，await它、表示让出控制权
            progress.update({'progress': int((i+1)/time_cost*100)/100})
            on_progress(progress)
        result.update({'result': time_cost})

        # test self-defined time consuming routine
        image_path = Config.QuickTest.GrayscaleImagePath
        image_mat = load_image_mat(image_path)
        show_image_mat(image_mat, block=False)
        # async def _async001(): show_image_mat(image_mat)
        # await _async001()

        on_succeeded(result)
        on_done(result)
        return result
    INFO(f'coro_task: {coro_task}')

    __test_flow__ = ('single_thread', 'dual_thread')[1]
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == 'single_thread':
        loop = asyncio.get_event_loop()
        # loop.run_forever()  # NOTE: if run_forever() before create_task(), task will not be run
        tasks = []
        total = 5
        data = [(i+1, random.randint(0, 5)) for i in range(total)]  # task_id, time_cost
        cbs = (cb_on_done, cb_on_succeeded, cb_on_failed, cb_on_progress)
        coroutines = [coro_task(data[i], cbs) for i in range(total)]
        for coroutine in coroutines:
            task = loop.create_task(coroutine)  # task depends on loop
            # task.add_done_callback(callback)
            # print(task)
            tasks.append(task)
        start = now()
        print("Ready, Run!")
        loop.run_until_complete(asyncio.gather(*tasks))  # everything runs well. including callbacks will be called.
        # loop.run_forever()  # NOTE: if use this instead of above line, tasks will not be run either..
        print("Time:", now()-start)
    elif __test_flow__ == 'dual_thread':
        def thread_for_asynctask_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        loop = asyncio.new_event_loop()
        from threading import Thread
        loop_thread = Thread(target=thread_for_asynctask_loop, args=(loop,))
        loop_thread.setDaemon(True)  # will close if master thread closed
        loop_thread.start()
        total = 5
        data = [(random.randint(0, 5), i) for i in range(total)]  # time_cost, ..
        cbs = (cb_on_done, cb_on_succeeded, cb_on_failed, cb_on_progress)
        start = now()
        __test_flow__ = ('use_run_coro', 'use_create_task')[1]
        print("[master thread]Ready, Run!")
        import functools
        for i in range(total):
            # NOTE: cannot append args to a coro object.
            # coro = coro_task(data[i], cbs, None)
            # coro = functools.partial(coro, id=i+1)
            coro = coro_task(data[i], cbs, id=i+1)
            if __test_flow__ == 'use_run_coro':
                # NOTE: if using coro by run_coro() instead of Task, it will be run instantly.
                future = asyncio.run_coroutine_threadsafe(coro, loop=loop)
                # future.add_done_callback(callback)  # NOTE: callback messages will be printed, but after pytest finished.
            elif __test_flow__ == 'use_create_task':
                task = loop.create_task(coro)

        if __test_flow__ == 'use_create_task':
            # NOTE: loop.create_task() after loop.run_forever() will not be run, unless activate an arbitrary call_soon
            loop.call_soon_threadsafe(lambda: {})
        print("[master thread]Time:", now()-start)
        # test results:
        # [2020/06/09]
        # 1.all messages printed in messed order
        # 2.'Time:' will be printed very quick(0.000997s), even before some 'waiting..'.
        # 3.pytest will keep running even after test PASSED -- because loop_thread is still running.
        # 4.some chars are lost e.g.'R' of 'Ready'
        import time
        from helpers.util import adjust_interrupt_handlers
        adjust_interrupt_handlers()
        try:
            for i in range(3600):
                INFO(f"[master thread]Waiting...{i} seconds")
                time.sleep(1)
        except KeyboardInterrupt:
            INFO("----- Keyboard interruption -----")
            # loop.stop()
            loop.call_soon_threadsafe(loop.stop)


def test_tf_misc(run_legacy=False):
    import tensorflow as tf
    # -------------------------------------------------------------------------------------------
    # [2020-07-13] 再确认图片等比缩放且自动补背景的tf实现
    resize_h, resize_w = 299, 299
    __test_flow__ = ("not_only_jpeg", "only_jpeg")[0]
    INFO(f"__test_flow__: {__test_flow__}")
    if __test_flow__ == "not_only_jpeg":
        from helpers.util import quick_load_image_tensor, show_image_mat
        image = quick_load_image_tensor()
        # image = tf.image.resize(image, [resize_h, resize_w], preserve_aspect_ratio=True)  # default=bilinear
        image = tf.image.resize_with_pad(image, resize_h, resize_w)  # background will be black, no method to set it..
        show_image_mat(image.numpy())
    elif __test_flow__ == "only_jpeg":
        from helpers.util import show_image_mat
        from config import Config
        image_jpeg_string_t = tf.io.read_file(Config.QuickTest.InputImagePath)

        def decode_and_center_crop(image_jpeg_string_t, image_size, crop_padding=32):
            shape = tf.image.extract_jpeg_shape(image_jpeg_string_t)
            image_height, image_width = shape[:2]
            padded_center_crop_size = tf.cast(
                ((image_size / (image_size + crop_padding)) *
                 tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                tf.int32)
            # FIXME: offset calculation in this sample is incorrect
            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                    padded_center_crop_size, padded_center_crop_size])
            image_t = tf.image.decode_and_crop_jpeg(image_jpeg_string_t, crop_window, channels=3)
            image_t = tf.image.resize(image_t, [image_size, image_size], preserve_aspect_ratio=True)
            return image_t

        image_t = decode_and_center_crop(image_jpeg_string_t, resize_h, crop_padding=0)
        image_t /= 255.0
        show_image_mat(image_t.numpy())

    pass

    # --- Done ----------------------------------------------------------------------------------
    if run_legacy:
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


def test_np_misc(run_legacy=False):
    import numpy as np
    # -------------------------------------------------------------------------------------------
    pass

    # --- Done ----------------------------------------------------------------------------------
    if run_legacy:
        # [2020-07-09] probs -> top_k probs
        np.random.seed(2020)
        probs = np.random.random((5, 10))
        INFO(f'probs:{probs}')
        top_k = 3
        idxs = np.argsort(probs, axis=-1)  # ascending order..
        idxs = np.flip(idxs, axis=-1)
        INFO(f'idxs sorted:{idxs}')
        idxs = idxs.take(np.arange(top_k), axis=-1)
        INFO(f'idxs top_k:{idxs}')
        probs = np.take_along_axis(probs, idxs, axis=-1)  # not np.take()
  