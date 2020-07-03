
import enum
import os.path as osp

from helpers.util import DEBUG, INFO, WARN, ERROR, Params, dump_iterable_data, path_possibly_formatted, ensure_web_app

__all__ = [
    "DataManager",
]

class _DataSignature(enum.Enum):
    """
    加载/保存数据。包含：预处理、分组(train/eval/test)、Feed导入(shuffle,repeat,batch)。
    支持：1.np.ndarray或tf.Tensor，或两者的list或dict（keras.Model.fit/eval/predict的需求） 2.tf.data.Dataset
    """
    # -- TF2.x ---------------------------------------
    TFDataset = ("tf.data.Dataset", "")
    TFKerasDataset = ("tf.keras.datasets.load_data", "Name",
                      "mnist, fashion_mnist, boston_housing, imdb, cifar10/100, reuters. "
                      "Result: tuple of numpy array `(x_train, y_train), (x_test, y_test)`")
    UI_Web_Files = ("ui_web_files", "Paths", "listen to ui web event (uploads) to get data")
    # -- TF1.x ---------------------------------------
    LabeledFolders = ("labeled_folders", "Path")
    SingleFile = ("single_file", "Path")
    TFFixedLengthRecordDataset = ("tf.data.FixedLengthRecordDataset", ["Path", "Url"],
                                  "e.g. 'train-images-idx3-ubyte'(images), 'train-labels-idx1-ubyte'(labels) of mnist. "
                                  "`header_bytes` required to truncate file header")

    def __init__(self, signature: str, formats, memo=None):
        self.signature = signature
        self.formats = formats if isinstance(formats, list) else [formats]
        self.__doc__ = memo


class DataManager:
    @staticmethod
    def _validate_format(format_, data_signature: _DataSignature):
        if format_ not in data_signature.formats:
            raise ValueError(f"Acceptable formats: {data_signature.formats}, while given: {format_}")
        return format_

    @staticmethod
    def _validate_path(path):
        from config import __abspath__
        path = path_possibly_formatted(path)
        path = __abspath__(path) if not osp.isabs(path) else path
        if not osp.exists(path):
            raise ValueError(f"Given path is invalid: {path}")
        return path

    @staticmethod
    def load_data(data_signature: str, category="all", meta_info=None, **params) -> object:
        """
        :param data_signature:
        :param category: 'train', 'test' or 'all'
        :param meta_info: if given as a dict, caller may get meta info of the dataset through it
        :param params:
        :return: if `category`='all', 'train' and 'test' dataset will be returned as a tuple
        """
        data = None
        params_data = Params(
            timeout=0,
            test_split=0.2, fixed_seed=None,
            decode_x=Params(colormode=None, resize_w=None, resize_h=None, preserve_aspect_ratio=True,
                            normalize=True, reshape=[]),
            decode_y=Params()).update_to(params)
        if data_signature == _DataSignature.LabeledFolders.signature:
            params_data = Params(file_exts=['jpg'], labels_ordered_in_train=None).update_to(params_data)
            # TODO: consider fix random seeds or control shuffle flags here and there
            import modules.data.dataset_labeled_folders as dataset_labeled_folders
            # format_ = DataManager._validate_format(kwargs['format'], _DataSignature.LabeledFolders)
            path = DataManager._validate_path(params_data.path)
            ds = dataset_labeled_folders.dataset(path, category=category, meta_info=meta_info, **params_data)
            DEBUG(f"loaded tf.data.Dataset: {ds}")
            data = ds
        elif data_signature == _DataSignature.TFKerasDataset.signature:
            # TODO: extract as modules.data.dataset_tf_keras_dataset :: dataset(name, **params)
            from importlib import import_module
            # format_ = DataManager._validate_format(kwargs['format'], _DataSignature.TFKerasDataset)
            lib_dataset = import_module(f"tensorflow.keras.datasets.{params_data.name}")
            (x_train, y_train), (x_test, y_test) = lib_dataset.load_data()  # Tensors
            WARN(f"Keras dataset {params_data.name} loaded as is. Ignored configs: colormode, resize_w/h, preserve_aspect_ratio")
            if params_data.decode_x.normalize:
                x_train, x_test = x_train/255.0, x_test/255.0
            if params_data.decode_x.reshape.__len__() > 0:
                x_train = x_train.reshape(params_data.decode_x.reshape)
                x_test = x_test.reshape(params_data.decode_x.reshape)
            DEBUG(f"loaded data: y_train={y_train}, y_test={y_test}")
            if category == 'all':
                data = ((x_train, y_train), (x_test, y_test))
            elif category == 'train':
                data = (x_train, y_train)
            elif category == 'test':
                data = (x_test, y_test)
            else:
                raise ValueError(f"Unknown category: {category}")
            # IGNORED: meta_info returns no value. test_split has no use. fixed_seed not used.
        elif data_signature == _DataSignature.SingleFile.signature:
            path = DataManager._validate_path(params_data.path)
            params_decode = Params(encoding='jpg', colormode=None, reshape=None,
                                   preserve_aspect_ratio=True,
                                   color_transform=None, normalize=True).left_join(params_data.decode_x)
            import modules.data.decode_tf as decode_tf
            import tensorflow as tf
            # IMPROVE: accept different kinds of input data
            data_t = decode_tf.decode_image_file(tf.convert_to_tensor(path), **params_decode)
            data = tf.data.Dataset.from_tensors(data_t)
        elif data_signature == _DataSignature.UI_Web_Files.signature:
            # path = DataManager._validate_path(params_data.path)
            params_decode = Params(encoding='jpg', colormode=None, reshape=None,
                                   preserve_aspect_ratio=True,
                                   color_transform=None, normalize=True).left_join(params_data.decode_x)
            data = None

            webapp = ensure_web_app()  # will load config from Path.DeployConfigAbs
            INFO(f'waiting for data input from web app {webapp.host}:{webapp.port}')  # IMPROVE: hint upload url
            from async_ import AsyncLoop, AsyncManager, amend_blank_cbs
            from helpers.util import track_entry_and_exit, load_image_mat, async_show_image_mats
            import asyncio
            this_task: asyncio.Task = None

            @track_entry_and_exit.coro()
            async def coro_consume_files(abspath_or_list, cbs):
                nonlocal this_task
                assert this_task is not None, '`this_task` should have been assigned before entering related coro.'

                import modules.data.decode_tf as decode_tf
                import tensorflow as tf

                DEBUG(f'[coro_consume_inputs]: {locals()}')
                on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
                filepaths = abspath_or_list if isinstance(abspath_or_list, list) else [abspath_or_list]
                result = {}  # data: tf.data.Dataset::{image_t}, error: optional(str)

                # from helpers.tf_helper import image_example
                # IMPROVE: try to use TFRecordDataset.from_tensors([tf_example])

                def map_path_to_image(path_t):
                    return decode_tf.decode_image_file(path_t, **params_decode)

                data = tf.data.Dataset.from_tensor_slices(filepaths)
                this_task.progress = 0.5
                data = data.map(map_path_to_image)
                this_task.progress = 1.0
                result.update({'data': data})
                # # if show inputs
                # try:
                #     asynctask = async_show_image_mats(image_mats)
                #     result.update({'asynctask_id': asynctask.id})
                # except Exception as e:
                #     result.update({'error': e.__repr__()})
                on_done(result)
                # TODO: how to link to the next task (e.g. model.predict) so user can monitor process.
                return result    # == this_task.set_result(result)

            def on_done_consume_inputs(result):
                """
                If using task.set_result, set_exception etc and wait for task instead of data,
                callbacks will be optional.
                """
                nonlocal data
                INFO(f'on_done_consume_inputs: {result}')
                data = result.get('data', None)

            @webapp.on_uploads(namespace="data_manager::ui_web_files", onetime=True)
            def handle_ui_web_files(abspath_or_list):
                nonlocal this_task
                this_task = AsyncManager.run_task(coro_consume_files(abspath_or_list, (on_done_consume_inputs,)))
                handler_result = {'asynctask_id': this_task.id}
                return handler_result

            # wait until get data uploaded
            import asyncio
            loop = asyncio.get_event_loop()
            async def coro_simple_wait(timeout=None):
                while data is None:  # IMPROVE: implement timeout. maybe wait_for(this_task)
                    await asyncio.sleep(1)
            loop.run_until_complete(coro_simple_wait(timeout=None))
            pass
        else:
            raise ValueError(f"Unsupported data signature: {data_signature}")
        # TODO: consider shuffle, repeat(epoch), batch(batch_size), prefetch(1) for train/predict, use tf.data.Database
        # if isinstance(data, tf.data.Dataset):...
        return data
