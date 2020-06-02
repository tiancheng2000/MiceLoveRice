import enum
import os.path as osp

import tensorflow as tf
# IMPROVE: compatibility with TF1.x
# if tf.version.VERSION.startswith('1.'):
#     import tensorflow as tf1
# else:  # tf v2.x
#     import tensorflow.compat.v1 as tf1
#     tf1.disable_v2_behavior()

from helpers.util import DEBUG, INFO, WARN, ERROR, Params, safe_get_len, ensure_dir_exists, path_possibly_formatted, \
    show_image_mats, save_image_mats
from helpers.tf_helper import preload_gpu_devices

__all__ = [
    "ModelManager",
]

preload_gpu_devices()

class _ModelSignature(enum.Enum):
    """
    加载/保存模型、模型权重参数用的签名。
    Model与weights/variables可能分开 e.g.Keras支持Model.load_weights('h5'或'ckpt')和models.load_model/model_from_json
    """
    # -- TF2.x ---------------------------------------
    TFSavedModel = ("tf.saved_model.load", "SavedModelDir", "Downloaded KerasApplications e.g.`inception_resnet_v2/3`")
    KerasSequential = ("keras.Sequential", "[Name, Config]", "Predefined sequential model, or by configuration")
    KerasFunctional = ("keras.Model", "Name", "Predefined model by specifying `inputs` and `outputs`, or subclassing")

    # -- TF1.x ---------------------------------------
    KerasApplications = ("keras.applications", "Name")
    KerasModels_LoadModel = ("keras.models.load_model", ["SavedModelDir", "SavedModelFile", "HDF5"])
    TFHub_LoadModuleSpec = ("tf_hub.load_module_spec", "SavedModelDir")
    TF_ImportGraphDef = ("tf.import_graph_def", "PB", "for TF2.x use tf.graph_util.import_graph_def")  # Protocol Buffer File
    TFTrain_LatestCKPT = ("tf.train.latest_checkpoint", "CKPT")

    def __init__(self, signature: str, formats, comment=None):
        self.signature = signature
        self.formats = formats if isinstance(formats, list) else [formats]
        self.__doc__ = comment


class ModelManager:
    @staticmethod
    def _validate_format(format_, model_signature: _ModelSignature):
        if format_ not in model_signature.formats:
            raise ValueError(f"Acceptable formats: {model_signature.formats}, while given: {format_}")
        return format_

    @staticmethod
    def _validate_path(path):
        path = path_possibly_formatted(path)
        if not osp.exists(path):
            raise ValueError(f"Given path is invalid: {path}")
        return path

    @staticmethod
    def load_model(model_signature: str, **params) -> object:
        """
        NOTE: Keras常见陷阱：1.TF卷积核与Theano卷积核shape相同，加载时需用测试样本验证其表现，Keras无法区别
        :param model_signature:
        :param params:
        """
        model = None
        if model_signature == _ModelSignature.TFSavedModel.signature:
            # format_ = ModelManager._validate_format(params['format'], _ModelSignature.TFSavedModel)
            path = ModelManager._validate_path(params.get('path', None))
            model = tf.saved_model.load(path, params.get('tags', None))
            if params.get('signature_', None) is not None:
                model = model.signatures[params['signature_']]
            output_shape = model.structured_outputs['default']
            DEBUG(f"loaded model={model}")
            DEBUG(f"  output_shape={output_shape}")
            pass
        elif model_signature == _ModelSignature.KerasSequential.signature:
            name = params['name']
            if name == '{conv-pool}*2-flat-dense-drop-dense':
                # origin: TF_1x_to_2x_3
                model = tf.keras.Sequential([
                    # NOTE: 1.TF2.x已无需限定Input层的维度，甚至各层间都能自动衔接
                    #      2.Conv层中无需设定上一层的(h,w)，只需设定filter数、kernel维度、padding(使h,w保持)等
                    #      3.但若不指定input_shape，Optimizer将无法加载其之前被保存的参数，只能重新初始化
                    tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    tf.keras.layers.Flatten(),  # 下面的神经网络需要1维的数据
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.Dropout(0.5),  # TODO: 关闭Dropout @evluate,predict
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
            else:
                raise ValueError(f"Undefined model: {name}")
            pass
        elif model_signature == _ModelSignature.KerasModels_LoadModel.signature:
            format_ = ModelManager._validate_format(params['format'], _ModelSignature.KerasModels_LoadModel)
            params_model = Params(path='', path_formatted='').update_to(params)
            path = ModelManager._validate_path(params_model.path)
            model = tf.keras.models.load_model(path)
            # DEBUG(f"loaded model={model}")
        elif model_signature == _ModelSignature.TF_ImportGraphDef.signature:
            format_ = ModelManager._validate_format(params['format'], _ModelSignature.TF_ImportGraphDef)
            params_model = Params(inputs='', outputs='').update_to(params)
            path = ModelManager._validate_path(params_model.path)

            # import PB model (frozen) in TF2.x. ref:https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # ref:https://www.tensorflow.org/api_docs/python/tf/compat/v1/wrap_function
            def wrap_frozen_graph(pb_path, inputs, outputs, prefix=""):
                def _imports_graph_def():
                    tf.compat.v1.import_graph_def(graph_def, name=prefix)  # turn off the default prefix "import/"
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(open(pb_path, 'rb').read())
                wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
                import_graph = wrapped_import.graph
                return wrapped_import.prune(
                    tf.nest.map_structure(import_graph.as_graph_element, inputs),
                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

            model = wrap_frozen_graph(path, inputs=params_model.inputs, outputs=params_model.outputs)
            test_img = tf.ones([1, 224, 224, 3], dtype=tf.float32)
            DEBUG(f"wrap_func test result: {model(test_img).shape}")
        else:
            raise ValueError(f"Unsupported model signature: {model_signature}")
        return model

    @staticmethod
    def model_load_weights(model: object, signature: str, **params) -> object:
        """
        NOTE: Keras常见陷阱：1.BN层载入权重的顺序应是:[gamma, beta, mean, std]，Caffe等与此不同但shape相同、Keras无法区别
        :param model:
        :param signature:
        :param params:
        """
        return model

    @staticmethod
    def model_train(model: object, data: tuple or tf.data.Dataset, **params):
        """
        NOTE: Keras常见陷阱：1.Keras先validation_split再shuffle，因此data中如果是负样本排在最后、宜自行先shuffle
        :param model:
        :param data:
        """
        # TODO: confirm if params for tf.Model.compile can be combined with those for tf.Model.fit
        params_train = Params(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'],
                              epochs=5,
                              checkpoint=Params(
                                  load_weights="latest",
                                  save_weights=Params(
                                      frequency="epoch",
                                      max_to_keep=5
                                  )
                              ),
                              show_result=Params()).update_to(params)
        x_train, y_train = None, None
        if isinstance(data, tuple) and len(data) == 2:
            if [isinstance(_, tuple) and len(_) == 2 for _ in data] == [True, True]:
                (x_train, y_train), (x_test, y_test) = data  # unpack: tuple of 4 np.ndarrays
            else:
                x_train, y_train = data  # unpack: tuple of 2 np.ndarrays
        else:
            import numpy as np
            if isinstance(data, np.ndarray):  # only x np array
                x_train, y_train = data, None
        if x_train is None:
            raise TypeError(f"Unsupported data type: {type(data)}")
        if isinstance(model, tf.keras.Model):
            # 1.compile and load variables from checkpoint
            model.compile(**params_train.fromkeys(['optimizer', 'loss', 'metrics']))
            # CKPT signatures: "tf.train.Checkpoint.restore", "tf.keras.Model.load_weights"
            ckpt_dir, ckpt_path_to_load = None, None
            if params_train.checkpoint.format == "CKPT_dir":
                ckpt_dir = path_possibly_formatted(params_train.checkpoint.path)
                ensure_dir_exists(ckpt_dir)
                ckpt_path_to_load = tf.train.latest_checkpoint(ckpt_dir)
            if params_train.checkpoint.load_weights == "latest" \
                    and params_train.checkpoint.signature == "tf.keras.Model.load_weights" \
                    and ckpt_path_to_load is not None:
                model.load_weights(ckpt_path_to_load)

            # 2.prepare callbacks
            callbacks = []
            # callback :: save medium CKPT
            if params_train.checkpoint.save_weights.has_attr() and ckpt_dir is not None:
                ckpt_path_to_save = osp.join(ckpt_dir, "ckpt.{epoch:02d}-{val_loss:.2f}")
                # NOTE: if save_freq is not equal to 'epoch', which means num of steps, it's will be less reliable
                _params = Params(save_freq='epoch').left_join(params_train.checkpoint.save_weights,
                                                              key_map={"save_freq": "frequency"})
                _callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path_to_save,  # not checkpoint_dir
                                                               save_weights_only=True,
                                                               save_best_only=True,
                                                               verbose=1, **_params)
                callbacks.append(_callback)
            # callback :: early stop
            if params_train.early_stop.has_attr():
                _params = Params(monitor='val_loss', patience=10).left_join(params_train.early_stop)
                _callback = tf.keras.callbacks.EarlyStopping(**_params)
                callbacks.append(_callback)
            # callback :: progress indicator
            # _callback = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None) # PyTest时不能实时输出
            _callback = tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: INFO(f"batch{batch:05d}: loss={logs['loss']:.4f},acc={logs['acc']:.4f}"))
            callbacks.append(_callback)

            # 3.train the model, and save checkpoints if configured
            # NOTE: core API for model training
            history = model.fit(x_train, y_train, **params_train.fromkeys(['epochs', 'validation_split']),
                                callbacks=callbacks if len(callbacks) > 0 else None)

            # 4.save checkpiont at last
            if params_train.save_model.has_attr() and ckpt_dir is not None:
                _params = Params(format="SavedModel").left_join(params_train.save_model)
                _ext = "tf" if _params.format != "HDF5" else "h5"
                ckpt_path_to_save = osp.join(ckpt_dir, f"model_trained.{_ext}")
                model.save(ckpt_path_to_save, save_format=_ext)

            # Optional: output history
            if params_train.show_result.__len__() > 0:
                plot_history = None
                if params_train.show_result.plotter == 'matplot':
                    from helpers.plt_helper import plot_history_by_metrics as plot_history
                if params_train.show_result.plotter.__len__() > 0 and plot_history is None:
                    WARN(f"Unsupported history plotter: {params_train.show_result.plotter}")
                if plot_history is not None:
                    plot_history(history, params_train.show_result.get('metrics', None))
                else:
                    # TODO: check this section
                    hist = history.history
                    INFO(f"Last epoch: "
                         f"ACC(train,val)=({hist['accuracy'][-1]}, {hist['val_accuracy'][-1]}), "
                         f"MSE(train,val)=({hist['mse'][-1]}, {hist['val_mse'][-1]})")
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        return model

    @staticmethod
    def model_evaluate(model: object, data: tuple or tf.data.Dataset, **params) -> object:
        eval_metrics = None
        if isinstance(data, tuple):
            x_test, y_test = data  # unpack: tuple of 2 np.ndarrays
        else:
            import numpy as np
            if isinstance(data, np.ndarray):
                x_test, y_test = data, None
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")
        if isinstance(model, tf.keras.Model):
            # NOTE: core API for model evaluation
            eval_metrics = model.evaluate(x_test, y_test)
            dumps = [f"{name}={value:8.4}" for name, value in zip(model.metrics_names, eval_metrics)]
            INFO("Evaluation: " + ", ".join(dumps))
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
        return eval_metrics

    # IMPROVE: predict -> infer, prediction -> inference (method names and config keys)
    @staticmethod
    def model_predict(model: object, x, y=None, **params) -> object:
        params_predict = Params(decode_prediction=Params(),
                                show_result=Params(top_k=20, only_difference=True)).update_to(params)
        predictions = None
        import numpy as np
        if not (isinstance(x, (np.ndarray, tf.data.Dataset, tf.Tensor))):
            raise TypeError(f"Unsupported input data type: {type(x)}")
        if y is not None and not isinstance(y, np.ndarray):
            raise TypeError(f"Unsupported target data type: {type(y)}")
        if isinstance(x, tf.data.Dataset) and y is None:  # unpack possible y
            # TODO: cannot deal with tf.data.Dataset yet..the following code only work on caller side?
            # from helpers.tf_helper import tf_data_to_np_array
            # # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API for unzip?
            # x = tf_data_to_np_array(data.map(lambda _x, _y: _x))
            # y = tf_data_to_np_array(data.map(lambda _x, _y: _y))
            # # y.prefetch(1)
            pass

        # wrapper for different model types
        def _predict(inputs):
            # NOTE: core API for prediction
            if isinstance(model, tf.keras.Model):
                # NOTE: if x is ndarray, result will be ndarray too
                return model.predict(inputs)
            elif callable(model):
                # type(model).__name__ == "tensorflow.python.eager.wrap_function.WrappedFunction"
                if isinstance(inputs, tf.data.Dataset):
                    # IMPROVE: stack result as a tensor
                    result = []
                    for t in inputs:
                        result.append(model(t))
                    return tf.stack(result)
                else:
                    return model(inputs)
            else:
                raise TypeError(f"Unsupported model type: {type(model)}")

        predictions = _predict(x)
        if predictions is None or safe_get_len(predictions) == 0:
            WARN("Predictions is blank.")
            return None

        if params_predict.decode_prediction.__len__() > 0:
            if params_predict.decode_prediction.name == 'logits_to_index':
                # one-hot array -> index
                if isinstance(predictions, np.ndarray):
                    predictions = np.argmax(predictions, 1)
                elif isinstance(predictions, tf.Tensor):
                    predictions = tf.math.argmax(predictions, 1)
                else:
                    raise TypeError(f"Unsupported type for logits_to_index: {type(predictions)}")
            else:
                raise ValueError(
                    f"Unsupported result decoding: {params_predict.decode_prediction.name}")
        if predictions is None or safe_get_len(predictions) == 0:
            WARN("Predictions is blank (after decoding).")
            return None

        if params_predict.show_result.__len__() > 0 and isinstance(predictions, np.ndarray):
            x_show, p_show, y_show = x, predictions, y  # NOTE: y is optional (default:None)
            if params_predict.show_result.only_difference:
                if hasattr(y_show, '__len__'):
                    if p_show.__len__() == y_show.__len__():
                        differences = p_show != y_show
                        x_show, p_show, y_show = x_show[differences], p_show[differences], y_show[differences]
                    else:
                        WARN(f"Cannot dump differences: len of targets is not same as predictions"
                             f"({y_show.__len__()} vs {p_show.__len__()})")
                else:
                    WARN(f"Cannot dump differences: unsupported y type(={type(y_show)})")
                INFO(f"Number of mismatch between prediction and truth: {len(p_show)}")
            if params_predict.show_result.get('top_k', None) is not None:
                top_k = params_predict.show_result.top_k
                x_show, p_show, y_show = x_show[:top_k], p_show[:top_k], y_show[:top_k]
            if len(p_show) > 0:
                dumps = []
                for i, p in enumerate(p_show):
                    if not hasattr(y_show, '__len__') or y_show.__len__() <= i:
                        dumps.append(f"{p}")
                    else:
                        dumps.append(f"({p} vs {y_show[i]})")
                need_to_show_or_save = (params_predict.show_result.plotter.__len__() > 0) \
                                       or (params_predict.show_result.save_path.__len__() > 0)
                if need_to_show_or_save:
                    if hasattr(x_show, "dtype") and x_show.dtype.name.startswith('float'):
                        # IMPROVE: use signature to match normalize and `un-normalize` routines
                        x_show = x_show * 255
                        x_show.astype(np.int32)
                if params_predict.show_result.plotter == "matplot":
                    show_image_mats(x_show, dumps)
                else:
                    INFO(f"Predictions{'(only diff)' if 'differences' in vars() else ''}: " + ", ".join(dumps))
                if params_predict.show_result.save_path.__len__() > 0:
                    save_dir = path_possibly_formatted(params_predict.show_result.save_path)
                    save_paths = [osp.join(save_dir, _+'.jpg') for _ in dumps]
                    save_image_mats(x_show, save_paths)
        else:
            INFO(f"Predictions(top): {predictions[0]}")
        return predictions
