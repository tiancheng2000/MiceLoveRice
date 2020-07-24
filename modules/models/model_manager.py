import enum
import os.path as osp

from helpers.util import DEBUG, INFO, WARN, ERROR, Params, safe_get_len, ensure_dir_exists, path_possibly_formatted, \
    show_image_mats, save_image_mats, safe_slice, np_top_k

from helpers import tf_helper
# TODO: move this to member functions, when supporting frameworks other than TF.
# tf_helper.preload_gpu_devices()

__all__ = [
    "ModelManager",
]

class _ModelSignature(enum.Enum):
    """
    加载/保存模型、模型权重参数用的签名。
    Model与weights/variables可能分开 e.g.Keras支持Model.load_weights('h5'或'ckpt')和models.load_model/model_from_json
    """
    # -- TF2.x ---------------------------------------
    TFSavedModel = ("tf.saved_model.load", "SavedModelDir", "Downloaded KerasApplications e.g.`inception_resnet_v2/3`")
    KerasSequential = ("keras.Sequential", "[Name, Config]", "Predefined sequential model, or by configuration")
    KerasFunctional = ("keras.Model", "Name", "Predefined model by specifying `inputs` and `outputs`, or subclassing")
    TFHub_KerasLayer = ("tf_hub.KerasLayer", "[SavedModelDir, SavedModelFile]", "")

    # -- TF1.x ---------------------------------------
    KerasApplications = ("keras.applications", "Name")
    KerasModels_LoadModel = ("keras.models.load_model", ["SavedModelDir", "SavedModelFile", "HDF5"])
    TFHub_LoadModuleSpec = ("tf_hub.load_module_spec", "SavedModelDir")
    TF_ImportGraphDef = ("tf.import_graph_def", "PB", "for TF2.x use tf.graph_util.import_graph_def")  # Protocol Buffer File
    TFTrain_LatestCKPT = ("tf.train.latest_checkpoint", "CKPT")

    def __init__(self, signature: str, formats, memo=None):
        self.signature = signature
        self.formats = formats if isinstance(formats, list) else [formats]
        self.__doc__ = memo


class ModelManager:
    @staticmethod
    def _validate_format(format_, model_signature: _ModelSignature):
        if format_ not in model_signature.formats:
            raise ValueError(f"Acceptable formats: {model_signature.formats}, while given: {format_}")
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
    def _validate_input(data) -> (object, object):
        x, y = None, None
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data  # unpack: tuple of x-y pair
            else:
                raise TypeError(f"Expect a (x, y) tuple, len=2, but len={len(data)}")
        else:
            x, y = data, None
        if x is None:
            raise TypeError(f"x cannot be None. input data={data}")

        import numpy as np
        import tensorflow as tf   # IMPROVE: check availability of ml backends
        allowed_types = (np.ndarray, tf.data.Dataset, tf.Tensor)
        if not isinstance(x, allowed_types):
            raise TypeError(f"Unsupported data type for part x(input): {type(x)}")
        if y is not None and not isinstance(y, allowed_types):
            raise TypeError(f"Unsupported data type for part y(target): {type(y)}")
        if isinstance(x, tf.data.Dataset) and y is None:  # unpack y from a ZipDataset
            # TODO: cannot deal with tf.data.Dataset yet..the following code only work on caller side?
            # # IMPROVE: unzip the ZipDataset by dataset.map(lambda). Any tf API?
            # from helpers.tf_helper import tf_data_to_np_array
            # x = tf_data_to_np_array(data.map(lambda _x, _y: _x))
            # y = tf_data_to_np_array(data.map(lambda _x, _y: _y))
            # # y.prefetch(1)
            pass
        return x, y

    @staticmethod
    def load_model(model_signature: str, **params) -> object:
        """
        NOTE: Keras常见陷阱：1.TF卷积核与Theano卷积核shape相同，加载时需用测试样本验证其表现，Keras无法区别
        :param model_signature:
        :param params:
        """
        model = None
        inputs, outputs = {}, {}   # {name: shape} dicts
        if model_signature == _ModelSignature.TFSavedModel.signature:
            import tensorflow as tf  # IMPROVE: check availability of ml backends
            # format_ = ModelManager._validate_format(params['format'], _ModelSignature.TFSavedModel)
            path = ModelManager._validate_path(params.get('path', None))
            model = tf.saved_model.load(path, params.get('tags', None))  # == core ==
            if params.get('signature_', None) is not None:
                model = model.signatures[params['signature_']]
            # TODO: append inputs, outputs spec to model object? so that predict() can adapt the fed inputs
            if hasattr(model, 'inputs') and hasattr(model, 'structured_outpus'):
                inputs = {model.inputs[0].name: model.inputs[0].shape}
                outputs = {'default': model.structured_outputs['default']}  # IMPROVE: iterate
            pass
        elif model_signature == _ModelSignature.TFHub_KerasLayer.signature:
            import tensorflow_hub as tf_hub
            # format_ = ModelManager._validate_format(params['format'], _ModelSignature.TFSavedModel)
            path = ModelManager._validate_path(params.get('path', None))
            params_model = Params(input_shape=None, trainable=False).update_to(params)
            if params_model.input_shape.__len__() == 4:
                params_model.input_shape = params_model.input_shape[1:]
            # NOTE: it will be delayed-build pattern when `input_shape` is None. no weights info available until build.
            model = tf_hub.KerasLayer(path, input_shape=params_model.input_shape)
            model.trainable = params_model.trainable
            pass
        elif model_signature == _ModelSignature.KerasSequential.signature:
            # IMPROVE: check availability of ml backends
            from tensorflow.keras import Sequential, layers
            name = params['name']
            # IMPROVE：parse name -> layers, or use structural config for iteration
            if name == '{conv-pool}*2-flat-dense-drop-dense':
                # NOTE: only for _test_\TF_1x_to_2x_3, output is len=10 logits
                model = Sequential([
                    # NOTE: 1.TF2.x已无需限定Input层的维度，甚至各层间都能自动衔接
                    #      2.Conv层中无需设定上一层的(h,w)，只需设定filter数、kernel维度、padding(使h,w保持)等
                    #      3.但若不指定input_shape，Optimizer将无法加载其之前被保存的参数，只能重新初始化
                    layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                    layers.Flatten(),  # 下面的神经网络需要1维的数据
                    layers.Dense(1024, activation='relu'),
                    layers.Dropout(0.5),  # TODO: 关闭Dropout @evluate,predict
                    layers.Dense(10, activation='softmax')
                ])
            elif name == 'dense-dense_softmax':
                params_model = Params(embedding_size=1024, class_count=None).update_to(params)
                if params_model.class_count is None:
                    raise ValueError('class_count must be specified')
                model = Sequential([
                    layers.Dense(params_model.embedding_size, activation='relu'),
                    layers.Dense(params_model.class_count, activation='softmax')
                ])
                # TODO: need to return intermediate tf.Tensor required by embedding, loss calculation and evaluation.
            else:
                raise ValueError(f"Undefined model: {name}")
            pass
        elif model_signature == _ModelSignature.KerasModels_LoadModel.signature:
            import tensorflow as tf  # IMPROVE: check availability of ml backends
            format_ = ModelManager._validate_format(params['format'], _ModelSignature.KerasModels_LoadModel)
            params_model = Params(path='', path_formatted='').update_to(params)
            path = ModelManager._validate_path(params_model.path)
            model = tf.keras.models.load_model(path)        # == core ==
        elif model_signature == _ModelSignature.TF_ImportGraphDef.signature:
            import tensorflow as tf  # IMPROVE: check availability of ml backends
            format_ = ModelManager._validate_format(params['format'], _ModelSignature.TF_ImportGraphDef)
            params_model = Params(inputs='', outputs='').update_to(params)
            path = ModelManager._validate_path(params_model.path)

            # import PB model (frozen) in TF2.x. ref:https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # ref:https://www.tensorflow.org/api_docs/python/tf/compat/v1/wrap_function
            def wrap_frozen_graph(pb_path, inputs, outputs, prefix=""):
                def _imports_graph_def():
                    tf.compat.v1.import_graph_def(graph_def, name=prefix)  # turn off the default prefix "import/"
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(open(pb_path, 'rb').read())          # == core ==
                wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])     # == core ==
                import_graph = wrapped_import.graph
                return wrapped_import.prune(
                    tf.nest.map_structure(import_graph.as_graph_element, inputs),
                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

            model = wrap_frozen_graph(path, inputs=params_model.inputs, outputs=params_model.outputs)
            test_img = tf.ones([1, 224, 224, 3], dtype=tf.float32)  # fixed shape is for test ONLY
            DEBUG(f"wrap_func test result: {model(test_img).shape}")
        else:
            raise ValueError(f"Unsupported model signature: {model_signature}")
        INFO(f"type of loaded model={type(model)}")
        INFO(f"  inputs={inputs}, outputs={outputs}")
        return model

    @staticmethod
    def model_load_weights(model: object, signature: str, **params) -> object:
        """
        NOTE: Keras常见陷阱：1.BN层载入权重的顺序应是:[gamma, beta, mean, std]，Caffe等与此不同但shape相同、Keras无法区别
        :param model:
        :param signature:
        :param params:
        """
        assert NotImplementedError()

    @staticmethod
    def model_train(model: object, data: object, **params):
        """
        NOTE: Keras常见陷阱：1.Keras先validation_split再shuffle，因此data中如果是负样本排在最后、宜自行先shuffle
        :param model:
        :param data: accept `np.ndarray`, `tf.data.Dataset` or `tf.Tensor`, or a pair of such data if y is available.
        """
        # TODO: confirm if params for tf.Model.compile can be combined with those for tf.Model.fit
        params_train = Params(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['acc'],
                              validation_split=0.1,
                              epochs=5,
                              batch_size=None,
                              checkpoint=Params(
                                  load_weights="latest",
                                  save_weights=Params(
                                      frequency="epoch",
                                      max_to_keep=5
                                  )
                              ),
                              show_result=Params()).update_to(params)
        x_train, y_train = ModelManager._validate_input(data)

        import tensorflow as tf   # IMPROVE: check availability of ml backends
        tf_helper.preload_gpu_devices()
        if isinstance(model, tf.keras.Model):
            # 1.compile and load variables from checkpoint
            model.compile(**params_train.fromkeys(['optimizer', 'loss', 'metrics']))
            # CKPT signatures: "tf.train.Checkpoint.restore", "tf.keras.Model.load_weights"
            ckpt_dir, ckpt_path_to_load = None, None
            if params_train.checkpoint.format == "CKPT_dir":
                from config import __abspath__
                ckpt_dir = path_possibly_formatted(params_train.checkpoint.path)
                ckpt_dir = __abspath__(ckpt_dir) if not osp.isabs(ckpt_dir) else ckpt_dir
                ensure_dir_exists(ckpt_dir)
                ckpt_path_to_load = tf.train.latest_checkpoint(ckpt_dir)
            # NOTE: 当使用delayed-build模式时，仅当调用build(batch_input_shape)或compile()+fit(x,y,batch_size)后才能确定weights
            #  ref:https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
            if params_train.checkpoint.load_weights == "latest" \
                    and params_train.checkpoint.signature == "tf.keras.Model.load_weights" \
                    and ckpt_path_to_load is not None:
                model.load_weights(ckpt_path_to_load)

            # 2.prepare callbacks
            callbacks = []
            # callback :: save medium CKPT
            if params_train.checkpoint.save_weights.is_defined() and ckpt_dir is not None:
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
            if params_train.early_stop.is_defined():
                _params = Params(monitor='val_loss', patience=10).left_join(params_train.early_stop)
                _callback = tf.keras.callbacks.EarlyStopping(**_params)
                callbacks.append(_callback)
            # callback :: progress indicator / verbose
            # IMPROVE: use config for training verbose / progress indicator
            # _callback = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None) # PyTest时不能实时输出
            _callback = tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: INFO(f"batch{batch:05d}: loss={logs.get('loss',None):.4f},acc={logs.get('acc',None):.4f}"))
            callbacks.append(_callback)
            cb_batch_stats = None
            if params_train.collect_batch_stats:
                # when only train several epochs, may collect stats of each batch instead of the epoch average.
                class CallbackCollectBatchStats(tf.keras.callbacks.Callback):
                    def __init__(self):
                        self.current_batch = 0
                        self.batch = []
                        self.loss = []
                        self.acc = []
                    def on_train_batch_end(self, batch, logs=None):
                        self.batch.append(self.current_batch)
                        self.loss.append(logs['loss'])
                        self.acc.append(logs['acc'])
                        self.model.reset_metrics()
                        self.current_batch += 1
                cb_batch_stats = CallbackCollectBatchStats()  # TODO: can plot batch_lsses and batch_acc using this
                callbacks.append(cb_batch_stats)
            if len(callbacks) == 0:
                callbacks = None

            # 3.train the model, and save checkpoints if configured
            # TODO: use model.fit_generator() for batch feeding. `steps_per_epoch` = np.ceil(samples / param.batch_size)
            # NOTE: core API for model training
            params_train_fit = params_train.fromkeys(['validation_split', 'batch_size', 'epochs'])
            INFO(f"Beginning to train: {params_train_fit}")
            history = model.fit(x_train, y_train, **params_train_fit, callbacks=callbacks)  # == core ==
            if cb_batch_stats is not None:
                history.history['batch'] = cb_batch_stats.batch  # accumulated batch number through epoches
                history.history['batch_loss'] = cb_batch_stats.loss
                history.history['batch_acc'] = cb_batch_stats.acc

            # 4.save checkpiont at last
            if params_train.save_model.is_defined() and ckpt_dir is not None:
                _params = Params(format="SavedModel").left_join(params_train.save_model)
                save_format, ckpt_path_to_save = None, None
                if _params.format == "HDF5":
                    save_format = _ext = "h5"
                    ckpt_path_to_save = osp.join(ckpt_dir, f"model_trained.{_ext}")
                else:  # default=SavedModel
                    save_format = "tf"
                    ckpt_path_to_save = osp.join(ckpt_dir, f"model_trained")
                    ensure_dir_exists(ckpt_path_to_save)
                # IMPROVE: consider using tf.saved_model.save()
                model.save(ckpt_path_to_save, save_format=save_format)  # by default, TF2 saves as 'tf' (SavedModel)

            # Optional: output history
            if params_train.show_result.is_defined():
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
    def model_evaluate(model: object, data: object, **params) -> object:
        eval_metrics = None
        x_test, y_test = ModelManager._validate_input(data)

        import tensorflow as tf   # IMPROVE: check availability of ml backends
        tf_helper.preload_gpu_devices()
        if isinstance(model, tf.keras.Model):
            # NOTE: core API for model evaluation
            eval_metrics = model.evaluate(x_test, y_test)
            dumps = [f"{name}={value:8.4}" for name, value in zip(model.metrics_names, eval_metrics)]
            INFO("Evaluation: " + ", ".join(dumps))
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
        return eval_metrics

    # IMPROVE: predict => infer, prediction => inference (method names and config keys)?
    @staticmethod
    def model_predict(model: object, data: object, **params) -> object:
        params_predict = Params(decode_prediction=Params(name='logits_to_index'),
                                show_result=Params(top_k=20, only_difference=True)).update_to(params)
        predictions = None
        x, y = ModelManager._validate_input(data)

        import numpy as np
        import tensorflow as tf  # IMPROVE: check availability of ml backends
        tf_helper.preload_gpu_devices()
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

        if params_predict.decode_prediction.is_defined():
            if params_predict.decode_prediction.name == 'logits_to_index':
                # one-hot array -> index
                if isinstance(predictions, np.ndarray):
                    predictions = np.argmax(predictions, axis=-1)
                elif isinstance(predictions, tf.Tensor):
                    predictions = tf.math.argmax(predictions, axis=-1)
                else:
                    raise TypeError(f"Unsupported type for logits_to_index: {type(predictions)}")
            elif params_predict.decode_prediction.name == 'logits_to_indices_and_probs':
                # for retrain, prediction should be a probs array and need to be sorted by `top_k`
                # NOTE: length of each prediction must be equivalent.
                top_k = params_predict.decode_prediction.get('top_k', safe_get_len(predictions[0]))
                # returns: top_values(=probs), top_idxs
                if isinstance(predictions, np.ndarray):
                    predictions = np_top_k(predictions, top_k)
                elif isinstance(predictions, tf.Tensor):
                    predictions = tf.math.top_k(input=predictions, k=top_k)
                else:
                    raise TypeError(f"Unsupported type for logits_to_indices_and_probs: {type(predictions)}")
            else:
                raise ValueError(
                    f"Unsupported result decoding: {params_predict.decode_prediction.name}")
        if predictions is None or safe_get_len(predictions) == 0:
            WARN("Predictions is blank (after decoding).")
            return None

        if params_predict.show_result.is_defined() and isinstance(predictions, np.ndarray):
            x_show, p_show, y_show = x, predictions, y  # NOTE: y(=label) is optional (default:None)
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
                # TODO: sorting? 1.use tf.math.top_k  2.diff algorithm need to be specified
                x_show, p_show, y_show = (safe_slice(_, end=top_k) for _ in (x_show, p_show, y_show))
            if len(p_show) > 0:
                dumps = []
                for i, p in enumerate(p_show):
                    if not hasattr(y_show, '__len__') or y_show.__len__() <= i:
                        dumps.append(f"{p}")
                    else:
                        dumps.append(f"({p} vs {y_show[i]})")
                need_to_show = params_predict.show_result.plotter.__len__() > 0
                need_to_save = params_predict.show_result.save_path.__len__() > 0
                only_save = params_predict.show_result.only_save
                if need_to_show or need_to_save:
                    def denormalize(x):
                        x = x * 255
                        if hasattr(x, 'astype'):  # np.ndarray
                            return x.astype(np.int32)
                        else:
                            return tf.cast(x, tf.int32)  # tf.Tensor
                    # IMPROVE: use signature to match normalize and `un-normalize` routines
                    if hasattr(x_show, "dtype") and x_show.dtype.name.startswith('float'):
                        x_show = denormalize(x_show)
                    elif hasattr(x_show, "element_spec") and \
                        hasattr(x_show.element_spec, "dtype") and x_show.element_spec.dtype.name.startswith('float'):
                        x_show = x_show.map(denormalize)
                save_dir, save_paths = None, None
                if need_to_save:
                    save_dir = path_possibly_formatted(params_predict.show_result.save_path)
                    # save_paths = [osp.join(save_dir, _+'.jpg') for _ in dumps]
                if params_predict.show_result.plotter == "matplot":
                    onlysave_path = None
                    if only_save:
                        if need_to_save:
                            from helpers.util import tmp_filename_by_time
                            onlysave_path = osp.join(save_dir, tmp_filename_by_time('jpg'))
                            need_to_save = False
                        else:
                            WARN('only_save is true, but save_path is not specified. ignored')
                    show_image_mats(x_show, texts=dumps, title="Predictions", onlysave_path=onlysave_path)
                else:
                    INFO(f"Predictions{'(only diff)' if 'differences' in vars() else ''}: " + ", ".join(dumps))
                # if need_to_save:
                #     save_image_mats(x_show, save_paths)
        else:
            top_k = params_predict.show_result.top_k
            INFO(f"Predictions(top{top_k}): {safe_slice(predictions, end=top_k)}")
        return predictions
