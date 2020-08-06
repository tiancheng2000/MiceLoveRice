from __future__ import absolute_import, division, print_function

import logging
import enum
import contextlib
import json
import os
import os.path as osp
import sys
import time
import functools
from contextlib import ContextDecorator
# ABCs (from collections.abc), Super-special typing primitives, Concrete collection types
from typing import Iterable, Iterator, Mapping, Hashable, Sequence, Sized, Awaitable, Collection, ContextManager, \
    Any, Optional, Callable, ClassVar, Type, TypeVar, Tuple, \
    Dict, OrderedDict, List, Set, Generator

__all__ = [
    # -- Output :: Logging ------------
    "init_logging",
    "VerbosityLevel",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "whereami",
    "dump_iterable_data",
    "track_entry_and_exit",
    # -- Structure Design -------------
    "singleton",
    "classproperty",
    # -- Input :: Config --------------
    "Params",
    "ConfigSerializer",
    # -- Input :: User Interactive --------------
    "adjust_interrupt_handlers",
    "ensure_web_app",
    # -- Libraries, Packages ---------------
    "safe_import_module",
    # -- File Operations --------------
    "ensure_dir_exists",
    "tmp_filename_by_time",
    "tmp_filename_by_uuid",
    "get_new_name_if_exists",
    "show_image_mat",
    "load_image_mat",
    "save_image_mat",
    "show_image_mats",
    "load_image_mats",
    "save_image_mats",
    "coro_show_image_mat",
    "coro_show_image_mats",
    "async_show_image_mat",
    "async_show_image_mats",
    "cache_object",
    "path_possibly_formatted",
    "path_regulate_to_slash",
    "is_abs_path",
    "walk",
    # -- Data Process ---------------
    "urlsafe_uuid",
    "dump_to_json",
    "safe_get_len",
    "safe_slice",
    "np_top_k",
    "dict_compare",
    "dict_left_join",
    "dict_cross_join",
    "dict_update",
    "indices_in_vocabulary_list",
    "safe_get",
    "hasmethod",
    # -- Thread Process ---------------
    "print_time_consumed",
    # -- Quick Test -------------------
    "quick_load_image_tensor",
    "quick_load_imagenet_labels",
]

# -----------------------------------------------------#
# -- Output :: Logging ------------
class VerbosityLevel(enum.Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR


__logger__ = logging.getLogger()  # default logger

# NOTE: if you use pytest, log_level should be configured in a pytest.ini located in project root dir
def init_logging(log_level: VerbosityLevel = VerbosityLevel.INFO, log_path=None, logger_name=__name__):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    logging.warn("...")
    logging.error("...")
    ```

    :param log_path: (string) where to log
    :param log_level: VerbosityLevel, it will be mapped to logging levels (integers)
    :param logger_name: logger name
    """
    global __logger__
    __logger__ = logging.getLogger(logger_name)
    __logger__.setLevel(log_level.value)

    if not __logger__.handlers:
        # Logging to a file
        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:[%(levelname)-5s]%(message)s'))
            __logger__.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(levelname)-5s]%(message)s'))
        __logger__.addHandler(stream_handler)

    __logger__.info("Logging initialized.")

# origins: logging module
def whereami(back=1, filename_only=True, return_dict=False):
    """
    :return: filepath, lineno, funcname, stack_info(not implemented) of caller
    """
    f = sys._getframe(0)
    rv = "(unknown file)", 0, "(unknown function)", None
    while hasattr(f, "f_code"):
        if f.f_code.co_filename == __file__:
            for i in range(back):
                f = f.f_back
            rv = (os.path.basename(f.f_code.co_filename) if filename_only else f.f_code.co_filename,
                  f.f_lineno, f.f_code.co_name, None)
            break
        else:
            f = f.f_back
    return rv if not return_dict else {'file': rv[0], 'line_num': rv[1], 'func_name': rv[2], 'stack_info': rv[3]}

# IMPROVE: consider using pprint lib
def dump_iterable_data(data: Any, indent=1):
    """
    Suppose sub-items in an Iterable have same data type -- Iterable or not
    """
    if not isinstance(data, Iterable):
        return str(data)
    # NOTE: scalar Tensor "isinstance" Iterable but will throw exception when asking for an Iterator
    try:
        for item in data:
            break
    except (TypeError, RuntimeError):
        return f"{type(data).__name__}{f'({data.dtype.name})' if hasattr(data,'dtype') else ''})"

    dump_items = []
    for item in data:
        if isinstance(item, Iterable):
            dump_item = dump_iterable_data(item, indent=indent+2)
        else:
            dump_item = f"{type(item).__name__}"
        # IMPROVE: ZipDataset with `element_spec` in type of `tuple` can be extended to its element dim
        if not isinstance(data, Tuple):
            dump_items.append(f"{safe_get_len(data)}* {dump_item}")
            break
        else:
            dump_items.append(dump_item)
    dump_indent = '' if len(dump_items) < 2 else '\n'+' '*indent
    return f"{type(data).__name__}({dump_indent}{str(','+dump_indent).join(dump_items)})"

def __log__(msg, level: VerbosityLevel, tag=None):
    global __logger__
    if __logger__ is not None:
        filename, lineno, funcname, stack_info = __logger__.findCaller()
        filename = os.path.basename(filename)
        # IMPROVE: trace true caller ==> send caller.filename through `extra={'filename': filename}`
        #   ==> subclass Logger, overwrite makeRecord() to allow overwrite `filename`
        __logger__.log(level.value, msg='%s %s (%s, %s:%s)' % (tag+':' if tag is not None else '', msg,
                                                               funcname, filename, lineno))
        # extra={'filename_ex': filename, 'lineno_ex': lineno, 'funcName_ex': funcname, 'stack_info_ex': stack_info})

def DEBUG(msg, tag=None):
    __log__(msg, VerbosityLevel.DEBUG, tag=tag)

def INFO(msg, tag=None):
    __log__(msg, VerbosityLevel.INFO, tag=tag)

def WARN(msg, tag=None):
    __log__(msg, VerbosityLevel.WARN, tag=tag)

def ERROR(msg, tag=None):
    __log__(msg, VerbosityLevel.ERROR, tag=tag)


# NOTE: hack ContextDecorator to support both regular and async functions
# class track_entry_and_exit(ContextDecorator):
class track_entry_and_exit:
    """
    Debug level logging helper. Support both regular and async functions:
    @track_entry_and_exit.coro()
    async def activity():
        print('Some time consuming activity goes here')
        load_widget()
    @track_entry_and_exit()
    def activity():
        print('Some time consuming activity goes here')
        load_widget()
    """

    class coro:
        def _recreate_cm(self, func):
            self.func_name = func.__name__
            return self

        def __call__(self, func):
            """
            :param func: an async(coroutine) function
            :return:
            """
            @functools.wraps(func)
            async def inner(*args, **kwargs):
                with self._recreate_cm(func=func):
                    return await func(*args, **kwargs)
            return inner

        def __enter__(self):
            DEBUG(f'Async Entering: {self.func_name}')

        def __exit__(self, exc_type, exc, exc_tb):
            DEBUG(f'Async Exiting: {self.func_name}')


    def _recreate_cm(self, func):
        self.func_name = func.__name__
        return self

    def __call__(self, func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func=func):
                return func(*args, **kwargs)
        return inner

    # def __init__(self, func_name=None):
    #     self.func_name = whereami(back=3, return_dict=True)['func_name'] if func_name is None else func_name

    def __enter__(self):
        DEBUG(f'Entering: {self.func_name}')

    def __exit__(self, exc_type, exc, exc_tb):
        DEBUG(f'Exiting: {self.func_name}')



# -----------------------------------------------------#
# -- Structure Design -------------
# decorator
def singleton(cls):
    """ Sample use:
    @singleton
    class Foo:
        def __new__(cls):
            cls.x = 10
            return object.__new__(cls)

        def __init__(self):
            assert self.x == 10
            self.x = 15

    assert Foo().x == 15
    Foo().x = 20
    assert Foo().x == 20
    """
    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)  # 使得被decorator修饰的类，仍能通过__name__正确获取自身名字
    def singleton_new(cls, *args, **kw):
        it = cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__

    return cls  # decorator

# decorator
class classproperty(property):
    """ modifiable class level property

        @classproperty
        def property_a(cls): return cls._property_a

        @property_a.setter
        def property_a(cls, value): cls._property_a = value
    """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()

# TODO: a decorator for suppressing exceptions on helper functions
# decorator

# -----------------------------------------------------#
# -- Input :: Config --------------
def _save_dict_to_json(dict_, json_path):
    # TODO: not completely implemented
    with open(json_path, 'w', encoding='utf-8') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        dict_ = {k: float(v) for k, v in dict_.items()}
        json.dump(dict_, f, indent=4)


class Params(dict):
    __class_initialized__ = False
    _key_missed_value = object()  # NOTE: will be initialized in first call to __init__

    def __init__(self, seq=None, **kwargs):
        if seq is not None:
            super(self.__class__, self).__init__(seq)
        else:
            super(self.__class__, self).__init__(**kwargs)
        if not self.__class__.__class_initialized__:
            self.__class__.__class_initialized__ = True
            self.__class__._key_missed_value = Params({})  # sentinel, allows check validity by `is_defined()`

    def __getattr__(self, attr_name):
        # NOTE: only called when has not found the attribute
        return self[attr_name]

    def __setattr__(self, attr_name, attr_value):
        self[attr_name] = attr_value

    def __str__(self):
        return '' if len(self) == 0 else super(self.__class__, self).__str__()

    def __missing__(self, key):
        # NOTE: if Params[key] missed, returns this value instead of raising KeyError (or IndexError)
        return Params._key_missed_value

    def is_defined(self):
        """
        NOTE: only use this checking a :class:`Params` node. For leaf nodes (str, list)
          :func:`__len__()` or `isinstance(node, type)` is suggested
        """
        return len(self) > 0

    def get(self, key, *args):
        # deprecated: return super(Params, self).get(key, default if default is not None else {})
        default = args[0] if len(args) > 0 else Params._key_missed_value
        return super(self.__class__, self).get(key, default)

    def left_join(self, other: dict, key_map: dict = None, **others):
        """
        Similar to `update_to()` except that only keys defined in left sides (this instance)
        will have their values updated. No extra key-value pairs will be appended.
        :param other:
        :param key_map: dict of key_left -> key_right, for key translation
        """
        other = other.copy() if other else None
        others = others.copy() if other else None
        key_map = {} if key_map is None else key_map
        for key, value in self.items():
            # 1.key translation and replace mapped keys of other and others
            key_right = key_map.get(key, None)
            if key_right in other and key not in other:
                other[key] = other[key_right]
                del other[key_right]
            if key_right in others and key not in others:
                others[key] = others[key_right]
                del others[key_right]
            value_in_other = other.get(key, None)
            value_in_others = others.get(key, None)
            # 2.recursively update if node is a Params object
            if isinstance(value, self.__class__):
                value.left_join(value_in_other or {}, key_map, **(value_in_others or {}))
                self[key] = value
            else:
                if value_in_others is not None:
                    self[key] = value_in_others
                elif value_in_other is not None:
                    self[key] = value_in_other
        return self

    # TODO: support recursive processing, like left_join
    def cross_join(self, other: dict, key_map: dict = None, **others):
        """
        Similar to `update_to()` except that only keys defined in both sides
        will have their values updated. Those not defined in right side will be deleted.
        :param other:
        :param key_map: dict of key_left -> key_right, for key translation
        """
        to_delete = []
        for key in self.keys():
            if key_map is not None and key in key_map:
                key_right = key_map.get(key)  # key translation
            else:
                key_right = key
            if key in others:
                self.__setitem__(key, others.get(key))
            elif key_right in others:
                self.__setitem__(key, others.get(key_right))
            elif key in other:
                self.__setitem__(key, other.get(key))
            elif key_right in other:
                self.__setitem__(key, other.get(key_right))
            else:
                to_delete.append(key)
        for key in to_delete:
            self.__delitem__(key)
        return self

    def update(self, other: dict, key_map: dict = None, **others):
        # NOTE: dict.update() always returns `None`, so should not be used in assignment
        other = other.copy() if other else {}
        others = others.copy() if others else {}
        key_map = {} if key_map is None else key_map
        for key, value in self.items():
            # 1.key translation and replace mapped keys of other and others
            key_right = key_map.get(key, None)
            if key_right in other and key not in other:
                other[key] = other[key_right]
                del other[key_right]
            if key_right in others and key not in others:
                others[key] = others[key_right]
                del others[key_right]
            # 2.recursively update if node is a Params object
            if isinstance(value, self.__class__):
                value_in_other = other[key] if key in other else {}
                value_in_others = others[key] if key in others else {}
                value.update(value_in_other, key_map, **value_in_others)
                self[key] = value
                if key in other:
                    del other[key]
                if key in others:
                    del others[key]
        super(self.__class__, self).update(other, **others)
        return None  # always return `None` as dict's behavior

    # -- deprecated --
    def update_v1(self, other: dict, key_map: dict = None, **others):
        # NOTE: dict.update() always returns `None`, so should not be used in assignment
        if key_map is not None:
            other = other.copy()
            others = others.copy()
            for key_left, key_right in key_map.items():
                if key_right in other:
                    other.__setitem__(key_left, other.get(key_right))
                    del other[key_right]
                if key_right in others:
                    others.__setitem__(key_left, others.get(key_right))
                    del others[key_right]
        return super(self.__class__, self).update(other, **others)  # always return `None` as dict's behavior

    def update_to(self, other: dict, key_map: dict = None, **others):
        """
        Not like `update()` (which always return `None`), can be used in assignment.
        Replace values of this Params instance with values in `other` and `**others`, if exists.
        And then return this instance.
        :param other: dict or another Params object used to specify key-value pairs
        :param key_map: map local key names to possible aliases
        :param others: kwargs used to specify other key-value pairs
        :return: the Params instance when calling this method, with its values updated.
        """
        self.update(other, key_map=key_map, **others)
        return self

    def fromkeys(self, keys: Iterable[str], key_map: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Create a new Params object with keys specified in `keys` (or translate to its alias defined
        in `key_map` if available), and values as mapped to these keys in this instance (if available)
        :param keys:
        :param key_map:
        :return: a new Params instance
        """
        params = Params()
        for key in keys:
            if key_map is not None and key in key_map:
                key = key_map.get(key)  # key translation
            if key not in self.keys():
                continue
            params.__setitem__(key, self.get(key))
        return params


def _recursive_replace_dict_attrs(dict_: dict, to_cls=Params):
    for key, value in dict_.items():
        if isinstance(value, dict):
            _recursive_replace_dict_attrs(value, to_cls)
            obj_attr = to_cls(value) if to_cls is not dict else value
            # obj_attr.__dict__ = value
            dict_[key] = obj_attr

class ConfigSerializer:
    """Class that loads/saves dictionary style ConfigObject from/to a json file.

    Example:
    ```
    params = ConfigSerializer.load(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    @staticmethod
    def save(dict_, json_path):
        _save_dict_to_json(dict_, json_path)

    @staticmethod
    def load(json_path) -> Params:
        dict_to = Params()
        with open(json_path, encoding='utf-8') as f:
            # TODO: use parse_constant param to parse 'None' (used in tf.Shape)
            dict_from = json.load(f, parse_constant=None)
            _recursive_replace_dict_attrs(dict_from, to_cls=Params)
            dict_to = Params(dict_from)
        return dict_to


# -----------------------------------------------------#
# -- Input :: User Interactive ---------------
# NOTE: If you have imported Scipy or alike, Interrupt Handler might have been injected.
def adjust_interrupt_handlers():
    import os
    import imp
    import ctypes
    import _thread
    import win32api

    # Load the DLL manually to ensure its handler gets
    # set before our handler.
    basepath = imp.find_module('numpy')[1]

    def try_to_load(dll_path):
        try:
            ctypes.CDLL(dll_path)
        except OSError as e:
            pass
    try_to_load(os.path.join(basepath, 'core', 'libmmd.dll'))
    try_to_load(os.path.join(basepath, 'core', 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event
    # types will chain to the next handler.
    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler

    win32api.SetConsoleCtrlHandler(handler, 1)

def ensure_web_app():
    from config import Path, __abspath__
    import os.path as osp
    config_deploy = ConfigSerializer.load(Path.DeployConfigAbs)
    params_webapp = Params(upload_folder=None).cross_join(config_deploy.web)
    # NOTE: relative path should relate to project root, not webapp's
    if not osp.isabs(params_webapp.upload_folder):
        params_webapp.upload_folder = __abspath__(params_webapp.upload_folder)

    from web import get_webapp
    webapp = get_webapp(**params_webapp)

    params_webapp_run = Params(host="127.0.0.1", port="2020", ssl_context=None) \
        .left_join(config_deploy.web, {"host": "local_ip", "port": "local_port"})
    if config_deploy.web.use_https:
        params_webapp_run.ssl_context = (config_deploy.web.certfile_path, config_deploy.web.keyfile_path)
    webapp.async_run(**params_webapp_run)
    return webapp


# -----------------------------------------------------#
# -- Thread Process ---------------
@contextlib.contextmanager
def print_time_consumed(format_: str = "[INFO] time consumed: {:.5f}s", file=sys.stdout):
    """
    Usage: `with print_time_consumed(): ...`
    """
    time_begin = time.time()
    yield
    time_now = time.time()
    print(format_.format(time_now - time_begin), file=file)

# -----------------------------------------------------#
# -- Libraries, Packages ---------------
def safe_import_module(name):
    module = None
    try:
        from importlib import import_module
        module = import_module(name)
    except ModuleNotFoundError as e:
        WARN(f'failed in trying to load the module {name}')
    return module

# -----------------------------------------------------#
# -- File Operations --------------
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def tmp_filename_by_time(file_ext=None):
    # IMPROVE: for milliseconds precision:
    #  t = time.time(); ms = int(round(t * 1000)); us = int(round(t * 1000000))
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + f".{file_ext}" if file_ext is not None else ""

def tmp_filename_by_uuid(file_ext=None, method="uuid4"):
    return urlsafe_uuid(method) + ".{}".format(file_ext) if file_ext is not None else ""

def get_new_name_if_exists(path, method="auto_inc", format_=" ({:d})", max_auto_inc=99999):
    path_splitted = osp.splitext(path)
    auto_inc = 1
    path_test = path
    while osp.exists(path_test) and auto_inc <= max_auto_inc:
        path_test = path_splitted[0] + format_.format(auto_inc) + path_splitted[1]
        auto_inc += 1
    if auto_inc > max_auto_inc:
        raise SystemError(f"Auto-incremental postfix exceeds its limit: {max_auto_inc}")
    return path_test

def show_image_mat(image_mat, text=None, title=None, cell_size: tuple = None, block=None, onlysave_path=None):
    if any([type(image_mat).__name__.endswith(_) for _ in ('Tensor', 'Dataset')]):
        from helpers.tf_helper import tf_obj_to_np_array
        image_mat = tf_obj_to_np_array(image_mat)
    if image_mat.shape.__len__() == 4:
        image_mat = image_mat[0]
    from helpers.plt_helper import plot_image_mat as plt_show_image_mat
    return plt_show_image_mat(image_mat, text, title, cell_size, block, onlysave_path)

def load_image_mat(image_path, format_=None):
    from helpers.plt_helper import load_image_mat as plt_load_image_mat
    return plt_load_image_mat(image_path, format_)

def save_image_mat(image_mat, image_path, **kwargs):
    if any([type(image_mat).__name__.endswith(_) for _ in ('Tensor', 'Dataset')]):
        from helpers.tf_helper import tf_obj_to_np_array
        image_mat = tf_obj_to_np_array(image_mat)
    if image_mat.shape.__len__() == 4:
        image_mat = image_mat[0]
    from helpers.plt_helper import save_image_mat as plt_save_image_mat
    plt_save_image_mat(image_mat, image_path, **kwargs)

def show_image_mats(image_mats, texts=None, title=None, num_rows=None, num_cols=None, cell_size: tuple = None,
                    block=None, onlysave_path=None):
    """
    :param image_mats: tf.Tensor or np.ndarray, in shape of [b,h,w,c]. Or enumerable of such elements.
    :param texts:
    :param title:
    :param num_rows:
    :param num_cols:
    :param cell_size:
    :param block:
    :param onlysave_path: if specified, save the figure and do not show
    """
    if any([type(image_mats).__name__.endswith(_) for _ in ('Tensor', 'Dataset')]):
        from helpers.tf_helper import tf_obj_to_np_array
        image_mats = tf_obj_to_np_array(image_mats)
    from helpers.plt_helper import plot_images as plt_show_image_mats
    return plt_show_image_mats(image_mats, texts, title, num_rows, num_cols, cell_size, block, onlysave_path)

def load_image_mats(root_path, format_=None):
    import glob
    import numpy as np
    format_ = "jpg" if format_ is None else format_
    file_ext = format_
    glob_pattern = osp.join(root_path, f"*/*.{file_ext}")
    image_paths = glob.glob(glob_pattern)
    image_mat_list = []
    for image_path in image_paths:
        image_mat_list.append(load_image_mat(image_path, format_))
    return np.array(image_mat_list)

def save_image_mats(image_mats, save_paths: list = None, save_dir: str = None, format_=None, **kwargs):
    """
    :param image_mats: tf.Tensor or np.ndarray, in shape of [b,h,w,c]. Or enumerable of such elements.
    :param save_paths: If save_dir is also valid, save_paths will be used as paths relative to save_dir
    :param save_dir:
    :param format_:
    :param kwargs:
    :return:
    """
    # if save_paths is not None and safe_get_len(image_mats) != safe_get_len(save_paths):
    #     raise ValueError("Size of image_mats and save_paths must be same.")
    if save_paths is None and save_dir is None:
        raise ValueError("Both save_paths and save_dir cannot be omitted.")
    if save_dir is not None:
        if save_paths is None:
            save_paths = []
            format_ = "jpg" if format_ is None else format_
            file_ext = format_
            for i in range(safe_get_len(image_mats)):
                save_path = osp.join(save_dir, tmp_filename_by_uuid(file_ext))
                save_paths.append(get_new_name_if_exists(save_path))
        else:
            for i in range(safe_get_len(image_mats)):
                save_paths[i] = osp.join(save_dir, save_paths[i])
    len_save_paths = safe_get_len(save_paths)
    for idx, image_mat in enumerate(image_mats):
        if idx >= len_save_paths:
            raise ValueError(f"Size of save_paths(={len_save_paths}) is less than image_mats.")
        if hasmethod(image_mat, 'numpy'):
            image_mat = image_mat.numpy()
        save_path = get_new_name_if_exists(save_paths[idx])  # no overwrites
        save_image_mat(image_mat, save_path, **kwargs)

@track_entry_and_exit.coro()
async def coro_show_image_mat(image_mat, text=None, title=None,
                              cell_size: tuple = None, block=False, image_name=None,
                              cbs=None, task_id=None):
    """
    :param image_name: used as callback arguments
    :param cbs: a tuple in order of (on_done, on_succeeded, on_failed, on_progress).
      only `on_done` is dispatched.
    :return: no return value
    """
    # NOTE: matplot backend is single-threaded, and can only be accessed from its host thread.
    import matplotlib.pyplot as plt

    import asyncio
    from async_ import amend_blank_cbs
    on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
    ret = {'image_name': image_name}
    try:
        fig = show_image_mat(image_mat, text=text, title=title, cell_size=cell_size, block=block)
    except Exception as e:
        ret.update({'error': e})
        on_done(ret)
        return
    if not block:
        try:
            async def coro_pause(delay_s):
                plt.pause(delay_s)
                await asyncio.sleep(0.1)  # give control to other tasks in the same loop

            while True:   # IMPROVE: figure on_close() => manager.destroy(), but cannot hook to it.
                fig.show()
                await coro_pause(1)
        except Exception as e:  # for TKinter backend, a _tkinter.TclError
            DEBUG(f'[coro_show_image_mats] show loop ended: {e}')
    plt.close(fig)
    on_done(ret)

@track_entry_and_exit.coro()
async def coro_show_image_mats(image_mats, texts=None, title=None, num_rows=None, num_cols=None,
                               cell_size: tuple = None, block=False, image_names=None,
                               cbs=None, task_id=None):
    """
    :param image_names: used as callback arguments
    :param cbs: a tuple in order of (on_done, on_succeeded, on_failed, on_progress).
      only `on_done` is dispatched.
    :return: no return value
    """
    # NOTE: matplot backend is single-threaded, and can only be accessed from its host thread.
    import matplotlib.pyplot as plt

    import asyncio
    from async_ import amend_blank_cbs
    on_done, on_succeeded, on_failed, on_progress = amend_blank_cbs(cbs)
    ret = {'image_names': image_names}
    try:
        fig = show_image_mats(image_mats, texts=texts, title=title, num_rows=num_rows, num_cols=num_cols,
                              cell_size=cell_size, block=block)
    except Exception as e:
        ret.update({'error': e})
        on_done(ret)
        return
    if not block:
        try:
            async def coro_pause(delay_s):
                plt.pause(delay_s)
                await asyncio.sleep(0.1)  # give control to other tasks in the same loop

            while True:   # IMPROVE: figure on_close() => manager.destroy(), but cannot hook to it.
                fig.show()
                await coro_pause(1)
        except Exception as e:  # for TKinter backend, a _tkinter.TclError
            DEBUG(f'[coro_show_image_mats] show loop ended: {e}')
    plt.close(fig)
    on_done(ret)


def async_show_image_mat(image_mat, text=None, title=None, cell_size: tuple = None, image_name=None):
    """
    :return: an async task object
    """
    from async_ import AsyncLoop, AsyncManager
    ui_loop = AsyncManager.get_loop(AsyncLoop.UIThread)
    coro = coro_show_image_mat(image_mat, text=text, title=title,
                               cell_size=cell_size, block=False, image_name=image_name)
    task = AsyncManager.create_task(coro, loop=ui_loop)
    return AsyncManager.run_task(task, loop=ui_loop)  # possibly only one task in this batch

def async_show_image_mats(image_mats, texts=None, title=None, num_rows=None, num_cols=None,
                          cell_size: tuple = None, image_names=None):
    """
    :return: an async task object
    """
    from async_ import AsyncLoop, AsyncManager
    ui_loop = AsyncManager.get_loop(AsyncLoop.UIThread)
    coro = coro_show_image_mats(image_mats, texts=texts, title=title, num_rows=num_rows, num_cols=num_cols,
                                cell_size=cell_size, block=False, image_names=image_names)
    task = AsyncManager.create_task(coro, loop=ui_loop)
    return AsyncManager.run_task(task, loop=ui_loop)  # possibly only one task in this batch


def cache_object(src, dest_dir, src_type="Path", suppress_exception=True, **kwargs):
    """
    :param src:
    :param dest_dir:
    :param src_type: Path, Image_mat, Content_str
    :param suppress_exception:
    :return:
    """
    from shutil import copyfile
    ensure_dir_exists(dest_dir)
    # TODO: guess src_type
    try:
        if src_type == "Path":
            if osp.exists(src):
                message = f"Path doesn't exist: {src}"
                if suppress_exception:
                    WARN(message)
                else:
                    raise ValueError(message)
            dest_path = get_new_name_if_exists(osp.join(dest_dir, osp.basename(src)))
            copyfile(src, dest_path)
        elif src_type == "Image_mat":
            file_ext = kwargs.get("file_ext", "jpg")
            dest_path = get_new_name_if_exists(osp.join(dest_dir, tmp_filename_by_time(file_ext)), format_="_{:d}")
            save_image_mat(src, dest_path)
        elif src_type == "Content_str":
            file_ext = kwargs.get("file_ext", None)
            dest_path = get_new_name_if_exists(osp.join(dest_dir, tmp_filename_by_time(file_ext)), format_="_{:d}")
            with open(dest_path, 'wb') as f:
                f.write(src)
        else:
            raise ValueError(f"Unsupported source type: {src_type}")
    except Exception as e:
        if suppress_exception:
            WARN(f"Exception occurred: {e}")
        else:
            raise e

def path_possibly_formatted(path: str or dict):
    if isinstance(path, str):
        return path
    elif isinstance(path, dict):
        args = []
        for arg_name in path.get('arg_names', []):
            from config import Config, Path
            if arg_name == 'experiment_name':
                args.append(Config.ExperimentName)
            elif arg_name == 'experiment_path':
                args.append(Path.ExperimentFolderAbs)
        if len(args) == 0:
            raise ValueError(f"Unparsable arg_names: {path.get('arg_names', [])}")
        return path.get('format', '{}').format(*args)
    else:
        raise ValueError(f"Unsupported path type: {type(path).__name__}")

def path_regulate_to_slash(path, no_driver=False):
    # path = path.replace(r'\/'.replace(os.sep, ''), os.sep)
    path = path.replace('\\', '/')
    if no_driver and 0 < path.find(':') < path.find('/'):
        path = path[path.find(':')+1:]  # get rid of driver
    return path


def is_abs_path(path):
    # path = path.replace('\\', '/')
    # return 0 < path.find(':') < path.find('/') or path[0] is '/'
    raise NotImplementedError('use os.path.isabs() instead')


# Origin: LuPY Path, not used yet.
def walk(root: str, depth: int = None) -> [{str, list, list, int}]:
    """
    Os.walk with specified Depth Level
    :param root: Dir to walk
    :param depth: None = list all folders recursively
                    0 = just root folders
                    1...N = any depth
    :return: (dir_path, subdir_names, file_names, depth)
    """
    root = root.rstrip(os.path.sep)
    assert os.path.isdir(root), f"Path ‘{root}’ is not a dir"
    sep_count = root.count(os.path.sep)

    for current_root, current_subdirs, current_files in os.walk(root):
        current_depth = current_root.count(os.path.sep) - sep_count
        if (depth is not None) and (current_depth >= depth):
            continue
        yield current_root, current_subdirs, current_files, current_depth
    pass

# -----------------------------------------------------#
# -- Data Process ---------------
def urlsafe_uuid(method='uuid4'):
    """
    :return: 24 bytes unique string generated by uuid module
    """
    import uuid
    import base64
    uuid_func = getattr(uuid, method, uuid.uuid4)
    return str(base64.urlsafe_b64encode(uuid_func().bytes), 'utf-8').strip(r'=+')

def dump_to_json(obj, key_if_not_dict=None):
    if hasattr(obj, '__dict__'):
        return json.dumps(obj.__dict__)
    elif isinstance(obj, dict):
        return json.dumps(obj)
    else:
        key = key_if_not_dict or 'data'
        return f'{{"{key}": {obj}}}'

def safe_get_len(obj, default_value=-1):
    len_ = default_value
    if hasattr(obj, "__len__"):
        len_ = obj.__len__()
    elif hasattr(obj, "shape") and safe_get_len(obj.__getattribute__("shape")) > 0:
        len_ = obj.shape[0]
    return len_
    # raise TypeError(f"Cannot get length info from type: {type(obj)}")

def safe_slice(obj: Sequence, start=0, end=0, step=None):
    """
    IMPROVE: implement step for `tf.data.Dataset` obj
    """
    if obj is None:
        return None
    if hasattr(obj, '__getitem__'):
        return obj[start:end:step]
    if hasattr(obj, 'take') and hasattr(obj, 'skip'):
        # duck type of tf.data.Dataset, with `take()` and `skip()` method
        if start > 0:
            obj = obj.skip(start)
        if end > start:
            obj = obj.take(end - start)
        return obj
    raise TypeError(f'Unexpected type to safe_slice: {type(obj).__name__}')

def np_top_k(arr, top_k, axis=-1, order=1):
    """
    :param arr:
    :param top_k:
    :param axis:
    :param order: 1 for descending, 0 for ascending
    :return:
    """
    import numpy as np
    # IMPROVE: use np.argpartition for efficiency
    #   full sorting is not necessary. ref:blog.csdn.net/SoftPoeter/article/details/86629329
    idxs = np.argsort(arr, axis=axis)  # ascending order
    if order == 1:
        idxs = np.flip(idxs, axis=axis)
    idxs = idxs.take(np.arange(top_k), axis=axis)
    arr = np.take_along_axis(arr, idxs, axis=-1)  # not np.take()
    return arr, idxs

def np_softmax(arr):
    import numpy as np
    arr -= np.max(arr, axis=-1, keepdims=True)
    return np.exp(arr) / np.sum(np.exp(arr), axis=-1, keepdims=True)


def dict_compare(before, after):
    # IMPROVE: removed
    new = {}
    changed = {}
    for k in after:
        v = after[k]
        if k not in before:
            new[k] = after[k]
        elif v != before[k]:
            changed[k] = v
    return new, changed

def dict_left_join(self, other: dict, key_map: dict = None, **others):
    """
    Similar to `update_to()` except that only keys defined in left sides (this instance)
    will have their values updated. No extra key-value pairs will be appended.
    :param self:
    :param other:
    :param key_map: dict of key_left -> key_right, for key translation
    """
    other = other.copy()
    others = others.copy()
    key_map = {} if key_map is None else key_map
    for key, value in self.items():
        # 1.key translation and replace mapped keys of other and others
        key_right = key_map.get(key, None)
        if key_right in other and key not in other:
            other[key] = other[key_right]
            del other[key_right]
        if key_right in others and key not in others:
            others[key] = others[key_right]
            del others[key_right]
        value_in_other = other.get(key, None)
        value_in_others = others.get(key, None)
        # 2.recursively update if node is a Params object
        if isinstance(value, self.__class__):
            value.left_join(value_in_other or {}, key_map, **(value_in_others or {}))
            self[key] = value
        else:
            if value_in_others is not None:
                self[key] = value_in_others
            elif value_in_other is not None:
                self[key] = value_in_other
    return self


def dict_cross_join(self, other: dict, key_map: dict = None, **others):
    """
    Similar to `update_to()` except that only keys defined in both sides
    will have their values updated. Those not defined in right side will be deleted.
    :param self:
    :param other:
    :param key_map: dict of key_left -> key_right, for key translation
    """
    to_delete = []
    for key in self.keys():
        if key_map is not None and key in key_map:
            key_right = key_map.get(key)  # key translation
        else:
            key_right = key
        if key in others:
            self.__setitem__(key, others.get(key))
        elif key_right in others:
            self.__setitem__(key, others.get(key_right))
        elif key in other:
            self.__setitem__(key, other.get(key))
        elif key_right in other:
            self.__setitem__(key, other.get(key_right))
        else:
            to_delete.append(key)
    for key in to_delete:
        self.__delitem__(key)
    return self

def dict_update(self, other: dict, key_map: dict = None, **others):
    # NOTE: dict.update() always returns `None`, so should not be used in assignment
    other = other.copy()
    others = others.copy()
    key_map = {} if key_map is None else key_map
    for key, value in self.items():
        # 1.key translation and replace mapped keys of other and others
        key_right = key_map.get(key, None)
        if key_right in other and key not in other:
            other[key] = other[key_right]
            del other[key_right]
        if key_right in others and key not in others:
            others[key] = others[key_right]
            del others[key_right]
        # 2.recursively update if node is a Params object
        if isinstance(value, self.__class__):
            value_in_other = other[key] if key in other else {}
            value_in_others = others[key] if key in others else {}
            value.update(value_in_other, key_map, **value_in_others)
            self[key] = value
            if key in other:
                del other[key]
            if key in others:
                del others[key]
    self.update(other, **others)
    return None  # always return `None` as dict's behavior

def dict_fromkeys(self, keys: Iterable[str], key_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Create a new Params object with keys specified in `keys` (or translate to its alias defined
    in `key_map` if available), and values as mapped to these keys in this instance (if available)
    :param keys:
    :param key_map:
    :return: a new Params instance
    """
    result = {}
    for key in keys:
        if key_map is not None and key in key_map:
            key = key_map.get(key)  # key translation
        if key not in self.keys():
            continue
        result.__setitem__(key, self.get(key))
    return result


def indices_in_vocabulary_list(values: list, vocabulary: list, default_value=-1):
    """
    Find values in vocabulary and returns their indices
    :param values:
    :param vocabulary:
    :param default_value: for oov (out-of-vocabulary) values
    :return:
    """
    indices = []
    for value in values:
        index = default_value
        try:
            index = vocabulary.index(value)
        except ValueError:
            pass
        finally:
            indices.append(index)
    return indices

def safe_get(collection: Any, key, default_value=None):
    try:
        if hasattr(collection, '__getitem__'):
            return collection.__getitem__(key)
        elif hasattr(collection, 'get'):
            return collection.get(key, default_value)
    except RuntimeError:
        pass
    return default_value

def hasmethod(obj, name):
    return hasattr(obj, name) and callable(obj.__getattribute__(name))

# -----------------------------------------------------#
# -- Quick Test -------------------
def quick_load_image_tensor(grayscale=False, normalize=True, rank=4):
    from config import Config
    import numpy as np
    image_path = Config.QuickTest.InputImagePath if not grayscale else Config.QuickTest.GrayscaleImagePath
    image_mat = load_image_mat(image_path)
    if normalize and any([str(image_mat.dtype).startswith(_) for _ in ('int', 'uint')]):
        image_mat = image_mat.astype(np.float32) / 255.0  # normalize = true
    if rank == 4 and image_mat.shape.__len__() == 3:
        image_mat = np.expand_dims(image_mat, axis=0)
    import tensorflow as tf
    return tf.convert_to_tensor(image_mat)

def quick_load_imagenet_labels():
    from config import Config
    import numpy as np
    return np.array(open(Config.QuickTest.ImagenetLabelsPath).read().splitlines())
