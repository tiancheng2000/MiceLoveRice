from __future__ import absolute_import, division, print_function

import logging
import enum
import contextlib
import json
import os
import os.path as osp
import sys
import time
from queue import Queue
from threading import Thread
import functools
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
    # -- Structure Design -------------
    "singleton",
    "classproperty",
    # -- Input :: Config --------------
    "Params",
    "ConfigSerializer",
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
    "cache_object",
    "path_possibly_formatted",
    "path_regulate_to_slash",
    "is_abs_path",
    "walk",
    # -- Data Process ---------------
    "safe_get_len",
    "indices_in_vocabulary_list",
    "safe_get",
    # -- Thread Process ---------------
    "asynchronous",
    "print_time_consumed",
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
def whereami(filename_only=True):
    """
    :return: filepath, lineno, funcname, stack_info(not implemented)
    """
    f = sys._getframe(0)
    rv = "(unknown file)", 0, "(unknown function)", None
    while hasattr(f, "f_code"):
        if f.f_code.co_filename == __file__:
            f = f.f_back
            rv = (os.path.basename(f.f_code.co_filename) if filename_only else f.f_code.co_filename,
                  f.f_lineno, f.f_code.co_name, None)
            break
        else:
            f = f.f_back
    return rv

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
    with open(json_path, 'w', encoding='utf-8') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        dict_ = {k: float(v) for k, v in dict_.items()}
        json.dump(dict_, f, indent=4)


class Params(dict):
    __class_initialized__ = False
    _key_missed_value = object()  # NOTE: will be initialized in first call to __init__

    def __init__(self, seq=None, **kwargs):
        if seq is not None:
            super(Params, self).__init__(seq)
        else:
            super(Params, self).__init__(**kwargs)
        if not Params.__class_initialized__:
            Params.__class_initialized__ = True
            Params._key_missed_value = Params({})  # sentinel, allows check validity by `has_attr()`

    def __getattr__(self, attr_name):
        # NOTE: only called when has not found the attribute
        return self[attr_name]

    def __missing__(self, key):
        # NOTE: if Params[key] missed, returns this value instead of raising KeyError (or IndexError)
        return Params._key_missed_value

    def has_attr(self):
        """
        NOTE: only use this checking a :class:`Params` node. For leaf nodes (str, list)
          :func:`__len__()` or `isinstance(node, type)` is suggested
        """
        return len(self) > 0

    def get(self, key, *args):
        # deprecated: return super(Params, self).get(key, default if default is not None else {})
        default = args[0] if len(args)>0 else Params._key_missed_value
        return super(Params, self).get(key, default)

    def left_join(self, other: dict, key_map: dict = None, **others):
        """
        Similar to `update_to()` except that only keys defined in left sides (this instance)
        will have their values updated. No extra key-value pairs will be appended.
        :param other:
        :param key_map: dict of key_left -> key_right, for key translation
        """
        for key in self.keys():
            if key_map is not None and key in key_map:
                key_right = key_map.get(key)  # key translation
            else:
                key_right = key
            if key_right in other:
                self.__setitem__(key, other.get(key_right))
            if key_right in others:
                self.__setitem__(key, others.get(key_right))
        return self

    def update(self, other: dict, key_map: dict = None, **others):
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
        return super(Params, self).update(other, **others)  # always return `None` as dict's behavior

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
        dict_to = None
        with open(json_path, encoding='utf-8') as f:
            # TODO: use parse_constant param to parse 'None' (used in tf.Shape)
            dict_from = json.load(f, parse_constant=None)
            _recursive_replace_dict_attrs(dict_from, to_cls=Params)
            dict_to = Params(dict_from)
        return dict_to


# -----------------------------------------------------#
# -- Thread Process ---------------
# decorator
# NOTE: experimental. use AsyncTask or asyncio lib (await, asyncio.create_task(), non-blocking)
class asynchronous(object):   # consider use AsyncTask instead (support process feedback)
    """Sample usage:
    import time
    @asynchronous
    def long_process(num):
        time.sleep(10)
        return num * num

    async_result = long_process.start(12)
    for i in range(20):
        print(i)
        time.sleep(1)
        if async_result.is_done():
            print("async_result {0}".format(async_result.get_result()))

    async_result2 = long_process.start(13)
    try:
        print("async_result2 {0}".format(async_result2.get_result()))
    except asynchronous.NotYetDoneException as ex:
        print(ex.message)
    """

    def __init__(self, func):
        self.func = func
        self.result_queue = None

        def threaded(*args, **kwargs):
            self.result_queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.result_queue = Queue()
        # IMPROVE: to avoid GIL, use Process (and multi-core CPU) to mimic multi-thread
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs)
        thread.start()
        return asynchronous.AsyncResult(self.result_queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class AsyncResult:
        def __init__(self, result_queue, thread):
            self.result_queue = result_queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException('the call has not yet completed its task')
            return self.result_queue.get()

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
# -- File Operations --------------
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def tmp_filename_by_time(file_ext=None):
    # IMPROVE: for milliseconds precision:
    #  t = time.time(); ms = int(round(t * 1000)); us = int(round(t * 1000000))
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + f".{file_ext}" if file_ext is not None else ""

def tmp_filename_by_uuid(file_ext=None, method="uuid4"):
    import uuid
    if method == "uuid1":
        return str(uuid.uuid1()) + ".{}".format(file_ext) if file_ext is not None else ""
    elif method == "uuid4":
        return str(uuid.uuid4()) + ".{}".format(file_ext) if file_ext is not None else ""
    else:
        raise ValueError(f"Unsupported method: {method}")

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

def show_image_mat(image_mat, text=None, title=None, cell_size: tuple = None, block=None):
    from helpers.plt_helper import plot_image_mat as plt_show_image_mat
    return plt_show_image_mat(image_mat, text, title, cell_size, block)

def load_image_mat(image_path, format_=None):
    from helpers.plt_helper import load_image_mat as plt_load_image_mat
    return plt_load_image_mat(image_path, format_)

def save_image_mat(image_mat, image_path, **kwargs):
    from helpers.plt_helper import save_image_mat as plt_save_image_mat
    plt_save_image_mat(image_mat, image_path, **kwargs)

def show_image_mats(image_mats, texts=None, title=None, num_rows=None, num_cols=None, cell_size: tuple = None,
                    block=None):
    from helpers.plt_helper import plot_images as plt_show_image_mats
    return plt_show_image_mats(image_mats, texts, title, num_rows, num_cols, cell_size, block)

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
    :param image_mats:
    :param save_paths: If save_dir is also valid, save_paths will be used as paths relative to save_dir
    :param save_dir:
    :param format_:
    :param kwargs:
    :return:
    """
    if save_paths is not None and safe_get_len(image_mats) != safe_get_len(save_paths):
        raise ValueError("Size of image_mats and save_paths must be same.")
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
    for idx, image_mat in enumerate(image_mats):
        save_path = get_new_name_if_exists(save_paths[idx])  # no overwrites
        save_image_mat(image_mat, save_path, **kwargs)

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
def safe_get_len(obj, default_value=-1):
    len_ = default_value
    if hasattr(obj, "__len__"):
        len_ = obj.__len__()
    elif hasattr(obj, "shape") and safe_get_len(obj.__getattribute__("shape")) > 0:
        len_ = obj.shape[0]
    return len_
    # raise TypeError(f"Cannot get length info from type: {type(obj)}")

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
