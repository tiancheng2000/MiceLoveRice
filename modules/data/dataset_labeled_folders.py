import os.path as osp

import collections
import glob
import random
import tensorflow as tf

import modules.data.decode_tf as decode_tf
from helpers.util import Params, indices_in_vocabulary_list


__all__ = [
    "dataset",
]

"""
参考：retrain.py(TF1.x)
- MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M, used as hash and random base
- 初始化图像列表。返回映射集：label_name -> {目录, training图片[], testing图片[], validation图片[]}
  def create_image_lists(root_dir, testing_percentage=10, validation_percentage=0):
    pass
- 获得某label、第index个、位于某image_dir目录下、属于某category(training/testing/validation)的image的路径
  def get_image_path(image_lists, label_name, category, index, root_dir):
    pass
- 获得bottleneck(即特征向量)缓存文件名(.txt)
  def get_image_feature(image_lists, label_name, category, index, bottleneck_dir, module_name):
    pass
  def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                    bottleneck_dir, save_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_input_tensor,
                                    bottleneck_tensor, module_name):
    pass
"""

__full_sets__ = None  # root_path --> data_paths, labels

# noinspection SpellCheckingInspection
def _load_as_data_label_lists(root_path, file_exts: set, labels_ordered_in_train=None,
                              need_shuffle=False, shuffle_seed=None, force_reload=False):
    global __full_sets__
    if __full_sets__ is None:
        __full_sets__ = collections.OrderedDict()  # IMPROVE: use dict ?
    elif not force_reload and isinstance(__full_sets__, dict) and __full_sets__[root_path] is not None:
        return

    # TODO: support glob multiple file extentions (e.g .JPG, .jpeg, .JPEG for non-Windows platform)
    # IMPROVE: may use tf.data.list_files(glob_pattern, shuffle=False) instead
    data_paths = []
    for file_ext in file_exts:
        glob_pattern = osp.join(root_path, f"*/*.{file_ext}")
        # TODO: use glob.iglob() and returns Iterators which yield filenames and labels? to deal with Big file set.
        #   tf.data.Dataset.from_generator() can consume yields. Or, tf.data.Dataset+map_fn already load by batch_size?
        #   tf.data.Dataset.list_files() can directly return globed result!
        data_paths.extend(glob.glob(glob_pattern))
    # IMPROVE: use (shuffled_idx=) np.random.permutation(len(data)) instead
    if type(shuffle_seed) is int:
        random.seed(shuffle_seed)
    if need_shuffle:
        random.shuffle(data_paths)

    # Use data filenames instead of full path. Consumer may join(root_path, label, data_filename)
    data_paths_splitted = [osp.split(_) for _ in data_paths]
    del data_paths
    data_fns, labels = zip(*[(_[1], osp.basename(_[0])) for _ in data_paths_splitted])
    del data_paths_splitted
    # convert label from string to index (of a fixed vocabulary). caller may convert to `categorical` format (one-hot)
    data_fns, labels = list(data_fns), list(labels)  # zip returns generator since python 3.x, convert it to lists
    if labels_ordered_in_train is not None:
        for label in set(labels):
            if label not in labels_ordered_in_train:
                raise ValueError(f"given labels_ordered_in_train lost the folder name: {label}")
        vocabulary_list = labels_ordered_in_train
    else:
        vocabulary_list = list(set(labels))
        vocabulary_list.sort()  # sorted
    # IMPROVE: TF操作可用tf.unique():返回一个元组tuple(y,idx)，y为x的列表的唯一化数据列表，idx为x数据对应y元素的index
    labels = indices_in_vocabulary_list(labels, vocabulary_list)

    total = len(labels)
    # divide into 2 groups
    # train_percent = 1 - test_split
    # train_data_fns = data_fns[0:int(train_percent * total)]
    # train_labels = labels[0:int(train_percent * total)]
    # test_data_fns = data_fns[int((train_percent +eval_percent) * total):]
    # test_labels = labels[int((train_percent +eval_percent) * total):]

    __full_sets__[root_path] = {
        'total': total,
        'vocabulary': vocabulary_list,
        'all': (data_fns, labels),
        # 'train': (train_data_fns, train_labels),
        # 'eval': (eval_data_fns, eval_labels),
        # 'test': (test_data_fns, test_labels)
    }


def dataset(root_path, file_exts={'jpg', 'jpeg', 'png', 'gif', 'bmp'}, category='all', need_shuffle=False, shuffle_seed=None,
            labels_ordered_in_train=None, test_split=0.2, decode_x={}, decode_y={},
            meta_info=None, **unused) -> tf.data.Dataset or (tf.data.Dataset, tf.data.Dataset):
    """
    TODO: `images, labels = tf.keras.preprocessing.image.ImageDataGenerator::flow_from_directory` implemented this.
    Retrieves data files from each labeled folders.
    :param root_path: absolute path of root of labeled folders
    :param file_exts: set of file extensions to be matched
    :param category: 'train': with 'eval'(=validation) together, used in training phase; 'test': used in predicting phase
    :param need_shuffle: need to shuffle before validation_split in some cases, otherwise validation data could be the last class
    :param shuffle_seed: only effective when shuffle. given same random seed, the order of returned files will be same.
    :param labels_ordered_in_train: folder names must be sorted in the same order that used in training for prediction
    :param test_split: deprecated. category split should be done by the caller  # TODO: deprecation
    :param decode_x: params of decoding x (data)
    :param decode_y: params of decoding y (labels)
    :param meta_info: if given a dict, caller can retrieve 'total' and 'vocabulary'() info
    :return: tf.data.Database, if category='all', 'train' and 'test' dataset will be returned as a tuple
    """
    global __full_sets__

    # UPDATE: return all categories
    # if category is None:
    #     raise ValueError("category must be specified for when fetching dataset_labeled_folders.")

    # IMPROVE: try directly use `tf.data.dataset.list_files(file_pattern)`
    _load_as_data_label_lists(root_path, file_exts, need_shuffle=need_shuffle, shuffle_seed=shuffle_seed,
                              labels_ordered_in_train=labels_ordered_in_train)
    total = __full_sets__[root_path]['total']
    vocabulary_list = __full_sets__[root_path]['vocabulary']  # vocabulary of label names
    data_fns, labels = __full_sets__[root_path]['all']  # fn: filename, labels: label's index in vocabulary_list
    if category == 'train':
        data_fns = data_fns[0:int((1-test_split) * total)]
        labels = labels[0:int((1-test_split) * total)]
    elif category == 'test':
        data_fns = data_fns[int((1-test_split) * total):]
        labels = labels[int((1-test_split) * total):]

    root_path_t = tf.constant(root_path, dtype=tf.dtypes.string)
    vocabulary_t = tf.constant(vocabulary_list, dtype=tf.dtypes.string)
    # Updating: consider use **decode_x in map_func
    # colormode = decode_x.get('colormode', None)
    # resize_w, resize_h = decode_x.get('resize_w', None), decode_x.get('resize_h', None)
    # normalize = decode_x.get('normalize', True)
    # preserve_aspect_ratio = decode_x.get('preserve_aspect_ratio', True)
    # TODO: after support arbitrary image format in `decode_tf`, change default encoding to `None`
    params_decode = Params(encoding='jpg', colormode=None, reshape=None, preserve_aspect_ratio=True,
                           color_transform=None, normalize=True).left_join(decode_x)

    def wrapper_decode_image_file(file_name_t, folder_id_t):
        # NOTE: cannot use ordinary string operations here, use Tensor instead
        # path = osp.join(root_path, folder_name, file_name)
        folder_name_t = vocabulary_t[folder_id_t]
        path_t = tf.strings.join([root_path_t, folder_name_t, file_name_t], '/')
        # Updating: consider use **decode_x in map_func
        # return decode_tf.decode_image_file(
        #             path_t, encoding=None, colormode=colormode,
        #             resize_w=resize_w, resize_h=resize_h,
        #             normalize=normalize, preserve_aspect_ratio=preserve_aspect_ratio
        # )
        return decode_tf.decode_image_file(path_t, **params_decode)

    ds_data = tf.data.Dataset.from_tensor_slices((data_fns, labels))
    ds_data = ds_data.map(wrapper_decode_image_file)
    # NOTE: labels must be mapped to int32 for model_fn processing, furthermore, for triplet_loss mask calc
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)
    # TODO: remove this kinda decode_y, since already used indices_in_vocabulary_list() for conversion
    if decode_y.get('name', None) == 'decode_integer_label':
        ds_labels = ds_labels.map(decode_tf.decode_integer_label)
    ds = tf.data.Dataset.zip((ds_data, ds_labels))

    if meta_info is not None:
        meta_info['total'] = total
        meta_info['vocabulary'] = vocabulary_list
        meta_info['filenames'] = data_fns

    if category == 'all':
        # split 'train' and 'test'
        return ds.take(int((1-test_split) * total)), ds.skip(int((1-test_split) * total))
    else:
        return ds
