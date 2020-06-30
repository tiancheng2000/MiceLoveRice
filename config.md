
# Syntax of config_experiment.json

## model_set
Key can be name of the models employed for different purposes, e.g.`model_base`, `model_trained`.
- signature = `"tf.saved_model.load"` | `"keras.Sequential"` | ... 
  - ref:`model_manager.py :: _ModelSignature` 
- format = `"SavedModelDir"` | `"[Name, Config]"` | ...
  - ref:`model_manager.py :: _ModelSignature` 
- path: _string_, or definition of a _formatted string_ like the following:
  ```json
  "path": {
      "format": "experiments/{}/model_trained.tf",
      "arg_names": ["experiment_name"]
  }
  ```
- inputs, outputs: for signature `"tf.import_graph_def"`. _string_ or _list of string_. specify the inputs and outputs _Tensor_ from and to which should be pruned.


## data_set
Key can be name of the data source used, e.g.`data`, `data_simple_test`
- signature = `"tf.data.Dataset"` | `"labeled_folders"` | ...
  - ref:`data_manager.py :: _DataSignature` 
- format = `"Name"` | `"Path"` | ...
  - ref:`data_manager.py :: _DataSignature` 
- path: _string_, or definition of a _formatted string_ like the following:
  ```json
  "path": {
      "format": "experiments/{}/data",
      "arg_names": ["experiment_name"]
  }
  ```
- timeout: _(only implemented for `ui_web_files`)_ _int_, `0` means will wait until data retrieved.
- labels_ordered_in_train: e.g.`["0","1","3","10","11"]`
- fixed_seed: _int_
- category = `"train"` | `"test"` | `"all"`
- test_split: percentage of data split to test set. _float_.
- decode_x
  - name: e.g.`decode_image_file`, name of decode_x func.
    > TODO: allow multiple decode_x funcs.
  - encoding = `"jpg"`
  - colormode = `"grayscale"` | `"rgb"`
  - reshape: e.g.`[-1, 299, 299, 1]`
  - preserve_aspect_ratio = `true` | `false` 
  - color_transform = `"complementary"` | `None`
  - normalize = `true` | `false`


## train
- enable = `true` | `false`
- learning_rate: e.g.`1e-3`, _float_
- validation_split: percentage of train data split to validation set. _float_.
- epochs: _int_
- batch: _int_
- checkpoint
  - signature = `"tf.keras.Model.load_weights"`
  - format = `"CKPT_dir"`
  - path: _string_, or definition of a _formatted string_
  - load_weights = `"latest"`
  - save_weights
    - frequency = `"epoch"`
      > NOTE: if frequency is not equal to `'epoch'`, which represents num of steps instead, the weights would be _half-way_ and less reliable.
- early_stop
  - monitor: e.g.`"val_loss"`, ref:`tf.keras.callbacks.EarlyStopping`
  - patience: _int_, ref:`tf.keras.callbacks.EarlyStopping`
- save_model
  - format = `"SavedModel"` | `"HDF5"`
  - path: cannot be specified. will be saved to checkpoint path with filename of `model_trained` and fileext of `.tf` or `.h5`
- show_result
  - plotter = `"matplot"` | `"remotetask"`
  - metrics: e.g.`["loss", "acc", "precision", "recall"]`


## predict
- enable = `true` | `false`
- remote_task = `true` | `false`
- data_inputs = a key defined in `data_set`
- decode_prediction:
  - name: e.g.`"logits_to_index"`, name of decode_y_ func.
    > TODO: allow multiple decode_y_ funcs.
- show_result
  - plotter = `"matplot"`
  - save_path: _string_, or definition of a _formatted string_
  - top_k: _int_
  - only_difference = `true` | `false`


