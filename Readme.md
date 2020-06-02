
# A Deep Learning Test Yard, based on Tensorflow2

## Aims
- Wraps _implementation_ details (esp. there are so many alternatives) so as to highlight DL domain concepts and workflows.
- Allows beginners to focus on _theoretical_ points, while being able to validate _applicational_ results. 
- Allows adapting new _implementation_ from different sources into this framework to ease future utilizing.
- Allows developers to walk through one kind of workflow and to reuse _implementation_ codes.


## Principles
- Use _Keras_ workflow for most of the case.


## Inputs
- _data_: generally provided in type acceptable by `keras.Model.fit()`, i.e. `np.ndarray` or `tf.Tensor`, or `list` or `dict` of these two types.

## Dependencies
- _modules_: stable pre-trained models, algorithms
  - _experimental_: models, algorithms for research or tuning
- _helpers_: helper libraries, inc. 3rd party

## Outputs
- _experiments_: name_convention ::= Base + Experimental(algorithm, params) + Inputs(num_classes, dataset, params)

  [Configures]
  - _config_main_: configs loaded by the _main_ module
  - _config_experimental_: configs loaded by experimental modules 

  [Intermediates Cache]
  - _checkpoints_: model variables(weights/bias), model graph _(optional)_
  - _features_: 1.bottleneck output of base model, for retraining
  - _embeddings_: 1.output of target algorithm, e.g. from a dense layer
  - _clusters_

  [Outputs Cache]
  - _models_: graph + variables as training results
  - _metrics_: 1.summary 2.projector_data(embeddings, labels, sprite_images)
  - _results_: 1.predictions(esp. misses) 2.relevant raw data(e.g.predicted Ps Ns by a triplet-loss algorithm)


