---
MiceLoveRice -- a Machine Learning test Reinforcement
---
> under construction, support _TF2_ by now.

## Aims
- Wrap _implementation_ details (esp. there are so many alternatives) so as to highlight DL domain concepts and workflows.
- Allow beginners to focus on _theoretical_ points, while being able to verify _applicational_ results. 
- Allow adapting new _implementation_ from different sources into this framework to ease comparing and utilizing.
- Allow developers to work out one combination of workflow and to reuse relevant _implementation_ codes.


## Principles
- Use _Keras_ workflow for most cases.


## Inputs
- _data_: generally provided in type acceptable by `keras.Model.fit()`, i.e. `np.ndarray` or `tf.Tensor`, or `list` or `dict` of these two types.

## Dependencies
- _modules_: stable pre-trained models, algorithms
  - _experimental_: models, algorithms for research or tuning
- _helpers_: helper libraries, inc. 3rd party

## Outputs
- _experiments_: name_convention ::= Base + Experimental(algorithm, params) + Inputs(num_classes, dataset, params)

  [Configures]
  - _config_experiment_: configs loaded by each experiments 
  - _config_main_: _optional_. configs loaded by the _main_ module

  [Intermediates Cache]
  - _checkpoints_: to record/load intermediate model variables(weights/bias) and model graph _(optional)_
  - _features_: derived from raw inputs data, usually through bottleneck layer of a base model, and can be used in further transaction e.g. retraining (transfer learning)
  - _embeddings_: output of a target algorithm (e.g. several dense layers trained by a loss algorithm) to map features into a target space. Quality of the embedding determines the efficiency of inference. 
  - _clusters_

  [Outputs Cache]
  - _models_: graph + variables as training results
  - _metrics_: 1.summary 2.projector_data(embeddings, labels, sprite_images)
  - _results_: 1.predictions(esp. misses) 2.relevant raw data(e.g.predicted Ps Ns by a triplet-loss algorithm)

