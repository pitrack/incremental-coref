# Incremental Neural Coreference Resolution in Constant Memory

This repository contains the code corresponding to [Incremental Neural Coreference Resolution in Constant Memory](https://arxiv.org/abs/2005.00128).
It is primarily a PyTorch-based incremental document reader. Some of the underlying components are built off of the recent neural models for Coreference Resolution (Lee et al., 2017; 2018, Joshi et al., 2019)

## Data and Setup

We use OntoNotes 5.0. See the [official LDC release](https://catalog.ldc.upenn.edu/LDC2013T19) for more information, and
the [(Span)BERT-coref baseline model](https://github.com/mandarjoshi90/coref) for preprocessing steps. The `conll-2012/`
directory, along with the `minimize.py` and `minimize_json.py` files are copied over or adapted from that repo.

Download the SpanBERT coref model, which includes fine-tuned SpanBERT for coreference resolution on OntoNotes 5.0.
To convert tf-based checkpoints from the baseline model to PyTorch tensors, set `$data_dir` to the location of the
checkpoint and run `convert_tf_to_pt.sh tf_checkpoint_dir pt_checkpoint_dir`. This requires both TensorFlow and PyTorch.

Create `local.jsonnet`, the local configuration file. It should contain the following information

```
{
  data_dir: "", // location of data files
  encoders_dir: "", // location of encoders, could be the same as above
  log_root: "", // location of logs
  gpu_gb: "" // max gpu limit, in GB. Used in gradient accumulation
}
```

Training or running the model works with PyTorch 1.5+, HuggingFace Transformers, and [`pytorch_memlab`](https://github.com/Stonesjtu/pytorch_memlab).
`pytorch_memlab` is not used by default, but it is useful for debugging and so it is left in our code as a required import.

## Training

To train, pick one of the experiments, e.g. `spb_on_512` (or define your own) in `definitions.jsonnet` and run

```
python trainer.py spb_on_512
```

## Evaluation

To evaluate, pick a configuration and run the following. An optional final command-line argument to
the gold conll annotations can be provded for (slower) evaluation against the official scorer. Most of the settings,
like evaluation data, encoder, hyperparameters, etc are all configured in either `definitions.jsonnet` (for
experimental settings with dev/test sets) or `base.jsonnet` (for hyperparameters and eviction).

The metrics reported at the end consist of several slices of information. Edit `eval_all.py`
to select fewer (or more).

```
python inference.py spb_on_512
```

To evaluate our best model, download the [`checkpoint.bin` here](https://nlp.jhu.edu/incremental-coref/models/checkpoint.bin) (1.7GB),
and place it under `$log_dir/spanbert_large/spb_on_512/`, which is the experimental configuration it matches.


## Contact

This contains the main code for our paper, but additional instructions,
e.g. on reproducing our tables, won't be available until later. Feel
free to raise issues to request anything specifically or if you have
any questions about running the code in this repository. You can also
email paxia@cs.jhu.edu with questions about the paper.

## Citing our work

If you wish to cite our work:

```
@inproceedings{xia-etal-2020-incremental,
  title={Incremental Neural Coreference Resolution in Constant Memory},
  author={Patrick Xia and Jo\~ao Sedoc and Benjamin {Van Durme}},
  year={2020},
  booktitle={Proceedings of EMNLP},
}
```
