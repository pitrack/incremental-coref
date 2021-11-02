# Incremental Coreference Resolution

The code in this repository is associated with the following papers:

* [current branch] [Moving on from OntoNotes: Coreference Resolution Model Transfer](https://arxiv.org/abs/2104.08457). Patrick Xia and Benjamin Van Durme. EMNLP 2021.

* [[Docker Hub](https://hub.docker.com/r/hltcoe/lome)] [LOME: Large Ontology Multilingual Extraction](https://aclanthology.org/2021.eacl-demos.19/). Patrick Xia*, Guanghui Qin*, Siddharth Vashishtha, Yunmo Chen, Tongfei Chen, Chandler May, Craig Harman, Kyle Rawlins, Aaron Steven White, Benjamin Van Durme. EACL 2021 System Demos. [Demo](https://nlp.jhu.edu/demos/lome/). 

* [[#emnlp2020 branch]](https://github.com/pitrack/incremental-coref/tree/emnlp2020) [Incremental Neural Coreference Resolution in Constant Memory](https://aclanthology.org/2020.emnlp-main.695/). Patrick Xia, João Sedoc, Benjamin Van Durme. EMNLP 2020.

In its current state, there are likely still lingering issues/missing documentation in this repository. If you have any questions, please raise an issue or email me. I would also be happy to debug issues related to this codebase over video/voice calls.

## About

The icoref model is a PyTorch-based incremental document reader. Some of the underlying components, like the span representations, are built of prior work on Coreference Resolution (Lee et al., 2017; 2018, Joshi et al., 2019). Details of the model can be found in the [EMNLP 2020 paper](https://aclanthology.org/2020.emnlp-main.695/). 

## Getting Started

To use the code, the high-level process is to convert data into a minimized jsonlines format and create a [jsonnet](https://jsonnet.org/) experiment configuration. Then, running `python inference.py <exp name>` or `python trainer.py <exp name>` should run inference or training over the files.

### Local setup

In the main repo, create `local.jsonnet`, which is the local configuration file. It should contain the following information

```
{
  data_dir: "", // location of data files
  encoders_dir: "", // location of encoders, could be the same as above
  log_root: "", // location of logs
  gpu_gb: "" // max gpu limit, in GB. Used in gradient accumulation
}
```

For packages, the primary ones are jsonnet(=0.16.0), torch(=1.7.0), and transformers(=3.4.0), and these are also included in requirements.txt. The repo is not tested on the most recent versions of these libraries. 

To set up all of this, run `pip install -r requirements.txt`.

### Defining experiments

The config files are sorted into [jsonnets](https://jsonnet.org/), which is a data templating language. The main model parameters are defined in `jsonnets/base.jsonnet`, encoder parameters in `jsonnets/encoder.jsonnet`, and data parameters in `jsonnets/data.jsonnet`. Local paths, as mentioned above, are in `local.jsonnet`. The remaining files are individual files related to different experiments/projects. `jsonnets/verify_jsonnet.py` wraps the jsonnet import and load function to quickly check for syntax errors by running `python verify_jsonnet.py <jsonnet_file>`.

The experiments themselves are then imported at the top-level `experiments.jsonnet`. This is the file ultimately read by the code.

#### OntoNotes example

For example, we can use [OntoNotes 5.0, official LDC release](https://catalog.ldc.upenn.edu/LDC2013T19). Run `conversion_scripts/minimize.py` (same as in [prior work](https://github.com/mandarjoshi90/coref)) to convert the OntoNotes format to jsonlines. Place these into `$data_dir/ontonotes`. Next, in `jsonnets/data.jsonnet`, create a `Dataset` config, specifying the paths to the exact files, e.g. 

```
 Ontonotes_example: Dataset("ontonotes/train.english.512.jsonlines",
                            "ontonotes/dev.english.512.jsonlines",
                            "ontonotes/test.english.512.jsonlines",
                            30, false),
```
30 is the max span width hyperparameter, which varies by dataset while "false" indicates that we are not using genre embeddings. With the data configuration defined, make a new file (e.g. `ontonotes.jsonnet`) and import `base.jsonnet` (which contains the default model hyperparameters), `data.jsonnet` (which contains all the information about our datasets), and `encoder.jsonnet` (which contains information about all the encoders). Then define an experiment, e.g. 

```
example: (base.base + encoders.spanbert_alrge + data.OntoNotes + base.Name("example"))
```

`base.Name` specifies the output dir for the experiment. Then, run `python trainer.py example`, which trains the icoref model from EMNLP2020. On a GPU like 1080TI, it should take 1-1.5 hours per epoch and converge within ~10-15 epochs. 

In the EMNLP 2020 paper, we reused trained parameters. To get these parameters, download [the baseline model](https://github.com/mandarjoshi90/coref), which includes both the fine-tuned SpanBERT for coreference resolution on OntoNotes 5.0 and the coref model. Then, convert those weights from tf to pytorch by running and placing these in the encoder dir:

```
python conversion_scripts/convert_tf_to_pytorch.py <path_to_model>/model.max.ckpt <path_to_model>/torch_scorer_vars.bin
``` 

After the first epoch, the average coref on the dev set F1 should be around 71-72.



#### Dataless Example with XLMR tokenizer

It is possible to run models without data by loading a model without specifying a path to the data. This could be useful for using the coref model as part of a larger system (as we do in LOME), or for loading data from different sources (e.g. for analysis of specific examples). To do so, use the `Dataless` constructor in `data.jsonnet`. As an example, `xlmr_ml_true_24_dataless` corresponds to the `xlmr_ml_true_24` config except the paths to the data are empty strings.

To load/use the model, use `run_xlmr.py` in the following way (turn off logging by commenting out or changing `logging.basicConfig(level=logging.INFO)`):

```
# run()
python run.py xlmr_ml_true_24_dataless <input jsonlines file> <output jsonlines file> False
# run_concrete()
python run.py xlmr_ml_true_24_dataless <input concrete comm> <output concrete comm> True
```


## Training and evaluation

To train, pick one of the experiments, e.g. `spb_on_512` in `definitions.jsonnet` and run

```
python trainer.py spb_on_512
```

To evaluate, pick a configuration and run the following. An optional final command-line argument to
the gold conll annotations can be provded for (slower) evaluation against the official scorer. Most of the settings,
like evaluation data, encoder, hyperparameters, etc are all configured in either `definitions.jsonnet` (for
experimental settings with dev/test sets) or `base.jsonnet` (for hyperparameters and eviction).

The metrics reported at the end consist of several slices of information. Edit `eval_all.py`
to select fewer (or more). There are additional options for aggregating results across many runs.

```
python inference.py spb_on_512
```

To evaluate the best model (from EMNLP 2020), download the [`checkpoint.bin` here](https://nlp.jhu.edu/incremental-coref/models/checkpoint.bin) (1.7GB),
and place it under `$log_dir/spanbert_large/spb_on_512/`, which is the experimental configuration it matches.


### Pretrained Models

In the EMNLP2021 paper, we make use of three "base" models that can be downloaded here. These only contain the checkpoint.bin files, so they need to be used with the corresponding encoder and jsonnet config. They are different sizes because transfer-on reuses publicly available encoder while transfer-en includes finetuned XLM-R and transfer-pc includes a fine-tuned SpanBERT model. If requested, we can reduce the size of these files and upload the encoders onto HuggingFace models too.

1. transfer-on: [1.7G](https://nlp.jhu.edu/coref-transfer/models/ontonotes.tar)
2. transfer-en: [6.7G](https://nlp.jhu.edu/coref-transfer/models/ontonotes_en.tar)
3. transfer-pc: [4.2G](https://nlp.jhu.edu/coref-transfer/models/preco.tar)

## Other

### Overriding arguments

Most arguments can be overridden in the command-line with by adding the name and argument after the experiment name, e.g. `python inference.py spb_on_512 threshold=-16 method="mean"`. Overriding dicts is trickier and may not work. Always check the printed config to make sure the arguments were correctly overridden.

Overriding arguments is useful for loading specific checkpoints, we do in the EMNLP 2021 paper. See `command_list` in `domain/base_data_curve.py` for another example.

### Domain Transfer

In the EMNLP 2021 paper, we perform extensive experiments for domain transfer. Most of experiments follow the pattern described in this README of curating and processing a dataset, writing a config, training a lot of models, and then evaluating them. To do this, we wrote a job queuing/submission scripts (in the `domain/` subdirectory), added more features to `eval_all.py`, and wrote plotting code (in `analysis/`). There were a few modeling changes around 1) how frequent we perform the backwards pass; 2) handling singletons; and 3) finetuning the encoder. These are all updated in the `base.jsonnet` file to match what was done in the EMNLP2020 paper and then overridden in specific experiment configs.

Additional instructions for reproducing the EMNLP 2021 paper can be found in `domain/README.md`

### Verifying Setup

We demonstrate one way to verify the basic setup from the EMNLP2020 paper.  In this example, `torch_scorer_vars.bin` exists but `checkpoint.bin` does not.

As a quick unit test, we can run `python inference.py spb_on_512` (no training). Since we are doing no training, these numbers should be deterministic.

```
$ python inference.py spb_on_512
[..]
2020-12-03 17:49:17 [INFO] Loaded 343 examples.
2020-12-03 17:49:17 [INFO] Did not found tokenizer, using spanbert-cased
2020-12-03 17:49:26 [INFO] Loaded model from /home/local_dir/incremental_coref/encoders/spanbert_large/torch_scorer_vars.bin
2020-12-03 17:49:26 [INFO] Putting Encoder, GenreEmbedder, SpanScorer, and ScoringModule all on cuda
2020-12-03 17:49:26 [INFO] Checkpoint not found at /home/local_dir//spb_on_512/checkpoint.bin
2020-12-03 17:49:26 [INFO] Updating threshold to 0
[..]
2020-12-03 17:55:42 [INFO] cn: 59931 and wn: 12977
2020-12-03 17:55:42 [INFO] ca: 489 and wa: 364
2020-12-03 17:55:42 [INFO] Evaluation on 343 documents [329.802] took 375.5 seconds
2020-12-03 17:55:42 [INFO] __@ALL: 0.770, 0.048, 0.088, (343 docs)
[..]
muc: 0.9367 0.0547 0.1033
b_cubed: 0.9421 0.0301 0.0583
ceafe: 0.4314 0.0582 0.1025
mentions: 0.9604 0.0735 0.1366
```

These numbers are a bit disappointing (high precision, very low recall!). `wn: 12977` means that 12977 mentions were incorrectly marked as "make a new cluster" instead of "join an existing one." These poor numbers are expected since we did zero training! However, if we switch to "`threshold=-16`" and "`method="mean"`", running on `python inference.py spb_on_512 threshold=-16 method="mean"` results in the following. In this example we overrode the arguments on the command-line but we can also do so directly in `jsonnets/base.jsonnet` or even in the definition of `spb_on_512`.

```
$ python inference.py spb_on_512 threshold=-16 method="mean"
[..]
2020-12-03 17:48:34 [INFO] Loaded 343 examples.
2020-12-03 17:48:34 [INFO] Did not found tokenizer, using spanbert-cased
2020-12-03 17:48:43 [INFO] Loaded model from /home/incremental_coref/encoders/spanbert_large/torch_scorer_vars.bin
2020-12-03 17:48:43 [INFO] Putting Encoder, GenreEmbedder, SpanScorer, and ScoringModule all on cuda
2020-12-03 17:48:43 [INFO] Checkpoint not found at /home/local_dir//spb_on_512/checkpoint.bin
2020-12-03 17:48:43 [INFO] Updating threshold to -16
[..]
2020-12-03 17:54:22 [INFO] cn: 54180 and wn: 1591
2020-12-03 17:54:22 [INFO] ca: 8820 and wa: 9170
2020-12-03 17:54:22 [INFO] Evaluation on 343 documents [128.078] took 338.2 seconds
2020-12-03 17:54:22 [INFO] __@ALL: 0.514, 0.678, 0.582, (343 docs)
[..]
muc: 0.6043 0.7441 0.6670
b_cubed: 0.5290 0.6248 0.5729
ceafe: 0.4074 0.6646 0.5051
mentions: 0.6392 0.8478 0.7289
```

If `checkpoint.bin` does exist, then it should be found and it would reproduce the numbers in the paper (without adjusting the threshold).

### Another unused feature: Caching embeddings

We can also run `python encoder.py <exp_name>` and it will cache the embeddings (top layer) into the experiment's directory. Then run `training.py` or `inference.py` as before. These files can get large (OntoNotes train + dev is around 6.4GB), but they can speed up training/inference substantially. Of course, this is not compatible with finetuning -- finetuning will silently fail if this embedding file exists! Also, no part of these embeddings are dropped out in training, as these are static. Under the same setup as above (note that this time the model finds `embeddings.pt`), we can see how much faster it is (Answer: not very much).

```
$ python encoder.py spb_on_512
2020-12-03 18:12:45 [INFO] Did not found tokenizer, using spanbert-cased
2020-12-03 18:12:54 [INFO] Loaded 2802 examples.
2020-12-03 18:12:55 [INFO] Loaded 343 examples.
2020-12-03 18:13:03 [INFO] Cached 99 documents
2020-12-03 18:13:27 [INFO] Cached 299 documents
[..]
2020-12-03 18:19:04 [INFO] Cached 3099 documents
2020-12-03 18:19:24 [INFO] Saved 4984 embeddings to /home/local_dir//spb_on_512/embeddings.pt

$ du -h ../local_dir/spb_on_512/embeddings.pt
6.4G    ../local_dir/spb_on_512/embeddings.pt

$ python inference.py spb_on_512
2020-12-03 17:38:21 [INFO] Did not found tokenizer, using spanbert-cased
2020-12-03 17:38:28 [INFO] Found cached embeddings at /home/local_dir//spb_on_512/embeddings.pt. Using them
2020-12-03 17:38:30 [INFO] Loaded model from /home/incremental_coref/encoders/spanbert_large/torch_scorer_vars.bin
2020-12-03 17:38:30 [INFO] Putting Encoder, GenreEmbedder, SpanScorer, and ScoringModule all on cuda
2020-12-03 17:38:30 [INFO] Checkpoint not found at /home/local_dir//spb_on_512/checkpoint.bin
2020-12-03 17:38:30 [INFO] Updating threshold to -16
[..]
2020-12-03 17:43:26 [INFO] cn: 54180 and wn: 1591
2020-12-03 17:43:26 [INFO] ca: 8820 and wa: 9170
2020-12-03 17:43:26 [INFO] Evaluation on 343 documents [128.078] took 295.6 seconds
2020-12-03 17:43:26 [INFO] __@ALL: 0.514, 0.678, 0.582, (343 docs)
```

## Contact

This contains the main code for our papers, but some additional 
instructions, e.g. on reproducing our tables, might not be available. 
Feel free to raise issues to request anything specifically or if you have
any questions about running the code in this repository. You can also
email paxia@cs.jhu.edu with questions about the paper.

## Citing our work

If you use this repo or the icoref model, please cite:

```
@inproceedings{xia-etal-2020-incremental,
    title = "Incremental Neural Coreference Resolution in Constant Memory",
    author = "Xia, Patrick  and
      Sedoc, Jo{\~a}o  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    url = "https://aclanthology.org/2020.emnlp-main.695",
    doi = "10.18653/v1/2020.emnlp-main.695",
}
```

If you are performing work related to domain transfer,
and (re)using configs or performing experiments on a dataset besides
OntoNotes English, you may also wish to cite:

```
@inproceedings{xia-van-durme-2021-moving,
  title={Moving on from OntoNotes: Coreference Resolution Model Transfer},
  author={Patrick Xia and Benjamin {Van Durme}},
  year={2021},
  booktitle={Proceedings of EMNLP},
}
```

Finally, if you specifically use the [LOME](https://hub.docker.com/r/hltcoe/lome) 
docker container (or the contents therein), then instead cite:

```
@inproceedings{xia-etal-2021-lome,
    title = "{LOME}: Large Ontology Multilingual Extraction",
    author = "Xia, Patrick  and
      Qin, Guanghui  and
      Vashishtha, Siddharth  and
      Chen, Yunmo  and
      Chen, Tongfei  and
      May, Chandler  and
      Harman, Craig  and
      Rawlins, Kyle  and
      White, Aaron Steven  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    year = "2021",
    url = "https://aclanthology.org/2021.eacl-demos.19",
    doi = "10.18653/v1/2021.eacl-demos.19",
}

```
