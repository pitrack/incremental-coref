# Domain scheduler files

This directory documents some of infrastructure and job files used to train and evaluate all 
of the models. In particular, some hyperparameters might be found here if they aren't in 
the jsonnets, e.g. for the PreCo dev set analysis, I think this is the only place (besides the 
paper) where the value "60 epochs" appears. 

I recommend running the jobs directly based on the jsonnets. At some point mid-project, 
I extended this queue system to be more versatile. As a result, some configs 
are now broken (and those are noted below). Further, since some jobs would occasionally 
fail and since I would reuse the queue creation script for training and inference, almost
all of these files have some lines commented out.

If you are interested in using the code/files in this directory specifically, I would be 
happy to help improve them. Contact me via github issues, email, or video call.


## Data [WIP section]

There are several conversion scripts used to format all data into jsonlines. 

OntoNotes is can be preprocessed using `minimize.py`, see the main README.md or [prior work](https://github.com/mandarjoshi90/coref) for instructions. The script may need to modified a bit to process all the languages or for specific splits/segment sizes.

SemEval uses a very similar script of `minimize_semeval.py`, since the document input format is the same. This should be run the same way, with one command-line argument being the directory containing the files `<lang_id>.{train, devel}.txt` and another being the output dir. Use the XLM-R model.

Similarly, qb and litbank can be processed using `minimize_qb.py` and `minimize_litbank.py`. The SARA data is already in the correct format, while `convert_arrau.py` is used to convert ARRAU-RST corpus. Note that the ARRAU conversion makes some assumptions (e.g. deferring to minumum span/span heads token when split).

For QBCoref and SARA, the splits were made using `make_splits.py`. This script is not really runnable in its current state. It documents the process for creating the splits.

Some other scripts can be ignroed: `rucor_to_json.py`, `ancor_to_json.py`, `minimize_russian_json.py` (all Russian preprocessing) and `remove_es_trace.py` which I think we didn't end up using.

## General structure

Each compute node had 8 GPUs, so these were used to enqueue the jobs.
All checkpoints were stored in `exp_logs`, and each experiment would save a 
model checkpoint and predictions in `exp_logs/exp_name`. Each experiment
would also log (stdout/logging) to a `subprocess_logs` directory.

These files below contain references to absolute paths, although the
exact paths should be relatively straightforward to replace.

`base_data_curve.py` contains a basic template and documentation for how to queue up multiple jobs. Other functions call `start_jobs` with a list of job configs.

## Main experiments (Fig 1, Fig 2)

These all use a different version (V0) of `base_data_curve.py` where `start_jobs` would add enqueue an updated `config` (with the job name). 

* Preco: `preco_data_curve.py`
* ARRAU: `arrau_data_curve.py`
* QBCoref: `qbcoref_data_curve.py`

SARA and LitBank give a good example of how the previous 3 files can be modified.
* LitBank: `litbank_data_curve.py`
* SARA: `sara_data_curve.py`

For the XLM-R (multilingual) experiments:

* OntoNotes Arabic: `ar_data_curve.py`
* OntoNotes Chinese: `zh_data_curve.py`
* SemEval (Spanish, Italian, Catalan, Dutch): `semeval_data_curve.py`

## PreCo dev set (Fig 3)

This also uses the V0 `start_jobs` format, in `preco_dev_analysis.py`.

## Catastrophic Forgetting (Fig 4)

`zero_data_curve.py` evaluates all the models in a zero-shot 
setting. Many lines are commented out/fractured because I ran these on multiple 
identical 8-gpu nodes that did not share a file system.

On the other hand, `forget_data_curve.py` evaluates all the models on the 
source dataset after they are all trained. For similar reasons, many lines 
are commented out.

## Layer Analysis (Fig 5)

`layers_data_curve.py` trains several models based on different number 
of finetuned layers and data sizes. Again, a small section related to 
which encoder should be used had to be toggled manually, and a couple 
jobs related to Chinese OntoNotes were restarted manually too.
