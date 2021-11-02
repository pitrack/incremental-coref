import argparse
import itertools
from base_data_curve import start_jobs

path_model = {}

# Base models
path_model["on"] = "/srv/local1/paxia/exp_logs/public_icoref/ontonotes/checkpoint.bin"
path_model["preco"] = "/srv/local1/paxia/exp_logs/public_icoref/preco/checkpoint.bin"
path_model["en"] = "/srv/local1/paxia/exp_logs/public_icoref/ontonotes_en/checkpoint.bin"


# ENCODERS
# base_encoders = ["onb"]
# qb_encoders = ["fb", "on", "da"]
# lb_encoders = ["fb", "on", "da"]
qb_encoders = ["da"]
lb_encoders = ["da"]
zh_encoders = ["xlmr", "da"]


# LAYERS
large_layers = [0, 6, 12]
base_layers = [0, 3, 6]

# DATA
qb_samples = [15, 60, 240]
lb_samples = [10, 40, 80]
zh_samples = [50, 500, 1810]

trials = [0]
GPUs = [2, 3, 4, 5, 6, 7]

def add_job(name, encoder, lang, job_list):
  if encoder == "da" and lang == "en":
    job_list.append(
      {"name": name,
       "log_location": log_location + "preco",
       "load_path": path_model["preco"],
       "test_set": "true",
       })
    # job_list.append(
    #   {"name": name,
    #    "log_location": log_location + "on",
    #    "load_path": path_model["on"],
    #    "test_set": "true",
    #    })
  elif encoder == "da" and lang == "zh":
    job_list.append(
      {"name": name,
       "log_location": log_location + "onen",
       "load_path": path_model["en"],
       "test_set": "true",
       })
  else:
    job_list.append(
      {"name": name,
       "log_location": log_location + "_pretrained",
       "load_path": path_model["preco"],
       "test_set": "true",
       })


log_location = "layer_exps_da_"
job_list = []
# QB Jobs
large_job_names = [(f"qb_curve_{encoder}_0_{layers}_{num_samples}_{trial}", encoder)
                   for (encoder, layers, num_samples, trial) in
                   itertools.product(["da"], large_layers, qb_samples, trials)]
# small_job_names = [(f"qb_curve_{encoder}_0_{layers}_{num_samples}_{trial}", encoder)
#                    for (encoder, layers, num_samples, trial) in
#                    itertools.product(base_encoders, base_layers, qb_samples, trials)]
# for job, encoder in large_job_names: #+ small_job_names:
#   add_job(job, encoder, "en", job_list)

# litbank Jobs
large_job_names = [(f"litbank_curve_{encoder}_0_{layers}_{num_samples}_{trial}", encoder)
                   for (encoder, layers, num_samples, trial) in
                   itertools.product(lb_encoders, large_layers, lb_samples, trials)]
# small_job_names = [(f"litbank_curve_{encoder}_0_{layers}_{num_samples}_{trial}", encoder)
#                    for (encoder, layers, num_samples, trial) in
#                    itertools.product(base_encoders, base_layers, lb_samples, trials)]

# for job, encoder in large_job_names: # + small_job_names:
#   add_job(job, encoder, "en", job_list)

# zh
tmp_layers = [6]
large_job_names = [(f"onml_curve_zh_{encoder}_{layers}_{num_samples}_{trial}", encoder)
                   for (encoder, layers, num_samples, trial) in
                   itertools.product(zh_encoders, tmp_layers, zh_samples, trials)]

for job, encoder in large_job_names:
  add_job(job, encoder, "zh", job_list)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  log_location = log_location
  config = {
    "command": "trainer.py",
    "jobs": job_list,
  }
  start_jobs(config, GPUs)
