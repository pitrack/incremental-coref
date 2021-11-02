import argparse
import itertools
from base_data_curve import start_jobs

encoders = ["xlmr", "da"]
layers = [25]
num_samples = {
  "es": [0, 10, 25, 50, 100, 250, 875],
  "ca": [0, 10, 25, 50, 100, 250, 829],
  "it": [0, 10, 20, 40, 80],
  "nl": [0, 10, 20, 40, 80, 145],
}

languages = ["it", "ca", "es", "nl"]
GOLD = ["true", "false"]
trials = [0]
GPUs = [2, 3, 4, 5, 6, 7]

log_location = "onml_curve_se_q1_"
all_jobs = []
for language in languages:
  job_names = [(f"onml_curve_{language}_{encoder}_{layers}_{samples}_{trial}{'_gold' if gold == 'true' else ''}", gold)
               for (encoder, layers, samples, trial, gold) in
               itertools.product(encoders, layers, num_samples[language], trials, GOLD)]
  all_jobs.extend(job_names)

def add_job(name, gold, job_list):
  job_list.append(
    {"name": name,
     "log_location": log_location,
     "load_path": args.ckpt if gold != "true" else args.gold_ckpt,
     "test_set": "true",
     })

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  parser.add_argument("-g", "--gold_ckpt", type=str, help="Location gold checkpoint to continue from")
  args = parser.parse_args()
  log_location = log_location
  job_list = []
  for name, gold in all_jobs:
    add_job(name, gold, job_list)
  config = {
    "command": "trainer.py",
    "jobs": job_list,
  }
  start_jobs(config, GPUs)
