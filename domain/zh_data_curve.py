import argparse
import itertools
from base_data_curve import start_jobs

encoders = ["xlmr", "da"]
layers = [25]
#num_samples = [10, 25, 50, 100, 250, 500, 1810]
num_samples = [0, 25, 50, 100, 250, 500, 1000, 1810]
GOLD = ["true", "false"]
trials = [0]
GPUs = [4, 5, 6, 7]

log_location = "onml_curve_zh_full_rerun_q1_"
job_names = [(f"onml_curve_zh_{encoder}_{layers}_{num_samples}_{trial}{'_gold' if gold == 'true' else ''}", gold)
             for (encoder, layers, num_samples, trial, gold) in
             itertools.product(encoders, layers, num_samples, trials, GOLD)]

def add_job(name, gold, job_list):
  job_list.append(
    {"name": name,
     "log_location": log_location,
     "load_path": args.ckpt if not gold else args.gold_ckpt,
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
  for name, gold in job_names:
    add_job(name, gold, job_list)
  config = {
    "command": "trainer.py",
    "jobs": job_list,
  }

  start_jobs(config, GPUs)
