import argparse
import itertools
from base_data_curve import start_jobs

GOLD = False
# encoders = ["fb", "on", "da", "onb"]

encoders = ["da"]
layers = [25]
splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_samples = [5, 10, 20, 40, 80]
trials = [0]
GPUs = [1, 2, 3, 4, 5, 7]

def add_job(name, gold, job_list):
  job_list.append(
    {"name": name,
     "log_location": log_location + "on2pc",
     "load_path": args.ckpt,
     "test_set": "true",
   })

log_location = f"litbank_data_curve_q1_{'gold_' if GOLD else ''}"
job_names = [(f"litbank_curve_{encoder}_{split}_{layers}_{num_samples}_{trial}", GOLD)
             for (split, encoder, layers, num_samples, trial) in
             itertools.product(splits, encoders, layers, num_samples, trials)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  # parser.add_argument("-g", "--gold_ckpt", type=str, help="Location gold checkpoint to continue from")
  args = parser.parse_args()
  job_list = []
  for name, gold in job_names:
    add_job(name, gold, job_list)
  config = {
    "command": "trainer.py",
    "jobs": job_list,
  }

  start_jobs(config, GPUs)
