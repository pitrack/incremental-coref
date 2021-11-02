import argparse
import itertools
from base_data_curve import start_jobs

GOLD = True
encoders = ["fb", "on", "da", "onb"]

layers = [25]
splits = [0, 1, 2, 3, 4]
num_samples = [5, 15, 30, 60, 120, 240]
trials = [0]
GPUs = [0, 1, 2, 3, 4, 5, 6, 7]

log_location = f"qbcoref_data_curve_q1_{'gold_' if GOLD else ''}"
job_names = [f"qb_curve_{encoder}_{split}_{layers}_{num_samples}_{trial}"
             for (encoder, split, layers, num_samples, trial) in
             itertools.product(encoders, splits, layers, num_samples, trials)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  args = parser.parse_args()
  config = {
    "log_location": log_location + "_" + args.pretrained_model,
    "load_path": args.ckpt,
    "test_set": "true",
  }
  if GOLD:
    config["mentions"] = "false"
    config["use_gold_spans"] = "true"

  start_jobs(job_names, GPUs, config)
