import argparse
import itertools
from base_data_curve import start_jobs

GOLD = True
encoders = ["fb", "on", "da", "onb"]
# encoders = ["da"]
layers = [25]
num_samples = [10, 20, 40, 80, 160, 335]
trials = [0]
GPUs = [1, 2, 3, 5, 6, 7]


log_location = f"arrau_data_curve_q1_{'gold_' if GOLD else ''}"
job_names = [f"arrau_curve_{encoder}_{layers}_{num_samples}_{trial}"
             for (encoder, layers, num_samples, trial) in
             itertools.product(encoders, layers, num_samples, trials)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  args = parser.parse_args()
  config = {
    "log_location": log_location + args.pretrained_model,
    "load_path": args.ckpt,
    "test_set": "true",
  }
  if GOLD:
    config["mentions"] = "false"
    config["use_gold_spans"] = "true"

  start_jobs(job_names, GPUs, config)
