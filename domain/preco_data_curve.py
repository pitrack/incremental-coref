import argparse
import itertools
from base_data_curve import start_jobs

encoders = ["fb", "on", "da"]#, "onb"]
# encoders = ["onb"]
layers = [25]
num_samples = [5, 10, 25, 50, 100, 250, 500]
trials = [0]
GPUs = [0, 1, 2, 3, 4, 5, 6, 7]

log_location = "preco_data_curve_q1_inference_"
job_names = [f"preco_curve_{encoder}_{layers}_{num_samples}_{trial}"
             for (layers, encoder, num_samples, trial) in
             itertools.product(layers, encoders, num_samples, trials)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  # parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  args = parser.parse_args()
  config = {
    "log_location": log_location, # + args.pretrained_model,
    "load_model": "true",
    # "load_path": args.ckpt,
    "test_set": "true",
    "command": "inference.py"
  }
  # print(job_names)
  start_jobs(job_names, GPUs, config)
