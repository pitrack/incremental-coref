import argparse
import itertools
from base_data_curve import start_jobs

encoders = ["on", "da"]
layers = [25]
# num_samples = [500, 250, 150, 1, 5, 15, 25, 50, 100]
num_samples = [10]
trials = [0]
GPUs = [0, 1, 2, 3, 4, 5, 6, 7]


log_location = "preco_dev_analysis_q2_"
job_names = [f"preco_curve_{encoder}_{layers}_{num_samples}_{trial}"
             for (layers, num_samples, encoder, trial) in
             itertools.product(layers, num_samples, encoders, trials)]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--pretrained_model", required=True, type=str, help="Which pretrained checkpoint is being used?")
  parser.add_argument("-c", "--ckpt", required=True, type=str, help="Location of checkpoint to continue training from")
  args = parser.parse_args()
  config = {
    "log_location": log_location + args.pretrained_model,
    "load_path": args.ckpt,
    "test_set": "true",
    "num_epochs": 60,
    "patience": 999,
    "dev_path": "/srv/local1/paxia/incremental_coref/data/preco/nontrain.512.jsonlines",
  }

  start_jobs(job_names, GPUs, config)
