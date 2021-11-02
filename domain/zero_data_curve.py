from base_data_curve import start_jobs

PATHS = {
  "on": "/srv/local1/paxia/exp_logs/public_icoref/ontonotes/checkpoint.bin",
  "preco": "/srv/local1/paxia/exp_logs/public_icoref/preco/checkpoint.bin",
  "en": "/srv/local1/paxia/exp_logs/public_icoref/ontonotes_en/checkpoint.bin",
  "en_gold": "/srv/local1/paxia/exp_logs/public_icoref/mentions/baseline_onml_en_25_2802/checkpoint.bin",
  "preco_gold": "/srv/local1/paxia/exp_logs/public_icoref/mentions/fb_preco_25_36120/checkpoint.bin",
  "on_gold": "/srv/local1/paxia/exp_logs/public_icoref/mentions/on_0_2802/checkpoint.bin"
}

DATA = {
  "arrau": "/srv/local1/paxia/incremental_coref/data/arrau/test.512.jsonlines",
  "zh": "/srv/local1/paxia/incremental_coref/data/ontonotes_ml/chinese/test.chinese.512.jsonlines",
  "on": "MISSING",
  "litbank": "/srv/local1/paxia/incremental_coref/data/litbank/train.jsonlines",
  "qbcoref": "/srv/local1/paxia/incremental_coref/data/qbcoref/all_docs.512.jsonlines",
  "preco": "/srv/local1/paxia/incremental_coref/data/preco/dev.preco.512.jsonlines",
}
LOG_DIR = "/srv/local1/paxia/subprocess_logs/baseline_forgetting/logs/"

GPUs = [0, 1, 2, 3, 4, 5, 6, 7]

log_location = "baseline_forgetting"


def add_job(name, model_key, data_key, job_list):
  job_list.append({
    "name": name,
    "command": "inference.py",
    "log_location": log_location,
    "load_path": PATHS[model_key],
    "log_dir": LOG_DIR + "/" + model_key + "_" + name,
    "log_path": LOG_DIR + "/" + model_key + "_" + name + "/out.log",
    "dev_path": DATA[data_key],
    "load_model": "true",
  })
  job_list.append({
    "name": name,
    "command": "inference.py",
    "log_location": log_location,
    "load_path": PATHS[f"{model_key}_gold"],
    "mentions": "false",
    "use_gold_spans": "true",
    "log_dir": LOG_DIR + "/" + model_key + "_" + name + "_gold",
    "log_path": LOG_DIR + "/" + model_key + "_" + name + "_gold" + "/gold_out.log",
    "dev_path": DATA[data_key],
    "load_model": "true",
  })

job_list = []

# Zeros

# Litbank
#add_job("litbank_curve_da_0_25_5_0", "on", "litbank", job_list)
#add_job("litbank_curve_da_0_25_5_0", "preco", "litbank", job_list)

#add_job("preco_curve_da_25_5_0", "on", "preco", job_list)

# QBCoref
#add_job("qb_curve_da_0_25_5_0", "on", "qbcoref", job_list)
#add_job("qb_curve_da_0_25_5_0", "preco", "qbcoref", job_list)

# ARRAU
add_job("arrau_curve_da_25_80_0", "on", "arrau", job_list)
add_job("arrau_curve_da_25_80_0", "preco", "arrau", job_list)

# zh
#add_job("onml_curve_zh_da_25_5_0", "en", "zh", job_list)


if __name__ == "__main__":
  config = {
    "command": "inference.py",
    "log_location": "log_location",
    "jobs": job_list
  }
  start_jobs(config, GPUs)
