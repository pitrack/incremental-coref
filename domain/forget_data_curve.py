from base_data_curve import start_jobs

DATA = {
  "en": "/srv/local1/paxia/incremental_coref/data/ontonotes_ml/english/test.english.512.jsonlines",
  "on": "/srv/local1/paxia/incremental_coref/data/ontonotes/test.english.512.jsonlines",
  "preco": "/srv/local1/paxia/incremental_coref/data/preco/dev.preco.512.jsonlines",
}

LOG_DIR = "/srv/local1/paxia/subprocess_logs/forgetting_continued/logs/"
GPUs = [1, 2, 4, 5, 6, 7]

NAMES = {
  "on": "on_25_2802",
  "en": "onml_en_baseline",
  "preco": "fb_preco_25_10",
}

path_model = []

# Base models
path_model.append(("/srv/local1/paxia/exp_logs/public_icoref/ontonotes/checkpoint.bin", "on"))
path_model.append(("/srv/local1/paxia/exp_logs/public_icoref/preco/checkpoint.bin", "preco"))
path_model.append(("/srv/local1/paxia/exp_logs/public_icoref/ontonotes_en/checkpoint.bin", "en"))

# Litbank (601)
# for split in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#   for size in [5, 10, 20, 40, 80]:
#     # path_model.append((f"/srv/local1/paxia/exp_logs/preco_to_litbank/litbank_da_{split}_25_{size}_0/checkpoint.bin",
#     #                    "preco"))
#     path_model.append((f"/srv/local1/paxia/exp_logs/on_to_litbank/litbank_da_{split}_25_{size}_0/checkpoint.bin",
#                        "on"))

# # ARRAU (601)
    # path_model.append((f"/srv/local1/paxia/exp_logs/preco_to_litbank/litbank_da_{split}_25_{size}_0/checkpoint.bin",
    #                    "preco"))
    # path_model.append((f"/srv/local1/paxia/exp_logs/on_to_litbank/litbank_da_{split}_25_{size}_0/checkpoint.bin",
    #                    "on"))

# QBCoref (603)
# for split in [0, 1, 2, 3, 4]:
#   for size in [5, 15, 30, 60, 120, 240]:
#     path_model.append((f"/srv/local1/paxia/exp_logs/preco_to_qbcoref/qbcoref_da_{split}_25_{size}_0/checkpoint.bin",
#                        "preco"))
#     path_model.append((f"/srv/local1/paxia/exp_logs/on_to_qbcoref/qbcoref_da_{split}_25_{size}_0/checkpoint.bin",
#                        "on"))

# SARA

for split in [0, 1, 2, 3, 4, 5, 6]:
  for size in [10, 20, 40, 80, 200]:
    path_model.append((f"/srv/local2/paxia/exp_logs/local1/on_sara/sara_da_{split}_25_{size}_0/checkpoint.bin",
                       "on"))
    path_model.append((f"/srv/local2/paxia/exp_logs/local1/preco_sara/sara_da_{split}_25_{size}_0/checkpoint.bin",
                       "preco"))

# # PreCo (603)
# for size in [5, 10, 25, 50, 100, 250, 500]:
#   path_model.append((f"/srv/local1/paxia/exp_logs/preco_data_curve_2/preco_da_25_{size}_0/checkpoint.bin",
#                      "on"))

# Chinese (601)
# for size in [10, 25, 50, 100, 250, 500, 1810]:
#   path_model.append((f"/srv/local1/paxia/exp_logs/onen_to_zh/onml_zh_da_25_{size}_0/checkpoint.bin",
#                      "en"))

# Arabic + semeval (603)
# for size in [10, 20, 40, 80, 160, 359]:
#   path_model.append((f"/srv/local1/paxia/exp_logs/on_ar/onml_ar_da_25_{size}_0/checkpoint.bin",
#                      "en"))

# ca
for size in [10, 25, 50, 100, 250, 829]:
  path_model.append((f"/srv/local1/paxia/exp_logs/semeval/onml_ca_da_25_{size}_0/checkpoint.bin",
                     "en"))
# es
for size in [10, 25, 50, 100, 250, 875]:
  path_model.append((f"/srv/local1/paxia/exp_logs/semeval/onml_es_da_25_{size}_0/checkpoint.bin",
                     "en"))

# it
for size in [10, 20, 40, 80]:
  path_model.append((f"/srv/local1/paxia/exp_logs/semeval/onml_it_da_25_{size}_0/checkpoint.bin",
                     "en"))

# nl
for size in [10, 20, 40, 80, 145]:
  path_model.append((f"/srv/local1/paxia/exp_logs/semeval/onml_nl_da_25_{size}_0/checkpoint.bin",
                     "en"))



log_location = "forgetting_continued"


def add_job(name, model_path, data_key, job_list):
  model_key = model_path.split("/")[-2]
  job_list.append({
    "name": name,
    "command": "inference.py",
    "log_location": log_location,
    "load_path": model_path,
    "log_dir": LOG_DIR + "/" + name + "/" + model_key,
    "log_path": LOG_DIR + "/" + name + "/" + model_key + "/out.log",
    "dev_path": DATA[data_key],
    "load_model": "true",
  })

job_list = []

# Add models correctly to job list
for model_path, model in path_model:
  add_job(NAMES[model], model_path, model, job_list)


if __name__ == "__main__":
  config = {
    "command": "inference.py",
    "log_location": log_location,
    "jobs": job_list,
  }
  start_jobs(config, GPUs)
