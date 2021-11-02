import time
import queue
import subprocess
import os
import errno
from threading import Thread
import logging

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def manage_gpu(q, gpu, command):
  """
  This function is called per thread. It finds the next unclaimed job off the
  queue and starts running the command passed in. The command is usually
  `trainer.py` but it could also be `inference.py`.
  q: a Queue of job configs defined in start_jobs. Each job config needs:
    name: name of jsonnet exp config
    log_location: location of logs
    In addition, any key that needs to be overriden should be passed in the config.
  gpu: the number (e.g. 0) of the cuda device
  command: a single python function, e.g. `trainer.py`
  """
  new_env = os.environ.copy()
  new_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  while not q.empty():
    config = q.get()
    job = config["name"]
    mkdirs(f"/srv/local1/paxia/subprocess_logs/{config['log_location']}")
    logging.info(f"getting next task {job} on {gpu} ({q.qsize()} remain)")
    err_file = open(f"/srv/local1/paxia/subprocess_logs/{config['log_location']}/e.{job}", "a+")
    command_list = (["python", f"/srv/local1/paxia/incremental_coref/{command}", job] +
                    [f"{k}={v}" for k, v in config.items()])
    subprocess.run(command_list, env=new_env, stdout=err_file, stderr=err_file)
    err_file.close()
    logging.info(f"finished {job} at {config['log_location']}")
    q.task_done()

def start_jobs(config, gpus):
  """
  This function takes an experiment config and a list of available GPUs (ids)
  and sets up a queue before launching all of them until the queue is empty.
  config: a dict containing
    jobs: list of job configs to be enqueued
    command: Optional, command that is ultimately called (`inference.py` or `trainer.py`)
a list of job configs (dict) containing keys for:
  gpus: a list of integers.
  """
  if "command" not in config:
    config["command"] = "trainer.py"
  q = queue.Queue()
  for job_config in config["jobs"]:
    q.put(job_config)
  logging.info(f"Queued {q.qsize()} jobs")
  for gpu in gpus:
    time.sleep(1)
    worker = Thread(target=manage_gpu, args=(q, gpu, config["command"]))
    logging.info(f"Starting {gpu}")
    worker.start()
  logging.info("Waiting for all to finish")
  q.join()
  logging.info("Finished")
