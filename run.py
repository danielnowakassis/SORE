import subprocess
import smtplib
import email.message
import os
from pathlib import Path
import shlex
import tempfile
import time

import sys


from generate_synth import *

datasets = [
agrawal([1,2,3,4,5,6,7,8,9,10], abrupt = True),
agrawal([10,9,8,7,6,5,4,3,2,1], abrupt = False),
sea([1,2,3,4,3,2,1], abrupt = True),
sea([1,2,3,4,3,2,1], abrupt = False),
led([7,5,3,1,3,5,7], abrupt = True),
led([7,5,3,1,3,5,7], abrupt = False),
"-i 1000000 -f 1000 -s (generators.RandomRBFGeneratorDrift -c 5 -s .0001)",
"-i 1000000 -f 1000 -s (generators.RandomRBFGeneratorDrift -c 5 -s .001)",
"-i 1000000 -f 1000 -s (generators.HyperplaneGenerator  -k 10 -t .001)",
f"-f 1000 -s (ArffFileStream -f CaDrift_a.arff)",
f"-f 1000 -s (ArffFileStream -f CaDrift_g.arff)",
f"-f 1000 -s (ArffFileStream -f CaDrift_i.arff)",

f"-f 100 -s (ArffFileStream -f outdoor.arff)",
f"-f 1000 -s (ArffFileStream -f elecNormNew.arff)",
f"-f 100 -s (ArffFileStream -f gassensor.arff)",
f"-f 1000 -s (ArffFileStream -f sensorstream.arff)",
f"-f 1000 -s (ArffFileStream -f rialto.arff)",
f"-f 1000 -s (ArffFileStream -f covtypeNorm.arff)",
f"-f 1000 -s (ArffFileStream -f nomao.arff)",
f"-f 1000 -s (ArffFileStream -f poker-lsn.arff)",
f"-f 1000 -s (ArffFileStream -f NOAA.arff)",
f"-f 1000 -s (ArffFileStream -f INSECTS-abrupt_balanced_norm.arff)",
f"-f 1000 -s (ArffFileStream -f INSECTS-incremental_balanced_norm.arff)",
f"-f 1000 -s (ArffFileStream -f INSECTS-gradual_balanced_norm.arff)",
f"-f 1000 -s (ArffFileStream -f SmartMeter_LADPU.arff)",
f"-f 100 -s (ArffFileStream -f Asfault.arff)",

]

datasets_name = [
    "AGR_a",
    "AGR_g",
    "SEA_a",
    "SEA_g",
    "LED_a",
    "LED_g",
    "RBF_m",
    "RBF_f",
    "HYPER",
    "CaDrift_a",
    "CaDrift_g",
    "CaDrift_i",

    "Outdoor",
    "Elec",
    "Gas",
    "Sensor",
    "Rialto",
    "CovType",
    "Nomao",
    "Poker",
    "NOAA",
    "INSECTS_a",
    "INSECTS_i",
    "INSECTS_g",
    "LADPU",
    "Asfault"
]



seeds = [i for i in range(42,51)]



ensemble_size = 100
random_seed = 42


algs = {
    "SORE" : "SORE"
}


add_nj = True

def get_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    return rss_kb / 1024.0
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def run_with_ram_tracking(comando, log_file_path, interval_seconds=10):
    process = subprocess.Popen(shlex.split(comando))
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    first = False
    with log_path.open("a", encoding="utf-8") as log_file:
        while process.poll() is None:
            if first:
                rss_mb = get_rss_mb(process.pid)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if rss_mb is None:
                    log_file.write(f"[{timestamp}] PID {process.pid} RAM: unavailable\n")
                else:
                    log_file.write(f"[{timestamp}] PID {process.pid} RAM: {rss_mb:.2f} MB\n")
                log_file.flush()
            first = True
            time.sleep(interval_seconds)
    return process.returncode


for (alg_name, alg_str) in algs.items():
    diretorio = Path(f"{os.getcwd()}/resultados/{alg_name}_{ensemble_size}_{random_seed}")
    diretorio.mkdir(parents=True, exist_ok=True)
    add_nj = alg_name == "SGBT"
    for dataset, dataset_name in zip(datasets, datasets_name):
        if not (alg_name == "SGBT" and dataset_name in ["CaDrift_a","CaDrift_g","CaDrift_i", "Outdoor", "Sensor"]):
            comando = f"java -Xms4g -Xmx27g -cp {os.getcwd()}/moa.jar  moa.DoTask \"EvaluatePrequential -l (meta.{alg_str} -s {ensemble_size} -S {random_seed} {'-j -1' if not add_nj else ''} ) -e (BasicClassificationPerformanceEvaluator -o) {dataset} -d {os.getcwd()}/resultados/{alg_name}_{ensemble_size}_{random_seed}/{dataset_name}.csv \""
            ram_log_file = f"{os.getcwd()}/resultados/{alg_name}_{ensemble_size}_{random_seed}/{dataset_name}_ram.log"
            run_with_ram_tracking(comando, ram_log_file, interval_seconds=15)
print("enviado")
