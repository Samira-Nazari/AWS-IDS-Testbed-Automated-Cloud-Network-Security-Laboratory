from Feature_extraction import Feature_extraction
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Process
import numpy as np
import pandas as pd

DEFAULT_PCAP_DIRECTORY = "/home/ubuntu/aws_ids_testbed/input"
DEFAULT_CSV_OUTPUT_DIRECTORY = "/home/ubuntu/aws_ids_testbed/output/csv"
DEFAULT_WORK_DIRECTORY = "/home/ubuntu/aws_ids_testbed/tmp/pcap_to_csv"
DEFAULT_SUBFILE_SIZE_MB = 10
DEFAULT_THREADS = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IDS PCAP files to CSV feature files."
    )
    parser.add_argument(
        "--pcap-file",
        help="Convert one PCAP file.",
    )
    parser.add_argument(
        "--pcap-directory",
        default=DEFAULT_PCAP_DIRECTORY,
        help="Directory containing PCAP files to convert.",
    )
    parser.add_argument(
        "--csv-output-directory",
        default=DEFAULT_CSV_OUTPUT_DIRECTORY,
        help="Directory where converted CSV files will be saved.",
    )
    parser.add_argument(
        "--work-directory",
        default=DEFAULT_WORK_DIRECTORY,
        help="Temporary working directory for split PCAP and CSV chunk files.",
    )
    parser.add_argument(
        "--subfile-size-mb",
        type=int,
        default=DEFAULT_SUBFILE_SIZE_MB,
        help="Split PCAP chunk size in MB.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Number of parallel feature extraction processes.",
    )
    return parser.parse_args()


def clear_directory(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == '__main__':

    start = time.time()
    args = parse_args()
    print("========== IDS PCAP to CSV feature extraction ==========")
    
    pcap_directory = os.path.abspath(args.pcap_directory)
    csv_output_directory = os.path.abspath(args.csv_output_directory)
    work_directory = os.path.abspath(args.work_directory)

    if args.subfile_size_mb < 1:
        raise ValueError("--subfile-size-mb must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    if args.pcap_file:
        pcap_file = os.path.abspath(args.pcap_file)
        if not os.path.isfile(pcap_file):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
        if not pcap_file.endswith(".pcap"):
            raise ValueError(f"Input file must end with .pcap: {pcap_file}")
        pcap_directory = os.path.dirname(pcap_file)
        pcapfiles = [pcap_file]
    else:
        pcapfiles = []
        for root, dirs, files in os.walk(pcap_directory):
            for file_name in files:
                if file_name.endswith('.pcap'):
                    pcapfiles.append(os.path.join(root, file_name))
    pcapfiles.sort()
    subfiles_size = args.subfile_size_mb
    split_directory = os.path.join(work_directory, 'split_temp') + os.sep
    destination_directory = os.path.join(work_directory, 'output') + os.sep
    os.makedirs(split_directory, exist_ok=True)
    os.makedirs(destination_directory, exist_ok=True)
    os.makedirs(csv_output_directory, exist_ok=True)
    n_threads = args.threads

    print(f"PCAP input directory: {pcap_directory}")
    print(f"CSV output directory: {csv_output_directory}")
    print(f"Work directory: {work_directory}")
    print(f"Split size: {subfiles_size} MB")
    print(f"Threads: {n_threads}")
    print(f"PCAP files found: {len(pcapfiles)}")
    
    address = "./"
    
        

    
    for i in range(len(pcapfiles)):
        lstart = time.time()
        pcap_file = pcapfiles[i]
        relative_pcap_path = os.path.relpath(pcap_file, pcap_directory)
        output_csv_file = os.path.join(csv_output_directory, os.path.splitext(relative_pcap_path)[0] + '.csv')
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        if os.path.exists(output_csv_file) and os.path.getsize(output_csv_file) > 0:
            print(f"Skipping existing output: {output_csv_file}")
            continue
        clear_directory(split_directory)
        clear_directory(destination_directory)
        print(pcap_file)
        print(">>>> 1. splitting the .pcap file.")
        subprocess.run(
            [
                'tcpdump',
                '-r', pcap_file,
                '-w', os.path.join(split_directory, 'split_temp'),
                '-C', str(subfiles_size),
            ],
            check=True,
        )
        subfiles = sorted(
            f for f in os.listdir(split_directory)
            if os.path.isfile(os.path.join(split_directory, f))
        )
        if not subfiles:
            raise RuntimeError(f"No split files were created for {pcap_file}")
        print(">>>> 2. Converting (sub) .pcap files to .csv files.")
        processes = []
        errors = 0
        
        n_batches = max(1, int(np.ceil(len(subfiles) / n_threads)))
        subfiles_threadlist = np.array_split(subfiles, n_batches)
        for f_list in tqdm(subfiles_threadlist):
            n_processes = min(len(f_list), n_threads)
            assert n_threads >= n_processes
            assert n_threads >= len(f_list)
            processes = []
            for i in range(n_processes):
                fe = Feature_extraction()
                f = f_list[i]
                subpcap_file = split_directory + f
                p = Process(target=fe.pcap_evaluation, args=(subpcap_file,destination_directory + f.split('.')[0]))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"Feature extraction process failed with exit code {p.exitcode}")
        print('The length of subfiles : ', len(subfiles))
        print('The length of destination directory : ', len(os.listdir(destination_directory)))
     #   assert len(subfiles)==len(os.listdir(destination_directory))
        print(">>>> 3. Removing (sub) .pcap files.")
        for sf in subfiles:
            os.remove(split_directory + sf)

        print(">>>> 4. Merging (sub) .csv files (summary).")
        
        csv_subfiles = sorted(
            f for f in os.listdir(destination_directory)
            if os.path.isfile(os.path.join(destination_directory, f))
        )
        if not csv_subfiles:
            raise RuntimeError(f"No CSV chunk files were created for {pcap_file}")
        mode = 'w'
        for f in tqdm(csv_subfiles):
            d = pd.read_csv(destination_directory + f)
            d.to_csv(output_csv_file, header=mode=='w', index=False, mode=mode)
            mode='a'

        print(">>>> 5. Removing (sub) .csv files.")
        for cf in tqdm(csv_subfiles):
            os.remove(destination_directory + cf)
        print(f"CSV output: {output_csv_file}")
        print(f'done! ({pcap_file})(' + str(round(time.time()-lstart, 2))+ 's),  total_errors= '+str(errors))
        
    end = time.time()
    print(f'Elapsed Time = {(end-start)}s')
    
    
    
