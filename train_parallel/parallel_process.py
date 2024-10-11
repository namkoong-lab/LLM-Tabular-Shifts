import subprocess
import time
import torch

def parallel_process(command_list, max_concurrent_processes = 5):
    max_concurrent_processes = max_concurrent_processes
    processes = []
    try:
        # Loop over the command list and manage the process queue
        while command_list or processes:
            # If we haven't reached max concurrency and there are commands left to run
            while len(processes) < max_concurrent_processes and command_list:
                # Start a new process
                command = command_list.pop(0)
                #print(f"Starting process: {command}")
                processes.append(subprocess.Popen(command))
            # Check the status of active processes
            for process in list(processes):  # Use list to make a copy of the list for safe removal
                if process.poll() is not None:  # Check if process has finished
                    processes.remove(process)
            # Sleep briefly to avoid busy waiting
            time.sleep(1)
    finally:
        # Ensure all processes finish before script exits
        for process in processes:
            process.wait()
            #print("Process completed")
