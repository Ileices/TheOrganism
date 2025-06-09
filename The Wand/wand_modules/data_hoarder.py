import os
import time

def execute(config):
    target_dir = config.get("directory", os.getcwd())
    total_files = 0
    total_size = 0

    for root, dirs, files in os.walk(target_dir):
        total_files += len(files)
        for name in files:
            try:
                total_size += os.stat(os.path.join(root, name)).st_size
            except Exception:
                continue

    print("Data Hoarder Metrics:")
    print(f"Total files: {total_files}")
    print(f"Total size (bytes): {total_size}")
    # ...additional metrics like new files per day, processing frequency, etc.
