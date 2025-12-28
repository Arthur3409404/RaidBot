import os
import subprocess
import sys
import shutil
import time
import traceback

REPO_URL = "https://github.com/Arthur3409404/RaidBot.git"
TEMP_DIR = "RaidBot_update_tmp"
ENTRY_POINT = "main.py"


def main():
    base_dir = os.getcwd()
    temp_path = os.path.join(base_dir, TEMP_DIR)

    try:
        print("[UPDATE] Cloning latest repository...")

        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        subprocess.check_call(["git", "clone", REPO_URL, TEMP_DIR])

        print("[UPDATE] Replacing old files...")

        for item in os.listdir(temp_path):
            src = os.path.join(temp_path, item)
            dst = os.path.join(base_dir, item)

            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)

            shutil.move(src, dst)

        shutil.rmtree(temp_path)

        print("[UPDATE] Restarting bot...")

        time.sleep(1)
        os.chdir(base_dir)

        subprocess.Popen(
            [sys.executable, ENTRY_POINT],
            creationflags=subprocess.DETACHED_PROCESS
        )

        print("[UPDATE] Update completed successfully.")
        sys.exit(0)

    except Exception:
        print("[UPDATE] Update failed:")
        traceback.print_exc()
        time.sleep(3)
        sys.exit(1)


if __name__ == "__main__":
    main()