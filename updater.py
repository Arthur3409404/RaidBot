import os
import subprocess
import sys
import shutil
import time
import traceback
import re
import stat

# Public repository URL
REPO_URL = "https://github.com/Arthur3409404/RaidBot.git"
TEMP_DIR = "RaidBot_update_tmp"
ENTRY_POINT = "main.py"

# Files / folders that should never be replaced
EXCLUDE = {
    TEMP_DIR,
    ".git",
    ".env",
    "__pycache__",
    "updater.py",
}

VERSION_BRANCH_REGEX = re.compile(r"refs/heads/(v\d+\.\d+\.\d+)$")

# Windows equivalent of /dev/null
NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"

def remove_readonly(func, path, excinfo):
    """Clear read-only attribute and retry deletion (Windows)"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def get_latest_version_branch():
    """Get latest version branch from remote repo"""
    output = subprocess.check_output(
        ["git", "ls-remote", "--heads", REPO_URL],
        text=True
    )

    versions = []
    for line in output.splitlines():
        match = VERSION_BRANCH_REGEX.search(line)
        if match:
            versions.append(match.group(1))

    if not versions:
        raise RuntimeError("No version branches found (vX.Y.Z)")

    versions.sort(
        key=lambda v: tuple(map(int, v[1:].split("."))),
        reverse=True
    )

    return versions[0]

def main():
    base_dir = os.getcwd()
    temp_path = os.path.join(base_dir, TEMP_DIR)

    try:
        print("[UPDATE] Checking latest version branch...")
        branch = get_latest_version_branch()
        print(f"[UPDATE] Latest version: {branch}")

        # Remove temp folder if exists
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, onerror=remove_readonly)

        print("[UPDATE] Cloning repository...")
        subprocess.check_call([
            "git", "clone",
            "--branch", branch,
            "--single-branch",
            REPO_URL,
            TEMP_DIR
        ])

        print("[UPDATE] Applying update...")
        for item in os.listdir(temp_path):
            if item in EXCLUDE:
                continue

            src = os.path.join(temp_path, item)
            dst = os.path.join(base_dir, item)

            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst, onerror=remove_readonly)
                else:
                    os.remove(dst)

            shutil.move(src, dst)

        shutil.rmtree(temp_path, ignore_errors=True)

        print("[UPDATE] Restarting bot...")
        time.sleep(1)

        subprocess.Popen(
            [sys.executable, ENTRY_POINT],
            cwd=base_dir,
            creationflags=subprocess.DETACHED_PROCESS if os.name == "nt" else 0
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