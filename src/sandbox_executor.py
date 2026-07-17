import os
import uuid
import shutil
import docker
from docker.errors import ContainerError

SANDBOX_IMAGE = "ml-sandbox"


def run_in_sandbox(code: str, data_path: str, packages: list[str], data_path_in_container: str = "/workspace/data.csv") -> dict:
    """Runs `code` inside a fresh Docker container, with `data_path` mounted in.
    Returns {"success": bool, "stdout": str, "stderr": str}."""

    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.abspath(f"data/runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    fixed_code = code.replace(data_path, data_path_in_container)

    script_path = os.path.join(run_dir, "script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(fixed_code)

    shutil.copy(data_path, os.path.join(run_dir, "data.csv"))

    packages_str = " ".join(packages)
    # Enhanced pip install with retries and longer timeouts for network issues
    shell_cmd = f"pip install -q --retries 5 --default-timeout=100 {packages_str} && python script.py"

    client = docker.from_env()

    try:
        output = client.containers.run(
            SANDBOX_IMAGE,
            command=["sh", "-c", shell_cmd],
            volumes={run_dir: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            remove=True,
            stdout=True,
            stderr=True,
            network_mode="bridge",
            dns=["8.8.8.8", "8.8.4.4", "1.1.1.1"],
            cap_drop=["NET_RAW"],
        )
        return {"success": True, "stdout": output.decode("utf-8", errors="replace"), "stderr": "", "run_dir": run_dir}

    except ContainerError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        return {"success": False, "stdout": "", "stderr": stderr, "run_dir": run_dir}