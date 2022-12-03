import io
import re
from datetime import datetime
from pathlib import Path

from fabric import task, Connection, ThreadingGroup

from settings import (
    USER,
    HOSTS,
    REPO_PATH,
    CONDA_ENV,
    NIFLHEIM_SCRIPT_PREAMBLE,
    NIFLHEIM_VIRTUAL_ENV,
    NIFLHEIM_REPO_SCRATCH,
    NIFLHEIM_LOGIN_HOST,
    NIFLHEIM_PYTHON_MODULE,
)


# CogSys tasks


@task
def gpustat(c):
    """Run gpustat on all hosts.

    Example:
        $ fab gpustat
    """
    ThreadingGroup(*HOSTS, user=USER).run("gpustat")


@task
def screenls(c, host):
    """Run screen -ls on remote host.

    Example:
        $ fab screenls [host]
    """
    host = f"{host}.compute.dtu.dk"
    assert host in HOSTS, f"{host} not in list of known hosts."
    remote = Connection(host, user=USER)
    remote.run(f"screen -ls")


@task
def uptime(c, host):
    """Run uptime on remote host.

    Example:
        $ fab uptime [host]
    """
    host = f"{host}.compute.dtu.dk"
    assert host in HOSTS, f"{host} not in list of known hosts."
    remote = Connection(host, user=USER)
    remote.run(f"uptime")


@task
def run(c, host, device, script):
    """Run script on remote host.

    Example:
        $ fab run [host] [GPU index] "scripts/runner.py --target=U0"
    """
    host = f"{host}.compute.dtu.dk"
    assert host in HOSTS, f"{host} not in list of known hosts."
    assert device.isdigit(), "Device should be an integer."
    assert "--output_dir" not in script, "--output_dir is set automatically."
    commit_time_id = c.run("git show -s --format=%cI-%h HEAD", hide=True).stdout.strip()
    commit_id = commit_time_id.split("-")[-1]
    timestamp = datetime.now().isoformat()
    output_path = Path(f"runs/{timestamp}/")
    gitdetails_path = output_path / "gitdetails.txt"
    screenlog_path = output_path / "screenlog"
    screenrc_path = output_path / "screenrc"

    print(f"Process timestamp: {timestamp}")

    # Prepare commands
    # Source and activate python virtual environment
    env_cmds = f"""
    source /opt/miniconda3/bin/activate
    conda activate {CONDA_ENV}
    """
    # Run script in detached screen process
    cmd = f"screen -c {screenrc_path} -dmSL {timestamp} bash -c "
    cmd += f"'CUDA_VISIBLE_DEVICES={device} "
    cmd += f"python -u {script} --output_dir={output_path}'"
    # cmd += "python -c \"import torch; print(torch.cuda.is_available())\"'"

    # Run commands on remote host
    remote = Connection(host, user=USER)
    with remote.cd(REPO_PATH):
        # Checkout current commit on remote
        remote.run(f"git fetch && git checkout {commit_id}")
        # Create output directory
        if remote.run(f"test -d {output_path}", warn=True).failed:
            remote.run(f"mkdir -p {output_path}")
        # Write git details to file
        remote.run(f"echo {commit_time_id} > {gitdetails_path}")
        # Create screenrc config file
        remote.run(f"echo 'logfile {screenlog_path}' > {screenrc_path}")
        # Activate environment and run command
        with remote.prefix(chain(env_cmds)):
            remote.run("pip install --quiet -r requirements.txt")
            remote.run(cmd)
        # Tail screen log
        remote.run(f"tail -f {screenlog_path}")


@task
def tail(c, host, timestamp, n=10):
    """Tail screen log on remote host.

    Example:
        $ fab tail [host] [timestamp]
    """
    host = f"{host}.compute.dtu.dk"
    assert host in HOSTS, f"{host} not in list of known hosts."
    screenlog_path = Path(f"runs/{timestamp}/screenlog")
    remote = Connection(host, user=USER)
    with remote.cd(REPO_PATH):
        remote.run(f"tail -n {n} -f {screenlog_path}")


@task
def copy(c, host, timestamp):
    """Copy output folder from remote to local runs directory.

    Example:
        $ fab copy [host] [timestamp]
    """
    host = f"{host}.compute.dtu.dk"
    assert host in HOSTS, f"{host} not in list of known hosts."
    c.run(f"scp -r {host}:{REPO_PATH}/runs/{timestamp}/ runs/{timestamp}/")


# Niflheim tasks


@task
def submit(c, script, dataset_to_scratch=None):
    """Submit script as a GPU job to Niflheim cluster.

    Args:
        dataset_to_scratch: Absolute path to dataset that will be copied to /scratch.
                            This will set the --dataset argument of the [script]
                            to the new location on /scratch.

    Example:
        $ fab submit "scripts/runner.py --dataset path/to/dataset --target U0"
    """
    assert "--output_dir" not in script, "--output_dir is set automatically."

    commit_time_id = c.run("git show -s --format=%cI-%h HEAD", hide=True).stdout.strip()
    timestamp = datetime.now().isoformat()
    output_path = Path(f"~/graphnn_runs/{timestamp}/")

    common_submit(script, commit_time_id, timestamp, output_path, dataset_to_scratch)


@task
def submitn(c, n, script, dataset_to_scratch=None):
    """Submit multiple GPU jobs to Niflheim cluster with the same script and args.

    Args:
        dataset_to_scratch: Absolute path to dataset that will be copied to /scratch.
                            This will set the --dataset argument of the [script]
                            to the new location on /scratch.

    Example:
        $ fab submitn 2 "scripts/runner.py --dataset path/to/dataset --target U0"
    """
    assert n.isdigit(), "Device should be an integer."
    for i in range(int(n)):
        submit(c, script, dataset_to_scratch)


@task
def resume(c, timestamp, dataset_to_scratch=None):
    """Resume a submitted script at Niflheim cluster.

    Args:
        dataset_to_scratch: Absolute path to dataset that will be copied to /scratch.
                            This will set the --dataset argument of the [script]
                            to the new location on /scratch.

    Example:
        $ fab resume 2021-01-11T14:58:24.702153
    """
    load_path = Path(f"~/graphnn_runs/{timestamp}/")
    gitinfo_path = load_path / "gitdetails.txt"

    # Extract the script
    remote = Connection(NIFLHEIM_LOGIN_HOST, user=USER)
    commit_time_id = read_file(remote, f"{gitinfo_path}").strip()
    script = remote.run(
        f"grep -E '^[[:space:]]*python -u' {load_path}/submit_script.sh"
    ).stdout.strip()

    # Remove python -u --output_dir and --load_model arguments from script
    script_parts = re.split("\s+|=", script)
    todel = [0, 1]
    assert script_parts[0] == "python"
    assert script_parts[1] == "-u"
    todel.extend(
        [script_parts.index("--output_dir"), script_parts.index("--output_dir") + 1]
    )
    try:
        todel.extend(
            [script_parts.index("--load_model"), script_parts.index("--load_model") + 1]
        )
    except ValueError:
        pass
    script_parts = [s for i, s in enumerate(script_parts) if i not in todel]

    # Load splitfile if not already given
    if "--split_file" not in script_parts:
        script_parts.append("--split_file")
        script_parts.append(f"{load_path}/datasplits.json")
    # Load best model from previous run
    if "--load_model" not in script_parts:
        script_parts.append("--load_model")
        script_parts.append(f"{load_path}/best_model.pth")

    new_script = " ".join(script_parts)
    print("Resume command:", new_script)
    new_timestamp = datetime.now().isoformat()
    output_path = Path(f"~/graphnn_runs/{new_timestamp}/")

    common_submit(new_script, commit_time_id, new_timestamp, output_path, dataset_to_scratch)


@task
def qstat(c):
    """See your running jobs on Niflheim cluster"""
    remote = Connection(NIFLHEIM_LOGIN_HOST, user=USER)
    remote.run(f"qstat -u {USER}")


@task
def qstatf(c):
    """See more information about your running jobs on Niflheim cluster"""
    remote = Connection(NIFLHEIM_LOGIN_HOST, user=USER)
    remote.run(f"qstat -f  | grep -B 10 -A 5 'euser = {USER}'")


@task
def scancel(c, jobid):
    remote = Connection(NIFLHEIM_LOGIN_HOST, user=USER)
    remote.run(f"scancel {jobid}")


# Helpers


def chain(commands):
    """Chain commands from a multi-line string into a one-liner."""
    return " && ".join(c.strip() for c in commands.strip().splitlines())


def read_file(remote, file_path, encoding="utf-8"):
    if file_path.startswith("~"):
        file_path = file_path.replace("~", ".")
    io_obj = io.BytesIO()
    remote.get(file_path, io_obj)
    return io_obj.getvalue().decode(encoding)


def common_submit(script, commit_time_id, timestamp, output_path, dataset_to_scratch):

    commit_id = commit_time_id.split("-")[-1]
    submit_path = output_path / "submit_script.sh"
    repo_path = Path(NIFLHEIM_REPO_SCRATCH) / commit_time_id
    log_path = output_path / "slurmlog.log"
    gitdetails_path = output_path / "gitdetails.txt"

    remote = Connection(NIFLHEIM_LOGIN_HOST, user=USER)

    if dataset_to_scratch is not None:
        assert remote.run(f"test -f {dataset_to_scratch}", warn=True).ok
        dataset_filename = Path(dataset_to_scratch).name
        scratch_path = Path(f"/scratch/{USER}/{timestamp}/")

    modules = f"module load {NIFLHEIM_PYTHON_MODULE} foss"
    env_cmd = f"source {NIFLHEIM_VIRTUAL_ENV}/bin/activate"

    # Check that virtual environment exists and create if not
    if remote.run(f"test -d {NIFLHEIM_VIRTUAL_ENV}", warn=True).failed:
        print("Creating virtual environment...")
        with remote.prefix(modules):
            remote.run(f"virtualenv {NIFLHEIM_VIRTUAL_ENV}")
            with remote.prefix(env_cmd):
                # Asap3 does not detect its own numpy install dependency
                remote.run("pip install numpy")
                # We need newest version of torch to support RTX 3090
                remote.run(
                    "pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html"
                )

    # Install dependencies using requirements file
    remote.put("requirements.txt", ".")
    with remote.prefix(modules), remote.prefix(env_cmd):
        # Install remaining dependencies
        remote.run("pip install --quiet --quiet -r requirements.txt")
    remote.run("rm requirements.txt")

    # Create output directory
    if remote.run(f"test -d {output_path}", warn=True).failed:
        remote.run(f"mkdir -p {output_path}")

    # Write git details to file
    remote.run(f"echo {commit_time_id} > {gitdetails_path}")

    # Checkout required repository revision
    if remote.run(f"test -d {repo_path}", warn=True).failed:
        remote.run(f"git clone git@gitlab.gbar.dtu.dk:sure/graphnn.git {repo_path}")


    if dataset_to_scratch is not None:
        # Copy dataset to /scratch and clean up when job is done.
        script_body = f"""
        cd {repo_path}
        git fetch && git checkout {commit_id}
        mkdir {scratch_path}
        cp -v {dataset_to_scratch} {scratch_path}
        python -u {script} --dataset {scratch_path / dataset_filename} --output_dir {output_path}
        rm -rf {scratch_path}
        """
    else:
        script_body = f"""
        cd {repo_path}
        git fetch && git checkout {commit_id}
        python -u {script} --output_dir {output_path}
        """

    script = io.StringIO(NIFLHEIM_SCRIPT_PREAMBLE + script_body)

    # Put script in output path
    print(f"Writing job script to {submit_path}")
    # A bug in fabric means that remote.put does not expand tilde correctly
    string_path = f"{submit_path}"
    if string_path.startswith("~"):
        string_path = string_path.replace("~", ".", 1)
    remote.put(script, string_path)

    remote.run(
        f"sbatch --output {log_path} --job-name graphnn_{timestamp} {submit_path}"
    )
