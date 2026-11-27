"""Where trained models and results live, and how released models are fetched.

Everything an agent produces (model weights, training checkpoints, tensorboard logs and
evaluation results) goes under one ``local_data`` directory, with one subfolder per
environment, then per agent, then per run::

    local_data/<env_name>/<agent_name>/<run_name>/
        model/         # what the agent loads
        checkpoints/   # intermediate training checkpoints (PPO)
        logs/          # tensorboard
        eval/          # grid2op Runner output

``local_data`` is not tracked by git. The reference models used by the scripts, notebooks
and the app are published as assets of a GitHub release instead, and downloaded into
``local_data/<env_name>/<agent_name>/<release tag>/`` the first time they are needed.
See the "Pretrained models" section of the README.

Command line::

    python -m ExpertAgent.utils.local_data list
    python -m ExpertAgent.utils.local_data download            # all released models
    python -m ExpertAgent.utils.local_data download PPO_SB3     # only one agent
    python -m ExpertAgent.utils.local_data package --help      # maintainers: build a release asset
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: Environment variable that overrides the location of ``local_data``.
LOCAL_DATA_ENV_VAR = "EXPERTAGENT_LOCAL_DATA"
#: Environment variable that overrides the GitHub repository hosting the release.
MODELS_REPO_ENV_VAR = "EXPERTAGENT_MODELS_REPO"
#: Environment variables read (in this order) for a GitHub token, needed for a private repository.
TOKEN_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN")

#: GitHub repository (``owner/name``) whose releases hold the model files.
MODELS_REPO = "ainetus/T2.1_deep_expert"

#: Name of the folder, inside a run directory, that holds what the agent loads.
MODEL_DIRNAME = "model"
#: File written next to an extracted release model, recording which asset it came from.
MARKER_FILENAME = ".release.json"
#: Folder of ``local_data`` where downloaded release assets are kept.
DOWNLOADS_DIRNAME = ".downloads"

_CHUNK = 1 << 20


@dataclass(frozen=True)
class ReleaseModel:
    """One model published as a release asset.

    Attributes
    ----------
    agent_name: str
        The agent, as used in ``local_data/<env_name>/<agent_name>``.
    env_name: str
        The grid2op environment the model was trained on.
    tag: str
        The GitHub release (tag) the asset is attached to. A release is never changed once
        published: a new model goes into a new release, with a new tag.
    asset: str
        The file name of the asset in the release (a zip archive containing ``model/...``).
    sha256: str
        The checksum of the asset, checked after every download.
    load_path: str
        The path, relative to the run directory, to give to the agent's ``load``.
    description: str
        A short human readable description.
    """
    agent_name: str
    env_name: str
    tag: str
    asset: str
    sha256: str
    load_path: str
    description: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.agent_name, self.env_name)


#: The models the code uses by default. To publish a new version of a model, build its
#: asset with ``python -m ExpertAgent.utils.local_data package``, attach it to a NEW release
#: and replace the corresponding entry below (tag, asset and sha256).
RELEASE_MODELS: Dict[Tuple[str, str], ReleaseModel] = {
    m.key: m for m in (
        ReleaseModel(
            agent_name="DeepQExpert",
            env_name="l2rpn_case14_sandbox",
            tag="models-v1",
            asset="DeepQExpert_l2rpn_case14_sandbox_models-v1.zip",
            sha256="56d4d75caf728e93e5f9440f5bc49eaed5fb823babe247fb7a60e4c7f81613b9",
            load_path=MODEL_DIRNAME,
            description="Extended DeepQ agent, 1M training steps",
        ),
        ReleaseModel(
            agent_name="PPO_SB3",
            env_name="ai4realnet_small",
            tag="models-v1",
            asset="PPO_SB3_ai4realnet_small_models-v1.zip",
            sha256="4bb0211e9516a02cd47d3497e495ef801fd545f3cc29eb1e01fcefd3aae0ab96",
            load_path=f"{MODEL_DIRNAME}/PPO_SB3",
            description="PPO (stable-baselines3) policy used by ExpertAgentRL and the app",
        ),
    )
}


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------
def _repo_root() -> Optional[str]:
    """The source checkout containing the ``ExpertAgent`` package, if there is one."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(package_dir)
    if os.path.isfile(os.path.join(root, "setup.py")):
        return root
    return None


def get_local_data_dir(create: bool = True) -> str:
    """The ``local_data`` directory.

    In order: the ``EXPERTAGENT_LOCAL_DATA`` environment variable, ``<repo>/local_data``
    when running from a source checkout, else ``~/.cache/ExpertAgent/local_data``
    (for a non-editable installation).
    """
    path = os.environ.get(LOCAL_DATA_ENV_VAR)
    if not path:
        root = _repo_root()
        if root is not None:
            path = os.path.join(root, "local_data")
        else:
            path = os.path.join(os.path.expanduser("~"), ".cache", "ExpertAgent", "local_data")
    path = os.path.abspath(os.path.expanduser(path))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_env_name(env) -> str:
    """The name of a grid2op environment, as used for its folder in ``local_data``.

    ``env.env_name`` has the backend class appended (``ai4realnet_smallLightSimBackend``),
    so the name of the environment's data folder is used instead.
    """
    try:
        return os.path.basename(os.path.normpath(env.get_path_env()))
    except Exception:
        return env.env_name


def new_run_name(prefix: str = "run") -> str:
    """A run name that does not overwrite earlier runs, e.g. ``run_20261127-153000``."""
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}"


def get_run_dir(env_name: str,
                agent_name: str,
                run_name: Optional[str] = None,
                *subdirs: str,
                create: bool = True) -> str:
    """``local_data/<env_name>/<agent_name>/<run_name>[/<subdirs>...]``.

    Parameters
    ----------
    env_name: str
        The grid2op environment name (e.g. ``"ai4realnet_small"``).
    agent_name: str
        The agent name (e.g. ``"PPO_SB3"``, ``"DeepQExpert"``, ``"ExpertHeuristic"``).
    run_name: str, optional
        The run; a new time-stamped name (see :func:`new_run_name`) when ``None``.
    *subdirs: str
        Optional subfolders, e.g. ``"model"``, ``"logs"``, ``"eval"``.
    create: bool
        Create the directory if it does not exist.
    """
    if run_name is None:
        run_name = new_run_name()
    path = os.path.join(get_local_data_dir(create=create), env_name, agent_name, run_name, *subdirs)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------------------
# Release models
# ---------------------------------------------------------------------------------------
def get_release_model_info(agent_name: str, env_name: str) -> ReleaseModel:
    try:
        return RELEASE_MODELS[(agent_name, env_name)]
    except KeyError:
        available = ", ".join(f"{a} on {e}" for a, e in RELEASE_MODELS)
        raise KeyError(f"No released model for agent {agent_name!r} on environment {env_name!r}. "
                       f"Available: {available}") from None


def get_release_model_dir(agent_name: str, env_name: str, create: bool = False) -> str:
    """The run directory of a released model: ``local_data/<env>/<agent>/<tag>``."""
    info = get_release_model_info(agent_name, env_name)
    return get_run_dir(env_name, agent_name, info.tag, create=create)


def get_release_model(agent_name: str,
                      env_name: str,
                      force_download: bool = False,
                      verbose: bool = True) -> str:
    """Path to give to the agent's ``load``, downloading the model first if needed.

    The model is downloaded once, checked against its sha256, and extracted to
    ``local_data/<env_name>/<agent_name>/<tag>/model``. Later calls only return the path.

    Parameters
    ----------
    agent_name: str
        ``"DeepQExpert"`` or ``"PPO_SB3"`` (see ``RELEASE_MODELS``).
    env_name: str
        The environment the model was trained on.
    force_download: bool
        Download and extract again, even if the model is already there.
    verbose: bool
        Print what is being downloaded.

    Returns
    -------
    str
        For DeepQExpert the directory containing ``DeepQExpert/``; for PPO_SB3 the model
        path without the ``.zip`` extension, as ``PPO.load`` expects.
    """
    info = get_release_model_info(agent_name, env_name)
    run_dir = get_run_dir(env_name, agent_name, info.tag, create=True)
    model_dir = os.path.join(run_dir, MODEL_DIRNAME)
    load_path = os.path.join(run_dir, info.load_path)

    if not force_download and _is_installed(model_dir, info):
        return load_path

    if os.path.exists(model_dir) and not os.path.isfile(os.path.join(model_dir, MARKER_FILENAME)):
        if not force_download:
            raise RuntimeError(
                f"{model_dir} exists but was not extracted from release {info.tag} "
                f"({MARKER_FILENAME} is missing). Move it away, or call with force_download=True "
                f"(CLI: --force) to replace it with the released model.")

    archive = _download_asset(info, force=force_download, verbose=verbose)
    _extract(archive, run_dir, info)
    if verbose:
        print(f"[ExpertAgent] {agent_name} ({env_name}, {info.tag}) ready in {model_dir}")
    return load_path


def _is_installed(model_dir: str, info: ReleaseModel) -> bool:
    marker = os.path.join(model_dir, MARKER_FILENAME)
    if not os.path.isfile(marker):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            content = json.load(f)
    except (OSError, ValueError):
        return False
    return content.get("tag") == info.tag and content.get("sha256") == info.sha256


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_token() -> Optional[str]:
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def _models_repo() -> str:
    return os.environ.get(MODELS_REPO_ENV_VAR) or MODELS_REPO


def _open(url: str, token: Optional[str], accept: Optional[str] = None):
    request = urllib.request.Request(url, headers={"User-Agent": "ExpertAgent-model-download"})
    if accept:
        request.add_header("Accept", accept)
    if token:
        # An "unredirected" header is not forwarded when GitHub redirects the download
        # to its storage host, so the token is only ever sent to api.github.com.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=60)


def _asset_url(info: ReleaseModel, repo: str, token: Optional[str]) -> Tuple[str, Optional[str]]:
    """The URL to download the asset from, and the Accept header it needs."""
    if token is None:
        return f"https://github.com/{repo}/releases/download/{info.tag}/{info.asset}", None
    # With a token (needed for a private repository) go through the REST API.
    api = f"https://api.github.com/repos/{repo}/releases/tags/{info.tag}"
    with _open(api, token, accept="application/vnd.github+json") as response:
        release = json.load(response)
    for asset in release.get("assets", []):
        if asset.get("name") == info.asset:
            return asset["url"], "application/octet-stream"
    raise RuntimeError(f"Release {info.tag} of {repo} has no asset named {info.asset}.")


def _download_asset(info: ReleaseModel, force: bool = False, verbose: bool = True) -> str:
    """Download (or reuse) the asset in ``local_data/.downloads/<tag>/`` and check it."""
    if not _is_valid_sha256(info.sha256):
        raise RuntimeError(f"The checksum of {info.asset} is not set in RELEASE_MODELS "
                           f"(ExpertAgent/utils/local_data.py): the release is not published yet.")
    downloads = os.path.join(get_local_data_dir(), DOWNLOADS_DIRNAME, info.tag)
    os.makedirs(downloads, exist_ok=True)
    target = os.path.join(downloads, info.asset)

    if os.path.isfile(target) and not force:
        if _sha256(target) == info.sha256:
            return target
        os.remove(target)

    repo = _models_repo()
    token = _get_token()
    try:
        url, accept = _asset_url(info, repo, token)
        if verbose:
            print(f"[ExpertAgent] Downloading {info.asset} from release {info.tag} of {repo}")
        fd, tmp = tempfile.mkstemp(dir=downloads, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, _open(url, token, accept=accept) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    out.write(chunk)
                    done += len(chunk)
                    if verbose and total:
                        print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
            if verbose and total:
                print()
            actual = _sha256(tmp)
            if actual != info.sha256:
                raise RuntimeError(f"Checksum mismatch for {info.asset}: expected {info.sha256}, "
                                   f"got {actual}. The release asset differs from the one this "
                                   f"code expects.")
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code == 404:
            hint = (" The release or the asset does not exist, or the repository is private: "
                    f"set {TOKEN_ENV_VARS[0]} to a token that can read {repo}.")
        elif exc.code in (401, 403):
            hint = f" Check that the token in {'/'.join(TOKEN_ENV_VARS)} can read {repo}."
        raise RuntimeError(f"Could not download {info.asset} (HTTP {exc.code}).{hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {info.asset}: {exc.reason}. You can also download "
                           f"it by hand and put it in {downloads}.") from exc
    return target


def _is_valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _extract(archive: str, run_dir: str, info: ReleaseModel) -> None:
    """Extract the archive's ``model/`` folder into ``run_dir``, replacing an older copy."""
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:
            parts = name.replace("\\", "/").split("/")
            if parts[0] != MODEL_DIRNAME or name.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unexpected entry {name!r} in {archive}: every entry must be "
                                   f"under {MODEL_DIRNAME}/.")
        staging = tempfile.mkdtemp(dir=run_dir, prefix=".extract-")
        try:
            zf.extractall(staging)
            model_dir = os.path.join(run_dir, MODEL_DIRNAME)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.replace(os.path.join(staging, MODEL_DIRNAME), model_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    with open(os.path.join(run_dir, MODEL_DIRNAME, MARKER_FILENAME), "w", encoding="utf-8") as f:
        json.dump({"repo": _models_repo(), "tag": info.tag, "asset": info.asset,
                   "sha256": info.sha256}, f, indent=2)


# ---------------------------------------------------------------------------------------
# Building a release asset (maintainers)
# ---------------------------------------------------------------------------------------
def package_model(model_dir: str, agent_name: str, env_name: str, tag: str, out_dir: str) -> Tuple[str, str]:
    """Zip a model folder into a release asset.

    The archive holds the folder's files under ``model/``. It is reproducible: the same
    files always give the same archive, hence the same sha256.

    Returns
    -------
    (path, sha256)
    """
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{agent_name}_{env_name}_{tag}.zip")

    files = []
    for root, dirs, filenames in os.walk(model_dir):
        dirs.sort()
        for filename in sorted(filenames):
            if filename == MARKER_FILENAME:
                continue
            full = os.path.join(root, filename)
            files.append((full, os.path.relpath(full, model_dir).replace(os.sep, "/")))
    if not files:
        raise RuntimeError(f"{model_dir} is empty.")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for full, rel in files:
            entry = zipfile.ZipInfo(f"{MODEL_DIRNAME}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            with open(full, "rb") as src, zf.open(entry, "w") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
    return path, _sha256(path)


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------
def _cmd_list(_args) -> int:
    print(f"local_data: {get_local_data_dir(create=False)}")
    print(f"release repository: {_models_repo()}\n")
    for info in RELEASE_MODELS.values():
        model_dir = os.path.join(get_release_model_dir(info.agent_name, info.env_name), MODEL_DIRNAME)
        state = "downloaded" if _is_installed(model_dir, info) else "not downloaded"
        print(f"{info.agent_name:<12} {info.env_name:<22} {info.tag:<10} {state}")
        print(f"{'':<12} {info.description}")
    return 0


def _cmd_download(args) -> int:
    selected = [m for m in RELEASE_MODELS.values()
                if (not args.agent or m.agent_name == args.agent)
                and (not args.env or m.env_name == args.env)]
    if not selected:
        print("No released model matches.", file=sys.stderr)
        return 1
    for info in selected:
        print(get_release_model(info.agent_name, info.env_name, force_download=args.force))
    return 0


def _cmd_package(args) -> int:
    path, digest = package_model(args.model_dir, args.agent, args.env, args.tag, args.out)
    load_path = MODEL_DIRNAME if not args.load_path else f"{MODEL_DIRNAME}/{args.load_path}"
    print(f"asset:  {path}")
    print(f"size:   {os.path.getsize(path) / 2**20:.1f} MB")
    print(f"sha256: {digest}\n")
    print("Attach the asset to a NEW release with this tag, then put this entry in RELEASE_MODELS:\n")
    print(f"""        ReleaseModel(
            agent_name="{args.agent}",
            env_name="{args.env}",
            tag="{args.tag}",
            asset="{os.path.basename(path)}",
            sha256="{digest}",
            load_path="{load_path}",
            description="...",
        ),""")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ExpertAgent.utils.local_data",
                                     description="Manage local_data and the released models.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show the released models and whether they are downloaded.")

    p_down = sub.add_parser("download", help="Download released models into local_data.")
    p_down.add_argument("agent", nargs="?", help="Only this agent (e.g. PPO_SB3).")
    p_down.add_argument("--env", help="Only models trained on this environment.")
    p_down.add_argument("--force", action="store_true", help="Download and extract again.")

    p_pack = sub.add_parser("package", help="Build a release asset from a trained model folder.")
    p_pack.add_argument("--model_dir", required=True,
                        help="The folder the agent loads from, e.g. local_data/<env>/<agent>/<run>/model")
    p_pack.add_argument("--agent", required=True, help="Agent name, e.g. DeepQExpert or PPO_SB3.")
    p_pack.add_argument("--env", required=True, help="Environment the model was trained on.")
    p_pack.add_argument("--tag", required=True, help="The NEW release tag, e.g. models-v2.")
    p_pack.add_argument("--load_path", default="",
                        help="Path inside the model folder given to the agent's load (PPO_SB3: PPO_SB3).")
    p_pack.add_argument("--out", default="dist", help="Where to write the asset (default: dist).")

    args = parser.parse_args(argv)
    command = {"list": _cmd_list, "download": _cmd_download, "package": _cmd_package}[args.command]
    try:
        return command(args)
    except (RuntimeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
