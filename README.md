# T2.1_deep_expert
Expert Agent python package exploiting the expert knowledge in two ways:
1. Focus the exploration phase of an RL agent (DeepQ) on specific zones of a power grid
2. Reduce the action space to most relevant ones and improve the scalability or RL agents. There are two variants of this approach:
    - *Heuristic-based*: A set of well known heuristics with some greedy search over the reduced action space is used to solve the overload and congestion problems;
    - *Learning-based*: A PPO tries to learn the effective topological manipulations on the grid, by considering the reduced action space. Its combination with some heuristics helps to remedy most of the overload and congestion problems. 

![image](docs/imgs/T2.1_scheme_strategies.png)

# Credits
- Credits for Javaness [winning solution](https://github.com/lajavaness/l2rpn-2023-ljn-agent) at the L2RPN 2023 IDF AI challenge which has inspired the heuristic part of this work and most of the code is adapted and reused. The adapted code is on a forked repository which could be found [here](https://github.com/Mleyliabadi/l2rpn-2023-ljn-agent). 
- Credits for [CurriculumAgent](https://github.com/FraunhoferIEE/curriculumagent), which has inspired the search for reduced action space. Herein, we have replaced the greedy search over all the action space, by those suggested using expert knowlege. 
- The action suggested by expert knowledge uses the [ExpertOp4Grid](https://github.com/marota/ExpertOp4Grid) package.

# Installation guide
To be able to run the experiments in this repository, the following steps show how to install this package and its dependencies from source.

### Requirements
- Python >= 3.6
- [ExpertOp4Grid package (customized)](https://github.com/Mleyliabadi/ExpertOp4Grid)
- [LJN agent package (customized)](https://github.com/Mleyliabadi/l2rpn-2023-ljn-agent)

### Setup a Virtualenv (optional)
#### Create a Conda env (recommended)
```bash
conda create -n expert_agent python=3.10
conda activate venv_gnn
```
#### Create a virtual environment

```bash
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_expert_agent
source venv_expert_agent/bin/activate
```

### Install the prerequisites
> [!IMPORTANT] 
> These steps are mandatory to be able to use the package and its different functionalities
#### ExpertOp4Grid package
```bash
git clone git@github.com:Mleyliabadi/ExpertOp4Grid.git
cd ExpertOp4Grid
pip install -U .
```

#### LJN Agent package
```
git clone git@github.com:Mleyliabadi/l2rpn-2023-ljn-agent.git
cd l2rpn-2023-ljn-agent
pip install -U .
```

#### Prepare the environment 
```bash
git clone git@github.com:AI4REALNET/grid2op-scenario.git
cd grid2op-scenario
cp -r ai4realnet_small /home/<USERNAME>/data_grid2op/.
```

### Install the current package from source
```bash
git clone git@github.com:AI4REALNET/T2.1_deep_expert.git
cd T2.1_deep_expert
pip3 install -U .[recommended]
```



### To contribute
```bash
pip3 install -e .[recommended]
```

### Download the pretrained models
The trained models are not stored in git: they are files attached to a GitHub release (see [Pretrained models and `local_data`](#pretrained-models-and-local_data)). The scripts and notebooks download them automatically the first time they need them. To download them in advance (for example before working offline):
```bash
python -m ExpertAgent.utils.local_data download
```


## Overview of code structure
:open_file_folder: **ExpertAgent**

├── :open_file_folder: configs

│   └── ...

├── :open_file_folder: getting_started

│   &ensp;&ensp;&ensp;&ensp;└── 0_extract_actions.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 1_apply_deepqexpert.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 2_apply_expert_agent_heuristic.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 3_apply_expert_agent_rl.ipynb

├── :open_file_folder: ExpertAgent

│   └── :open_file_folder: assets

│     &ensp;&ensp;&ensp;&ensp;└── ...

│   └── :open_file_folder: DeepQExpert

│     &ensp;&ensp;&ensp;&ensp;└── ...

│   └── :open_file_folder: ExpertAgent

│     &ensp;&ensp;&ensp;&ensp;└── agentHeuristic.py

│     &ensp;&ensp;&ensp;&ensp;└── agentRL.py

│   └── :open_file_folder: utils

│     &ensp;&ensp;&ensp;&ensp;└── extractExpertActions.py

│     &ensp;&ensp;&ensp;&ensp;└── extractAttackingExpertActions.py

│     &ensp;&ensp;&ensp;&ensp;└── local_data.py &ensp;(paths in `local_data`, download of the released models)

│     &ensp;&ensp;&ensp;&ensp;└── ...

├── :open_file_folder: local_data &ensp;(not in git: models, checkpoints, logs and results, see below)

├── setup.py


## Pretrained models and `local_data`

### Where models and results are stored
Everything produced by an agent (model weights, training checkpoints, tensorboard logs and evaluation results) is written to a single `local_data` directory at the root of the repository, with **one folder per environment, then per agent, then per run**:

```
local_data/
├── l2rpn_case14_sandbox/
│   └── DeepQExpert/
│       ├── models-v1/              # the released model (downloaded)
│       │   ├── model/              # what the agent loads
│       │   └── eval/               # results of evaluating it
│       └── run_20261127-153000/    # one of your own training runs
│           ├── model/
│           └── logs/               # tensorboard
├── ai4realnet_small/
│   ├── PPO_SB3/
│   │   ├── models-v1/
│   │   │   ├── model/PPO_SB3.zip
│   │   │   └── eval/
│   │   └── run_<date-time>/
│   │       ├── model/              # final model, best model, evaluations.npz
│   │       ├── checkpoints/        # PPO_SB3_<n>_steps.zip
│   │       └── logs/
│   ├── ExpertHeuristic/
│   │   └── run_<date-time>/eval/
│   └── ExpertActions/              # output of the expert action extraction scripts
└── .downloads/                     # the downloaded release files, kept as a cache
```

- A training run gets a new folder named `run_<date-time>`, so it never overwrites an earlier one. The scripts print where they save. `main_expert_rl_train.py`, `main_expert_rl_eval.py` and `main_expert_heuristic.py` accept `--run_name` to choose the name.
- `local_data` is ignored by git (see `.gitignore`), so nothing in it is committed by accident.
- The inputs of the agents, such as the expert action spaces in `ExpertAgent/assets`, are small and stay in git.
- To put `local_data` somewhere else (a larger disk, a cluster scratch space, a Docker volume), set the environment variable `EXPERTAGENT_LOCAL_DATA` to the directory to use.

In code, the paths are built with the helpers of `ExpertAgent.utils.local_data`:
```python
from ExpertAgent.utils.local_data import get_run_dir, get_release_model

# local_data/ai4realnet_small/PPO_SB3/my_run/model (created if needed)
save_path = get_run_dir("ai4realnet_small", "PPO_SB3", "my_run", "model")

# the released model: downloaded on first use, then only returns the path to give to agent.load
load_path = get_release_model("PPO_SB3", "ai4realnet_small")
```

### Released models
The models used by the scripts, the notebooks and the app are published as files ("assets") of a GitHub release of [`ainetus/T2.1_deep_expert`](https://github.com/ainetus/T2.1_deep_expert/releases), rather than committed to git. This keeps the repository small, and each model version is published once and never changes.

| Agent | Trained on | Release | File | Content |
|---|---|---|---|---|
| DeepQExpert | `l2rpn_case14_sandbox` | `models-v1` | `DeepQExpert_l2rpn_case14_sandbox_models-v1.zip` | Extended DeepQ agent, 1M training steps (27 MB) |
| PPO_SB3 (`ExpertAgentRL`) | `ai4realnet_small` | `models-v1` | `PPO_SB3_ai4realnet_small_models-v1.zip` | PPO policy used by `ExpertAgentRL` and the app (60 MB) |

The list of released models, with their checksums, is `RELEASE_MODELS` in [ExpertAgent/utils/local_data.py](ExpertAgent/utils/local_data.py). The code always uses the release named there.

**How the download works**
- The first call to `get_release_model(...)`, made by the scripts and notebooks, downloads the file into `local_data/.downloads/<release>/`. It checks its sha256 checksum, then extracts it into `local_data/<env>/<agent>/<release>/model/`. Later calls do not download anything.
- The download can also be started by hand:
  ```bash
  python -m ExpertAgent.utils.local_data list              # released models, and whether they are downloaded
  python -m ExpertAgent.utils.local_data download          # all of them
  python -m ExpertAgent.utils.local_data download PPO_SB3  # only one agent
  ```
- **Private repository:** downloading needs a GitHub token that can read the repository, in the environment variable `GITHUB_TOKEN` (or `GH_TOKEN`). The token is only sent to `api.github.com`:
  ```bash
  export GITHUB_TOKEN=<your token>
  python -m ExpertAgent.utils.local_data download
  ```
- **Without internet access:** download the zip files from the release page in a browser, put them in `local_data/.downloads/models-v1/`, then run `python -m ExpertAgent.utils.local_data download`. It checks them and extracts them without connecting to GitHub.
- **Another repository:** to download from another repository (for example the fork), set `EXPERTAGENT_MODELS_REPO=<owner>/<name>`. That repository needs a release with the same tag and the same files.

**Docker:** rather than copying a model folder into the image, download the model while building it, after the package is installed:
```dockerfile
RUN python -m ExpertAgent.utils.local_data download PPO_SB3
```
For a private repository, give the token as a build secret, so that it is not stored in the image:
```dockerfile
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) python -m ExpertAgent.utils.local_data download PPO_SB3
```
```bash
docker build --secret id=github_token,env=GITHUB_TOKEN -t expert-agent .
```

### Publishing a new version of a model (maintainers)
A published release is never modified: replacing a file in it would make the copies already downloaded differ from new ones, and it would break the checksum the code checks. A new model always goes into a **new release** with a new tag (`models-v2`, `models-v3`, ...). Older releases stay available, so older versions of the code keep working.

1. Train the model. It is saved in `local_data/<env>/<agent>/<run>/model/`.
2. Build the release file. This prints its sha256 and the entry to put in `RELEASE_MODELS`:
   ```bash
   # DeepQExpert
   python -m ExpertAgent.utils.local_data package \
       --model_dir local_data/l2rpn_case14_sandbox/DeepQExpert/<run>/model \
       --agent DeepQExpert --env l2rpn_case14_sandbox --tag models-v2
   # PPO_SB3 (the agent loads model/PPO_SB3.zip, hence --load_path)
   python -m ExpertAgent.utils.local_data package \
       --model_dir local_data/ai4realnet_small/PPO_SB3/<run>/model \
       --agent PPO_SB3 --env ai4realnet_small --tag models-v2 --load_path PPO_SB3
   ```
   The file is written to `dist/` (ignored by git). Building it twice from the same model gives the same file and checksum.
3. Create the release `models-v2` on GitHub (*Releases → Draft a new release*, new tag `models-v2`) and attach the file. With the [GitHub CLI](https://cli.github.com/) this is:
   ```bash
   gh release create models-v2 dist/*_models-v2.zip --repo ainetus/T2.1_deep_expert \
       --title "Models v2" --notes "What changed and how the models were trained"
   ```
   Put in the release notes how each model was trained (environment, number of steps, main parameters) and its evaluation results.
4. Replace the entry of the model in `RELEASE_MODELS` with the printed one, test with `python -m ExpertAgent.utils.local_data download`, and commit. A model that did not change keeps its existing entry, and its older release.


## How to use
A set of jupyter notebooks are provided to ease the use of the package for the users. Here is the list of notebooks:
0. [00_action_extraction_explained.ipynb](getting_started/00_action_extraction_explained.ipynb) : It explains step by step how the reduced action spaces are extracted with the expert system (`ExpertAgent/utils/extractExpertActions.py` and `extractAttackingExpertActions.py`), and how the agents use them
1. [01_deep_q_expert.ipynb](getting_started/01_deep_q_expert.ipynb) : It shows how to use the extended DeepQ agent  
2. [02_expert_agent_heuistics.ipynb](getting_started/02_expert_agent_heuristic.ipynb) : It shows how to use the heuristic agent using the expert knowledge 
3. [03_expert_agent_RL.ipynb](getting_started/03_expert_agent_RL.ipynb) : It show how to use the RL based agent harnessing expert knowledge to reduce the action space.

## Reproducibility
### 1. DeepQExpert Agent
----
This agent applies an extended DeepQ algorithm for power grids and specifically works good with ``l2rpn_case14_sandbox`` environment.

#### Train
To train this agent, the following command could be executed from root and CLI:
```bash
python ExpertAgent/DeepQExpert/train.py \
    --num_train_steps=1000000 \
    --name="DeepQExpert"
```
At the end of the training, the weights of the model and some information concerning the neural network architecture are saved in `local_data/l2rpn_case14_sandbox/DeepQExpert/run_<date-time>/model`, and the tensorboard logs in `.../logs`. `--save_path` and `--logs_dir` can still be given to save elsewhere.

#### Evaluate
To evaluate the released model (downloaded on first use), the following command could be executed from root and using CLI:
```bash
python ExpertAgent/DeepQExpert/evaluate.py
```
The results are written to `local_data/l2rpn_case14_sandbox/DeepQExpert/models-v1/eval`. To evaluate one of your own runs instead, set `load_path = get_run_dir(env_name, name, "<run_name>", "model")` in the script.

At the end of the evaluation, a graphic representing the performance (reward/alive time) of the agent is visualized to the user.

![image](docs/imgs/DeepQExpert_Evaluation.png)


### 2. ExpertAgent Heuristics
--------------------------
The heuristic version of the `ExpertAgent` does not require any training and the evaluation could be run using a main function included in the root of the package. This agent is already provided to work for `ai4realnet_small` scenario of AI4REALNET project and power grid usecase (first).

```bash
python main_expert_heuristic.py --nb_episode=15 --nb_process=1 --max_step=2016 --verbose=True 
```
The results are written to `local_data/ai4realnet_small/ExpertHeuristic/run_<date-time>/eval` (or to `--logs_dir`).

At the end of the evaluation a graphic representing the performance (reward/alive time) of the agent is visualized to the user.

![image](docs/imgs/ExpertAgentHeuristic_Evaluation.png)



### 3. ExpertAgent RL
------------------
The RL based agent learning the reduced action space obtained using Expert Knowledge should be trained. The training could be launched using the corresponding `main` file provided in the `root` of the repository as:
```bash
python main_expert_rl_train.py --run_name=my_run
```
The run is saved in `local_data/ai4realnet_small/PPO_SB3/my_run/`: the final and best models in `model/`, the checkpoints in `checkpoints/` and the tensorboard logs in `logs/`. Without `--run_name`, the folder is named `run_<date-time>`.

An already trained agent is also provided as a [released model](#released-models). It can be loaded easily as follows (see [notebook](getting_started/03_expert_agent_RL.ipynb) for a full example):
```python
from ExpertAgent.utils.local_data import get_release_model
from ExpertAgent.ExpertAgent import ExpertAgentRL

env, env_gym = create_env(...)

nn_kwargs = {...}

# downloaded on first use into local_data/ai4realnet_small/PPO_SB3/models-v1/model
load_path = get_release_model("PPO_SB3", "ai4realnet_small")
agent = ExpertAgentRL(name="PPO_SB3",
                      env=env,
                      action_space=env.action_space,
                      gymenv=env_gym,
                      gym_act_space=env_gym.action_space,
                      gym_obs_space=env_gym.observation_space,
                      nn_kwargs=nn_kwargs
                      )

agent.load(load_path)
``` 

One the agent is trained or loaded, the evaluation could be done using the following command and the main file `main_expert_rl_eval.py`:
```bash
python main_expert_rl_eval.py --nb_episode=15 --nb_process=1 --max_step=2016 --verbose=True 
```
By default it evaluates the released model and writes the results to `local_data/ai4realnet_small/PPO_SB3/models-v1/eval`. Add `--run_name=my_run` to evaluate one of your own runs instead (results in `local_data/ai4realnet_small/PPO_SB3/my_run/eval`).
At the end of the evaluation a graphic representing the performance (reward/alive time) of the agent is visualized to the user.

![image](docs/imgs/ExpertAgentRL_Evaluation.png)