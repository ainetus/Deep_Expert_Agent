# command to execute the API
# uvicorn main_multiple:app --host 0.0.0.0 --port 8000
# Command to request a single recommendation from server (unchanged, kept for comparison)
# curl -X POST http://localhost:8000/api/v1/recommendation -H "Content-Type: application/json" --data @rte_recommendation.json
# Command to request several ranked recommendations from server (new)
# curl -X POST http://localhost:8000/api/v1/recommendations -H "Content-Type: application/json" --data @rte_recommendation.json
# docker build -t expert-agent-api .
# docker run -p 8000:8000 expert-agent-api
# docker run --rm -it --entrypoint bash expert-agent-api

import os
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import numpy as np

from LJNAgent.modules.rewards import PPO_Reward
from LJNAgent.modules.rewards import MaxRhoReward
from stable_baselines3.ppo import MlpPolicy
from ExpertAgent.utils.helper_functions import create_env
from ExpertAgent.utils import get_package_root
from ExpertAgent.ExpertAgent import ExpertAgentRL


# --- Define environment ---
env_name = "ai4realnet_small"
reward_class = PPO_Reward # LinesCapacityReward
reward_class = MaxRhoReward
seed = 12345
obs_attr_to_keep = ["rho"]
act_attr_to_keep = ["set_bus"]

env, env_gym = create_env(env_name=env_name,
                          reward_class=reward_class,
                          obs_attr_to_keep=obs_attr_to_keep,
                          action_space_path="read_from_file"
                          )
env.seed(seed)
obs = env.reset()

# --- Agent parameters and load path ---
name = "PPO_SB3"
load_path = os.path.join(get_package_root(), "..", name, "model", "PPO_SB3")
logs_dir = None

net_arch=[800, 1000, 1000, 800]
policy_kwargs = {}
policy_kwargs["net_arch"] = net_arch

nn_kwargs = {
        "policy": MlpPolicy,
        "env": env_gym,
        "verbose": True,
        "learning_rate": 3e-4,
        "tensorboard_log": logs_dir,
        "policy_kwargs": policy_kwargs,
        "device": "auto"
    }

# --- RL Agent instanciation ---
agent = ExpertAgentRL(name="PPO_SB3",
                      env=env,
                      action_space=env.action_space,
                      gymenv=env_gym,
                      gym_act_space=env_gym.action_space,
                      gym_obs_space=env_gym.observation_space,
                      nn_kwargs=nn_kwargs
                      )
agent.load(load_path)

# --- Chronic synchronisation -------------------------------------------------
# `obs.from_json()` restores the observation's state, but NOT the forecast that
# `obs.simulate()` rolls forward (`_forecasted_inj` is not serialised). Since every
# decision and KPI in the agent is simulate-driven, without synchronising the chronic
# the whole pipeline runs against whatever chronic/timestep this server happens to be
# sitting at -- which silently changes the recommendations and can even invert the
# `is_safe` verdicts.
#
# Fix: the simulator sends the chronic it is replaying (`env.chronics_handler.get_name()`,
# e.g. "2035-01-01_1") in the request context; we seek this env to that chronic and
# timestep, then overlay the real state with `from_json`. Both halves are needed --
# seeking alone would not reproduce the operator's actions or the overflow trips.
CHRONIC_INDEX = {
    os.path.basename(p): i
    for i, p in enumerate(env.chronics_handler.real_data.subpaths)
}
# keys accepted in `context` for the chronic, so the simulator side is free to name it
_CHRONIC_KEYS = ("chronic_name", "chronic", "chronic_path", "scenario_id", "scenario")
_TIMESTEP_KEYS = ("time_step", "timestep", "current_step")


def sync_env_to_context(context):
    """Seek the env to the chronic/timestep the request describes, then load the state.

    Args:
        context (dict): the request's `context` block. Read keys: the chronic under any
            of `_CHRONIC_KEYS` (a name like "2035-01-01_1", or a full path), the
            timestep under any of `_TIMESTEP_KEYS` (defaults to the observation's own
            `current_step`), and `observation`.

    Returns:
        dict: {"synced": bool, "chronic": str|None, "time_step": int|None, "reason": str}
            `synced` is False when no chronic was supplied or it isn't in this dataset;
            the observation is still loaded, but every simulate-derived KPI is then
            conditioned on this server's arbitrary forecast rather than the real one,
            which is why it is reported back to the caller rather than hidden.
    """
    # `env.reset()` hands back a NEW observation object carrying the fresh forecast;
    # the module-level `obs` keeps the forecast from startup. Rebinding the global is
    # what actually makes the seek take effect -- calling `from_json` on the stale
    # object would restore the state but leave `simulate()` on the old forecast.
    global obs

    obs_json = context.get("observation")
    raw = next((context[k] for k in _CHRONIC_KEYS if context.get(k)), None)
    name = os.path.basename(str(raw).rstrip("/")) if raw is not None else None

    time_step = next((context[k] for k in _TIMESTEP_KEYS if context.get(k) is not None), None)
    if time_step is None and obs_json is not None:
        cur = obs_json.get("current_step")
        time_step = cur[0] if isinstance(cur, list) else cur

    if name is None:
        obs.from_json(obs_json)
        return {"synced": False, "chronic": None, "time_step": time_step,
                "reason": "no chronic supplied in context"}
    if name not in CHRONIC_INDEX:
        obs.from_json(obs_json)
        return {"synced": False, "chronic": name, "time_step": time_step,
                "reason": f"chronic {name!r} not found in this dataset"}

    options = {"time serie id": CHRONIC_INDEX[name]}
    if time_step:                      # "init ts" only makes sense beyond the first step
        options["init ts"] = int(time_step)
    obs = env.reset(options=options)
    # the seek fixes the forecast; this overlays the actual state on top of it
    obs.from_json(obs_json)
    return {"synced": True, "chronic": name, "time_step": time_step, "reason": "ok"}


# --- API Schema ---
class RecommendationRequest(BaseModel):
    event: dict
    context: dict
    # observation: Optional[Any] = None
    # observation: List[float] = None


class MultipleRecommendationRequest(RecommendationRequest):
    num_reco: Optional[int] = None  # when omitted, recommend()'s own default is used

# class RecommendationResponse(BaseModel):
#     recommendation: List[Any]

app = FastAPI()

@app.post("/api/v1/recommendation")#, response_model=RecommendationResponse)
def get_recommendation(request: RecommendationRequest):
    # Convert incoming data to the observation format your agent expects
    observation = {
        "event": request.event,
        "context": request.context,
        # "observation": request.observation
    }

    # Get recommendation from RL agent
    sync = sync_env_to_context(observation.get("context", {}))
    action = agent.act(obs, reward=None, done=False)
    result = get_parade_info(action, obs)
    result["kpis"]["context_synced"] = bool(sync["synced"])
    if result is not list:
        result = [result]
    # print("DEBUG OUTPUT: ", type(result))
    return result#{"recommendations": result}


@app.post("/api/v1/recommendations")
def get_recommendations(request: MultipleRecommendationRequest):
    """Same as `get_recommendation`, but returns several ranked candidate
    recommendations at once instead of a single one.
    """
    # Convert incoming data to the observation format your agent expects
    observation = {
        "event": request.event,
        "context": request.context,
    }

    # Seek the env to the chronic/timestep this observation came from before asking
    # the agent, so the simulate-driven ranking and KPIs reflect the real forecast.
    sync = sync_env_to_context(observation.get("context", {}))
    # Only forward top_k when the caller actually set it, so recommend()'s own
    # default applies otherwise (passing top_k=None explicitly would instead hit
    # recommend()'s "fall back to self.top_k" branch, which is a different knob --
    # the NN candidate pool size set at agent construction, not the response length).
    recommend_kwargs = {"top_k": request.num_reco} if request.num_reco is not None else {}
    ranked_candidates = agent.recommend(obs, reward=None, done=False, **recommend_kwargs)

    result = [
        get_parade_info_ranked(candidate, obs, rank=rank, context_synced=sync["synced"])
        for rank, candidate in enumerate(ranked_candidates)
    ]
    return result


# Titles for the supplementary operator heuristics. These replace the title that
# `get_parade_info` derives from the action's impacts, so a user can tell at a glance
# that the entry is a manual heuristic option rather than something the agent is
# recommending.
HEURISTIC_TITLES = {
    "ref_topology": "Return to the reference topology (heuristic)",
    "line_reconnect": 'Reconnect the disconnected line "{line_id}" (heuristics)',
}


def get_parade_info_ranked(candidate, obs, rank=None, context_synced=None):
    """Thin wrapper around `get_parade_info` that also attaches ranking / safety
    information to the resulting KPIs, for use by the multi-recommendation endpoint.

    Args:
        candidate (dict): one entry of `ExpertAgentRL.recommend()`'s return value,
            i.e. {"action": ..., "reward": ..., "rho_max": ..., "is_safe": ...}
        rank (int, optional): position of this candidate in the ranked list (0 = best)

    Returns:
        dict: Same format as `get_parade_info`, with `rank`/`predicted_reward`/
            `simulated_rho_max`/`is_safe` kpis added. `is_safe` is False when this
            candidate didn't fully resolve the overload (it is still returned,
            ranked last, as the "best attempt" rather than dropped silently).
            Supplementary heuristic entries get their own title (HEURISTIC_TITLES)
            instead of the impact-derived one.
    """
    result = get_parade_info(candidate["action"], obs)
    heuristic_title = HEURISTIC_TITLES.get(candidate.get("source"))
    if heuristic_title is not None:
        result["title"] = heuristic_title.format(line_id=candidate.get("line_id"))
    if rank is not None:
        result["kpis"]["rank"] = rank
    if candidate.get("reward") is not None:
        result["kpis"]["predicted_reward"] = float(candidate["reward"])
    if candidate.get("is_safe") is not None:
        result["kpis"]["is_safe"] = bool(candidate["is_safe"])
    if candidate.get("is_primary") is not None:
        result["kpis"]["is_primary"] = bool(candidate["is_primary"])
    # Distinguishes the agent's own recommendations from the supplementary operator
    # heuristics ("return to reference topology", "reconnect line N"), which are
    # offered as manual options a user may pick rather than as agent decisions.
    if candidate.get("is_agent_action") is not None:
        result["kpis"]["is_agent_action"] = bool(candidate["is_agent_action"])
    if candidate.get("source") is not None:
        result["kpis"]["source"] = str(candidate["source"])
    if candidate.get("line_id") is not None:
        result["kpis"]["line_id"] = int(candidate["line_id"])
    # False means the chronic could not be synchronised, so `simulated_rho_max`,
    # `is_safe` and `efficiency_of_the_reco` were computed against this server's
    # forecast rather than the real one and should not be shown as predictions.
    if context_synced is not None:
        result["kpis"]["context_synced"] = bool(context_synced)
    return result


def get_parade_info(act, obs):
    """Compile unitary recomendation in json format for InteractiveAI's frontend compliance

    Args:
        act (): Unitary action object

    Returns:
        dict: Recomendations data in json format
    """
    kpis = {}
    title = []
    description = []
    impact = act.impact_on_objects()

    # redispatch
    if act._modif_redispatch:
        kpis["type_of_the_reco"] = (
            "Redispatch"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Injection recommendation: production source redispatch"
        )
        if description:
            description.append(" | ")
        cpt = 0
        for gen_idx in range(act.n_gen):
            if act._redispatch[gen_idx] != 0.0:
                gen_name = act.name_gen[gen_idx]
                r_amount = act._redispatch[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'"{gen_name}" de {r_amount:.2f} MW'
                )

    # storage
    if act._modif_storage:
        kpis["type_of_the_reco"] = (
            "Storage"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Storage recommendation")
        if description:
            description.append(" | ")
        cpt = 0
        for stor_idx in range(act.n_storage):
            amount_ = act._storage_power[stor_idx]
            if np.isfinite(amount_) and amount_ != 0.0:
                name_ = act.name_storage[stor_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Ask unit "{name_}" to '
                    f'{"charge" if amount_ > 0.0 else "discharge"} '
                    f'{abs(amount_):.2f} MW (setpoint: {amount_:.2f} MW)'
                )

    # curtailment
    if act._modif_curtailment:
        kpis["type_of_the_reco"] = (
            "Injection"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Injection recommendation")
        if description:
            description.append(" | ")
        cpt = 0
        for gen_idx in range(act.n_gen):
            amount_ = act._curtail[gen_idx]
            if np.isfinite(amount_) and amount_ != -1.0:
                name_ = act.name_gen[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Limit unit "{name_}" to '
                    f'{100.0 * amount_:.1f}% of its maximum capacity '
                    f'(setpoint: {amount_:.3f})'
                )

    # force line status
    force_line_impact = impact["force_line"]
    if force_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: connection/disconnection of line"
        )
        if description:
            description.append(" | ")
        reconnections = force_line_impact["reconnections"]
        if reconnections["count"] > 0:
            description.append(
                f"Reconnection of {reconnections['count']} lines "
                f"({reconnections['powerlines']})"
            )

        disconnections = force_line_impact["disconnections"]
        if disconnections["count"] > 0:
            description.append(
                f"Disconnection of {disconnections['count']} lines "
                f"({disconnections['powerlines']})"
            )

    # swtich line status
    swith_line_impact = impact["switch_line"]
    if swith_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Topological: change a line state")
        if description:
            description.append(" | ")
        description.append(
            f"Change the state of {swith_line_impact['count']} lines "
            f"({swith_line_impact['powerlines']})"
        )

    # topology
    bus_switch_impact = impact["topology"]["bus_switch"]
    if len(bus_switch_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(bus_switch_impact["substation"])
        )
        if description:
            description.append(" | ")
        description.append("Busbar change:")
        for switch in bus_switch_impact:
            description.append(
                f"\t \t - Switch bus of {switch['object_type']} id "
                f"{switch['object_id']} [at station {switch['substation']}]"
            )

    assigned_bus_impact = impact["topology"]["assigned_bus"]
    disconnect_bus_impact = impact["topology"]["disconnect_bus"]
    if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(assigned_bus_impact[0]["substation"])
        )
        if description:
            description.append(" | ")
        if assigned_bus_impact:
            description.append("")
        cpt = 0
        for assigned in assigned_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f" Assign bus {assigned['bus']} to "
                f"{assigned['object_type']} id {assigned['object_id']}"
            )
        if disconnect_bus_impact:
            description.append("")
        cpt = 0
        for disconnected in disconnect_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f"Disconnect {disconnected['object_type']} with id "
                f"{disconnected['object_id']} [at the substation level "
                f"{disconnected['substation']}]"
            )

    # Any of the above cases,
    # then the recommendation is most likely "Do nothing"
    if not title and act == env.action_space({}):
        kpis["type_of_the_reco"] = (
            "Do nothing"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Poursuivre")
        if description:
            description.append(" | ")
        description.append(
            "Continuation of the scenario without operator action"
        )

    title = [t for t in title
             if not any(t != other and t in other for other in title)]
    title = " + ".join(dict.fromkeys(title))
    description = "".join(description)

    if title:
        obs_simulate, _, _, _ = (
            obs.simulate(act, time_step=1)
        )
        kpis["efficiency_of_the_reco"] = float(
            np.float32(obs_simulate.rho.max())
        )  # pour renvoyer le kpi efficiency_of_the_reco

    return {
        "title": title,
        "description": description,
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [act.to_json()],
        "kpis": kpis,
    }
