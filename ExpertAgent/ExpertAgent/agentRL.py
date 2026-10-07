import os
import logging
from typing import Optional
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymEnv
from grid2op.Environment import Environment

import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback 
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from ExpertAgent.DeepQExpert.heuristics import RecoPowerlineModule, RecoverInitTopoModule, TopoNNTopKModulePPO
from LJNAgent.modules.convex_optim import OptimModule
from LJNAgent.modules.rewards import PPO_Reward


from l2rpn_baselines.PPO_SB3.utils import SB3Agent

logging.basicConfig(level=logging.INFO, filename="agent_RL.log", filemode="w")
logger = logging.getLogger(__name__)

AI4REALNET_SMALL_DEFAULT_OPTIM_CONFIG = {
    "margin_th_limit": 0.93,
    "alpha_por_error": 0.5,
    "rho_danger": 0.99,
    "rho_safe": 0.9,
    "penalty_curtailment_unsafe": 15,
    "penalty_redispatching_unsafe": 0.005,
    "penalty_storage_unsafe": 0.0075,
    "penalty_curtailment_safe": 0.0,
    "penalty_redispatching_safe": 0.0,
    "penalty_storage_safe": 0.0,
    "weight_redisp_target": 1.0,
    "weight_storage_target": 1.0,
    "weight_curtail_target": 1.0,
    "margin_rounding": 0.01,
    "margin_sparse": 5e-3,
    "max_iter": 100000,
    "areas": False,
    "sim_range_time_step": 1,
}

class ExpertAgentRL(SB3Agent, BaseAgent):
    def __init__(self,
                 name: str,
                 env: Environment,
                 action_space: ActionSpace,
                 gymenv: GymEnv,
                 gym_act_space,
                 gym_obs_space,
                 nn_type=PPO,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 iter_num=None,
                 rho_danger: float=0.99,
                 rho_safe: float=0.9,
                 top_k: int=20,
                 device: str="cpu"
                 ):
        self.name = name
        
        SB3Agent.__init__(self, action_space, gym_act_space, gym_obs_space, nn_type=nn_type, 
                          nn_path=nn_path, nn_kwargs=nn_kwargs, custom_load_dict=custom_load_dict, 
                          gymenv=gymenv, iter_num=iter_num)
        BaseAgent.__init__(self, action_space=action_space)
        
        self.env = env
        self.reconnect = RecoPowerlineModule(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Continuous control
        self.optim = OptimModule(env, self.action_space, config=AI4REALNET_SMALL_DEFAULT_OPTIM_CONFIG)
        
        self.topo_search = None
        
        self.top_k = top_k
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        self.device = device
        
        self.action_list = []
        
        self._loaded = False
        self._trained = False
        
    def act(self,
            observation: BaseObservation,
            reward: float,
            done: bool=False) -> BaseAction:
        
        if not self._loaded and not self._trained:
            raise Exception("You should first load the model or train it.")
        
        act = self.action_space({})

        # Try to perform reconnection if necessary
        reconnect_act = self.reconnect.get_act(observation, act, reward)
        
        if reconnect_act is not None:  
            _obs, _rew, _done, _info = observation.simulate(reconnect_act, time_step=1)
            if (reconnect_act is not None
                and not _done
                and reconnect_act != self.action_space({})
                and 0. < _obs.rho.max() < 2.
                and (len(_info["exception"]) == 0)
            ):
                logger.info("calling reconnection module")
                act += reconnect_act

        if observation.rho.max() > self.rho_danger:
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                logger.info("Calling recovery action")
                act += recovery_act
            else:
                # get the action from the trained RL (PPO)
                topo_act = self.topo_search.get_act(observation, act, reward, done)
                
                if topo_act is not None:        
                    logger.info(topo_act)
                    act += topo_act
                    
            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                logger.info("calling optim module")
                act = self.optim.get_act(observation, act, reward)
            
        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                logger.info("Calling recovery action (grid is safe)")
                act += recovery_act
        else:
            # Update the observed storage power.
            self.optim._update_storage_power_obs(observation)
            self.optim.flow_computed[:] = observation.p_or
            
        if act != self.action_space({}):
            self.action_list.append(act)
        return act

    def recommend(self,
                   observation: BaseObservation,
                   reward: float,
                   top_k: Optional[int]=3,
                   done: bool=False) -> list:
        """Similar to `act`, but returns several ranked candidate actions instead of
        committing to a single one. Intended for advisory / decision-support use cases
        (e.g. a recommendation API shown to a human operator) where several options
        should be surfaced at once. `act` should still be used to actually step a
        grid2op episode (e.g. via a Runner), which expects exactly one action per step.

        The first entry is always exactly what `act` would do for this observation --
        `recommend` calls `act` itself to get it, rather than re-implementing its
        decision logic, so the two can never drift apart. The remaining entries (only
        populated when the grid is in the "danger" regime, i.e. `rho.max() > rho_danger`)
        are other topology candidates and a dispatch-only alternative, ranked after the
        primary one, so an operator can see what else was considered.

        Parameters
        ----------
        top_k: int, optional
            Maximum number of candidate actions to return (including the primary one
            from `act`). Defaults to 3.

        Returns
        -------
        list of dict
            Ranked list of candidate recommendations, most recommended first. Entry 0
            is always the `act`-equivalent primary recommendation. Each dict has keys:
              - "action": BaseAction
              - "reward": Optional[float], simulated reward backing the ranking when
                 available (only set for alternative candidates coming from the ranked
                 topology search; None for the primary and for heuristic/optim-only
                 candidates, since those aren't comparable on the same scale)
              - "rho_max": Optional[float], simulated `observation.rho.max()` for this
                 candidate; None if that simulation errored
              - "is_safe": bool, whether this candidate's `rho_max` stays under
                 `self.rho_safe`. An alternative that isn't fully safe is still
                 included (ranked after the safe ones) rather than silently dropped,
                 so an operator can see the "best attempt" options even when nothing
                 but the primary recommendation resolves the overload.
              - "is_primary": bool, True only for entry 0 (the `act`-equivalent pick).
              - "is_agent_action": bool, False for the supplementary operator
                 heuristics appended at the end of the list (see `_heuristic_options`),
                 which are manual options a human may pick rather than anything the
                 agent is recommending. True for everything else.
              - "source": str, which strategy produced the entry -- "act" (primary),
                 "topo_search", "dispatch_only", "ref_topology" or "line_reconnect".
              - "line_id": Optional[int], the line to reconnect, set only when
                 "source" is "line_reconnect".

            Note that `top_k` caps the agent's candidates only; the supplementary
            heuristic entries are always appended after them, so the returned list
            can be longer than `top_k`.
        """
        if not self._loaded and not self._trained:
            raise Exception("You should first load the model or train it.")

        if top_k is None:
            top_k = self.top_k

        # The primary recommendation is produced by calling `act` directly, with no
        # state isolation whatsoever: `OptimModule` warm-starts each solve from
        # `flow_computed` (as a linearization-error correction term), so its solution
        # genuinely depends on that state. Snapshotting/restoring around this call
        # would make `recommend` return a cold-start dispatch solution while a plain
        # `act` call returns a warm-started one -- i.e. a different action. Letting
        # `act` mutate `flow_computed` and `action_list` exactly as it normally does
        # is what keeps this entry identical to what `act` alone would return.
        primary_act = self.act(observation, reward, done)

        # Only the *alternatives* below are hypotheticals that must not leave a trace,
        # so they are isolated against this post-`act` state (and against each other).
        flow_computed_snapshot = self.optim.flow_computed.copy()

        _obs, _rew, _done, _info = observation.simulate(primary_act, time_step=1)
        has_error = len(_info["exception"]) != 0
        ranked_candidates = [{
            "action": primary_act,
            "reward": None,
            "rho_max": float(_obs.rho.max()) if not has_error else None,
            "is_safe": bool((not has_error) and _obs.rho.max() < self.rho_safe),
            "is_primary": True,
            "is_agent_action": True,
            "source": "act",
            "line_id": None,
        }]

        if observation.rho.max() > self.rho_danger:
            # Gather distinct alternatives beyond the primary pick: other topology
            # candidates (even ones that don't fully resolve the overload, so an
            # operator can see the best attempts that weren't chosen) plus a
            # dispatch-only option, ranked after the primary recommendation.
            act0 = self.action_space({})
            reconnect_act = self.reconnect.get_act(observation, act0, reward)
            if reconnect_act is not None:
                _r_obs, _r_rew, _r_done, _r_info = observation.simulate(reconnect_act, time_step=1)
                if (not _r_done
                    and reconnect_act != self.action_space({})
                    and 0. < _r_obs.rho.max() < 2.
                    and (len(_r_info["exception"]) == 0)
                ):
                    act0 += reconnect_act

            alt_candidates = []

            ranked_topo_acts = self.topo_search.get_ranked_acts(
                observation, act0, reward, done, top_n=top_k
            )
            for c in ranked_topo_acts:
                alt_candidates.append({"action": act0 + c["action"], "reward": c["reward"],
                                       "source": "topo_search"})

            # A dispatch-only alternative (no topology change), in case an operator
            # wants to compare a pure redispatch/curtailment/storage strategy.
            alt_candidates.append({"action": act0, "reward": None, "force_optim": True,
                                   "source": "dispatch_only"})

            for candidate in alt_candidates:
                candidate_act = candidate["action"]
                _obs, _rew, _done, _info = observation.simulate(candidate_act, time_step=1)
                needs_optim = (
                    candidate.get("force_optim", False)
                    or _obs.rho.max() > self.rho_safe
                    or (len(_info["exception"]) != 0)
                )
                if needs_optim:
                    logger.info("calling optim module for one of the alternative candidates")
                    self.optim.flow_computed[:] = flow_computed_snapshot
                    candidate_act = self.optim.get_act(observation, candidate_act, reward)
                    self.optim.flow_computed[:] = flow_computed_snapshot
                    _obs, _rew, _done, _info = observation.simulate(candidate_act, time_step=1)
                if candidate_act == primary_act:
                    continue
                has_error = len(_info["exception"]) != 0
                ranked_candidates.append({
                    "action": candidate_act,
                    "reward": candidate["reward"],
                    "rho_max": float(_obs.rho.max()) if not has_error else None,
                    "is_safe": bool((not has_error) and _obs.rho.max() < self.rho_safe),
                    "is_primary": False,
                    "is_agent_action": True,
                    "source": candidate["source"],
                    "line_id": None,
                })

            # Keep the primary first, then rank the alternatives: safe ones ahead of
            # unsafe ones, broken by how much each one helped.
            primary, rest = ranked_candidates[0], ranked_candidates[1:]
            rest.sort(
                key=lambda c: (
                    not c["is_safe"],
                    c["rho_max"] if c["rho_max"] is not None else float("inf"),
                )
            )
            ranked_candidates = [primary] + rest

        # `top_k` caps the agent's own candidates only. The supplementary operator
        # heuristics appended below are a separate category -- not agent decisions --
        # so they are always offered rather than truncated away by the cap.
        if top_k is not None:
            ranked_candidates = ranked_candidates[:top_k]

        ranked_candidates += self._heuristic_options(observation, reward, primary_act)

        return ranked_candidates

    def _heuristic_options(self,
                           observation: BaseObservation,
                           reward: float,
                           primary_act: Optional[BaseAction]=None) -> list:
        """Standard operator heuristics offered *alongside* the agent's recommendations.

        These are deliberately not agent decisions -- they are well-understood manual
        options a human operator may want to pick instead -- so every entry carries
        `"is_agent_action": False`, and none of them is run through the optim module,
        which keeps each one a single, clearly readable discrete action instead of a
        composite with redispatch mixed in.

        Two kinds are produced:

        1. `"ref_topology"`: a step back towards the reference topology. This
           environment has `MAX_SUB_CHANGED == 1`, so a full revert can take several
           steps; this offers the best single-substation revert currently available
           (`RecoverInitTopoModule` already filters out substations under cooldown).
           `get_ranked_acts` is used rather than `get_act` so the option is still
           offered when it doesn't improve rho -- whether that trade is worth making
           is the operator's call here, not the agent's.
        2. `"line_reconnect"`: one action per disconnected line that can legally be
           reconnected right now, using the same legality test as grid2op's own
           `RecoPowerlineAgent` (`not connected` and no remaining line cooldown), so
           a line still serving a cooldown is never suggested.

        Returns
        -------
        list of dict
            Same keys as `recommend`'s entries, where "source" is "ref_topology" or
            "line_reconnect" and "line_id" is set only for "line_reconnect". Ordered
            safe-first, then by how much each option helped. Empty when the topology
            is already at reference and no line can be reconnected.
        """
        options = []

        for c in self.recover_topo.get_ranked_acts(
            observation, self.action_space({}), reward, top_n=1
        ):
            options.append({"action": c["action"], "reward": c["reward"],
                            "source": "ref_topology", "line_id": None})

        can_be_reco = (~observation.line_status) & (observation.time_before_cooldown_line == 0)
        for line_id in can_be_reco.nonzero()[0]:
            options.append({
                "action": self.action_space({"set_line_status": [(int(line_id), +1)]}),
                "reward": None,
                "source": "line_reconnect",
                "line_id": int(line_id),
            })

        results = []
        for opt in options:
            if primary_act is not None and opt["action"] == primary_act:
                continue
            _obs, _rew, _done, _info = observation.simulate(opt["action"], time_step=1)
            has_error = len(_info["exception"]) != 0
            results.append({
                "action": opt["action"],
                "reward": opt["reward"],
                "rho_max": float(_obs.rho.max()) if not has_error else None,
                "is_safe": bool((not has_error) and _obs.rho.max() < self.rho_safe),
                "is_primary": False,
                "is_agent_action": False,
                "source": opt["source"],
                "line_id": opt["line_id"],
            })

        results.sort(
            key=lambda c: (
                not c["is_safe"],
                c["rho_max"] if c["rho_max"] is not None else float("inf"),
            )
        )
        return results

    def load(self, load_path: Optional[str]=None):
        if load_path is None:
            if self._nn_path is None:
                raise Exception("The path variable should be set before loading the model.")
        else:
            self._nn_path = load_path
        
        super().load()
        
        self._init_topoNNModule(top_k=self.top_k)
        self._loaded = True
    
    def _init_topoNNModule(self, top_k=20):
        self.topo_search = TopoNNTopKModulePPO(self.action_space, self.gymenv, self.nn_model, top_k=top_k)
        
    def train_model(self,
                    env_gym: GymEnv, 
                    total_timesteps: int=1000,
                    n_eval_episodes: int=5,
                    load_path: Optional[str]=None,
                    save_path: Optional[str]=None,
                    save_freq: int=2000,
                    eval_freq: int=1000):
        
        if load_path is not None:
            self.nn_model = PPO.load(path=load_path,
                                     custom_objects={"observation_space" : env_gym.observation_space,
                                                     "action_space": env_gym.action_space})
            self.nn_model.set_env(env_gym)
        
        if save_path is None:
            save_path = os.path.join("logs", self.name)
        
        callbacks = []
        callbacks.append(CheckpointCallback(save_freq=save_freq,
                                            save_path=save_path,
                                            name_prefix=self.name))
        
        if env_gym is not None:
            callbacks.append(EvalCallback(eval_env=env_gym,
                                          best_model_save_path=save_path,
                                          log_path=save_path,
                                          eval_freq=eval_freq,
                                          deterministic=True,
                                          render=False,
                                          verbose=True,
                                          n_eval_episodes=n_eval_episodes,
                                         ))
            
        # Train the model
        self.nn_model.learn(total_timesteps=total_timesteps,
                            progress_bar=True,
                            callback=CallbackList(callbacks))
        
        # save the model
        self.nn_model.save(os.path.join(save_path, self.name))
        self._init_topoNNModule(top_k=self.top_k)
        self._trained = True    
        
    def predict(self, observation, top_k=1) -> tuple:
        if self.gymenv is not None and self.gymenv.observation_space is not None:
            gym_obs = self.gymenv.observation_space.to_gym(observation)
        else:
            raise Exception("the gymenv not correclty loaded")
        input = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).to(self.device)
        distribution = self.nn_model.policy.get_distribution(input)
        if distribution.distribution is not None:
            logits = distribution.distribution.logits

            action_id_list = torch.topk(logits, k=top_k)[1].cpu().numpy()[0]
            g2op_action = [self.gymenv.action_space.from_gym(action_id) for action_id in action_id_list]
            return (g2op_action, distribution, logits)
        else:
            logits = None
            return (logits, distribution, logits)
    
    @staticmethod
    def train_static(agent,
                     env_gym: GymEnv, 
                     train_steps: int=1000,
                     n_eval_episodes: int=5,
                     load_path: Optional[str]=None,
                     load_path_fine_tune: Optional[str]=None,
                     save_path: Optional[str]=None,
                     save_freq: int=2000,
                     eval_freq: int=1000):
        
        if load_path is not None:
            if agent._trained:
                load_path = load_path_fine_tune
            agent.nn_model = PPO.load(path=load_path,
                                     custom_objects={"observation_space" : env_gym.observation_space,
                                                     "action_space": env_gym.action_space})
            agent.nn_model.set_env(env_gym)
        
        if save_path is None:
            save_path = os.path.join("logs", agent.name)
        
        callbacks = []
        callbacks.append(CheckpointCallback(save_freq=save_freq,
                                            save_path=save_path,
                                            name_prefix=agent.name))
        
        if env_gym is not None:
            callbacks.append(EvalCallback(eval_env=env_gym,
                                          best_model_save_path=save_path,
                                          log_path=save_path,
                                          eval_freq=eval_freq,
                                          deterministic=True,
                                          render=False,
                                          verbose=True,
                                          n_eval_episodes=n_eval_episodes,
                                         ))
            
        # Train the model
        agent.nn_model.learn(total_timesteps=train_steps,
                            progress_bar=True,
                            callback=CallbackList(callbacks))
        
        # save the model
        agent.nn_model.save(os.path.join(save_path, agent.name))
        agent._init_topoNNModule(top_k=agent.top_k)
        agent._trained = True
        
        return agent
        
    @staticmethod
    def evaluate(agent, env_gym, **kwargs):
        mean_reward, std_reward = evaluate_policy(agent.nn_model, 
                                                  env_gym, 
                                                  **kwargs)
        return np.mean(mean_reward), np.mean(std_reward)
        
if __name__ == "__main__":
    from ExpertAgent.utils.helper_functions import create_env
    
    env_name = "ai4realnet_small"
    reward_class = PPO_Reward # LinesCapacityReward
    seed = 1234
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    # create the training and testing environments
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                              )
    
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        model_path = os.path.join(logs_dir, "PPO_SB3")
    
    net_arch=[800, 1000, 1000, 800]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 1e-3,
            "tensorboard_log": model_path,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    agent = ExpertAgentRL(name="PPO_SB3",
                          env=env,
                          action_space=env.action_space,
                          gymenv=env_gym,
                          gym_act_space=env_gym.action_space,
                          gym_obs_space=env_gym.observation_space,
                          nn_kwargs=nn_kwargs
                          )
    
    agent.train_model(env_gym=env_gym,
                      n_eval_episodes=1,
                      total_timesteps=int(1e4)
                      )
