
import os
import argparse
from LJNAgent.modules.rewards import PPO_Reward, MaxRhoReward
from stable_baselines3.ppo import MlpPolicy

from ExpertAgent.ExpertAgent import ExpertAgentRL
from ExpertAgent.utils.helper_functions import create_env
from ExpertAgent.utils.local_data import get_run_dir, new_run_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO model of the RL expert agent")
    parser.add_argument("--run_name", required=False, default=None, type=str,
                        help="Name of the run folder in local_data/<env>/PPO_SB3/ (default: run_<date-time>)")
    args = parser.parse_args()

    env_name = "ai4realnet_small"
    reward_class = PPO_Reward # LinesCapacityReward
    reward_class = MaxRhoReward
    seed = 1234
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                             )
    
    name = "PPO_SB3"
    # Everything of this run goes to local_data/<env_name>/PPO_SB3/<run_name>/:
    # final and best model in model/, intermediate checkpoints in checkpoints/, tensorboard in logs/.
    run_name = args.run_name if args.run_name is not None else new_run_name()
    save_path = get_run_dir(env_name, name, run_name, "model")
    checkpoints_path = get_run_dir(env_name, name, run_name, "checkpoints")
    logs_dir = get_run_dir(env_name, name, run_name, "logs")
    print(f"INFO: this run is saved in {os.path.dirname(save_path)}")


    net_arch=[800, 1000, 1000, 800]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch

    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 1e-3,
            "tensorboard_log": logs_dir,
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
                      total_timesteps=int(1e5),
                      save_path=save_path,
                      checkpoints_path=checkpoints_path,
                      save_freq=int(1e4)
                      )