import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class EpochCheckpointCallback(BaseCallback):
    def __init__(self, n: int, freq: int, name: str, verbose=0):
        super(EpochCheckpointCallback, self).__init__(verbose)
        self.n = n
        self.freq = freq
        self.name = name
        self.last_saved_epoch = -1

    def _on_step(self):
        cur_epoch = self.num_timesteps // self.n
        if cur_epoch % self.freq == 0 and cur_epoch != self.last_saved_epoch:
            self.last_saved_epoch = cur_epoch
            model_path = os.path.join( f"{self.name}_epoch_{cur_epoch}")
            self.model.save(model_path)
            print(f"{cur_epoch}")
        return True
    
from overtake_env import OvertakeEnv

def train():
    device = "cpu"
    env = DummyVecEnv([lambda: OvertakeEnv()])

    checkpoint_callback = EpochCheckpointCallback(
        n=50,
        freq=5,
        name="ppo_overtake"
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        learning_rate=1e-5,
        n_steps=8192,
        batch_size=512,
        n_epochs=20,
        gamma=0.999,
        ent_coef=0.1,
        clip_range=0.2,
        verbose=1,
        device=device
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    model.save("ppo_overtake")

train()