import numpy as np
from gym.utils import EzPickle
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv

import matplotlib.pyplot as plt
import os


class PointMassEnv(MujocoEnv, EzPickle):
    def __init__(
            self,
            action_scale=0.01,
            boundary_dist=0.15,
            include_final_transition=False,
            interaction_length=50,
            save_interactions=True,
    ):
        EzPickle.__init__(**locals())
        self._action_scale = action_scale
        self._boundary_dist = boundary_dist
        self._save_interactions = save_interactions

        self._target_position = None
        self._position = np.zeros(2)
        self._center = np.array([0.0, -0.05])

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self._boundary_dist * np.ones(2)
        self.observation_space = spaces.Box(-o, o, dtype='float32')

        self._include_final_transition = include_final_transition
        self._num_interactions = -1
        self._interaction_length = interaction_length
        self._timestep = 0

        self._dtheta = 0.2
        self._index = -1
        self._interaction = []

        self._reset_interaction()

    def step(self, velocities):
        reward = self.compute_reward()
        
        move_to_start = self._include_final_transition and (self._timestep + 1) % 50 == 0
        
        if move_to_start:
            self._reset_interaction()
        else:
            assert self._action_scale <= 1.0
            velocities = np.clip(velocities, a_min=-1, a_max=1) * self._action_scale
            self._position += velocities
            self._position = np.clip(
                self._position,
                a_min=-self._boundary_dist,
                a_max=self._boundary_dist,
            )

        self._interaction.append(self._position.copy())

        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )

        ob = self._get_obs()
        # reward = self.compute_reward()
        info = {
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
        }
        done = False

        self._timestep += 1
        return ob, reward, done, info

    def _get_goal(self):
        goal = np.zeros(self.observation_space.low.size)
        goal[0] = self._center[0] + 0.1 * np.cos(self._dtheta * self._index)
        goal[1] = self._center[1] + 0.1 * np.sin(self._dtheta * self._index)
        return goal

    def _move_goal(self):
        inside_of_circle = int(np.linalg.norm(self._interaction[-1] - self._center) < 0.1)
        if inside_of_circle:
            self._index += 1
        else:
            self._index -= 1

    def _reset_interaction(self):
        if self._interaction:
            self._move_goal()
            if self._save_interactions:
                self.plot_interaction()
        
        self._interaction = []
        self._target_position = self._get_goal()
        self._position = np.zeros(2)
        self._num_interactions += 1

        return self._get_obs()

    def reset(self):
        if not self._include_final_transition:
            self._reset_interaction()

        self._timestep = 0

        return self._get_obs()

    def plot_interaction(self):
        if self._num_interactions > 3000:
            interaction = np.array(self._interaction)

            plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal')

            circle = plt.Circle(self._center, 0.1, color='dimgray', fill=False)
            ax.add_artist(circle)

            colors = [(i+1) / interaction.shape[0] for i in range(interaction.shape[0])]
            plt.scatter(interaction[:, 0], interaction[:, 1], c=colors, cmap='winter', s=20)
            plt.scatter(self._target_position[0], self._target_position[1], s=30, c='dimgray')

            plt.xlim(self._center[0] - 0.12, self._center[0] + 0.12)
            plt.ylim(self._center[1] - 0.12, self._center[1] + 0.12)

            plt.savefig('{}.png'.format(self._num_interactions), format='png')
            plt.close()

    def _get_obs(self):
        return self._position.copy()

    def compute_reward(self):
        return -np.linalg.norm(self._position - self._target_position)

if __name__ == "__main__":
    env = PointMassEnv(include_final_transition=False, save_interactions=True)

    for i in range(100):
        obs_seq = [env.reset()]
        for t in range(50):
            obs, rew, info, done = env.step(env.action_space.sample())
            obs_seq.append(obs)
