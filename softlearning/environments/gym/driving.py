# REMOVE ALL INSTANCES OF PYGAME

import numpy as np
import pygame

import gym
from gym.utils import EzPickle

import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

JITTER = True

NUM_STRATEGIES = 3
STRATEGY_COLORS = ['red', 'green', 'blue']
PROB_CHANGE = 0.0
LANE_DISTANCE = 0.08

ORACLE = True
ORACLE_STRATEGIES = True

class Opponent():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update(self, x, y):
        self.x = x
        self.y = y


class Player():

    def __init__(self, x, y):

        # initial conditions
        self.x = x
        self.y = y


    def update(self, x, y):
        self.x = x
        self.y = y

# driving environment
class Driving(gym.Env, EzPickle):
    
    LATERAL = 0.1
    MIN_FORWARD = 0.1 # fixed rate for now
    MAX_FORWARD = 0.1 
    WIDTH = 0.13 # give some leeway
    def __init__(
                self, oracle=False,
                min_episode_log=900,
                gif_index_from_end=100,
                exp_name="PLACEHOLDER",
                save_path="/Users/Documents/",):
        #EzPickle.__init__(**locals())
        self.mode = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.3])
        self.timestep = 0

        self.agent1 = Player(self.state[0], self.state[1])
        self.agent2 = Opponent(self.state[2], self.state[3])
        
        self.oracle = ORACLE
        # States: each car can be on x-axis (-WIDTH, WIDTH) and y-axis (0, inf)
        self.observation_space = gym.spaces.Box(np.array([-self.WIDTH, 0, -self.WIDTH, 0] + ([-1] if self.oracle else []) + ([0] if ORACLE_STRATEGIES else [])), #([0,0,0] if oracle else [] )),
                                                np.array([self.WIDTH, 1e10, self.WIDTH, 1e10] + ([1] if self.oracle else []) + ([2] if ORACLE_STRATEGIES else [])))#([1,1,1] if oracle else [])))

        # Action: ego agent can move left-right by (-0.1, 0.1) and forward (0.01, 0.1)
        self.action_space = gym.spaces.Box(low=-self.LATERAL, high=self.LATERAL, shape=(1,)) #gym.spaces.Box(np.array([-self.LATERAL, self.MIN_FORWARD]),
                            #               np.array([self.LATERAL, self.MAX_FORWARD]))

        self._max_episode_steps = 10
        self.just_switched = None
        self.next_mode = 0
        self.next_mode_decided = False

        self.predicted_latents = []
        self.ego_positions = []
        self.opponent_positions = []
        self._meta_time = -1

        self.strategy_hist = [0, 0, 0]
        self.all_episode_rewards = [] # tuples of (task reward, stable reward)
        self.curr_episode_task_reward = 0
        self.curr_episode_stable_reward = 0
        self.stabilized = False

        self.gif_images_paths = []

        self.min_episode_log = min_episode_log
        self.gif_index_from_end = gif_index_from_end
        self.exp_name = exp_name
        self.save_path = save_path
        os.mkdir(self.save_path + '{}/'.format(self.exp_name))

    def get_oracle_state(self):
        #return np.eye(3)[self.mode].tolist()
        oracle_state = []
        if ORACLE_STRATEGIES:
            oracle_state.append(self.mode)
        if self.stabilized:
            oracle_state.append(1.0)
        else:
            oracle_state.append(0.0)

        return oracle_state

    def PCA_plot_predicted_latents(self):
        time_from_end_to_plot = 100

        pca = PCA(n_components=2)
        #all_latents = np.array(self.predicted_latents[-time_from_end_to_plot:]) # Only plot last time_from_end_to_plot latents
        all_latents = np.array(self.predicted_latents)
        #dim_reduced_latents = all_latents 
        # print(all_latents.shape)
        dim_reduced_latents = pca.fit_transform(all_latents)

        if JITTER:
            dim_reduced_latents = dim_reduced_latents + (np.random.randn(*dim_reduced_latents.shape) * 0.1)
        
        # print(dim_reduced_latents.shape)

        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        cmap = matplotlib.colors.ListedColormap(STRATEGY_COLORS)

        colors = self.strategy_hist[-dim_reduced_latents.shape[0] - 1:-1]
        scatter = plt.scatter(dim_reduced_latents[:, 0], dim_reduced_latents[:, 1], c=colors, cmap=cmap, s=20)
        #plt.scatter(dim_reduced_latents[:, 0], dim_reduced_latents[:, 1], c=[(i+1)/dim_reduced_latents.shape[0] for i in range(dim_reduced_latents.shape[0])], cmap='winter', s=20)
        plt.colorbar(scatter, ticks=np.linspace(0, NUM_STRATEGIES - 1, NUM_STRATEGIES))
        plt.title("PCA of All Latent Strategies")
        plt.savefig(self.save_path + '{}/{}_pca.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

        # Plot only the last time_from_end_to_plot latent strategies
        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        cmap = matplotlib.colors.ListedColormap(STRATEGY_COLORS)
        last_latents = dim_reduced_latents[-time_from_end_to_plot:]
        colors = self.strategy_hist[-last_latents.shape[0]-1:-1]
        scatter = plt.scatter(last_latents[:, 0], last_latents[:, 1], c=colors, cmap=cmap, s=20)
        plt.colorbar(scatter, ticks=np.linspace(0, NUM_STRATEGIES - 1, NUM_STRATEGIES))
        plt.title("PCA of Last " + str(time_from_end_to_plot) + " Latent Strategies")
        plt.savefig(self.save_path + '{}/{}_pca_end.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

        # Plot previous strategies throughout training
        all_strategies = np.array(self.strategy_hist)
        plt.figure()
        ax = plt.gca()
        ax.set_aspect('auto')
        plt.scatter(np.arange(all_strategies.shape[0]), all_strategies, c=[(i+1)/all_strategies.shape[0] for i in range(all_strategies.shape[0])], cmap='winter', s=0.05)
        plt.savefig(self.save_path + '{}/{}_strategy_history.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

        plt.figure()
        ax = plt.gca()
        ax.set_aspect('auto')
        plt.scatter(np.arange(time_from_end_to_plot), all_strategies[-time_from_end_to_plot:], c=[(i+1)/time_from_end_to_plot for i in range(time_from_end_to_plot)], cmap='winter', s=0.5)
        plt.savefig(self.save_path + '{}/{}_strategy_history_end.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

        # Compute average pairwise change in latents and write to a text file. 
        avg_change_in_latents = np.mean(np.abs(all_latents[1:] - all_latents[:-1]))
        f = open(self.save_path + '{}/{}_change_in_latents.txt'.format(self.exp_name, self.exp_name), "w")
        f.write("Average pairwise change in latents: " + str(avg_change_in_latents) + "\n")
        f.close()

        with open(self.save_path + '{}/{}_rewards.npy'.format(self.exp_name, self.exp_name), 'wb') as f:
            np.save(f, self.all_episode_rewards)

        plt.plot(np.arange(len(self.all_episode_rewards)), [curr_reward[0] for curr_reward in self.all_episode_rewards])
        plt.title("Task Reward")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.savefig(self.save_path + '{}/{}_task_reward.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

        plt.plot(np.arange(len(self.all_episode_rewards)), [curr_reward[1] for curr_reward in self.all_episode_rewards])
        plt.title("Stable Reward")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.savefig(self.save_path + '{}/{}_stable_reward.png'.format(self.exp_name, self.exp_name), format='png')
        plt.close()

    def save_gif_interactions(self):
        all_images = [Image.open(im_path) for im_path in self.gif_images_paths[-self.gif_index_from_end:]]
        all_images[0].save(self.save_path + '{}/{}_interactions.gif'.format(self.exp_name, self.exp_name), save_all=True, append_images=all_images[1:], optimize=False, duration=500, loop=0)

    # resets agent to start, updates target position
    def reset(self):
        self._meta_time += 1
        if self.ego_positions and self._meta_time >= self.min_episode_log:            
            pos_hist = np.array(self.ego_positions)
            opp_hist = np.array(self.opponent_positions)
            plt.figure()
            ax = plt.gca()
            #ax.set_aspect('equal')
            
            plt.scatter(pos_hist[:, 0], pos_hist[:, 1], c=[(i+1)/pos_hist.shape[0] for i in range(pos_hist.shape[0])], cmap='winter', s=100)
            plt.plot(pos_hist[:, 0], pos_hist[:, 1])
            plt.scatter(opp_hist[:, 0], opp_hist[:, 1], c=[(i+1)/opp_hist.shape[0] for i in range(opp_hist.shape[0])], cmap='inferno', s=100)
            plt.plot(opp_hist[:, 0], opp_hist[:, 1])

            plt.plot([-self.WIDTH, self.WIDTH], [0.1, 0.1], color="red") # reaction time line
            plt.plot([-LANE_DISTANCE, -LANE_DISTANCE], [0.0, 1.0], color="grey", linestyle='dashed') # reaction time line
            plt.plot([LANE_DISTANCE, LANE_DISTANCE], [0.0, 1.0], color="grey", linestyle='dashed') # reaction time line

            plt.xlim(-self.WIDTH, self.WIDTH)
            plt.ylim(0.0, 1.0)
            plt.savefig(self.save_path + '{}/{}.png'.format(self.exp_name, self._meta_time), format='png')
            plt.close()

            # save the path to create a gif later
            self.gif_images_paths.append(self.save_path + '{}/{}.png'.format(self.exp_name, self._meta_time))

        self.state = np.array([0.0, 0.0, 0.0, 0.3]) # (ego x, ego y, opp x, opp y) where y-axis is forward movement
        self.timestep = 0
        self.just_switched = None
        self.mode = self.next_mode
        self.next_mode_decided = False
        self.stabilized = False

        self.strategy_hist.append(self.mode)

        self.ego_positions = [(self.state[0], self.state[1])]
        self.opponent_positions = [(self.state[2], self.state[3])]
        return np.array(self.state.tolist() + (self.get_oracle_state() if self.oracle else []))

    # returns next state, reward, done, and info
    def step(self, action):
        #print(self.just_switched)
        self.timestep += 1
        # action for ego
        ego_action = [action[0], 0.1] # ego currently only chooses left-right movement

        # action for other
        other_action = [0.0, 0.05]
        # compare vertical distance down road; want to make sure small enough so it can't totally slow down and evade every time
        if self.just_switched is None and np.abs(self.state[1] - self.state[3]) <= 0.05 + 1e-5:#self.MIN_FORWARD :
            self.just_switched = self.timestep
            if self.mode == 0:
                other_action = [-self.WIDTH, 0.0]
            elif self.mode == 1:
                other_action = [0.0, 0.0]
            elif self.mode == 2:
                other_action = [self.WIDTH, 0.0]
        # next state

        deltax = np.array(ego_action + other_action)
        next_state = np.array(self.state + deltax)

        # ego agent colliding against road boundaries -- project onto (-WIDTH, WIDTH) to stay on road
        if np.abs(next_state[0]) >= self.WIDTH:
            next_state[0] = next_state[0] * self.WIDTH / np.abs(next_state[0])

        self.ego_positions.append((next_state[0], next_state[1]))
        self.opponent_positions.append((next_state[2], next_state[3]))

        # reward for current state
        reward = 0.0
        dist_x = np.abs(self.state[0] - self.state[2]) 
        dist_y = np.abs(self.state[1] - self.state[3])
        dist = np.sqrt(dist_x**2 + dist_y**2)
        #print(dist_x, dist_y)

        # 0.1 is not touching at all
        if dist_x < 0.1 and dist_y <= 0.05 + 1e-6:
            #print('COLLISSION')
            reward = -1.0

        # done if trajectory reaches full length
        # Passing driving env dynamics
        # if self.just_switched == self.timestep - 1:
        #     if self.state[0] < -0.05:
        #         self.next_mode = 0
        #     elif self.state[0] > 0.05:
        #         self.next_mode = 2
        #     else:
        #         self.next_mode = 1

        if not self.next_mode_decided and (self.just_switched == self.timestep - 1):
            if self.state[0] < -0.05:
                self.next_mode = 0
            elif self.state[0] > 0.05:
                self.next_mode = 2
            else:
                self.next_mode = 1
            if np.random.random() < PROB_CHANGE: # change to the opposite mode (left -> right, right -> left)
                if self.next_mode == 0:
                    self.next_mode = 2
                elif self.next_mode == 2:
                    self.next_mode = 0

        # different ego vs. opp velocities allow for multiple collisions actually-- fixed timestep is interesting
        if self.timestep == self._max_episode_steps:
            done = True
        else:
            done = False
            self.state = next_state
        # mode if we reset from current state (info)
        info = self.mode

        # Stable Driving Env dynamics: There are 3 lanes and a road hazard upcoming in the center lane, so both the ego and other car need to 
        # merge to a new lane. If the ego agent merges to a lane before the red barrier (giving opponent enough reaction time), 
        # then the opponent merges to the opposite lane as the ego agent. Otherwise, the opponent aggressively choosees to merge to the lane the
        # ego agent passed in and with PROB_CHANGE chooses to nicely merge into the other lane. 
        if self.timestep == 1: # before the red barrier
            if self.state[0] < -LANE_DISTANCE:
                self.next_mode = 2 # opponent goes right next trajectory 
                self.next_mode_decided = True
                self.stabilized = True
            # elif self.state[0] > LANE_DISTANCE:
            #     self.next_mode = 0 # opponent goes left next trajectory
            #     self.next_mode_decided = True
            else:
                self.next_mode_decided = False

        if done:
            self.all_episode_rewards.append((self.curr_episode_task_reward, self.curr_episode_stable_reward))
            self.curr_episode_task_reward = 0
            self.curr_episode_stable_reward = 0
        self.curr_episode_task_reward += reward
        change_in_strategy = None
        if self.strategy_hist[-1] != self.strategy_hist[-2]: # change in strategy
            change_in_strategy = 0.0
        else:
            change_in_strategy = 0.1 # stable for entire episode of horizon 10 gives cumulative reward 1
        self.curr_episode_stable_reward += change_in_strategy

        return np.array(self.state.tolist() + (self.get_oracle_state() if self.oracle else [])), reward, done, {'mode': info*1.0}
#        return np.array(self.state.tolist() + (self.get_oracle_state() if self.oracle else [])), reward, done, [info*1.0]

    def render(self, mode=None):
        if mode is not None:
            return None
        self.agent1.update(self.state[0], self.state[1])
        self.agent2.update(self.state[2], self.state[3])
        # SKIP ANIMATION
