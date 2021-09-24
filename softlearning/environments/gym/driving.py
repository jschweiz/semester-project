import numpy as np
import pygame

import gym
from gym.utils import EzPickle

class Opponent(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,20))
        self.image.fill((135, 206, 235))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,20))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2

# driving environment
class Driving(gym.Env):
    
    LATERAL = 0.1
    MIN_FORWARD = 0.1 # fixed rate for now
    MAX_FORWARD = 0.1 
    WIDTH = 0.13 # give some leeway
    def __init__(self, oracle=False):
        #EzPickle.__init__(**locals())
        self.mode = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.3])
        self.timestep = 0
        pygame.init()
        self.clock = pygame.time.Clock()
        self.world = pygame.display.set_mode([400,400])
        self.agent1 = Player(self.state[0], self.state[1])
        self.agent2 = Opponent(self.state[2], self.state[3])
        self.sprite_list = pygame.sprite.Group()
        self.sprite_list.add(self.agent1)
        self.sprite_list.add(self.agent2)
        
        self.oracle = oracle
        # States: each car can be on x-axis (-WIDTH, WIDTH) and y-axis (0, inf)
        self.observation_space = gym.spaces.Box(np.array([-self.WIDTH, 0, -self.WIDTH, 0] + ([-1] if oracle else [])), #([0,0,0] if oracle else [] )),
                                                np.array([self.WIDTH, 1e10, self.WIDTH, 1e10] + ([1] if oracle else [])))#([1,1,1] if oracle else [])))

        # Action: ego agent can move left-right by (-0.1, 0.1) and forward (0.01, 0.1)
        self.action_space = gym.spaces.Box(low=-self.LATERAL, high=self.LATERAL, shape=(1,)) #gym.spaces.Box(np.array([-self.LATERAL, self.MIN_FORWARD]),
                            #               np.array([self.LATERAL, self.MAX_FORWARD]))

        self._max_episode_steps = 10
        self.just_switched = None
        self.next_mode = 0

    def get_oracle_state(self):
        #return np.eye(3)[self.mode].tolist()
        return [self.mode]

    # resets agent to start, updates target position
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.3]) # (ego x, ego y, opp x, opp y) where y-axis is forward movement
        self.timestep = 0
        self.just_switched = None
        self.mode = self.next_mode
        return np.array(self.state.tolist() + (self.get_oracle_state() if self.oracle else []))

    # returns next state, reward, done, and info
    def step(self, action):
        self.timestep += 1
        # action for ego
        ego_action = [action[0], 0.1] # ego currently only chooses left-right movement

        # action for other
        other_action = [0.0, 0.05]
        # compare vertical distance down road; want to make sure small enough so it can't totally slow down and evade every time
        if self.just_switched is None and np.abs(self.state[1] - self.state[3]) <= 0.05 + 1e-5:#self.MIN_FORWARD :
            self.just_switched = self.timestep
            if self.mode == -1:
                other_action = [-self.WIDTH, 0.0]
            elif self.mode == 0:
                other_action = [0.0, 0.0]
            elif self.mode == 1:
                other_action = [self.WIDTH, 0.0]
        # next state

        deltax = np.array(ego_action + other_action)
        next_state = np.array(self.state + deltax)

        # ego agent colliding against road boundaries -- project onto (-WIDTH, WIDTH) to stay on road
        if np.abs(next_state[0]) >= self.WIDTH:
            next_state[0] = next_state[0] * self.WIDTH / np.abs(next_state[0])

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
        if self.just_switched == self.timestep - 1:
            if self.state[0] < -0.05:
                self.next_mode = -1
            elif self.state[0] > 0.05:
                self.next_mode = 1
            else:
                self.next_mode = 0
        
        # different ego vs. opp velocities allow for multiple collisions actually-- fixed timestep is interesting
        if self.timestep == self._max_episode_steps:
            done = True
        else:
            done = False
            self.state = next_state
        # mode if we reset from current state (info)
        info = self.mode
        return np.array(self.state.tolist() + (self.get_oracle_state() if self.oracle else [])), reward, done, [info*1.0]

    def render(self, mode=None):
        if mode is not None:
            return None
        self.agent1.update(self.state[0], self.state[1])
        self.agent2.update(self.state[2], self.state[3])

        # animate
        self.world.fill((255,255,255))
        pygame.draw.rect(self.world, (0, 0, 0), (150, 0, 100, 400), 0)
        self.sprite_list.draw(self.world)
        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        pygame.quit()
