#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import imageio
from datetime import datetime

import carla
import gym
from gym.utils import EzPickle

import queue
import random
import time
import pygame
import numpy as np


### SYNCHRNOZING SIMULATION TIMESTEPS ######
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, should_render, *sensors, **kwargs):
        self.world = world
        self.should_render = should_render
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=not self.should_render,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            # try-catch added for null collision sensing
            try:
                data = sensor_queue.get(timeout=1.0)
                if data.frame == self.frame:
                    return data
            except:
                return None
            
####  Utility methods for pygame rendering   ####
def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


####  General Utility Methods ####
def to_np(vec):
    return np.array([vec.x, vec.y])
    #return np.array([vec.x, vec.y, vec.z]) don't give z since never changes

def get_vehicle_state(v):
    vel = to_np(v.get_velocity())
    angular = to_np(v.get_angular_velocity())
    pos = to_np(v.get_location())
    return np.concatenate((pos, vel, angular))


DEBUG = True
def debug_print(*args):
    if DEBUG:
        print(*args)


#####  GYM ENV   #####
class CarlaEnv(gym.Env, EzPickle):

    OPP_THROTTLE = 0.5
    EGO_THROTTLE = 0.6
    TRIGGER_THROTTLE = 0.6
    MAX_STEER = 0.15

    LANE_BOUNDS = (-248.0, -239.3)  # the actual carla boundaries
    LANE_TOLERANCE = 0.1  # constant to add to widen the lanes for easier training?
    STEER_BOUNDS = 1.25  # distance within lane boundary to apply steering cutoffs
    STEER_FACTOR = 0.05
    VEER_TRIGGER = -4.0  # distance at which the opponent is triggered to veer into the other lanes
    SUCCESS_DIST = 3.5    # distance the ego_agent has to be past the opponent before ending the episode
    
    BOUND = 1e3 # arbitrary bound on pos,vel, angular vel
    TICKS_PER_STEP = 5  # how many simulation ticks per action

    def __init__(self, render=False, should_save=False, oracle=False):
        
        EzPickle.__init__(**locals())
        # Internal Bookkeeping
        self.should_render = render
        self.should_save = should_save
        self.oracle = oracle
        self.mode = -1  # -1: veer left;  1: veer right
        self.next_mode = -1
        self.done_cache = False  # weird quitting errors???

        self.triggered = False  # keep track of opponent veering trigger time
        self.timestep = 0
        self._max_episode_steps = 400  # should probably be 200ish tbh

        # Gym Bookkeeping
        state_size = (3 * 2) * 2 + (1 if self.oracle else 0)
        self.observation_space = gym.spaces.Box(low=-self.BOUND, high=self.BOUND, shape=(state_size, ))
        self.action_space = gym.spaces.Box(low=-self.MAX_STEER, high=self.MAX_STEER, shape=(1,))

        # Run initializations for the CARLA environment
        self.actor_list = []
        self.initialize_simulation()
        self.initialize_agents()
        self.initialize_sensors()
        self.set_initial_agent_conditions()

        ##### Start Synchronization #####
        self.sync_mode = CarlaSyncMode(self.world, self.should_render, self.rgb_camera, fps=30) #self.collision_sensor, fps=30)
        self.sync_mode.__enter__() # start synchronization
        self.tick_data = None

    def initialize_simulation(self):
        # Initalize PyGame + World
        pygame.init()
        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(3.0)
        self.world = self.client.load_world('Town05')
        self.blueprint_library = self.world.get_blueprint_library()

    def initialize_agents(self):
        #### ADD SENSORS AND SPAWN AGENTS #### 
        debug_print(self.blueprint_library.filter('vehicle'))
        bp = self.blueprint_library.filter('vehicle')[0]

        # Spawn Ego Agent and Set Location
        bp.set_attribute('color', '106, 156, 226')
        ego_transform = self.world.get_map().get_spawn_points()[0]
        self.ego_agent = self.world.spawn_actor(bp, ego_transform)
        self.actor_list.append(self.ego_agent)
        debug_print('created ego agent')
        
        # add an opponent car
        bp.set_attribute('color', '160, 160, 160')
        opp_transform = self.world.get_map().get_spawn_points()[1]
        self.opponent = self.world.spawn_actor(bp, opp_transform)
        self.actor_list.append(self.opponent)
        debug_print('created opponent %s' % self.opponent.type_id)
        
    def set_initial_agent_conditions(self):
        self.ego_start_location = carla.Location(x=-244.0, y=97.1, z=10)
        self.opponent_start_location = carla.Location(x=-243.6, y=80.0, z=10)

        self.ego_agent.set_transform(carla.Transform(self.ego_start_location, carla.Rotation(0, -89.5, 0)))
        ## For now...testing
        self.ego_agent.apply_control(carla.VehicleControl(throttle=self.EGO_THROTTLE, steer=0))

        ## Set opponent here
        self.opponent.set_transform(carla.Transform(self.opponent_start_location, carla.Rotation(0, -89.5, 0)))
        self.opponent.apply_control(carla.VehicleControl(throttle=self.OPP_THROTTLE, steer=0))

        # Issues with resetting the env and maintaining previous velocities...
        ego_v = carla.Vector3D(0.05, -20, 0)
        opp_v = carla.Vector3D(0.05, -15, 0)
        zero_v = carla.Vector3D(0, 0, 0)
        self.ego_agent.set_angular_velocity(zero_v)
        self.opponent.set_angular_velocity(zero_v)
        self.ego_agent.set_velocity(ego_v)
        self.opponent.set_velocity(opp_v)
        debug_print('finished setting initial conditions')

    def initialize_sensors(self):
        # Create Ego Agent Camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "120")
        camera_transform = carla.Transform(carla.Location(x=-4, y=0, z=3), carla.Rotation(-20, 0, 0))
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_agent, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.rgb_camera)
        debug_print('created ego camera %s' % self.rgb_camera.type_id)

        # Create Ego Agent Collision Detector
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_agent)
        self.actor_list.append(self.collision_sensor)
        debug_print('created collision sensor')

        self.collision_detect = queue.Queue()
        self.collision_sensor.listen(self.collision_detect.put)


    def get_state(self):
        ego_state = get_vehicle_state(self.ego_agent)
        opp_state = get_vehicle_state(self.opponent)

        # we care about distance from start location for the ego, and distance between ego and opp
        opp_state[:2] -= ego_state[:2]  # order matters here
        ego_state[:2] -= to_np(self.ego_start_location)
        arrs = [ego_state, opp_state] + ( [np.array([self.mode])] if self.oracle else [])
        return np.concatenate(arrs)
         

    def step(self, action):

        # Bound the action to not hit the side road too much
        vel_x = self.ego_agent.get_velocity().x
        pos_x = self.ego_agent.get_location().x
        if pos_x < self.LANE_BOUNDS[0] + self.STEER_BOUNDS  and vel_x < 0: # so within left boundary, don't steer left
            offset = vel_x * -self.STEER_FACTOR  * 1/max(abs(self.LANE_BOUNDS[0] - pos_x)**2, 1/2)  # penalize if leftwards, scale up to _x and make it go other direction
            offset = min(offset, self.MAX_STEER)
            if offset != 0:
                print(f'Forced steering to the right {vel_x}')
            action = np.maximum(offset, action)
        if pos_x > self.LANE_BOUNDS[1] - self.STEER_BOUNDS  and vel_x > 0: # so within right boundary, don't steer right
            offset = vel_x * -self.STEER_FACTOR * 1/max(abs(self.LANE_BOUNDS[1] - pos_x) ** 2, 1/2) # penalize if rightwards, scale and make it go other direction
            offset = max(offset, -self.MAX_STEER)
            if offset != 0:
                print(f'Forced steering to the left {vel_x}')
            action = np.minimum(offset, action)

        # take the action lol
        self.ego_agent.apply_control(carla.VehicleControl(throttle=self.EGO_THROTTLE, steer=float(action[0])))  # action: (1,) np.float32 dtype
        self.clock.tick()

        self.tick_data = self.sync_mode.tick(timeout=2.0)
        snapshot, image_rgb = self.tick_data
 
        state = self.get_state()
        reward = 0
        done = False
        info = {}

        dist_y = (self.ego_agent.get_location().y - self.opponent.get_location().y) * -1  # negative means behind, since highway orientation goes from high y => low y
        dist_x = self.ego_agent.get_location().x - self.opponent.get_location().x  # negative means to left of, since highway orientation has standard x-axis
 
        # If we're out of bounds (the LANE_BOUNDS ) or collide with opponent...terminate
        with self.collision_detect.mutex:
            if self.collision_detect.queue:
                debug_print(f'Collision Detected!  DistY: {dist_y}  DistX: {dist_x}')
                if self.timestep < 15:  # small number where a collision shouldn't happen by now
                    debug_print('Clearing collision mutex from buffered events')
                    self.collision_detect.queue.clear()
                else:
                    done = True
                    reward = -1.0

        if (pos_x < self.LANE_BOUNDS[0] or pos_x > self.LANE_BOUNDS[1]):
            debug_print(f'Car out of bounds for {"left" if pos_x < self.LANE_BOUNDS[0] else "right"} lane')
            if self.timestep < 5:
                debug_print('Stale out of bounds position data...skipping')
            else:
                done = True
                reward = -1.0


       
        # ego agent nearby and approaching the opponent
        if dist_y > self.VEER_TRIGGER and dist_y < 0 and not self.triggered:
            # veer based on the mode
            self.opponent.apply_control(carla.VehicleControl(throttle=self.TRIGGER_THROTTLE, steer=self.mode * 0.2))  # playing around with steering magnitude...can't be so harsh that you can avoid from behind/staying in middle lane
            self.triggered = True
            debug_print('triggered veering')
            
                       
        # ego agent has passed the opponent or gone past timesteps (which shouldn't happen i think based on forward velocities)
        if dist_y > self.SUCCESS_DIST or self.timestep >= self._max_episode_steps:
            debug_print('succeeded in passing')
            done = True
            # Note: if collided from above conditional, then reward will still be -1.0


        # Rendering and saving
        if image_rgb is not None:
            if self.should_save:
                # TOO SLOW rn
                cc = carla.ColorConverter.Raw
                image_rgb.save_to_disk('/home/ryan/_out/%06d.png' % image_rgb.frame, cc)

            if self.should_render:
                self.render()

        
        self.timestep += 1
        
        # set mode to block off ego agent's lane next episode; do it every step in case the episode early terminates via algorithm
        if pos_x < self.opponent.get_location().x:  # roughly the middle of the middle lane, so < means to left of
            self.next_mode = -1
        else:
            self.next_mode = 1

        self.done_cache = done
        return state, reward, done, info


    def reset(self):
        with self.collision_detect.mutex:
            self.collision_detect.queue.clear()
        self.set_initial_agent_conditions()
        
        debug_print(f'\n\n Resetting after {self.timestep} steps. Cache was {self.done_cache} \n')
        self.done_cache = False
        self.triggered = False
        self.timestep = 0

        debug_print(f'Next mode is {"left" if self.next_mode==-1 else "right"}')
        self.mode = self.next_mode  # go to the next mode (need extra var, since self.mode used in self.get_state)
        
        return self.get_state()

    def close(self):
        ## Pygame and sync cleanup
        pygame.quit()
        self.sync_mode.__exit__()


        debug_print('destroying actors')
        self.rgb_camera.destroy()
        self.collision_sensor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])


        if self.should_save:
            # make a recording instead
            debug_print('ffpmeg the recordings')
            # haxy asf
            os.system(f"ffmpeg -r 60 -f image2 -s 800x800 -pattern_type glob -i '_out/pic*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p _out/test_{datetime.now().strftime('%m_%d_%H_%M_%S')}.mp4")
            os.system("rm /home/ryan/_out/*.png")
            debug_print('done.')
    

    def render(self):
        if self.tick_data is None:
            return

#        snapshot, image_rgb, collision_sensor = self.tick_data
        snapshot, image_rgb = self.tick_data
        if self.should_render:
            #image_semseg.convert(carla.ColorConverter.CityScapesPalette)
            fps = round(1.0 / snapshot.timestamp.delta_seconds)

            # Draw the display.
            draw_image(self.display, image_rgb)
            self.display.blit(
                self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                (8, 10))
            self.display.blit(
                self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                (8, 28))
            pygame.display.flip()

'''
if __name__=='__main__':

    env = CarlaEnv(render=True)
    try:
        for _ in range(10):
            for i in range(10000):
                s, r, d, _ = env.step(env.action_space.sample())
                if d:
                    debug_print('Hit a collision!', f'after {i} steps')
                    break
            env.reset() 
    finally:
        env.close()
    
''' 
