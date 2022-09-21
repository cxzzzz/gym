"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import random
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.core import ObservationWrapper
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

from PIL import Image
import pygame


#space : 0:forward 1:left 2:right
class MyCarEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def cal_displacement(self,dist:float,radian:float):
        dx = dist * math.sin(radian)
        dy = -dist * math.cos(radian)
        return (dx,dy)

    def __init__(self, render_mode: Optional[str] = None):

        assets_path = "/home/cxzzzz/Programming/rl/gym/gym/envs/classic_control/assets"
        self.car_map = np.array(Image.open(f"{assets_path}/map.png")).transpose(1,0,2)
        self.car_map_size = (self.car_map.shape[0],self.car_map.shape[1])
        self.car_size = (35,50)

        self.VEL = 10
        self.RADIAN_VEL = math.pi * 0.01
        self.DEFAULT_CAR_POS = (100,100)
        self.DEFAULT_CAR_RADIAN = math.pi / 2

        self.detector_size = 100
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0 , high = 1 , shape = [self.detector_size])

        self.render_mode = render_mode
        self.clock = None
        self.step_cnt = 0


        self.screen = None
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode(self.car_map_size)
            pygame.display.set_caption('car')

            self.car_map_img = pygame.image.load(f"{assets_path}/map.png")
            self.car_img_orig = pygame.transform.scale( pygame.image.load(f"{assets_path}/car.png") , self.car_size )

        self.reset()

    
    def get_detector_poses(self):
        start_pos = ( self.car_pos[0] + self.detector_size/2 * math.sin( self.car_radian - math.pi/2 )  ,  
            self.car_pos[1] -self.detector_size/2 * math.cos( self.car_radian - math.pi/2 )  )
        end_pos = (  self.car_pos[0] + self.detector_size/2 * math.sin( self.car_radian + math.pi/2 )  ,  
            self.car_pos[1] -self.detector_size/2 * math.cos( self.car_radian + math.pi/2 )  )

        poses = []
        for i in range(self.detector_size):
            pos = (round(start_pos[0]+(end_pos[0]-start_pos[0])*(i/(self.detector_size-1))),
                round(start_pos[1]+(end_pos[1]-start_pos[1])*(i/(self.detector_size-1))) )
            poses.append(pos)
        return poses

    def step(self,action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        reward = 0
        observation = 0
        car_move_forward = False

        if action == 0: #forward
            dx,dy = self.cal_displacement( self.VEL , self.car_radian )
            self.car_pos[0] += dx
            self.car_pos[1] += dy
            car_move_forward = True
        elif action == 1: #left
            self.car_radian -= self.RADIAN_VEL
        elif action == 2: #right
            self.car_radian += self.RADIAN_VEL
        
        self.car_pos[0] = max(0,min(self.car_pos[0],self.car_map_size[0]))
        self.car_pos[1] = max(0,min(self.car_pos[1],self.car_map_size[1]))


        start_cnt = 0
        middle_cnt = 0
        end_cnt = 0
        observation = np.zeros( self.detector_size )

        for idx,pos in enumerate(self.get_detector_poses()):
            if( pos[0] >= 0 and pos[0] < self.car_map_size[0] and pos[1] >= 0 and pos[1] < self.car_map_size[1]):
                if (self.car_map[pos[0]][pos[1]] == [0,255,0]).all(): #start
                    start_cnt += 1
                elif (self.car_map[pos[0]][pos[1]] == [0,0,0]).all(): #middle
                    middle_cnt += 1
                    observation[idx] = 1
                elif (self.car_map[pos[0]][pos[1]] == [255,0,0]).all(): #middle
                    end_cnt += 1
        
        done = False
        reward = -0.1
        #observation = middle_cnt

        if(end_cnt > 0):
            done = True
            reward = 100000
        elif middle_cnt > 0 or start_cnt > 0:
            if car_move_forward and middle_cnt>0:
                reward =  20*self.detector_size/( 1+(np.sum( observation)* np.sum( np.abs(np.argwhere(observation>0) - (self.detector_size/2))) )) 
        else:
            reward = -2000
            done = True
        
        self.step_cnt += 1
        if self.step_cnt > 10000:
            done = True

        if(self.render == 'human'):
            self.render()
        
        
        #print( self.step_cnt , observation, reward, done )
        return  observation, reward, done, False, {}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        self.car_pos = list( self. DEFAULT_CAR_POS)
        self.car_radian = ( math.pi / 2 ) * random.uniform(0.9,1.1)

        if self.render_mode == "human":
            self.render()
        
        return np.zeros(self.detector_size)

    def render(self):
        self.screen.fill((255,255,255))
        self.screen.blit(self.car_map_img,(0,0))

        car_img = pygame.transform.rotate(self.car_img_orig, angle = -self.car_radian/math.pi*180)
        self.screen.blit(car_img,(self.car_pos[0]-self.car_size[0]/2 , self.car_pos[1]-self.car_size[1]/2))

        start_pos = ( self.car_pos[0] + self.detector_size/2 * math.sin( self.car_radian - math.pi/2 )  ,  
            self.car_pos[1] -self.detector_size/2 * math.cos( self.car_radian - math.pi/2 )  )
        end_pos = (  self.car_pos[0] + self.detector_size/2 * math.sin( self.car_radian + math.pi/2 )  ,  
            self.car_pos[1] -self.detector_size/2 * math.cos( self.car_radian + math.pi/2 )  )

        pygame.draw.line(self.screen,(255,0,255),
            start_pos = start_pos,
            end_pos = end_pos ,
            width = 8
        ) 
        pygame.display.flip()


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            #self.isopen = False
