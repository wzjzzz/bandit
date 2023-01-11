# -*- coding: utf-8 -*-

from itertools import product
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class cube:
    def __init__(self, x, ys, xs, cube_length, d=1, reward=0):
        '''
        Parameters
        ----------
        x : TYPE: list or numpy.array
            'bottom left' point of initial cube, 
            'bottom left' means the coordinates of other points in the cube are not less than this point
            for example: the 'bottom left' point of cube [(0,0),(1,0),(0,1),(1,1)] is (0,0)
        cube_length : TYPE: float
            edge length of cube
        d : TYPE, int
            dimension of cube
        reward: TYPE, float
            mean reward of the cube
        '''
        self.x = x
        self.cube_length = cube_length
        self.d = d
        self.ys = ys
        self.xs = xs       
        self.children = []
        
        
    def split(self):
        '''split by dyadic'''
        new_cube_length = self.cube_length / 2
        candidate_idx = np.array(list(product([0,1], repeat=self.d)))
        for idx in candidate_idx:
            new_x = self.x + idx * new_cube_length
            if not np.array(self.xs).any():
                new_cube = cube(new_x, [], [], new_cube_length)
            else:
                new_xs, new_ys = self.update_xy(self.xs, self.ys, new_x, new_cube_length)
                new_cube = cube(new_x, new_ys, new_xs, new_cube_length)
            new_cube.parent = self
            self.children.append(new_cube)
        return self.children
    
    
    def insert_point(self, x, y):
        if (x >= self.x).all() and (x <= (self.x+self.cube_length)).all():
            self.xs.append(x)
            self.ys.append(y)
    

        
        
        
    def update_xy(self, parent_xs, parent_ys, child_x, child_cube_length):
        xs = np.array(parent_xs).copy()
        ys = np.array(parent_ys).copy()
        condition = ((xs >= child_x) & (xs <= child_x + child_cube_length)).all(axis=1)
        try:
            return xs[condition], ys[condition]
        except:
            print(child_x)
            print(ys.shape)
    
    
    
    def mean_reward(self):
        return np.mean(self.ys)
        
        

            
        
class BMO_E:
    def __init__(self, T, init_x, init_cube_length, eta, epsilon, delta, c, split_times=1):
        '''
        Parameters
        ----------
        B : TYPE: int
            total batch number
        init_x : TYPE: list or numpy.array
            'bottom left' point of initial cube, 
            'bottom le8ft' means the coordinates of other points in the cube are not less than this point
            for example: the 'bottom left' point of cube [(0,0),(1,0),(0,1),(1,1)] is (0,0)
        init_cube_length :TYPE: float
            edge length of initial cube

        '''
        
        self.init_cubes = [cube(np.array(init_x), [], [], init_cube_length)]
        #self.regrets = []  #regret list of each batch
        self.T = T   #time horizon
        self.s_cubes = []  #cubes surviving each batch
        self.cubes = []    #cubes played each batch
        self.count = 0     #number of times arm has been played
        self.eta = eta
        self.epsilon = epsilon
        self.delta = delta
        self.regrets = []   #regret list
        self.cur_regret = 0  #current regret
        self.rewards = []
        self.arms = []
        self.delta_regret = 0
        self.split_times = split_times
        self.c = c
        self.tmp = []

        
    
    def initialize(self):
        
        # calcalate f_delta
        print('------initialize--------')
        linspace = np.linspace(-1,1,100000)
        arms = list(product(linspace, repeat=1))
        fun_values = []
        print('get f_delta:')
        for arm in arms:
            if arm == 0 or arm == 1/2:
                continue
            fun_values.append(self.get_reward(arm)[1])
        self.tmp = fun_values
        fun_values = np.array(fun_values)
        self.delta_regret = np.quantile(fun_values, 1-self.delta)
        self.m_regret = np.max(fun_values)
        
        
        print('----play batch 0 ------')
        cubes = self.init_cubes
        for i in range(self.split_times):
            cubes = self.partition(cubes)
        self.cubes.append(cubes)
        
        nb, Hb = self.values4batch()
        print('Hb', Hb)
        self.play_one_batch(cubes, nb)
        print('have play {} times totally'.format(self.count))
        
        s_cubes = self.elimination(cubes, Hb)
        self.s_cubes.append(s_cubes)
        
        
          
    def values4batch(self):
        '''
        get some values related to batch
        
        Returns
        -------
        nb : playtimes of batch
        Hb : Hoeffding bound of batch
        '''
        phi = np.max([np.log(self.T**2/self.epsilon), 2*np.log2(1/self.eta)])
        denominator = phi * np.sqrt(2*np.log(2*self.T**2/self.epsilon))
        
#         denominator = np.sqrt(np.log(self.T/self.epsilon))
        
        cube_length = self.cubes[-1][-1].cube_length
        d = self.cubes[-1][-1].d
        mu_cube = pow(cube_length, d)
        numerator = self.c * np.log(mu_cube/self.eta)
        
        nb_sqrt = np.floor(denominator / numerator)
        
        nb = pow(nb_sqrt, 2)
        print(nb)
        Hb = denominator / np.sqrt(nb)
        
        return int(nb), Hb
        
        
    def partition(self, cubes):
        '''
        cubes: list of cubes to be split
        '''
        new_cubes = []
        for cube in cubes:
            new_cubes_ = cube.split()
            new_cubes += new_cubes_
        return new_cubes
    
    
    
    def play_one_batch(self, cubes, nb):
        print('every cube need to be played {} times this batch'.format(nb))
        
        for cube in cubes:
            cube_nb = max(0, nb - len(cube.xs))
            for t in range(cube_nb):
                cube_nb = max(0, nb - len(cube.xs))
                self.count += 1
                if self.count > self.T:
                    break
                arm = self.sample(cube)
                self.arms.append(arm)
                observation, reward = self.get_reward(arm)
                cube.xs = list(cube.xs) + [arm]
                cube.ys = list(cube.ys) + [observation]
                
                self.cur_regret += max(self.delta_regret - reward, 0)
                self.regrets.append(self.cur_regret)
                self.rewards.append(reward)
                
            if self.count > self.T:
                break
                
                

        
            
    def elimination(self, cubes, Hb):
        rewards = [cube.mean_reward() for cube in cubes]
        max_reward = np.max(rewards)
        rewards_diff = max_reward - rewards
        s_cubes_idx = np.where(rewards_diff <= (2+1/self.c)*Hb)[0]
        s_cubes = [cubes[idx] for idx in s_cubes_idx]
        return s_cubes
                
        
    def sample(self, cube):
        arm = np.random.uniform(cube.x, cube.x+cube.cube_length)
        while arm[0] == 0 or arm[0] == 1/2:
            arm = np.random.uniform(cube.x, cube.x+cube.cube_length)
        return arm
        
    
    def get_reward(self, arm):
        '''  get function value of arm
        function: bmo function in [-1, 1]
        arm: shape (1, 1) in this example
        '''
        x = arm[0]
        
        reward = -np.log(abs(x)) - np.log(abs(x-1/2)) / 3
        reward = reward * 20
        observation = np.random.normal(reward, 0.1)
        return observation, reward
        
    
    
    def play(self):
        self.initialize()
        for b in range(1, self.T):
            print('----play batch {} ------'.format(b))
            cubes = self.partition(self.s_cubes[-1])
            if self.count > self.T:
                break
                
            
            self.cubes.append(cubes)
            nb, Hb = self.values4batch()
            print('Hb', Hb)

            self.play_one_batch(cubes, nb)
            print('have play {} times totally'.format(self.count-1))
            
            s_cubes = self.elimination(cubes, Hb)
            self.s_cubes.append(s_cubes)
            
            
    def plot(self):
        self.play()
        scale_rewards = self.rewards
        delta_regret = self.delta_regret
        m_regret = self.m_regret
        #scale_rewards = [(np.log(reward*8000) - 8.693) / 2.427 for reward in self.rewards]
        #delta_regret = (np.log(self.delta_regret*8000) - 8.693) / 2.427
        scale_regrets = [max(0, delta_regret - reward) for reward in scale_rewards]
        regrets = [(m_regret - reward) for reward in scale_rewards]
        #plt.plot(np.cumsum(scale_regrets))
        #plt.plot(np.cumsum(regrets))
        #plt.show()
        
        return scale_regrets, regrets
