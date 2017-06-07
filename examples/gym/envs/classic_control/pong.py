import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import random
import pygame, sys
from pygame.locals import *

#colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

#globals
WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0,0]
ball_vel = [0,0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0

class PongEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    
    #keydown handler
    #def keydown(event):
    #    global paddle1_vel, paddle2_vel
    #    if event.key == K_UP:
    #        paddle2_vel = -8
    #   elif event.key == K_DOWN:
    #        paddle2_vel = 8
    #    elif event.key == K_w:
    #        paddle1_vel = -8
    #    elif event.key == K_s:
    #        paddle1_vel = 8

    #keyup handler
    #def keyup(event):
    #    global paddle1_vel, paddle2_vel
    #    if event.key in (K_w, K_s):
    #        paddle1_vel = 0
    #    elif event.key in (K_UP, K_DOWN):
    #        paddle2_vel = 0

    # helper function that spawns a ball, returns a position vector and a velocity vector
    # if right is True, spawn to the right, else spawn to the left
    def ball_init(right):
        global ball_pos, ball_vel # these are vectors stored as lists
        ball_pos = [WIDTH//2,HEIGHT//2]
        horz = random.randrange(2,4)
        vert = random.randrange(1,3)
        
        if right == False:
            horz = - horz
    
        ball_vel = [horz, -vert]

    # define event handlers
    def init():
        global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel,l_score,r_score  # these are floats
        global score1, score2  # these are ints
        paddle1_pos = [HALF_PAD_WIDTH - 1,HEIGHT//2]
        paddle2_pos = [WIDTH +1 - HALF_PAD_WIDTH,HEIGHT//2]
        l_score = 0
        r_score = 0
        if random.randrange(0,2) == 0:
            ball_init(True)
        else:
            ball_init(False)

    def __init__(self):
        pygame.init()
                        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high)
        
        self._seed()
        self.viewer = None
        self.state = None
        
        self.steps_beyond_done = None
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" %(action, type(action))
        state = self.state
        (paddle1_pos, paddle2_pos, ball_pos, paddle1_vel, paddle2_vel, ball_vel, l_score, r_score) = state
        
        #update padd velocity
        if action == -1:
            paddle2_vel = -8
        elif action == 1:
            paddle2_vel = 8
        elif ball_pos[1] < paddle1_pos[1]:
            paddle1_vel = -8
        elif ball_pos[1] > paddle1_pos[1]:
            paddle1_vel = 8
        if action == 0
            paddle2_vel = 0
        elif ball_pos[1] == paddle1_pos[1]
            paddle1_vel = 0
        
        # update paddle's vertical position, keep paddle on the screen
        if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
            paddle1_pos[1] += paddle1_vel
        elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
            paddle1_pos[1] += paddle1_vel
        elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
            paddle1_pos[1] += paddle1_vel
    
        if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
            paddle2_pos[1] += paddle2_vel
        elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
            paddle2_pos[1] += paddle2_vel
        elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
            paddle2_pos[1] += paddle2_vel
        
        # update ball
        ball_pos[0] += int(ball_vel[0])
        ball_pos[1] += int(ball_vel[1])
        
        # ball collision check on top and bottom walls
        if int(ball_pos[1]) <= BALL_RADIUS:
            ball_vel[1] = - ball_vel[1]
        if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
            ball_vel[1] = -ball_vel[1]
        
        # ball collison check on gutters or paddles
        if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(paddle1_pos[1] - HALF_PAD_HEIGHT,paddle1_pos[1] + HALF_PAD_HEIGHT,1):
            ball_vel[0] = -ball_vel[0]
            ball_vel[0] *= 1.1
            ball_vel[1] *= 1.1
        elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
            r_score += 1
            ball_init(True)

        if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(paddle2_pos[1] - HALF_PAD_HEIGHT,paddle2_pos[1] + HALF_PAD_HEIGHT,1):
            ball_vel[0] = -ball_vel[0]
            ball_vel[0] *= 1.1
            ball_vel[1] *= 1.1
        elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
            l_score += 1
            ball_init(False)

        self.state = (paddle1_pos, paddle2_pos, ball_pos, paddle1_vel, paddle2_vel, ball_vel, l_score, r_score)
        done =  l_score > self.l_score_threshold \
            or r_score > self.r_score_threshold
        
        done = bool(done)
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0

        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                pygame.quit()
                sys.exit()
                self.viewer = None
            return
    
        if self.viewer is None:
            fps = pygame.time.Clock()
            
            #canvas declaration
            window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
            pygame.display.set_caption('Hello World')
            
            canvas.fill(BLACK)
            pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0],[WIDTH // 2, HEIGHT], 1)
            pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
            pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
            pygame.draw.circle(canvas, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)
            
            # draw paddles and ball
            pygame.draw.circle(canvas, RED, ball_pos, 20, 0)
            pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
            pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)
            
            #update scores
            myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
            label1 = myfont1.render("Score "+str(l_score), 1, (255,255,0))
            canvas.blit(label1, (50,20))
            
            myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
            label2 = myfont2.render("Score "+str(r_score), 1, (255,255,0))
            canvas.blit(label2, (470, 20))
            
            init()
            
            #game loop
            #while True:
            #    for event in pygame.event.get():
                    
            #        if event.type == KEYDOWN:
            #            keydown(event)
            #        elif event.type == KEYUP:
            #            keyup(event)
            #        elif event.type == QUIT:
            #            pygame.quit()
            #            sys.exit()
                                                
            pygame.display.update()
            fps.tick(60)
                                                        
        if self.state is None: return None
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
