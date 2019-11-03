import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
#import matplotlib.pyplot as plt
import os
import time
import cv2


#plt.ion()
#plt.show(block=False)



class LmazeEnv_v3(gym.Env):
  metadata = {'render.modes': ['human']}

  """    
  """
  def __init__(self):

    self.state_type = "fullView"

    self.grid = np.array(       [['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'S', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'B', 'B', 'B', 'X', 'B', 'B', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W']])
    '''
    self.grid = np.array(       [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'S', 'B', 'B', 'B', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'W', 'W', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'B', 'B', 'B', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])
    '''
    '''
    self.grid = np.array(       [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'S', 'B', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'B', 'W', 'X', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])
    '''


    self.realgrid = self.grid.shape[0]
    self.expansionRatio = 4
    self.fovea = 18

    self.gridsize = self.fovea * self.expansionRatio

    if self.state_type == "oneState":
        self.channel = 4
        self.stateChannel = self.channel
    elif self.state_type == "twoState":
        self.channel = 2
        self.actionChannel = 1
        self.stateChannel = self.channel+self.actionChannel+self.channel
    elif self.state_type == "fullView":
        self.channel = 3
        self.stateChannel = self.channel

    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(self.stateChannel, self.gridsize, self.gridsize))
    self.action_space = spaces.Discrete(4)

    self.retStatelast = np.zeros((self.channel, self.fovea, self.fovea), dtype=np.float32)
    self.step_limit = 100
    self.negativeNominal = -1.0
    self.positiveNominal = -0.01
    self.positiveFull = 100.0
    self.RANDOM_BALL = True
    self.RANDOM_GOAL = True
    self.AUTO_VISUALIZE = False
    self.SAVEFRAME = False
    self.VISUALIZE = False
    self.localDone = False
    self.stepCount = 0
    self.fovealStepCount = 0


    self.boundary = np.zeros((self.fovea*self.expansionRatio, self.expansionRatio))
    self.boundary[:][:] = 250
    self.lowerboundary = np.zeros((self.expansionRatio, self.realgrid*self.expansionRatio))
    self.lowerboundary[:][:] = 250

    #self.goal_x, self.goal_y = 0, 0
    s1 = np.where(self.grid == 'X')
    self.goal_x, self.goal_y = s1[0][0], s1[1][0]

    self.ball_x1, self.ball_y1 = 0, 0
    self.ball_x0, self.ball_y0 = 0, 0

    self.retStateExpanded = np.zeros((self.stateChannel, self.fovea * self.expansionRatio,
                                      self.fovea * self.expansionRatio), dtype=np.float32)
    self.dir = "."
    self.reset()





  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self, mode="train"):                        # just reset

    if self.SAVEFRAME:
        print("creating directory")
        self.dir = "dir"+str(int(time.time()*1000))
        os.makedirs(self.dir)


    self.state = np.zeros((self.channel, self.realgrid, self.realgrid), dtype=np.float32)
    self.stepCount = 0

    if mode == "test":
        self.goal_x, self.goal_y = 8, 8
    elif self.RANDOM_GOAL:
        x, y = 0, 0
        while self.grid[x][y] == 'W':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.goal_x, self.goal_y = x, y

    if mode == "test":
        self.ball_x0, self.ball_y0 = 7, 8
    elif self.RANDOM_BALL:
        x, y = 0, 0
        while self.grid[x][y] == 'W' or (x == self.goal_x and  y == self.goal_y):
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.ball_x0, self.ball_y0 = x, y
    else:
        start = np.where(self.grid == 'S')
        self.ball_x0, self.ball_y0 = start[0][0], start[1][0]

    self.state[0] = [[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.state[1,self.ball_x0,self.ball_y0] = 1.0

    #self.image_wall_previous = np.zeros((self.fovea, self.fovea))
    #self.image_goal_previous = np.zeros((self.fovea, self.fovea))
    self.image_global_goal = np.zeros((self.fovea, self.fovea))
    self.image_wall_current = np.zeros((self.fovea, self.fovea))
    self.image_goal_current = np.zeros((self.fovea, self.fovea))

    #self.image_wall_previous = np.asarray(self.state[0,:,:])
    self.image_wall_current = np.asarray(self.state[0,:,:])
    #self.image_goal_previous =  np.asarray(self.state[1,:,:])
    #self.image_goal_previous =  np.copy(self.state[1,:,:])
    self.image_goal_current = np.asarray(self.state[1,:,:])

    #self.image_global_goal = [[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.image_global_goal[self.goal_x][self.goal_y] = 1.0

    if self.state_type == "oneState":
        retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0)], axis=0)
    elif self.state_type == "twoState":
        retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0),
                                   np.expand_dims(self.image_global_goal, axis=0),
                                   np.expand_dims(self.image_wall_previous,  axis=0),
                                   np.expand_dims(self.image_goal_previous, axis=0)], axis=0)
    elif self.state_type == "fullView":
        retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0),
                                   np.expand_dims(self.image_global_goal, axis=0)], axis=0)

    for channel in range(0, retState.shape[0]):
        for i in range(0, retState.shape[1]):
            for ii in range(0, self.expansionRatio):
                for j in range(0, retState.shape[2]):
                    for jj in range(0, self.expansionRatio):
                        self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                            retState[channel][i][j]

    return self.retStateExpanded



  """    
  """

  def rendering(self, msg):
      self.VISUALIZE = msg
      #print(self.VISUALIZE)

  def writing(self, msg):
      self.SAVEFRAME = msg

  def step(self, goal):

      #print(goal)

      self.originalReward = -0.0
      self.stepCount += 1

      #image_wall_previous_ = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))
      #image_goal_previous_ = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))
      image_global_goal_ = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))
      image_wall_current_ = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))
      image_goal_current_ = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))

      #print("ball pos", self.ball_x0, self.ball_y0)
      o_x = 0
      o_y = 0
      if goal == "left" or goal == "0":
        o_x = -1
        o_y = 0
      elif goal == "right" or goal == "1":
        o_x = 1
        o_y = 0
      elif goal == "up" or goal == "2":
        o_x = 0
        o_y = -1
      elif goal == "down" or goal == "3":
        o_x = 0
        o_y = 1

      #print(self.ball_x0, self.ball_y0, o_x, o_y)

      if self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'W':
        self.originalReward = self.negativeNominal

      #elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'B':
      else:
        self.state[1, self.ball_x0, self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[1, self.ball_x0, self.ball_y0] = 1.0
        self.originalReward = self.positiveNominal

        if self.ball_x0 + o_x == self.goal_x and self.ball_y0 + o_y == self.goal_y:
            self.originalReward = self.positiveFull

      #elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'X':
      #elif self.ball_x0 + o_x == self.goal_x and self.ball_y0 + o_y == self.goal_y:
      #  self.state[1, self.ball_x0, self.ball_y0] = 0.0
      #  self.ball_x0 = self.ball_x0 + o_x
      #  self.ball_y0 = self.ball_y0 + o_y
      #  self.state[1, self.ball_x0, self.ball_y0] = 1.0
      #  self.originalReward = self.positiveFull


      #print(self.ball_x0, self.ball_y0)
      self.image_wall_current = np.asarray(self.state[0,:,:])
      self.image_goal_current = np.asarray(self.state[1,:,:])
      #self.image_global_goal = np.asarray(self.state[1,:,:])

      if self.state_type == "oneState":
          retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0)], axis=0)
      elif self.state_type == "twoState":
          retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0),
                                   np.expand_dims(self.image_global_goal, axis=0),
                                   np.expand_dims(self.image_wall_previous,  axis=0),
                                   np.expand_dims(self.image_goal_previous, axis=0)], axis=0)
      elif self.state_type == "fullView":
          retState = np.concatenate([np.expand_dims(self.image_wall_current, axis=0),
                                   np.expand_dims(self.image_goal_current, axis=0),
                                   np.expand_dims(self.image_global_goal, axis=0)], axis=0)

      for channel in range(0, retState.shape[0]):
          for i in range(0, retState.shape[1]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, retState.shape[2]):
                      for jj in range(0, self.expansionRatio):
                          self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                              retState[channel][i][j]

      #print(self.VISUALIZE)
      if self.VISUALIZE:
          #
          # Build the global view
          #
          #print("saving image")
          image3 = np.zeros((self.realgrid*self.expansionRatio, self.realgrid*self.expansionRatio))
          image3_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in self.state[0]]), dtype=np.uint8)
          image3_[self.goal_x, self.goal_y] = 100
          image3_[self.ball_x0, self.ball_y0] = 150
          #
          #
          for i in range(0, image3_.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, image3_.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          image3[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image3_[i][j]

          #cv2.rectangle(image3, (self.ball_y0*self.expansionRatio-14, self.ball_x0*self.expansionRatio-14),
          #              (self.ball_y0*self.expansionRatio+21, self.ball_x0*self.expansionRatio+21), (100, 100, 00), 1)

          #cv2.rectangle(image3, (self.ball_y1*self.expansionRatio-14, self.ball_x1*self.expansionRatio-14),
          #              (self.ball_y1*self.expansionRatio+21, self.ball_x1*self.expansionRatio+21), (150, 150, 00), 1)

          for i in range(0, self.image_wall_current.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, self.image_wall_current.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          #image_wall_previous_[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = self.image_wall_previous[i][j]
                          #image_goal_previous_[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = self.image_goal_previous[i][j]
                          image_wall_current_[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = self.image_wall_current[i][j]
                          image_goal_current_[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = self.image_goal_current[i][j]
                          image_global_goal_[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = self.image_global_goal[i][j]

          #image_wall_previous_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in image_wall_previous_]), dtype=np.uint8)
          #image_goal_previous_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in image_goal_previous_]), dtype=np.uint8)
          image_wall_current_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in image_wall_current_]), dtype=np.uint8)
          image_goal_current_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in image_goal_current_]), dtype=np.uint8)
          image_global_goal_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in image_global_goal_]), dtype=np.uint8)

          ''' Arrange the visualization here '''
          image = np.concatenate([self.lowerboundary,
                                  self.lowerboundary,
                                  self.lowerboundary,
                                  self.lowerboundary,
                                  self.lowerboundary,
                                  image_wall_current_,
                                  #self.lowerboundary,
                                  #image_wall_previous_,
                                  #self.lowerboundary,
                                  #image_goal_previous_,
                                  self.lowerboundary,
                                  #image_wall_current_,
                                  #self.lowerboundary,
                                  image_goal_current_,
                                  self.lowerboundary,
                                  image_global_goal_,
                                  self.lowerboundary], axis = 0)
          # (200,255,155)
          font = cv2.FONT_HERSHEY_SIMPLEX
          line1 = "Step Number : "
          line2 = str(self.stepCount)

          cv2.putText(image,line1,(5,9), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          cv2.putText(image,line2,(5,18), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,line2x,(5,20), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,line2y,(5,30), font, 0.3, (0,0,0), 1, cv2.LINE_AA)

          #cv2.putText(image,"View",(50,160), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,"Previous",(50,170), font, 0.3, (0,0,0), 1, cv2.LINE_AA)

          #cv2.putText(image,"Goal",(50,200), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,"Previous",(50,210), font, 0.3, (0,0,0), 1, cv2.LINE_AA)

          #cv2.putText(image,"View",(50,240), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,"Current",(50,250), font, 0.3, (0,0,0), 1, cv2.LINE_AA)

          #cv2.putText(image,"Goal",(50,280), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,"Current",(50,290), font, 0.3, (0,0,0), 1, cv2.LINE_AA)

          #cv2.putText(image,"Main",(50,320), font, 0.3, (0,0,0), 1, cv2.LINE_AA)
          #cv2.putText(image,"Goal",(50,330), font, 0.3, (0,0,0), 1, cv2.LINE_AA)


          cv2.imshow('image',image)
          cv2.imwrite(self.dir+"/image"+ str(self.stepCount)+".png", image)
          cv2.waitKey(1)
          #time.sleep(1)

      #self.image_goal_previous = np.copy(self.image_goal_current)
      #self.image_wall_previous = self.image_wall_current

      self.ball_x1 = self.ball_x0
      self.ball_y1 = self.ball_y0

      if self.originalReward == self.positiveFull or self.stepCount > self.step_limit:
        #if self.originalReward == self.positiveFull:
        return self.retStateExpanded, self.originalReward, True, goal
      else :
        return self.retStateExpanded, self.originalReward, False, goal
