import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import os
import time
import cv2





# foveal view new logic for Rainbow


class LmazeEnv_v4(gym.Env):
  metadata = {'render.modes': ['human']}

  """    
  """
  def __init__(self):

    self.state_type = "twoState-threeLayers"
    
    self.expansionRatio = 7
    self.fovea = 5
    self.gridsize = self.fovea * self.expansionRatio

    if self.state_type == "oneState-twoLayers":
        self.channel = 4
        self.stateChannel = self.channel
    elif self.state_type == "twoState-twoLayers":
        self.channel = 2
        self.actionChannel = 1
        self.stateChannel = self.channel+self.actionChannel+self.channel
    elif self.state_type == "twoState-threeLayers":
        self.channel = 3
        self.actionChannel = 1
        self.stateChannel = self.channel+self.actionChannel+self.channel

    # Check this
    #self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(self.stateChannel, self.gridsize, self.gridsize))
    #self.action_space = spaces.Discrete(self.fovea*self.fovea)

    

    self.step_limit = 50
    self.stepCount = 0
    self.fovealStepCount = 0

    self.negativeNominal = -1.0
    self.positiveNominal = -0.01
    self.positiveFull = 100.0

    self.RANDOM_BALL = True
    self.RANDOM_GOAL = True
    self.AUTO_VISUALIZE = False
    self.SAVEFRAME = False
    self.VISUALIZE = False
    self.localDone = False



    self.retStatelast = np.zeros((self.channel, self.fovea, self.fovea), dtype=np.float32)
    self.retStateExpanded = np.zeros((self.stateChannel, self.fovea * self.expansionRatio, self.fovea * self.expansionRatio), dtype=np.float32)
    self.boundary = np.zeros((self.fovea*self.expansionRatio, self.expansionRatio))
    self.boundary[:][:] = 250
    self.lowerboundary = np.zeros((self.expansionRatio, self.fovea*self.expansionRatio))
    self.lowerboundary[:][:] = 0.5



    self.ball_x0, self.ball_y0, self.ball_x1, self.ball_y1 = 0, 0, 0, 0
    self.f_goal_x, self.f_goal_y, self.f_goal_x1, self.f_goal_y1 = 0, 0, 0, 0

    self.dir = "."
    self.dirhead = "eval_"


    #self.setGrid()
    #self.realgrid = self.grid.shape[0]

    self.reset()







  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.setGrid()
    self.realgrid = self.grid.shape[0]
    self.originalReward = -0.0
    self.fovealReward = -0.0
    self.stepCount = 0

    self.setGoal()
    self.setBall()



    action_value = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)



    self.state = np.zeros((self.channel, self.realgrid, self.realgrid), dtype=np.float32)
    self.state[0] = [[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.state[1][self.goal_x][self.goal_y] = 1.0

    visitMap = np.zeros((1,self.realgrid, self.realgrid))
    visitMap[0,self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2 : self.ball_y0 + 3] = 1.0

    self.state[2] = (self.state[2] + visitMap) /2




    observation = np.asarray(self.state[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2 : self.ball_y0 + 3])
    
    # build traversal layer



    #if self.SAVEFRAME:
    if self.SAVEFRAME and np.random.random() < 0.25:
        self.dir = str(self.dirhead)+"/dir"+str(int(time.time()*1000))
        os.makedirs(self.dir)
        self.SAVE = True
    else:
        self.SAVE = False




    if self.state_type == "oneState-twoLayers":
        retState = observation
    elif self.state_type == "twoState-twoLayers" or self.state_type == "twoState-threeLayers":
        self.retStatelast = observation
        retState = np.concatenate([observation, action_value, self.retStatelast], axis=0)






    for channel in range(0, retState.shape[0]):
        for i in range(0, retState.shape[1]):
            for ii in range(0, self.expansionRatio):
                for j in range(0, retState.shape[2]):
                    for jj in range(0, self.expansionRatio):
                        self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                            retState[channel][i][j]

    self.ball_x1, self.ball_y1 = self.ball_x0, self.ball_y0
    self.f_goal_x1, self.f_goal_y1 = self.f_goal_x, self.f_goal_y

    return self.retStateExpanded

  """    
  """
  def step(self, goal):
      #
      # goal should be an integer value (Explicit type-cast)
      #
      goal = int(goal)
      #
      # build action as a layer
      #
      action_value = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)
      action_value[0, int(goal/self.fovea), int(goal%self.fovea)] = 1.0
      #
      # for first step, set the history to current value
      #
      if self.stepCount == 0:
          self.f_goal_x1, self.f_goal_y1 = self.ball_x0, self.ball_y0
          self.ball_x1, self.ball_y1 = self.ball_x0, self.ball_y0
      #
      # set some more variables
      #
      self.originalReward = -0.0
      self.stepCount += 1
      #
      # build global goal value from local goal value
      #
      self.f_goal_x = self.ball_x0 + int(goal / self.fovea) - 2
      self.f_goal_y = self.ball_y0 + int(goal % self.fovea) - 2
      #
      #             MOVE AGENT
      # if the global goal is not out of boundary move the agent center to the given goalS
      #
      if self.f_goal_x < self.realgrid - 2 and self.f_goal_x > 1 and self.f_goal_y < self.realgrid - 2 and self.f_goal_y > 1:
          self.ball_x0 = self.f_goal_x
          self.ball_y0 = self.f_goal_y
      else:
          if self.f_goal_x >= self.realgrid - 2:
              self.ball_x0 = self.realgrid - 3
          if self.f_goal_x <= 1:
              self.ball_x0 = 2
          if self.f_goal_y >= self.realgrid - 2:
              self.ball_y0 = self.realgrid - 3
          if self.f_goal_y <= 1:
              self.ball_y0 = 2
              self.ball_y0 = 2

      visitMap = np.zeros((1, self.realgrid, self.realgrid))
      visitMap[0, self.ball_x0 - 2: self.ball_x0 + 3, self.ball_y0 - 2: self.ball_y0 + 3] = 1.0

      self.state[2] = (self.state[2] + visitMap) / 2

      #
      #              REWARD AGENT
      # if the agent is moved to a goal, wall or blank space; give relevant reward
      #
      #if self.grid[self.f_goal_x][self.f_goal_y] == 'X':
      if self.f_goal_x == self.goal_x and self.f_goal_y == self.goal_y:
              self.originalReward = self.positiveFull
      elif self.grid[self.f_goal_x][self.f_goal_y] == 'W':
          self.originalReward = self.negativeNominal
      elif self.grid[self.f_goal_x][self.f_goal_y] == 'B' or self.grid[self.f_goal_x][self.f_goal_y] == 'S':
          self.originalReward = self.positiveNominal
      #
      #              BUILD OBSERVATION
      # post update of agent position, build observation
      #
      observation = np.asarray(self.state[:, self.ball_x0 - int(self.fovea/2) : self.ball_x0 + int(self.fovea/2) + 1,
                               self.ball_y0 - int(self.fovea/2):self.ball_y0 + int(self.fovea/2) + 1])
      #
      # build state to return to the learner
      #
      if self.state_type == "oneState-twoLayers":
          retState = observation
      elif self.state_type == "twoState-twoLayers" or self.state_type == "twoState-threeLayers":
          retState = np.concatenate([observation, action_value, self.retStatelast], axis=0)
      #
      # expand state before returning to the learner
      #
      for channel in range(0, retState.shape[0]):
          for i in range(0, retState.shape[1]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, retState.shape[2]):
                      for jj in range(0, self.expansionRatio):
                          self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          retState[channel][i][j]
      #
      #             VISUALIZE / SAVE FRAMES
      #
      if self.VISUALIZE or self.SAVEFRAME:
        self.saveVisualize()
      #
      #             BUILD FRAME HISTORY for state generation
      #
      if self.state_type == "twoState-twoLayers" or self.state_type == "twoState-threeLayers":
          self.retStatelast = observation
      #
      #             BUILD FRAME HISTORY as variables for visualization / saving
      #
      self.ball_x1, self.ball_y1 = self.ball_x0, self.ball_y0
      self.f_goal_x1, self.f_goal_y1 = self.f_goal_x, self.f_goal_y
      #
      #
      #
      if self.originalReward == self.positiveFull or self.stepCount > self.step_limit:
          print(self.state[2])
          return self.retStateExpanded, self.originalReward, True, goal
      else :
          return self.retStateExpanded, self.originalReward, False, goal


  """
  """
  def rendering(self, msg):
    self.VISUALIZE = msg

  """    
  """
  def writing(self, msg):
    self.SAVEFRAME = msg

  """    
  """
  def setevaldir(self, msg):
    self.dirhead = msg

  """    
  """
  def saveVisualize(self):
      image1 = np.zeros((self.realgrid * self.expansionRatio, self.realgrid * self.expansionRatio))

      image = np.asarray([[254 if item == "B" or item == "S" or item == "X" else 0 for item in line] for line in self.grid])

      for i in range(0, image.shape[0]):
          for ii in range(0, self.expansionRatio):
              for j in range(0, image.shape[1]):
                  for jj in range(0, self.expansionRatio):
                      image1[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image[i][j]

      cv2.rectangle(image1, (self.ball_y0 * self.expansionRatio - 14, self.ball_x0 * self.expansionRatio - 14),
                (self.ball_y0 * self.expansionRatio + 21, self.ball_x0 * self.expansionRatio + 21), (150, 150, 00), 1)
      cv2.circle(image1, (self.f_goal_y * self.expansionRatio + 3, self.f_goal_x * self.expansionRatio + 3), 3,
             (150, 150, 00), -1)

      cv2.rectangle(image1, (self.ball_y1 * self.expansionRatio - 14, self.ball_x1 * self.expansionRatio - 14),
                (self.ball_y1 * self.expansionRatio + 21, self.ball_x1 * self.expansionRatio + 21), (50, 50, 00), 1)
      cv2.circle(image1, (self.f_goal_y1 * self.expansionRatio + 3, self.f_goal_x1 * self.expansionRatio + 3), 3,
             (50, 50, 00), -1)

      cv2.circle(image1, (self.goal_y * self.expansionRatio + 3, self.goal_x * self.expansionRatio + 3), 3, (0, 0, 0), -1)

      if self.VISUALIZE:
          cv2.imshow('i', image)
          cv2.waitKey(1)

      if self.SAVE:
          cv2.imwrite(self.dir + "/image" + str(self.stepCount) + ".png", image1)

  """    
  """
  def setGoal(self):
    if self.RANDOM_GOAL:
        x, y = 0, 0
        while self.grid[x][y] == 'W' or self.grid[x][y] == 'S':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.goal_x, self.goal_y = x, y
    else:
        start = np.where(self.grid == 'X')
        self.goal_x, self.goal_y = start[0][0], start[1][0]

  """    
  """
  def setBall(self):
    if self.RANDOM_BALL:
        x, y = 0, 0
        while self.grid[x][y] == 'W' or self.grid[x][y] == 'X' or (self.goal_x == x and self.goal_y == y):
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.ball_x0, self.ball_y0 = x, y
    else:
        start = np.where(self.grid == 'S')
        self.ball_x0, self.ball_y0 = start[0][0], start[1][0]

  """    
  """
  def setGrid(self):


    random = np.random.randint(1,6)

    if random == 1:
      self.grid = np.array([['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'S'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'X'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 2:
      self.grid = np.array([['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'S'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'X'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])
    if random == 3:
      self.grid = np.array([['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'S'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'X'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 4:
      self.grid = np.array([['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'S'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'X'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 5:
      self.grid = np.array([['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'S'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'X'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

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
