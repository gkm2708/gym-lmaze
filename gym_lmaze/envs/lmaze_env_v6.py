import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import os
import time
import cv2





# foveal view new logic for Rainbow


class LmazeEnv_v6(gym.Env):

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

    self.step_limit = 10
    self.foveal_step_limit = 50

    self.negativeNominal = -1.0
    self.positiveNominal = -0.01
    self.positiveFull = 100.0

    self.RANDOM_BALL = True
    self.RANDOM_GOAL = True
    self.AUTO_VISUALIZE = False
    self.SAVEFRAME = False
    self.VISUALIZE = False


    self.retStatelast = np.zeros((self.channel, self.fovea, self.fovea), dtype=np.float32)

    self.boundary = np.zeros((self.fovea*self.expansionRatio, self.expansionRatio))
    self.boundary[:][:] = 250
    self.lowerboundary = np.zeros((self.expansionRatio, self.fovea*self.expansionRatio))
    self.lowerboundary[:][:] = 0.5


    # to keep global goal index
    self.goal_x = 0
    self.goal_y = 0

    # to keep foveal goal index  (+1 history)
    self.f_goal_x0 = 0
    self.f_goal_y0 = 0
    #self.f_goal_x1 = 0
    #self.f_goal_y1 = 0

    # to keep agent index (+1 history)
    self.ball_x0 = 0
    self.ball_y0 = 0
    self.ball_x1 = 0
    self.ball_y1 = 0

    # to degenerate foveal position and agent position  (+1 history)
    self.fovea_x0 = 0
    self.fovea_y0 = 0
    self.fovea_x1 = 0
    self.fovea_y1 = 0


    self.dir = "."
    self.dirhead = "eval_"

    self.reset()





  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.setGrid()
    self.realgrid = self.grid.shape[0]

    self.originalReward = -0.0
    self.globalReward = -0.0

    self.fovealStepCount = 0
    self.stepCount = 0
    self.globalDone = False
    self.localDone = False

    self.setGoal() # sets self.goal_x self.goal_y
    self.setBall() # sets self.ball_x0 self.ball_y0

    #if self.SAVEFRAME:
    if self.SAVEFRAME:

        self.dir = str(self.dirhead)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.SAVE = True
    else:
        self.SAVE = False

    self.fovealGoal = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)
    self.fovealGoal[0,2,2] = 1.0

    self.state = np.zeros((self.channel, self.realgrid, self.realgrid), dtype=np.float32)

    self.state[0] = [[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.state[1][self.goal_x][self.goal_y] = 1.0


    # set first foveal values
    self.fovea_x0 = self.ball_x0  # initial fovea cenetred over agent start position
    self.fovea_y0 = self.ball_y0

    self.fovea_x1 = self.ball_x0  # initial fovea cenetred over agent start position
    self.fovea_y1 = self.ball_y0

    self.f_goal_x0 = self.ball_x0
    self.f_goal_y0 = self.ball_y0

    self.ball_x1 = self.ball_x0
    self.ball_y1 = self.ball_y0

    return self.buildFovealObservation()





  """    
  """
  def plannerStep(self, goal):

      # set some variables
      self.stepCount = 0
      self.globalReward = -0.
      self.localDone = False


      #             SET THE GOAL
      # goal should be an integer value (Explicit type-cast)
      self.foveal_goal = int(goal)
      #
      # build goal as a layer
      self.fovealGoal = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)
      self.fovealGoal[0, int(self.foveal_goal/self.fovea), int(self.foveal_goal%self.fovea)] = 1.0
      #
      # make the foveal goal and its history
      self.f_goal_x0 = self.ball_x0 + int(self.foveal_goal / self.fovea) - 2
      self.f_goal_y0 = self.ball_y0 + int(self.foveal_goal % self.fovea) - 2
      if self.fovealStepCount > 0:
          self.fovea_x1 = self.fovea_x0
          self.fovea_y1 = self.fovea_y0

      self.fovealStepCount += 1
      return self.buildLocalObservation()




  def step(self, goal):
      #
      # goal should be an integer value (Explicit type-cast)
      self.local_action = int(goal)

      #
      # Make ball History
      self.ball_x1 = self.ball_x0
      self.ball_y1 = self.ball_y0

      #
      # set some more variables
      self.originalReward = -0.0
      self.stepCount += 1

      #
      #             MOVE AGENT
      # if the global goal is not out of boundary move the agent center to the given goalS
      dx, dy = 0, 0
      if self.local_action == 0:
          dx = 1
          dy = 0
      elif self.local_action == 1:
          dx = -1
          dy = 0
      elif self.local_action == 2:
          dx = 0
          dy = 1
      elif self.local_action == 3:
          dx = 0
          dy = -1

      # projected move
      ballNew_x = self.ball_x0 + dx
      ballNew_y = self.ball_y0 + dy

      #
      # Reward and move
      #
      #     local level
      #
      # if new position cooincides with
      #     wall or it goes out of fovea            -ve
      #if (ballNew_x < self.fovea_x1 - 3 or ballNew_x > self.fovea_x1 + 2 or ballNew_y < self.fovea_y1 - 3 or ballNew_y > self.fovea_y1 + 2) \
      #        or self.grid[ballNew_x, ballNew_y] == 'W':
      if self.grid[ballNew_x, ballNew_y] == 'W':
          self.originalReward = self.negativeNominal
      #     local goal                              ++ve
      elif ballNew_x == self.f_goal_x0 and ballNew_y == self.f_goal_y0:
          self.originalReward = self.positiveFull
          self.ball_x0 = ballNew_x
          self.ball_y0 = ballNew_y
          self.localDone = True
      #     blank space                             +ve
      elif self.grid[ballNew_x, ballNew_y] == 'B' \
              or self.grid[ballNew_x, ballNew_y] == 'S' \
              or self.grid[ballNew_x, ballNew_y] == 'X':
          if (ballNew_x < self.fovea_x1 - 3 or ballNew_x > self.fovea_x1 + 2 or ballNew_y < self.fovea_y1 - 3 or ballNew_y > self.fovea_y1 + 2):
              self.localDone = True
          self.originalReward = self.positiveNominal
          self.ball_x0 = ballNew_x
          self.ball_y0 = ballNew_y

      
      #
      #     planning level
      #
      # if global goal      ++ve
      if ballNew_x == self.goal_x and ballNew_y == self.goal_y:
          self.globalReward = self.positiveFull
          self.globalDone = True
      # if on foveal goal   +ve
      elif ballNew_x == self.f_goal_x0 and ballNew_y == self.f_goal_y0:
          self.globalReward = self.positiveNominal
      # else                -ve
      else:
          self.globalReward = self.negativeNominal

      self.fovea_x0 = self.ball_x0
      self.fovea_y0 = self.ball_y0

      if self.stepCount >= self.step_limit:
          self.localDone = True
      if self.fovealStepCount >= self.foveal_step_limit:
          self.globalDone = True
          self.localDone = True

      #
      #             VISUALIZE / SAVE FRAMES
      #
      if self.VISUALIZE or self.SAVEFRAME:
        self.saveVisualize()
      #
      #


      fovobs = self.buildFovealObservation()
      locobs = self.buildLocalObservation()

      return fovobs, \
             locobs, \
             self.globalReward, \
             self.originalReward, \
             self.globalDone, \
             self.localDone, \
             self.fovealGoal, \
             self.local_action











  """
  """
  def buildFovealObservation(self):

      retStateExpanded = np.zeros(
          (self.stateChannel, self.fovea * self.expansionRatio, self.fovea * self.expansionRatio), dtype=np.float32)

      #
      #              BUILD OBSERVATION
      # post update of agent position, build observation
      #
      if self.localDone:
          visitMap = np.zeros((1, self.realgrid, self.realgrid))
          visitMap[0, self.fovea_x0 - 2: self.fovea_x0 + 3, self.fovea_y0 - 2: self.fovea_y0 + 3] = 1.0
          self.state[2] = (self.state[2] + visitMap) / 2
      #
      observation = np.asarray(self.state[:, self.fovea_x0 - int(self.fovea/2) : self.fovea_x0 + int(self.fovea/2) + 1,
                               self.fovea_y0 - int(self.fovea/2):self.fovea_y0 + int(self.fovea/2) + 1])
      if self.fovealStepCount == 0:
          self.retStatelast = observation
      #
      # build state to return to the learner
      #
      if self.state_type == "oneState-twoLayers":
          retState = observation
      elif self.state_type == "twoState-twoLayers" or self.state_type == "twoState-threeLayers":
          retState = np.concatenate([observation, self.fovealGoal, self.retStatelast], axis=0)
      #
      # expand state before returning to the learner
      #
      for channel in range(0, retState.shape[0]):
          for i in range(0, retState.shape[1]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, retState.shape[2]):
                      for jj in range(0, self.expansionRatio):
                          retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          retState[channel][i][j]
      #
      #             BUILD FRAME HISTORY for next state generation
      #
      if (self.state_type == "twoState-twoLayers" or self.state_type == "twoState-threeLayers") \
              and self.localDone:
          self.retStatelast = observation

      return retStateExpanded





  """
  """
  def buildLocalObservation(self):

      initialObservation = np.zeros((4,self.fovea,self.fovea),dtype=np.float32)
      retState_local = np.zeros((4,self.fovea*self.expansionRatio,self.fovea*self.expansionRatio), dtype = np.float32)

      initialObservation[0] = np.asarray(self.state[0, self.fovea_x0 - 2 : self.fovea_x0 + 3, self.fovea_y0 - 2 : self.fovea_y0 + 3])

      #print(self.ball_x0, self.fovea_x0, self.ball_x0 - self.fovea_x0 + 2, self.ball_y0, self.fovea_y0, self.ball_y0 - self.fovea_y0+3)

      initialObservation[1,self.ball_x0 - self.fovea_x1 + 2, self.ball_y0 - self.fovea_y1 + 2] = 1.0
      initialObservation[2,self.ball_x1 - self.fovea_x1 + 2, self.ball_y1 - self.fovea_y1 + 2] = 1.0

      initialObservation[3] = self.fovealGoal

      #
      # expand state before returning to the learner
      #
      for channel in range(0, initialObservation.shape[0]):
          for i in range(0, initialObservation.shape[1]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, initialObservation.shape[2]):
                      for jj in range(0, self.expansionRatio):
                          retState_local[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          initialObservation[channel][i][j]
      return retState_local









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

      cv2.rectangle(image1, (self.fovea_y0 * self.expansionRatio - 14, self.fovea_x0 * self.expansionRatio - 14),
                (self.fovea_y0 * self.expansionRatio + 21, self.fovea_x0 * self.expansionRatio + 21),
                (50, 50, 00), 1)

      cv2.circle(image1, (self.ball_y0 * self.expansionRatio + 3, self.ball_x0 * self.expansionRatio + 3), 3,
             (50, 50, 00), -1)

      cv2.rectangle(image1, (self.fovea_y1 * self.expansionRatio - 14, self.fovea_x1 * self.expansionRatio - 14),
                (self.fovea_y1 * self.expansionRatio + 21, self.fovea_x1 * self.expansionRatio + 21),
                    (150, 150, 00), 1)

      cv2.circle(image1, (self.fovea_y1 * self.expansionRatio + 3, self.fovea_x1 * self.expansionRatio + 3), 3,
             (150, 150, 00), -1)

      cv2.circle(image1, (self.goal_y * self.expansionRatio + 3, self.goal_x * self.expansionRatio + 3), 3,
                 (0, 0, 0), -1)




      #print(self.fovea_x0, self.fovea_y0, self.fovea_x1, self.fovea_y1)



      image2 = np.zeros((self.realgrid * self.expansionRatio, self.realgrid * self.expansionRatio))

      for i in range(0, image.shape[0]):
          for ii in range(0, self.expansionRatio):
              for j in range(0, image.shape[1]):
                  for jj in range(0, self.expansionRatio):
                      image2[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image[i][j]

      cv2.circle(image2, (self.ball_y0 * self.expansionRatio + 3, self.ball_x0 * self.expansionRatio + 3), 3,
             (50, 50, 00), -1)

      cv2.circle(image2, (self.ball_y1 * self.expansionRatio + 3, self.ball_x1 * self.expansionRatio + 3), 3,
             (150, 150, 00), -1)

      cv2.circle(image2, (self.f_goal_y0 * self.expansionRatio + 3, self.f_goal_x0 * self.expansionRatio + 3), 3,
             (200, 200, 00), -1)

      cv2.circle(image2, (self.goal_y * self.expansionRatio + 3, self.goal_x * self.expansionRatio + 3), 3,
            (0, 0, 0), -1)



      image3 = np.concatenate([image1, image2], axis=1)


      if self.VISUALIZE:
          cv2.imshow('i', image)
          cv2.waitKey(1)

      if self.SAVE:
          cv2.imwrite(self.dir + "/fovea_" + str(self.fovealStepCount) + "_Step_" + str(self.stepCount) + ".png", image3)

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
  def safeFovealGoal(self):
      #print("return a safe foveal goal for training local agent ")
      #print(self.ball_x0)
      #print(self.ball_y0)
      
      self.smallGrid = np.asarray(self.grid[self.ball_x0 - int(self.fovea/2) : self.ball_x0 + int(self.fovea/2) + 1,self.ball_y0 - int(self.fovea/2) : self.ball_y0 + int(self.fovea/2) + 1])
      

      action_local = np.random.randint(0, 25)

      
      while self.smallGrid[int(action_local/self.fovea)][action_local%self.fovea] == 'W': 
          action_local = np.random.randint(0, 25)

      #print(self.smallGrid)
      #print(action_local)
      #print(int(action_local/self.fovea), action_local%self.fovea)
      
      return action_local


  """    
  """
  def setGrid(self):


    random = np.random.randint(1,6)
    """
    if random == 1:
      self.grid = np.array([['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'S',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 2:
      self.grid = np.array([['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'S',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])
    if random == 3:
      self.grid = np.array([['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'S',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 4:
      self.grid = np.array([['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'S',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])

    if random == 5:
      self.grid = np.array([['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'S',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'B',	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'B'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'],
                            ['W',	 'W',	 'W',	 'W',	 'W',	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W'	,	 'W']])


    """
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


