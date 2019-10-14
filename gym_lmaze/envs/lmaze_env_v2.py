import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ion()
plt.show(block=False)

class LmazeEnv_v2(gym.Env):
  metadata = {'render.modes': ['human']}

  """    
  """
  def __init__(self):
    print("init-init")

    self.state_type = "twoState"

    self.realgrid = 14
    self.expansionRatio = 7
    self.fovea = 5
    self.gridsize = self.fovea * self.expansionRatio
    self.action_space = spaces.Discrete(self.fovea*self.fovea)

    if self.state_type == "oneState":
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4, self.gridsize, self.gridsize))
        self.channel = 4
    elif self.state_type == "twoState":
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4+1+4, self.gridsize, self.gridsize))
        self.channel = 9

    self.negativeNominal = -1.0
    self.positiveNominal = 0.1
    self.positiveFull = 1.0
    self.RANDOM_BALL = True
    self.AUTO_VISUALIZE = False
    self.VISUALIZE = False
    self.f_goal_x = 0
    self.f_goal_y = 0
    self.localDone = False
    self.stepCount = 0
    self.fovealStepCount = 0

    #self.visualFile = open("visualize.txt", "r")

    self.grid = np.array(       [['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'S', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'W', 'X', 'W', 'B', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])

    self.retStatelast = np.zeros((4, 5, 5), dtype=np.float32)
    self.boundary = np.zeros((35, 7))
    self.boundary[:][:] = 200
    self.lowerboundary = np.zeros((7, 98))
    self.lowerboundary[:][:] = 200

    self.goal_x0, self.goal_y0 = 0, 0
    self.ball_x1, self.ball_y1 = 0, 0

    self.retStateExpanded = np.zeros((self.channel, self.fovea * self.expansionRatio,
                                      self.fovea * self.expansionRatio), dtype=np.float32)

    self.reset()
    print("init-end")







  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.state = np.zeros((4, self.realgrid, self.realgrid), dtype=np.float32)
    self.originalReward = -0.0
    self.fovealReward = -0.0
    self.stepCount = 0
    self.img = None

    action_value = np.zeros((1, 5, 5), dtype=np.float32)

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    if self.RANDOM_BALL:
        x, y = 0, 0
        while self.grid[x][y] == 'W' or self.grid[x][y] == 'X':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.ball_x0, self.ball_y0 = x, y
        self.state[0][self.ball_x0][self.ball_y0] = 1.0

    else:
        start = np.where(self.grid == 'S')
        self.ball_x0, self.ball_y0 = start[0][0], start[1][0]
        self.state[0][self.ball_x0][self.ball_y0] = 1.0

    self.state[1] = [[1.0 if cell == "W" else 0.0 for cell in row] for row in self.grid]
    self.state[2] = [[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.state[3] = [[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]


    # global state as per grid and goal and start position
    # after this only get the view from this based on position of agent

    observation = np.asarray(self.state[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2 : self.ball_y0 + 3])

    #self.retStateExpanded = np.zeros((self.channel, observation.shape[1] * self.expansionRatio,
    #                                  observation.shape[2] * self.expansionRatio), dtype=np.float32)

    if self.state_type == "oneState":
        retState = observation
    elif self.state_type == "twoState":
        self.retStatelast = observation
        retState = np.concatenate([observation, action_value, self.retStatelast], axis=0)

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
  def step(self, goal):
      #
      self.originalReward = -0.0
      self.stepCount += 1
      #
      #
      image2 = np.zeros((35, 35))
      action_value = np.zeros((1, 5, 5), dtype=np.float32)
      #
      #
      if self.VISUALIZE:
          #
          # Build the local previous view
          #
          # last state wall structure
          im = np.asarray(self.retStatelast[1])
          #
          # change the value 1 to color 200
          #
          image2_temp = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          #
          # last goal given to the planner
          #
          image2_temp[self.goal_x0, self.goal_y0] = 50
          #
          # marking the center of the state
          #
          image2_temp[2, 2] = 150
          #
          # Expand the state in dimentions
          #
          for i in range(0, image2_temp.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, image2_temp.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          image2[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image2_temp[i][j]
      #
      # build action as a layer
      #
      action_value[0, int(goal/5), int(goal%5)] = 1.0
      #
      # build global goal value from local goal value
      #
      self.f_goal_x = self.ball_x0 + int(goal / 5) - 2
      self.f_goal_y = self.ball_y0 + int(goal % 5) - 2
      #print(self.f_goal_x,self.f_goal_y, self.ball_x0,self.ball_y0, goal, int(goal / 5), int(goal % 5))
      #
      # if the global goal is not out of boundary
      # move the agent center to the global goal
      #
      if self.f_goal_x < 12 and self.f_goal_x > 1 and self.f_goal_y < 12 and self.f_goal_y > 1:
          self.state[0][self.ball_x0][self.ball_y0] = 0.0
          self.ball_x0 = self.f_goal_x
          self.ball_y0 = self.f_goal_y
          self.state[0][self.ball_x0][self.ball_y0] = 1.0
      #else:
      #    self.originalReward = self.negativeNominal
      #
      # if the agent is moved to a goal, wall or blank space; give relevant reward
      #
      if self.grid[self.f_goal_x][self.f_goal_y] == 'X':
          self.originalReward = self.positiveFull
      #elif self.grid[self.f_goal_x][self.f_goal_y] == 'W':
      #    self.originalReward = self.negativeNominal
      #elif self.grid[self.f_goal_x][self.f_goal_y] == 'B':
      #    self.originalReward = self.positiveNominal
      #
      # post update of agent position, build observation
      #
      observation = np.asarray(self.state[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2:self.ball_y0 + 3])
      #
      # build state to return to the learner
      #
      if self.state_type == "oneState":
          retState = observation
      elif self.state_type == "twoState":
          retState = np.concatenate([observation, action_value, self.retStatelast], axis=0)
          self.retStatelast = observation
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
      #
      #
      if self.VISUALIZE:
          #
          # Build the global view
          #
          image3 = np.zeros((14*7, 14*7))
          image3_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in self.state[1]]), dtype=np.uint8)
          image3_[self.goal_x, self.goal_y] = 100
          image3_[self.f_goal_x, self.f_goal_y] = 50
          image3_[self.ball_x0, self.ball_y0] = 150
          #
          #
          #
          for i in range(0, image3_.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, image3_.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          image3[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image3_[i][j]
          #
          # Build the local current view
          #
          im = np.asarray(self.retStateExpanded[1])
          image1 = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          #
          for ii in range(0, self.expansionRatio):
              for jj in range(0, self.expansionRatio):
                  image1[2*7+ii][2*7+jj] = 150
                  image1[int(goal / 5)*7+ii][int(goal % 5)*7+jj] = 100
          #
          # Arrange the images over plot
          #
          image = np.concatenate([self.boundary, image1, self.boundary, self.boundary, image2, self.boundary], axis = 1)
          image = np.concatenate([image3, self.lowerboundary, image, self.lowerboundary], axis = 0)
          #
          # Plot the image
          #
          if self.img == None:
              plt.clf()
              self.img = plt.imshow(image)
          else:
              self.img.set_data(image)
          #
          #
          #
          plt.pause(.01)
          plt.draw()


      self.goal_x0 = int(goal/5)
      self.goal_y0 = int(goal%5)

      self.ball_x1 = self.ball_x0
      self.ball_y1 = self.ball_y0

      #if self.originalReward == self.positiveFull or self.stepCount > 10:
      if self.originalReward == self.positiveFull:
          return self.retStateExpanded, self.originalReward, True, goal
      else :
          return self.retStateExpanded, self.originalReward, False, goal