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

    self.state_type = "oneState"

    self.action_space = spaces.Discrete(4)
    self.realgrid = 14
    self.expansionRatio = 7
    self.fovea = 5
    self.gridsize = self.fovea * self.expansionRatio

    if self.state_type == "oneState":
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4, self.gridsize, self.gridsize))
    elif self.state_type == "twoState":
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4+1+4,self.gridsize,self.gridsize))

    self.negativeNominal = -1.0
    self.positiveNominal = 0.01
    self.positiveFull = 1.0
    self.RANDOM_BALL = True
    self.VISUALIZE = True
    self.f_goal_x = 0
    self.f_goal_y = 0
    self.localDone = False
    self.stepCount = 0
    self.fovealStepCount = 0

    self.visualFile = open("visualize.txt", "r")

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

    self.retState1 = np.zeros((4, 5, 5), dtype=np.float32)

    self.reset()
    print("init-end")

  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.state = np.zeros((4 * self.realgrid * self.realgrid), dtype=np.float32)
    action_value = np.zeros((1, 5, 5), dtype=np.float32)

    if self.RANDOM_BALL:

        x, y = 0, 0

        while self.grid[x][y] == 'W' or self.grid[x][y] == 'X':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)

        self.ball_x0, self.ball_y0 = x, y

        self.state[0 * self.realgrid * self.realgrid + (self.ball_x0) * self.realgrid + self.ball_y0] = 1.0

    else:

        start = np.where(self.grid == 'S')

        self.ball_x0, self.ball_y0 = start[0][0], start[1][0]

        self.state[0 * self.gridsize * self.gridsize + (self.ball_x0) * self.gridsize + self.ball_y0] = 1.0

    self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "W" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    self.originalReward = -0.0
    self.fovealReward = -0.0

    self.stepCount = 0

    self.img = None

    state1 = np.reshape(self.state, (4, self.realgrid, self.realgrid))

    retState = np.asarray(state1[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2 : self.ball_y0 + 3])

    if self.state_type == "oneState":
        self.retStateExpanded = np.zeros((4, retState.shape[1] * self.expansionRatio,
                                          retState.shape[2] * self.expansionRatio), dtype=np.float32)
    elif self.state_type == "twoState":
        self.retState1 = retState
        retState = np.concatenate([retState, action_value, self.retState1], axis=0)
        self.retStateExpanded = np.zeros((4+1+4, retState.shape[1] * self.expansionRatio,
                                      retState.shape[2] * self.expansionRatio), dtype=np.float32)

    channel = 0
    while channel < retState.shape[0]:
        i = 0
        while i < retState.shape[1]:
            ii = 0
            for ii in range(0, self.expansionRatio):
                j = 0
                while j < retState.shape[2]:
                    jj = 0
                    for jj in range(0, self.expansionRatio):
                        self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                            retState[channel][i][j]
                        jj += 1
                    j += 1
                ii += 1
            i += 1
        channel += 1

    return self.retStateExpanded



  """    
  """
  def step(self, goal):

      action_value = np.zeros((1, 5, 5), dtype=np.float32)

      self.originalReward = -0.0

      self.stepCount += 1

      action_value[0, int(goal/5), int(goal%5)] = 1.0

      self.f_goal_x = self.ball_x0 + int(goal / 5) - 2
      self.f_goal_y = self.ball_y0 + int(goal % 5) - 2

      if self.f_goal_x < 12 and self.f_goal_x > 1 and self.f_goal_y < 12 and self.f_goal_y > 1:
          self.state[0 * self.realgrid * self.realgrid + (self.ball_x0) * self.realgrid + self.ball_y0] = 0.0
          self.ball_x0 = self.f_goal_x
          self.ball_y0 = self.f_goal_y
          self.state[0 * self.realgrid * self.realgrid + (self.ball_x0) * self.realgrid + self.ball_y0] = 1.0

      if self.grid[self.f_goal_x][self.f_goal_y] == 'X':
          self.originalReward = self.positiveFull
      elif self.grid[self.f_goal_x][self.f_goal_y] == 'W':
          self.originalReward = self.negativeNominal

      state1 = np.asarray(np.reshape(self.state, (4, self.realgrid, self.realgrid)))

      retState = np.asarray(state1[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2:self.ball_y0 + 3])

      if self.state_type == "oneState":
          self.retStateExpanded = np.zeros((4, retState.shape[1] * self.expansionRatio,
                                            retState.shape[2] * self.expansionRatio), dtype=np.float32)
      elif self.state_type == "twoState":
          retState_temp = np.concatenate([retState, action_value, self.retState1], axis=0)
          self.retState1 = retState
          retState = retState_temp
          self.retStateExpanded = np.zeros((4 + 1 + 4, retState.shape[1] * self.expansionRatio,
                                        retState.shape[2] * self.expansionRatio), dtype=np.float32)

      channel = 0
      while channel < retState.shape[0]:
          i = 0
          while i < retState.shape[1]:
              ii = 0
              for ii in range(0, self.expansionRatio):
                  j = 0
                  while j < retState.shape[2]:
                      jj = 0
                      for jj in range(0, self.expansionRatio):
                          self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          retState[channel][i][j]
                          jj += 1
                      j += 1
                  ii += 1
              i += 1
          channel += 1

      if self.VISUALIZE:
          image = np.asarray(([200 if item == 1.0 else 0 for item in
                               self.state[1 * self.realgrid * self.realgrid: 2 * self.realgrid * self.realgrid]]),
                             dtype=np.uint8)
          image[(self.goal_x) * self.realgrid + self.goal_y] = 100
          image[(self.ball_x0) * self.realgrid + self.ball_y0] = 150

          image2 = np.reshape(image, (self.realgrid, self.realgrid))
          if self.img == None:
              plt.clf()
              self.img = plt.imshow(image2)
          else:
              self.img.set_data(image2)

          plt.pause(.01)
          plt.draw()


      if self.originalReward == self.positiveFull or self.stepCount > 10:
          return self.retStateExpanded, self.originalReward, True, goal
      else :
          return self.retStateExpanded, self.originalReward, False, goal