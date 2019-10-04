import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ion()
plt.show(block=False)

class LmazeEnv_v1(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("init-init")
    self.action_space = spaces.Discrete(4)
    self.realgrid = 14
    self.expansionRatio = 7
    self.fovea = 5
    #self.gridsize = self.realgrid * self.expansionRatio
    self.gridsize = self.fovea * self.expansionRatio
    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4,self.gridsize,self.gridsize))
    self.negativeNominal = -1.0
    self.positiveNominal = 0.0
    self.positiveFull = 1.0
    self.RANDOM_BALL = False
    self.VISUALIZE = True
    self.f_goal_x = 0
    self.f_goal_y = 0
    self.localDone = False

    """self.gri = np.array(  [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                            ['W', 'S', 'B', 'B', 'B', 'W', 'W', 'B'],
                            ['W', 'B', 'W', 'W', 'W', 'W', 'W', 'B'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'B', 'B'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W'],
                            ['W', 'B', 'B', 'B', 'W', 'X', 'W', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W']])"""

    self.grid = np.array(       [['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'S', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'W', 'X', 'W', 'B', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])

    """
    self.grid = np.chararray((self.gridsize,self.gridsize))

    i = 0
    while i < self.gri.shape[0]:
        ii = 0
        for ii in range(0,self.expansionRatio):
            j = 0
            while j < self.gri.shape[1]:
                jj = 0
                for jj in range(0,self.expansionRatio):
                    self.grid[i*self.expansionRatio + ii][j*self.expansionRatio + jj] = self.gri[i][j]
                    #print(i*self.expansionRatio + ii,j*self.expansionRatio + jj)
                    #print(i,j)
                    jj += 1
                j += 1
            ii += 1
        i += 1
    """

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)

    self.reset()
    print("init-end")







  def render(self, mode='human', close=False):
    if(mode == 'human'):
        self.VISUALIZE = True







  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)
    self.state[0 * self.realgrid * self.realgrid : 1 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "S" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "W" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    start = np.where(self.grid == 'S')
    self.ball_x0 = start[0][0]
    self.ball_y0 = start[1][0]

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    self.reward = -0.0
    self.stepCount = 0
    self.img = None

    return self.getGlobalView()





  """    
  """
  def setFovealGoal(self,msg0,msg1):
      # translate to global goal position for the loacl value
      # set it as the goal
      self.f_goal_x = self.ball_x0 + msg0 - 2
      self.f_goal_y = self.ball_y0 + msg1 - 2
      # return the environment view
      return self.getLocalView()




  """    
  """
  def getGlobalView(self):

      retState = np.concatenate(
          (np.reshape(self.state[0 * self.realgrid * self.realgrid: 1 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[1 * self.realgrid * self.realgrid: 2 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[2 * self.realgrid * self.realgrid: 3 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[3 * self.realgrid * self.realgrid: 4 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid)))), axis=0)

      retState = np.asarray(retState[:, self.ball_x0 - 2:self.ball_x0 + 3, self.ball_y0 - 2:self.ball_y0 + 3])

      retStateExpanded = np.zeros((4, retState.shape[1] * self.expansionRatio, retState.shape[2] * self.expansionRatio),
                                  dtype=np.float32)

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
                          retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          retState[channel][i][j]
                          jj += 1
                      j += 1
                  ii += 1
              i += 1
          channel += 1

      return retStateExpanded




  """    
  """
  def getLocalView(self):

      retState = np.concatenate(
          (np.reshape(self.state[0 * self.realgrid * self.realgrid: 1 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[1 * self.realgrid * self.realgrid: 2 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[2 * self.realgrid * self.realgrid: 3 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[3 * self.realgrid * self.realgrid: 4 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid)))), axis=0)

      retState = np.asarray(retState[:, self.f_goal_x - 2:self.f_goal_x + 3, self.f_goal_y - 2:self.f_goal_y + 3])

      retStateExpanded = np.zeros((4, retState.shape[1] * self.expansionRatio, retState.shape[2] * self.expansionRatio),
                                  dtype=np.float32)

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
                          retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          retState[channel][i][j]
                          jj += 1
                      j += 1
                  ii += 1
              i += 1
          channel += 1
      return retStateExpanded



  """
  """
  def step(self,msg):

    """ Build Action Value to be used for ball position update """
    self.stepCount += 1

    o_x = 0
    o_y = 0
    if msg == 0:
        #print("Action 0")
        o_x = -1
        o_y = 0
    elif msg == 1:
        #print("Action 1")
        o_x = 1
        o_y = 0
    elif msg == 2:
        #print("Action 2")
        o_x = 0
        o_y = -1
    elif msg == 3:
        #print("Action 3")
        o_x = 0
        o_y = 1

    if self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'W':
        # position unchanged small negative reward
        self.reward = self.negativeNominal

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'B':
        # position changed small positive reward
        self.state[0 * self.realgrid * self.realgrid + self.ball_x0 * self.realgrid + self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0

        # if the goal goes out of view of agent after this movement
        if (self.ball_x0 > self.f_goal_x - 2 or self.ball_x0 < self.f_goal_x +3 \
                or self.ball_y0 > self.f_goal_y - 2 or self.ball_y0 < self.f_goal_y +3):
            print(self.ball_x0 > self.f_goal_x - 2, self.ball_x0 < self.f_goal_x +3,
                 self.ball_y0 > self.f_goal_y - 2, self.ball_y0 < self.f_goal_y +3)
            self.stepCount = 0
            self.localDone = True

        self.reward = self.positiveNominal

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'X':
        # position changed full reward
        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0
        self.reward = self.positiveFull





    if self.VISUALIZE:
        image = np.asarray(([200 if item == 1.0 else 0 for item in self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid]]), dtype=np.uint8)
        image[(self.goal_x) * self.realgrid + self.goal_y] = 100
        image[(self.ball_x0) * self.realgrid + self.ball_y0] = 150

        image2 = np.reshape(image,(self.realgrid,self.realgrid))
        if self.img == None:
            plt.clf()
            self.img = plt.imshow(image2)
        else:
            self.img.set_data(image2)

        plt.pause(.01)
        plt.draw()

    if self.stepCount == 10:
        self.stepCount = 0
        self.localDone = True

    return self.getLocalView(), self.reward, self.isEpisodeFinished(), self.localDone, msg












    """    
    """
  def initState(self):
    return self.state, self.reward, self.isEpisodeFinished(), {'newState' : True}


  def isEpisodeFinished(self):
    if self.reward == self.positiveFull:
        return True
    return False
