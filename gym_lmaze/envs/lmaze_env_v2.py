import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
plt.ion()
plt.show(block=False)



class LmazeEnv_v2(gym.Env):
  metadata = {'render.modes': ['human']}

  """    
  """
  def __init__(self):

    self.state_type = "twoState"

    self.realgrid = 18
    self.expansionRatio = 7
    self.fovea = 5

    self.gridsize = self.fovea * self.expansionRatio

    if self.state_type == "oneState":
        self.channel = 4
        self.stateChannel = self.channel
    elif self.state_type == "twoState":
        self.channel = 2
        self.actionChannel = 1
        self.stateChannel = self.channel+self.actionChannel+self.channel

    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(self.stateChannel, self.gridsize, self.gridsize))
    self.action_space = spaces.Discrete(self.fovea*self.fovea)

    self.retStatelast = np.zeros((self.channel, self.fovea, self.fovea), dtype=np.float32)

    self.negativeNominal = -1.0
    self.positiveNominal = -0.01
    self.positiveFull = 1.0
    self.RANDOM_BALL = True
    self.AUTO_VISUALIZE = False
    self.VISUALIZE = False
    self.f_goal_x = 0
    self.f_goal_y = 0
    self.localDone = False
    self.stepCount = 0
    self.fovealStepCount = 0

    self.grid = np.array(       [['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'S', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'B', 'B', 'W', 'X', 'W', 'B', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W'],
                                ['W','W', 'W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W','W', 'W', 'W']])

    self.boundary = np.zeros((self.fovea*self.expansionRatio, self.expansionRatio))
    self.boundary[:][:] = 250
    self.lowerboundary = np.zeros((self.expansionRatio, self.realgrid*self.expansionRatio))
    self.lowerboundary[:][:] = 250

    self.goal_x0, self.goal_y0 = 0, 0
    self.ball_x1, self.ball_y1 = 0, 0

    self.retStateExpanded = np.zeros((self.stateChannel, self.fovea * self.expansionRatio,
                                      self.fovea * self.expansionRatio), dtype=np.float32)

    #self.fig, self.ax = plt.subplots()

    self.reset()






  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    #print("createing dir")
    self.dir = "dir"+str(int(time.time()*1000))
    os.makedirs(self.dir)

    self.state = np.zeros((self.channel, self.realgrid, self.realgrid), dtype=np.float32)
    self.originalReward = -0.0
    self.fovealReward = -0.0
    self.stepCount = 0
    self.img = None

    action_value = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    if self.RANDOM_BALL:
        x, y = 0, 0
        while self.grid[x][y] == 'W' or self.grid[x][y] == 'X':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)
        self.ball_x0, self.ball_y0 = x, y

    else:
        start = np.where(self.grid == 'S')
        self.ball_x0, self.ball_y0 = start[0][0], start[1][0]

    self.state[0] = [[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]
    self.state[1] = [[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]

    # global state as per grid and goal and start position
    # after this only get the view from this based on position of agent

    observation = np.asarray(self.state[:, self.ball_x0 - 2 : self.ball_x0 + 3, self.ball_y0 - 2 : self.ball_y0 + 3])

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

  def rendering(self, msg):
      self.VISUALIZE = msg


  def step(self, goal):
      #
      self.originalReward = -0.0
      self.stepCount += 1
      #
      #
      image_agent_current = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))
      image_agent_goal = np.zeros((self.fovea*self.expansionRatio, self.fovea*self.expansionRatio))

      action_value = np.zeros((self.actionChannel, self.fovea, self.fovea), dtype=np.float32)


      if self.VISUALIZE:
          im = np.asarray(self.retStatelast[0])

          image_agent_current_temp = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          image_agent_current_temp[2, 2] = 150
          for i in range(0, image_agent_current_temp.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, image_agent_current_temp.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          image_agent_current[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image_agent_current_temp[i][j]


          image_agent_goal_temp = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          image_agent_goal_temp[self.goal_x0, self.goal_y0] = 50
          for i in range(0, image_agent_goal_temp.shape[0]):
              for ii in range(0, self.expansionRatio):
                  for j in range(0, image_agent_goal_temp.shape[1]):
                      for jj in range(0, self.expansionRatio):
                          image_agent_goal[i * self.expansionRatio + ii][j * self.expansionRatio + jj] = image_agent_goal_temp[i][j]


      #
      # build action as a layer
      #
      action_value[0, int(goal/self.fovea), int(goal%self.fovea)] = 1.0
      #
      # build global goal value from local goal value
      #
      self.f_goal_x = self.ball_x0 + int(goal / self.fovea) - 2
      self.f_goal_y = self.ball_y0 + int(goal % self.fovea) - 2
      #
      # if the global goal is not out of boundary
      # move the agent center to the global goal
      #

      # To enable this, remove ball update in block below
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

      # if the agent is moved to a goal, wall or blank space; give relevant reward
      #
      if self.grid[self.f_goal_x][self.f_goal_y] == 'X':
      #    self.ball_x0 = self.f_goal_x
      #    self.ball_y0 = self.f_goal_y
      #    print("Goal")
          self.originalReward = self.positiveFull
      elif self.grid[self.f_goal_x][self.f_goal_y] == 'W':
          self.originalReward = self.negativeNominal
      elif self.grid[self.f_goal_x][self.f_goal_y] == 'B' or self.grid[self.f_goal_x][self.f_goal_y] == 'S':
      #    self.ball_x0 = self.f_goal_x
      #    self.ball_y0 = self.f_goal_y
          self.originalReward = self.positiveNominal
      #
      # post update of agent position, build observation
      #
      observation = np.asarray(self.state[:, self.ball_x0 - int(self.fovea/2) : self.ball_x0 + int(self.fovea/2) + 1,
                               self.ball_y0 - int(self.fovea/2):self.ball_y0 + int(self.fovea/2) + 1])
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


      if self.VISUALIZE:
          #
          # Build the global view
          #
          image3 = np.zeros((self.realgrid*self.expansionRatio, self.realgrid*self.expansionRatio))
          image3_ = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in self.state[0]]), dtype=np.uint8)
          image3_[self.goal_x, self.goal_y] = 100
          image3_[self.f_goal_x, self.f_goal_y] = 50
          image3_[self.ball_x0, self.ball_y0] = 150
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

          im = np.asarray(self.retStateExpanded[2])
          image_goal = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)


          im = np.asarray(self.retStateExpanded[0])
          image_agent_next = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          #
          for ii in range(0, self.expansionRatio):
              for jj in range(0, self.expansionRatio):
                  #image1[2*self.expansionRatio+ii][2*self.expansionRatio+jj] = 150
                  image_agent_next[int(goal / self.fovea)*self.expansionRatio+ii][int(goal % self.fovea)*self.expansionRatio+jj] = 100

          image_goal_next = np.asarray(([[200 if item == 1.0 else 0 for item in line] for line in im]), dtype=np.uint8)
          #
          for ii in range(0, self.expansionRatio):
              for jj in range(0, self.expansionRatio):
                  image_goal_next[2*self.expansionRatio+ii][2*self.expansionRatio+jj] = 150
                  #image11[int(goal / self.fovea)*self.expansionRatio+ii][int(goal % self.fovea)*self.expansionRatio+jj] = 100
          #
          # Arrange the images over plot
          #
          #image = np.concatenate([self.boundary,self.boundary,self.boundary, image1, self.boundary, self.boundary, image2, self.boundary,self.boundary,self.boundary], axis = 1)
          #image = np.concatenate([image3, self.lowerboundary, image, self.lowerboundary], axis = 0)
          imagel1 = np.concatenate([self.boundary, self.boundary, image_agent_current, self.boundary, self.boundary,
                                    self.boundary, self.boundary, image_agent_goal, self.boundary, self.boundary ], axis = 1)

          imagel2 = np.concatenate([self.boundary, self.boundary, image_agent_next, self.boundary, self.boundary,
                                    self.boundary, self.boundary, image_goal_next, self.boundary, self.boundary ], axis = 1)

          imagel3 = np.concatenate([self.boundary, self.boundary, self.boundary, self.boundary, self.boundary, self.boundary ,self.boundary , image_goal,
                                    self.boundary, self.boundary, self.boundary, self.boundary, self.boundary, self.boundary ], axis = 1)

          image = np.concatenate([image3, self.lowerboundary, imagel1, self.lowerboundary, imagel2, self.lowerboundary, imagel3, self.lowerboundary], axis = 0)
          #
          # Plot the image
          #

          if self.img == None:
              plt.clf()
              self.img = plt.imshow(image)
          else:
              self.img.set_data(image)

          title = "Step Number : " + str(self.stepCount) + \
                  "\nAction : "+ str(goal) +" (" + str(int(goal / self.fovea)) + ", " + str(int(goal % self.fovea)) + ")"

          plt.title(title)
          plt.savefig(self.dir+"/image"+ str(self.stepCount)+".png")
          plt.pause(0.5)
          plt.draw()


      self.goal_x0 = int(goal/self.fovea)
      self.goal_y0 = int(goal%self.fovea)

      self.ball_x1 = self.ball_x0
      self.ball_y1 = self.ball_y0

      if self.originalReward == self.positiveFull or self.stepCount > 100:
      #if self.originalReward == self.positiveFull:
          return self.retStateExpanded, self.originalReward, True, goal
      else :
          return self.retStateExpanded, self.originalReward, False, goal