import numpy as np


"""

env consists of collection of flavor-location paired associates
	i.e. tuples (Fi,Li) i={1...N}

- method for randomizing tuples


"""


## data generation parameters

class Task():

  def __init__(self,num_pas):
    """ num_sampels defines the size of the task space, 
      the number of (flavor,location) pairs
    """
    self.num_pas = num_pas
    return None

  def gen_env(self):
    """ returns a new random mapping of (flavor,location) pairs
    """
    F = [i for i in range(self.num_pas)]
    L = [i for i in range(self.num_pas)]
    np.random.shuffle(F)
    np.random.shuffle(L)
    env={F[i]:L[i] for i in range(self.num_pas)}
    return env

  def gen_session(self,num_episodes,pr_shift):
    """ 
    a session consists of multiple episodes (possibly from different environments)
    an episode consists of a full pass through all (flavor,location) pairs
		different episodes randomize the order at which the PAs are presented
		pr_shift controls the probabilty that a new environment will be drawn
			0 means same environment always, 1 means PAs always randomizing 
    """
    environment = self.gen_env()
    env_flavorL = [i for i in range(self.num_pas)]
    flavors = []
    locations = []
    for ep_num in range(num_episodes):
      # generate a session where sample order is randomized
      np.random.shuffle(env_flavorL)
      flavors.extend(env_flavorL)
      locations.extend([environment[flavor] for flavor in env_flavorL])
      # with probability shift environment
      if np.random.binomial(1,pr_shift):
        environment = self.gen_env()
    # pairs in random order
    return flavors,locations

  def gen_MLdataset(self,num_episodes,pr_shift,depth):
    flavors,locations = self.gen_session(num_episodes,pr_shift)
    X,Y = MLdataset(flavors,locations,depth=depth)
    return X,Y

  def format_Xeval(self,pathL):
    """
    given a list of paths [[0,10,1,3,5],[0,11,2,4,6]]
    returns an array with format expected by Trainer.predict_step
      (num_pas,depth,in_len)
    """
    Xeval = np.array(pathL)
    Xeval = np.expand_dims(Xeval,2)
    return Xeval


def MLdataset(Xseq,Yseq,depth=1):
    """ 
    given a series of Xseq and Yseq
    structures a meta-learning dataset
    returns:
      shape: (samples,depth,len)
    """
    X = [[yt1,xt] for xt,yt1 in zip(Xseq[1:],Yseq[:-1])]
    Y = [[yt] for yt in Yseq[1:]]
    X = slice_and_stride(X,depth)
    Y = slice_and_stride(Y,depth)
    return X,Y

def slice_and_stride(X,depth=1):
  """ 
  useful for including BPTT dim: 
    given (batch,in_len) 
    returns (batch,depth,in_len)
  stride step fixed to = 1
  tf.sliding_window_batch d/n support stride=depth=1
  """
  Xstr = []
  for idx in range(len(X)-depth+1):
    x = X[idx:idx+depth]
    Xstr.append(x)
  return np.array(Xstr)
