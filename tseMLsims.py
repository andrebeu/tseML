import numpy as np
import tensorflow as tf
from tseTask import Task as TseTask

""" 
GOAL

WORKFLOW: train & eval, save and restore data, analyze. 

separating RNNs allow me to modularize different tasks
separating trainers allows me to not have to rebuild graph every time 

NB depth == unroll_depth

----
TODO


"""


NUM_PAS = 3
IN_LEN = 2
OUT_LEN = 1
TRAIN_BATCH_SIZE = 1
DEPTH = 100 # must be less than number of samples in path 
NUM_EPISODES = 34



class NetGraph():

  def __init__(self,rnn_size,depth=DEPTH,random_seed=1):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.random_seed = random_seed
    # dimensions
    self.rnn_size = rnn_size
    self.embed_dim = rnn_size
    self.depth = depth 
    self.in_len = IN_LEN
    self.out_len = OUT_LEN
    self.num_classes = NUM_PAS
    # build
    self.build()

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.random_seed)
      print('initializing sub%.2i'%self.random_seed)
      # place holders
      self.setup_placeholders()
      # pipeline
      self.xbatch_id,self.ybatch_id = self.data_pipeline() # x(batches,bptt,in_tstep), y(batch,bptt,out_tstep)
      ## embedding 
      self.embed_mat = tf.get_variable('embedding_matrix',[self.num_classes,self.embed_dim])
      self.xbatch = tf.nn.embedding_lookup(self.embed_mat,self.xbatch_id,name='xembed') # batch,bptt,in_len,in_dim
      ## inference
      self.unscaled_logits,self.cell_state_op = self.RNN(self.depth,self.in_len,self.out_len) # batch,bptt*out_len,num_classes
      ## loss
      self.ybatch_onehot = tf.one_hot(indices=self.ybatch_id,depth=self.num_classes) # batch,bptt,out_len,num_classes
      self.loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,logits=self.unscaled_logits)
      self.minimizer_op = tf.train.AdamOptimizer(0.001).minimize(self.loss_op)
      ## accuracy
      self.yhat_sm = tf.nn.softmax(self.unscaled_logits)
      self.yhat_id = tf.argmax(self.yhat_sm,-1)
      # self.acc_op = setup_acc_op(self.yhat_id,self.ybatch_id)
      ## extra
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver()
    return None

  def setup_placeholders(self):
    self.xph = xph = tf.placeholder(tf.int32,
                  shape=[None,self.depth,self.in_len],
                  name="xdata_placeholder")
    self.yph = yph = tf.placeholder(tf.int32,
                  shape=[None,self.depth,self.out_len],
                  name="ydata_placeholder")
    self.batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    self.dropout_keep_prob = tf.placeholder(tf.float32,
                                shape=[],
                                name="dropout_ph")
    self.cellstate_ph = tf.placeholder(tf.float32,
                  shape=[None,self.rnn_size],
                  name = "initialstate_ph")
    return None

  def data_pipeline(self):
    """
    setup data iterator pipeline
    creates self.itr_initop and self.dataset
    returns x,y = get_next
    """
    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,self.yph))
    dataset = dataset.batch(self.batch_size_ph)
    # dataset = self.dataset = dataset.shuffle(100000)
    # iterator
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    xbatch,ybatch = iterator.get_next() 
    self.itr_initop = iterator.make_initializer(dataset)
    return xbatch,ybatch

  def reinitialize(self):
    print('**reinitializing weights** - NB: random_seed')
    self.random_seed += 1
    print('rand seed = ',self.random_seed)
    with self.graph.as_default():
      tf.set_random_seed(self.random_seed)
      self.sess.run(tf.global_variables_initializer())
    return None

  def RNN(self,depth,in_len,out_len):
    """ 
    general RNN structure that allows specifying 
      - depth: number of (input_seq,output_seq) that are unrolled
      - in_len: length of each input sequence
      - out_len: length of each output sequence
    consumes a sentence at a time
      
    RNN structure:
      takes in state and a filler
      returns prediction for next state and a filler
    returns unscaled logits
    """
    xbatch = self.xbatch
    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            self.rnn_size,dropout_keep_prob=self.dropout_keep_prob)

    xbatch = tf.layers.dense(xbatch,self.rnn_size,tf.nn.relu,name='inproj')
    # unroll RNN
    with tf.variable_scope('RNN_SCOPE') as cellscope:
      # initialize state
      # initial_state = state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32),tf.float32)
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      # unroll
      outputL = []
      for unroll_step in range(depth):
        xroll = xbatch[:,unroll_step,:,:]
        # input
        for in_tstep in range(in_len):
          __,state = cell(xroll[:,in_tstep,:], state)
          cellscope.reuse_variables()
        # output: inputs are zeroed out
        outputs_rs = []
        for out_tstep in range(out_len):
          zero_input = tf.zeros_like(xroll)
          cell_output, state = cell(zero_input[:,out_tstep,:], state) 
          outputs_rs.append(cell_output)
        outputs_rollstep = tf.stack(outputs_rs,axis=1)
        outputL.append(outputs_rollstep)
    # format for y_hat
    outputs = tf.stack(outputL,axis=1)
    # project to unscaled logits (to that outdim = num_classes)
    outputs = tf.layers.dense(outputs,self.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
    return outputs,state


""" TRAIN AND EVAL
separating net from trainer allows me to change trainer without having
to reinitialize net. more efficient for debugging 
"""

class Trainer():

  def __init__(self,net):
    self.net = net  

  def train_step(self,Xtrain,Ytrain,cell_state):
    """ updates model parameters using Xtrain,Ytrain
    """
    # initialize iterator with train data
    train_feed_dict = {
      self.net.xph: Xtrain,
      self.net.yph: Ytrain,
      self.net.batch_size_ph: TRAIN_BATCH_SIZE,
      self.net.dropout_keep_prob: .9,
      self.net.cellstate_ph: cell_state
      }
    self.net.sess.run([self.net.itr_initop],train_feed_dict)
    # train loop
    while True:
      try:
        _,train_step_loss,new_cell_state = self.net.sess.run(
          [self.net.minimizer_op,self.net.loss_op,self.net.cell_state_op],feed_dict=train_feed_dict)
        # print(loss)
      except tf.errors.OutOfRangeError:
        break
    return train_step_loss,new_cell_state

  def train(self,num_epochs,pr_shift,num_evals=1000):
    """ 
    return: 
      pred_data['yhat'], shape: (epochs,path,depth,len,num_classes)
    """
    ## SETUP
    # task: two graphs two filler ids
    task = TseTask(NUM_PAS)
    # array for recording data
    assert num_epochs % num_evals == 0
    train_loss = np.zeros(num_evals)
    ## main training loop
    # initial cell_state
    zero_cell_state = cell_state = np.zeros(shape=[TRAIN_BATCH_SIZE,self.net.rnn_size])
    eval_idx = -1
    for ep_num in range(num_epochs):
      # generate new sequence 
      Xdata,Ydata = task.gen_MLdataset(num_episodes=NUM_EPISODES,pr_shift=pr_shift,depth=self.net.depth)
      # zero cell state to begin each session
      step_loss,cell_state = self.train_step(Xdata,Ydata,zero_cell_state)
      # training data
      if ep_num%(num_epochs/num_evals) == 0:
        eval_idx += 1
        train_loss[eval_idx] = np.mean(step_loss)
      if ep_num%(num_epochs/100) == 0:
        print(ep_num,eval_idx,np.mean(step_loss)) 
    return train_loss

  def eval(self,Xpred,Ypred,cell_state=None):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    """
    batch_size = len(Xpred)
    if cell_state == None:
      cell_state = np.zeros(shape=[batch_size,self.net.rnn_size])
    # repeat cell state for each prediction
    # cell_state = np.repeat(cell_state,batch_size,axis=0)
    # initialize data datastructure for collecting data
    pred_array_dtype = [('xbatch','int32',(batch_size,self.net.depth,self.net.in_len)),
                        ('yhat','float32',(batch_size,self.net.depth,self.net.out_len,self.net.num_classes)),
                        ('loss','float32',(batch_size,self.net.depth)),
    ]
    pred_data_arr = np.zeros((),dtype=pred_array_dtype)
    # feed dict
    pred_feed_dict = {
      self.net.xph:Xpred,
      self.net.yph:Ypred,
      self.net.batch_size_ph: batch_size,
      self.net.dropout_keep_prob: 1.0,
      self.net.cellstate_ph: cell_state
    }
    # initialize iterator with eval data
    self.net.sess.run(self.net.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        loss,xbatch,yhat = self.net.sess.run([self.net.loss_op,self.net.xbatch_id,self.net.yhat_sm],feed_dict=pred_feed_dict)
        pred_data_arr['xbatch'] = xbatch
        pred_data_arr['yhat'] = yhat
        print(loss.shape)
        pred_data_arr['loss'] = loss.squeeze()
      except tf.errors.OutOfRangeError:
        break 
    return pred_data_arr

