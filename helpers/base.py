import numpy as np
import math
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

class DqnAgents:
    '''Implementation of various DQN agents.
    
    Attributes:
    S: numpy array with a batch of states.
    A: numpy array with a batch of actions.
    R: numpy array with a batch of rewards.
    S2: numpy array with a batch of next_states.
    T: bool, a batch of terminals.
    
    '''
    
    def __init__(self):
        '''Initializes transition batches.'''
        
        self.S = None
        self.A = None
        self.R = None
        self.S2 = None
        self.T = None
        
    def build_dqn_graph(self, s_shape, num_a, alpha, compiled=True):
        '''Instantiates a network graph for the DQN implementation.
        
        Arguments:
        s_shape: tuple of ints with the shape of input state.
        num_a: int, number of available actions.
        alpha: float, learning rate.
        compipled: bool, True for uncompiled target net. False for
                   compiled prediction net. 
        
        Returns:
        t_net: Keras model object, an uncompiled model.
        q_net_net: Keras model object, a compiled model.
        
        '''
        inputs = Input(shape=s_shape)
        x = Dense(400, activation='relu')(inputs)
        x = Dense(200, activation='relu')(x)
        outputs = Dense(num_a, activation='linear')(x)
        if not compiled:
            t_net = Model(inputs, outputs)
            return t_net
        q_net = Model(inputs, outputs)
        opt = RMSprop(learning_rate=alpha)
        q_net.compile(optimizer=opt, loss='mse')
        return q_net
    
    def build_dueling_graph(self, s_shape, num_a, alpha, batchsize, compiled=True):
        '''Instantiates a network graph for the Dueling DQN implementation.
        
        Arguments:
        s_shape: tuple of ints with the shape of input state.
        num_a: int, number of available actions.
        alpha: float, learning rate.
        batchsize: int.
        compipled: bool, True for uncompiled target net. False for
                   compiled prediction net. 
        
        Returns:
        t_net: Keras model object, an uncompiled model.
        q_net: Keras model object, a compiled model.
        q_net_policy: Keras model object with a seperate output.
        
        '''
        inputs = Input(shape=s_shape)
        x = Dense(400, activation='relu')(inputs)
        x = Dense(200, activation='relu')(x)
        v = Dense(100, activation='relu')(x)
        v = Dense(1, activation='linear')(v)
        a = Dense(100, activation='relu')(x)
        a = Dense(num_a, activation='linear')(a)
        outputs = Lambda(
            self._subtract_mean, arguments={'num_a': num_a,
                                            'batchsize': batchsize})([v, a])
        if not compiled:
            t_net = Model(inputs, outputs)
            return t_net
        q_net = Model(inputs, outputs)
        q_net_policy = Model(inputs, a)
        opt = RMSprop(learning_rate=alpha)
        q_net.compile(optimizer=opt, loss='mse')
        return q_net, q_net_policy
        
    def build_c51_graph(self, s_shape, num_a, num_atoms, alpha, compiled=True):
        '''Instantiates a network graph for the c51 implementation.
        
        Arguments:
        s_shape: tuple of ints with the shape of input state.
        num_a: int, number of available actions.
        num_atoms: int, number of atoms in support.
        alpha: float, learning rate.
        compipled: bool, True for uncompiled target net. False for
                   compiled prediction net. 
        
        Returns:
        t_net: Keras model object, an uncompiled model.
        q_net: Keras model object, a compiled model.
        
        '''
        inputs = Input(shape=s_shape)
        x = Dense(400, activation='relu')(inputs)
        x = Dense(200, activation='relu')(x)
        outputs = [Dense(num_atoms, activation='softmax')(x) for _ in range(num_a)]
        if not compiled:
            t_net = Model(inputs, outputs)
            return t_net
        q_net = Model(inputs, outputs)
        opt = RMSprop(learning_rate=alpha)
        q_net.compile(optimizer=opt, loss='categorical_crossentropy')
        return q_net
    
    def _subtract_mean(self, args, num_a, batchsize):
        '''Final layer module in the dueling architecture.
        
        Arguments:
        args: list of Keras tensor objects of layer outputs.
        num_a: int with the number of available actions.
        batchsize: int.
        
        Returns:
        action values: Keras tensor object, the model output.
        
        '''
        v, A = args
        A_mean = tf.math.reduce_mean(A)
        A_sub_mean = tf.math.subtract(A, A_mean)
        V = tf.broadcast_to(v, [batchsize, num_a])
        return tf.math.add(V, A_sub_mean)
    
    def policy(self, q_net, s, num_a, epsilon, c51=False):
        '''Generating a step given some state.
        
        Arguments: 
        q_net: Keras model object, action value approximator.
        s: numpy array, state agent is currently in.
        num_a: int, number of available actions.
        epsilon: float, e in epsilon greedy policy, exploration rate.
        c51: bool, takes the expected values of c51 outputs if True.
        
        Returns:
        a: numpy array of int, selected action.
        
        '''
        if np.random.random() < epsilon:
            action = np.random.randint(num_a)
        else:
            if c51:
                z = q_net.predict([[s]])
                q = [np.mean(z) for z in z]
            else:
                q = q_net.predict([[s]])[0]
            max_a = []
            for i, j in enumerate(q):
                if j == np.amax(q):
                    max_a.append(i)
            action = np.random.choice(max_a)
        return action
    
    def unpack_experience(self, transitions, batchsize, n_step=False):
        '''Unpacks a block of sampled experience into transition pieces.
        
        Arguments:
        transitions: list of tuples with a batch of sampled transitions.
        batchsize: int.
        n_step: bool, if to handle an n-step transition or not.
                               
        Returns:
        unpacked_transitions: tuple of numpy arrays, batch of transition 
                              components.
        
        '''
        if n_step:
            self.S = []
            self.A = []
            self.R = []
            self.S2 = []
            self.T = []
            for i in range(batchsize):
                self.S.append(transitions[i][:,:1].T)
                self.A.append(transitions[i][:,1:2].T)
                self.R.append(transitions[i][:,2:3].T)
                self.S2.append(transitions[i][:,3:4].T)
                self.T.append(transitions[i][:,4:5].T)
            self.S = np.array(self.S).squeeze()
            self.A = np.array(self.A).squeeze()
            self.R = np.array(self.R).squeeze()
            self.S2 = np.array(self.S2).squeeze()
            self.T = np.array(self.T).squeeze()
        else:
            self.S = np.array([transitions[i][0] for i in range(batchsize)])
            self.A = np.array([transitions[i][1] for i in range(batchsize)])
            self.R = np.array([transitions[i][2] for i in range(batchsize)])
            self.S2 = np.array([transitions[i][3] for i in range(batchsize)])
            self.T = np.array([transitions[i][4] for i in range(batchsize)])
    
    def dqn_targets(self, q_net, t_net, gamma, batchsize):
        '''Computes the TD error target term in the DQN implementation.
        
        Arguments:
        q_net: Keras model object, action value approximator.
        t_net: Keras model object, target model.
        gamma: float, the discount factor.
        
        Returns:
        targets: numpy array with a batch of targets.
        S: numpy array with a batch of states.
        
        '''
        t_nexts = t_net.predict(self.S2)
        t_maxs = [np.amax(t_nexts[i]) for i in range(batchsize)]
        deltas = []
        for i in range(batchsize):
            r = self.R[i]
            t = self.T[i]
            if t:
                deltas.append(r)
            else:
                deltas.append(r + (gamma * t_maxs[i]))
        targets = q_net.predict(self.S)
        self.deltas = deltas
        self.targets = targets
        for i in range(batchsize):
            targets[i, self.A[i]] = deltas[i]
        return self.targets, self.S
        
    def ddqn_targets(self, q_net, t_net, gamma, batchsize):
        '''Computes the target term in the Double DQN implementation.
        
        Arguments:
        q_net: Keras model object, action value approximator.
        t_net: Keras model object, target model.
        gamma: float, the discount factor.
        batchsize: int.
        
        Returns:
        targets: numpy array with a batch of targets.
        deltas: list with TD errors.
        S: numpy array with a batch of states.
        
        '''
        q_nexts = q_net.predict(self.S2)
        q_maxs = [np.argmax(q_nexts[i]) for i in range(batchsize)]
        t_nexts = t_net.predict(self.S2)
        deltas = []
        for i in range(batchsize):
            r = self.R[i]
            t = self.T[i]
            if t:
                deltas.append(r)
            else:
                deltas.append(r + (gamma * t_nexts[i, q_maxs[i]]))
        targets = q_net.predict(self.S)
        for i in range(batchsize):
            targets[i, self.A[i]] = deltas[i]
        return targets, deltas, self.S
    
    def per_targets(self, q_net, t_net, gamma, batchsize, is_w):
        '''Computes the target term in the PER DQN implementation.
        
        Arguments:
        q_net: Keras model object, action value approximator.
        t_net: Keras model object, target model.
        gamma: float, the discount factor.
        is_w: numpy array of floats, importance sampling weights.
        
        Returns:
        targets: numpy array with a batch of targets.
        deltas: list with TD errors.
        S: numpy array with a batch of states.
        
        '''
        q_nexts = q_net.predict(self.S2)
        q_maxs = [np.argmax(q_nexts[i]) for i in range(batchsize)]
        t_nexts = t_net.predict(self.S2)
        deltas = []
        for i in range(batchsize):
            r = self.R[i]
            t = self.T[i]
            if t:
                deltas.append(r)
            else:
                deltas.append(r + (gamma * t_nexts[i][q_maxs[i]]))
        targets = tmp = q_net.predict(self.S)
        deltas2 = []
        for i in range(batchsize):
            targets[i, self.A[i]] = deltas[i]
            d = deltas[i] - tmp[i][self.A[i]]
            deltas2.append(0.5 * d ** 2 if abs(d) < 1.0 else abs(d) - 0.5)
        return targets, deltas2, self.S
    
    def multi_step_targets(self, q_net, t_net, gamma, batchsize, n_steps):
        '''Computes the TD error target term in the n-step implementation.
        
        Arguments:
        q_net: Keras model object, action value approximator.
        t_net: Keras model object, target model.
        gamma: float, the discount factor.
        n_steps: int with the number of n_steps.
        
        Returns:
        targets: numpy array with a batch of targets.
        S_t: numpy array with a batch of states.
        
        '''  
        G_t = []
        S_t = []
        S2_t_n = []
        for i in range(batchsize):
            g = 0
            for j in range(1, n_steps):
                if not self.T[i, j]:
                    g += gamma**j-1 * self.R[i, j]
            G_t.append(g)
            S_t.append(self.S[i, 0])
            S2_t_n.append(self.S2[i, -1])
        S_t = np.array(S_t)
        S2_t_n = np.array(S2_t_n)
        targs_t_n = t_net.predict([S2_t_n])
        targs_max = [np.amax(targs_t_n[i]) for i in range(batchsize)]
        deltas = []
        for i in range(batchsize):
            r = G_t[i]
            t = self.T[i, 0]
            if t:
                deltas.append(r)
            else:
                deltas.append(r + (gamma * targs_max[i]))
        targets = q_net.predict([S_t])
        for i in range(batchsize):
            targets[i, self.A[i, 0]] = deltas[i]
        return targets, S_t
    
    def c51_targets(self, q_net, t_net, num_a, v_min, v_max, num_atoms, gamma, batchsize):
        '''Computes and projects a c51 target distribution.
        
        Arguments:
        t_net: Keras Model object, target model.
        num_a: int, number of available actions.
        v_min: float, minimum in the target support.
        v_max: float, maximum in the target support.
        num_atoms: int, number of atoms.
        gamma: float, discount factor.
        batchsize: int.
        
        Returns:
        targets: numpy array with a batch of targets.
        S_t, numpy array with a batch of states.
        
        '''
        z = q_net.predict(self.S2)
        z_ = t_net.predict(self.S2)      
        Z = np.linspace(v_min, v_max, num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)
        m_probs = [np.zeros((batchsize, num_atoms)) for _ in range(num_a)]
        for i in range(batchsize):
                if self.T[i]:
                    Z_rb = [max(v_min, min(self.R[i], v_max)) for z in Z]
                    B = [(z - v_min) / delta_z for z in Z_rb]
                    m_l = [math.floor(b) for b in B]
                    m_u = [math.ceil(b) for b in B]
                    for j in range(num_atoms):
                        m_probs[self.A[i]][i][m_l[j]] = (m_u[j] - B[j])
                        m_probs[self.A[i]][i][m_u[j]] = (B[j] - m_l[j])
                else:
                    Z_rb = [max(v_min, min(self.R[i] + (gamma * z), v_max)) for z in Z]
                    B = [(z - v_min) / delta_z for z in Z_rb]
                    m_l = [math.floor(b) for b in B]
                    m_u = [math.ceil(b) for b in B]
                    expectations = []
                    for a in range(num_a):
                        expectation = 0
                        for ind, val in enumerate(z[a][i]):
                            expectation += ind * val
                        expectations.append(expectation)
                    z_idx = np.argmax(expectations)
                    for j in range(num_atoms):
                        m_probs[self.A[i]][i][m_l[j]] = z_[z_idx][i][j] * (m_u[j] - B[j])
                        m_probs[self.A[i]][i][m_u[j]] = z_[z_idx][i][j] * (B[j] - m_l[j])
        return targets, self.S
    
class SumTree:
    '''Builds the sum tree structure for PER.
    
    Attributes:
    zero_idx (int): leaf starting indicie in the tree.
    step (int): tracks the number of added leaves.
    leaf_count (int): number of leaves that have non zero values.
    sum_tree (numpy array): sum tree data structure.
    data (numpy array): stored transitions data structure.
        
    '''

    def __init__(self, capacity):
        '''Initializes SumTree.
        
        Arguments:
        capacity (int): number of leaves in the tree.
        
        '''
        self.capacity = capacity
        self.zero_idx = self.capacity - 1
        self.step = 0
        self.leaf_count = 0
        self.sum_tree = np.zeros(((2 * capacity) - 1))
        self.data = np.zeros((capacity), dtype=object)
        
    def get_leaf_count(self):
        '''Gets the number of leaves in the tree.'''
        return self.leaf_count
    
    def get_leaf_sum(self):
        '''Gets the total sum of the leaves.'''
        return self.sum_tree[0]   
    
    def propagate(self, parent, value):
        '''Propagates a leaf value or difference to the root.
        
        Arguments:
        parent (int): parent of the leaf.
        value (float): value to be propagated.
              
        '''
        while (parent - 1) != 0:
            self.sum_tree[parent - 1] += value
            parent = parent // 2
        self.sum_tree[parent - 1] += value
        
    def add_leaf(self, priority, transition):
        '''Adds a leaf to the sum tree and a transition to data.
        
        - If the leaf_count equals the capacity, the leaf
          with the longest duration in the tree will be 
          replaced with the new.
        - If capacity is full, correspondning transition
          in data will be removed and replaced with the new.
        
        Arguments:
        priority (int or float): value to be stored as a leaf.
        transition (tuple): holds (s1, a, r, s2, t).
        
        '''
        # identify the parent to the leaf in question
        parent = (self.zero_idx + self.step + 1) // 2
        # if the tree is of full capacity
        if self.get_leaf_count() == self.capacity:
            # get difference between added and removed leaves
            current_value = self.sum_tree[self.zero_idx + self.step]
            diff = abs(current_value - priority)
            # set if propagated difference should be added or subtracted
            if priority < current_value:
                diff = -diff
            self.propagate(parent, diff)
        # if the tree is not full
        else: 
            self.propagate(parent, priority)
        # insert priority in the tree
        self.sum_tree[self.zero_idx + self.step] = priority
        # insert the transition in the data array
        self.data[self.step] = transition
        # update the step parameter
        if self.step == (self.capacity - 1):
            self.step = 0
        else:
            self.step += 1
        # update leaf count
        if self.leaf_count < self.capacity:
            self.leaf_count += 1
        else: 
            self.leaf_count = self.capacity
            
    def update_leaf(self, priority, idx):
        '''Updates a current leaf.
        
        Arguments:
        priority (float): value to be updated.
        idx (int): indicie of leaf to be updated.
        
        '''
        current_value = self.sum_tree[self.zero_idx + idx]
        diff = abs(current_value - priority)
        if priority < current_value:
            diff = -diff
        parent = (self.zero_idx + idx + 1) // 2
        # propagate difference up to the root node
        self.propagate(parent, diff)
        # replace the leaf value
        self.sum_tree[self.zero_idx + idx] = priority
        
    def get_leaf(self, s):
        '''Sample a leaf given a segment sample.
        
        Arguments:
        segment_sample (int): uniformly sampled integer.
        
        Returns:
        priority (float): the leaf value.
        idx (int): indicie of the leaf.
        transition (tuple): transition corresponding to the leaf.
        
        '''
        idx = 1
        while idx <= self.zero_idx:
            # identify children
            left = idx * 2
            right = left + 1
            # compare s and go left or right
            if s > self.sum_tree[left - 1]:
                s = s - self.sum_tree[left - 1]
                idx = right
            else: 
                idx = left
        priority = self.sum_tree[idx - 1]
        idx = idx - self.zero_idx - 1
        transition = self.data[idx]
        return priority, idx, transition
    

class PrioritizedExperienceReplay:
    '''Implements the prioritized replay.
        
        Attributes:
        sum_tree (class object): the sum tree class.
        max_priority (float): highest priority added to the tree.
    
    '''
    
    def __init__(self,
                capacity,
                batchsize,
                alpha=0.6,
                beta=0.4,
                beta_increment=0.001,
                epsilon=0.001):
        '''Initializes PrioritizedExperienceReplay.
        
        Arguments:
        capacity (int): sum tree size.
        batchsize (int): batchsize.
        alpha (float): priority exponent.
        beta (float): importance sampling exponent.
        beta_increment (float): beta increment rate.
        epsilon (float): priority additive constant. 
        
        '''
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 0.001
        self.batchsize = batchsize
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.
        
    def len_memory(self):
        '''Gives the current size of the memory.'''
        return self.sum_tree.get_leaf_count()

    def store_transition(self, delta, transition):
        '''Defines a priority and stores it in sum tree.

        Arguments:
        delta (float): TD error from training.
        transition (tuple or list): consist of (s1, a, r, s2, t)

        '''
        # define the priority as abs((delta) + epsilon)**alpha 
        priority = (abs(delta) + self.epsilon) ** self.alpha
        # set priority to max
        priority = max(priority, self.max_priority)
        # add the priority to the sum tree
        self.sum_tree.add_leaf(priority, transition)
        # update max_priority
        self.max_priority = max(priority, self.max_priority)
    
    def sample_transition(self):
        '''Samples a batch of transitions based on their proportional priorities.
        
        2. split up the range [0, tot_p] in batchsize equally sized segments.
        3. sample uniformly from each range and obtain a batch of idxs.
        4. obtain a batch of transitions according to sampled idxs.
        5. adjust the beta parameter.
        6. compute segment probabilities and importance sampling weights.
        
        #### 
        RETURNS:
        transitions (list of tuples): sampled transitions of size batchsize.
        idxs (list): indicies of sampled priorities and transitions.
        is_w (list): importance sampling weights
        
        '''
        priorities = []
        idxs = []
        transitions = []
        total_priority = self.sum_tree.get_leaf_sum()
        segment_size = total_priority // self.batchsize
        self.s = segment_size
        for i in range(self.batchsize):
            # sample uniformly from within each segmentself.
            s = np.random.randint(i * segment_size, i * segment_size + segment_size)
            priority, idx, transition = self.sum_tree.get_leaf(s)
            priorities.append(priority)
            idxs.append(idx)
            transitions.append(self.sum_tree.data[idx])
        # adjust beta according to global step
        self.beta = min(1., self.beta + self.beta_increment)
        # compute probabilities and importance sampling weights
        tot_p = self.sum_tree.get_leaf_sum()
        probabilities = priorities / total_priority
        is_w = np.power((self.sum_tree.get_leaf_count() * probabilities) + 1e-12, -self.beta)
        is_w = np.nan_to_num(is_w)
        is_w = is_w / max(is_w)
        if type(transitions[0]) != tuple:
            transitions[0] = transitions[1]
        return transitions, idxs, is_w
    
    def update_priorities(self, deltas, idxs):
        '''Takes TD errors from training and updates priorities.
        
        Arguments:
        deltas (list or tuple): TD errors from training.
        idxs (list): indicies of priorities from current batch.
        
        '''
        for i in range(self.batchsize):
            priority = (abs(deltas[i]) + self.epsilon) ** self.alpha  
            idx = idxs[i]
            self.sum_tree.update_leaf(priority, idx)
            self.max_priority = max(self.max_priority, priority)
            
        
class ExperienceReplay:
    '''Implements an experience memory replay.
        
    Attributes:
    memory (list): the memory data structure.
    
    '''
    
    def __init__(self, capacity, batchsize):
        '''
        Arguments:
        capacity (int): size of memory.
        batchsize (int): batchsize.

        '''
        self.capacity = capacity
        self.batchsize = batchsize
        self.memory = []
        
    def len_memory(self):
        '''Current size of the memory.
        
        Returns:
        int, length of the memory.
        '''
        return len(self.memory)
        
    def store_transition(self, _, transition):
        '''Loads a transition into the memory from the back.
        
        - If the memory is full, the transition with the longest
          duration in the memory will be removed from the front.
        
        Arguments:
        transition (tuple): a transition of (s1, a, r, s2, t)
        
        '''
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)
        
    def sample_transition(self, n_step=False):
        '''Samples a batch of transitions from memory.
        
        Arguments:
        n_step: bool, if True a tuple with n-steps and n_horizon.
        
        Returns:
        (numpy array): sampled transitions.
        
        '''
        samples = []
        if n_step:
            if len(self.memory) < self.capacity:
                for i in range(self.batchsize):
                    n_samples = [self.memory[
                            np.random.randint(len(self.memory))] for _ in range(n_step[0])]
                    samples.append(n_samples)
                transitions = np.array(samples)
                return transitions
            else:
                for i in range(self.batchsize):
                    min_h = (n_step[1]//2) + 10
                    max_h = self.capacity - min_h
                    n_samples = [self.memory[
                            np.random.randint(len(self.memory))] for _ in range(n_step[0])]
                    samples.append(n_samples)
                transitions = np.array(samples)
                return transitions
                
        for i in range(self.batchsize):
            sample = self.memory[np.random.randint(len(self.memory))]
            samples.append(sample)
        transitions = np.array(samples)
        return transitions