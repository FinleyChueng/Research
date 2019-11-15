import numpy as np
import logging

# Framework-relative abstraction.
class DQN:

    def __init__(self,
                 log_level=logging.INFO,
                 log_dir=None,
                 logger_name='DQN'
                 ):

        # config logger.
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # handler not exists.
        if not logger.handlers:
            # input log into screen.
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter('%(name)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            # input log into local file.
            if log_dir is not None:
                fh = logging.FileHandler(log_dir)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        self._logger = logger
        # clear the previous content.
        with open(log_dir, 'w') as f:
            f.write('')

    def train(self, epochs, max_iter):
        r'''
            Start to train the DQN agent in fixed epochs.

        Parameters:
            epochs: Indicates the training epochs.
            max_iter: Indicates the iterations for one epoch.
        '''
        raise NotImplementedError

    def test(self, iteration, is_validate):
        r'''
            Start to test the DQN agent in fixed epochs.

        Parameters:
            iteration: Indicates the iterations for one epoch in testing phrase.
            is_validate: Indicates whether it's "Validate" or real "Test".
        '''
        raise NotImplementedError



#------------------------------------------------
# "Prioritized Replay" relative data structure.
#------------------------------------------------

class SumTree(object):
    r'''
        The data structure used to support "Prioritized Replay", which maintains
            the latest probability of "experience" and contains the real data (experience)
            for DQN agent. (That is, this sum-tree will dynamically updating the
            probability of relative experience during "Training" stage.)
    '''

    def __init__(self,capacity):
        r'''
            The initialization method of this class.

        -------------------------------------------------------------------------------
        Parameters:
            capacity: The capacity of the "Replay Memory". (Here is just used to
                calculate the size of sum-tree.)
        '''

        # Normal initialization.
        self.capacity = capacity

        # The pointer used to indicates the next store cell within sum-tree.
        self.data_pointer = 0
        # Initialize the sum-tree, especially chains the size for priority tree.
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure.
        # Initialize the real data repository. That is, a array to really store the experience.
        self.data = np.zeros(capacity,dtype=object)

    def __len__(self):
        r'''
            Pre-defined method to return the current length of data repository.
        '''
        return np.sum(self.data != 0)

    def add(self,p,data):
        r'''
            Push experience into repository or replace the oldest experience when
                the size of memory reaches the threshold.

            Note that:
                    The real data (exp) is store in sum-tree as the leaf node,
                the middle node (include the root node) is the sum of left child
                and right child.
                    And we store the sum-tree into one-dimension (1D) array from
                top to bottom, from left to right.

        --------------------------------------------------------------------------------
        Parameters:
            p: The probability of new-come experience.
            data: That is, the experience.
        '''

        # Calculate the correct index (position) for new-come experience to store.
        tree_idx = self.data_pointer + self.capacity - 1    # The behind is the leaf.
        # Store the experience.
        self.data[self.data_pointer] = data
        # Update the value (sum of children) of relative middle-layer node.
        self.update(tree_idx,p)

        # Increase the store pointer.
        self.data_pointer += 1
        # If the store pointer beyond the capacity (of sum-tree), then replace oldest exp.
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0   # replace when exceed the capacity, i.e., reset the pointer.

    def update(self,tree_idx,p):
        r'''
            Update the probability of experience (leaf node) in specific position, meanwhile
                updates the value (sum of children) of relative upper layer node.

        -----------------------------------------------------------------------------------------
        Parameters:
            tree_idx: The index of experience (leaf node) to be updated.
            p: The new probability of specific experience (leaf node).
        '''

        # Compute the change (delta) between the previous and new probability, which is lately
        #   used to update the value of relative upper layer nodes.
        change = p - self.tree[tree_idx]
        # Update the probability of specific experience (leaf node).
        self.tree[tree_idx] = p

        # Recursively update the value of relative upper nodes.
        while tree_idx != 0:    # "equals 0" means finished all relative nodes (including root).
            tree_idx = (tree_idx - 1) // 2  # compute the parent node of current child node.
            self.tree[tree_idx] += change   # update the value (sum), simply plus the change.

    def get_leaf(self,v):
        r'''
            Get the experience (leaf node) whose probability (value) exactly contains a certain (fixed)
                number. (That is, the sum of values of previous leaf nodes and current leaf node will
                exactly larger than this number, then we regard this leaf node as the one exactly
                contains the specific number.)

        ---------------------------------------------------------------------------------------------------
        Parameters:
            v: The certain (fixed) number used to select the experience (leaf node).

        ---------------------------------------------------------------------------------------------------
        Return:
            The experience (leaf node) exactly contains the specific constant.
        '''

        # From root node to gradually find the satisfying leaf node.
        parent_idx = 0
        # Gradually search.
        while True:
            # check the children of current node.
            cl_idx = 2 * parent_idx + 1 # left
            cr_idx = cl_idx + 1 # right
            # if index of child is larger than the size of sum-tree,
            #   that means we found the satisfying leaf node.
            if cl_idx >= len(self.tree):
                # return the index of leaf node.
                leaf_idx = parent_idx
                break
            # otherwise children are middle layer node, go on.
            else:
                # if the constant smaller than the value of left child, then go left branch.
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                # otherwise go right branch.
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        # Finally we found the tree index of satisfying leaf node, but we should translate it into
        #   index for data repository. Note that, the former (capacity - 1) cells of sum-tree store
        #   the upper layer nodes, so the real index for data repository is [tree_index - (capacity - 1)].
        data_idx = leaf_idx - self.capacity + 1

        # Return the tuple of (leaf_index, probability, real_data). -- The leaf_index and probability
        #   may be used to update priority for current leaf.
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        r'''
            Get the total probability of this sum-tree.

        ----------------------------------------------------
        Return:
            The total probability of this sum-tree.
        '''

        return self.tree[0]  # the root


class PrioritizedPool(object):
    r'''
        The real data structure implements the "Prioritized Replay", including prioritized sample,
            update priority and so on. This class packages the @class{SumTree}.
    '''

    def __init__(self, capacity):
        r'''
            The initialization method of this class.

        ---------------------------------------------------
        Parameters:
            capacity: The capacity of experience memory.
        '''

        # Normal initialization.
        self.tree = SumTree(capacity)

        # Other parameters used in "Prioritized Replay".
        self.epsilon = 1e-9  # small amount to avoid zero priority  raw: 0.01
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

    def __len__(self):
        r'''
            Pre-defined method to return the current length of "Prioritized Pool".
        '''
        return len(self.tree)

    @property
    def size(self):
        r'''
            Return the current size (length) of "Prioritized Pool".
        '''
        return len(self.tree)

    def store(self, transition):
        r'''
            Store the new-come experience into memory.

        ------------------------------------------------------
        Parameters:
            transition: That is, the experience.
        '''

        # Set the max probability for new-come experience.
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        # The initial situation, set probability to "1" when max probability is "0"
        if max_p == 0:  # that means there is no exp in memory.
            max_p = self.abs_err_upper
        # Add (push) new-come experience into memory.
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self,n):
        r'''
            Sample the batch of "n" of experience according to the probability. -- "Prioritized Replay"

        ---------------------------------------------------------------------------------------------------
        Parameters:
            n: The batch size.

        ----------------------------------------------------------------------------------------------------
        Return:
            The n-batch of experience sampled from memory using "Prioritized Replay", including index
                and ISWeight.
        '''

        # Initialize the variables.
        b_idx = np.empty((n,), dtype=np.int32)
        # b_memory = np.empty((n, self.tree.data[0].size))
        b_memory = []
        ISWeights = np.empty((n,))

        # Separate the total probability into n-segments. Compute the size of segment.
        pri_seg = self.tree.total_p / n

        # Increase beta a little each "Sampling" stage. Note that there is a upper bound ("1").
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # Get the minimal probability of sum-tree (experience), and do normalization.
        min_prob = np.min(self.tree.tree[-self.tree.capacity: len(self.tree)-1+self.tree.capacity]) # for later calculate ISweight

        # his = []    # DEBUG

        # Sample the n-batch experiences, we have to use loop due to the dependence of probability.
        for i in range(n):
            # Note that, we do uniform sample in each segment, so here should calculate the bound.
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            # Use random number (constant) to select (experience) leaf.
            idx, p, data = self.tree.get_leaf(v)
            # Do normalization to probability of leaf.
            prob = p

            # his.append((idx, p))   # DEBUG

            # Compute the ISweight for each leaf (exp). Note that, ISWeight is a (n*1) matrix.
            ISWeights[i] = np.power(prob/min_prob, -self.beta)
            # Record the index (latter used to update probability) and real data of leaf (exp).
            b_idx[i] = idx
            b_memory.append(data)

        # print('The priority is: {}'.format(his))    # DEBUG

        return b_idx, b_memory, ISWeights   # Return the exp tuple.

    def batch_update(self, tree_idx, abs_errors):
        r'''
            Update probability for leafs (exps) of batch. Note that, the update method of
                sum-tree can only update single leaf.

        -----------------------------------------------------------------------------------------
        Parameters:
            tree_idx: The batch of tree index of leafs (exps) wait for updating probability.
            abs_errors: The batch of "absolute TD-errors", which are later used as new
                probability for (specific) leafs (exps).
        '''

        # Convert to abs and avoid 0. ("0" will lead to "NaN")
        abs_errors += self.epsilon
        # Avoid the TD-error beyond the upper bound ("1")
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # Calculate probability according to the formula.
        ps = np.power(clipped_errors, self.alpha)

        # his = []    # DEBUG

        # Update probability for each leaf (exp) recursively.
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

        #     his.append((ti, p))     # DEBUG
        # print("The updated priority is {}".format(his))     # DEBUG
