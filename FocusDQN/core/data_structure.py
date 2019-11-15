import numpy as np
import collections



# --------------------------------------------
# Data Structure.
# --------------------------------------------

class GridNode:
    r'''
        Basic cell of the tree-structure.
    '''

    def __init__(self,
                 bbox,
                 neighbour,
                 shape,
                 reward,
                 clazz
                 ):
        r'''
            Initialization.

        Parameters:
            bbox: The bounding-box coordinate of this grid.
            neighbour: The neighbour coordinate of this grid.
            shape: The shape of this grid.
            reward: The reward of this grid. Which indicates how many reward that agent can
                obtain when prediction is correct.
            clazz: The real class of this grid. It will be "Intercession" when this grid contains
                more than 2 class.
        '''

        # Check validity of arguments.
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise TypeError('The bbox should be a 4-D tuple !!!')
        if not isinstance(neighbour, tuple) or len(neighbour) != 4:
            raise TypeError('The neighbour should be a 4-D tuple !!!')
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError('The shape should be a 2-D tuple !!!')

        # Construct the data for node. A dictionary.
        data = {
            'bbox': bbox,
            'neighbour': neighbour,
            'shape': shape,
            'Q_vals': None,
            'reward': reward,
            'accu_value': None,
            'clazz': clazz,
            'is_fake': False
        }

        # The node data.
        self._data = data

        # The children nodes list.
        self._children = []

        # The parent node.
        self._parent = None

        # Finish initialization.
        return

    @property
    def bbox(self):
        return self._data['bbox']

    @property
    def neighbour(self):
        return self._data['neighbour']

    @property
    def shape(self):
        return self._data['shape']

    @property
    def Q_vals(self):
        r'''
            The Q values from DQN that it judge the class of this node.
        '''
        return self._data['Q_vals']

    @Q_vals.setter
    def Q_vals(self, q_values):
        r'''
            Set the Q values for this node.
        :param q_values: The Q values predicted by DQN.
        '''
        self._data['Q_vals'] = q_values

    @property
    def reward(self):
        return self._data['reward']

    @property
    def accu_value(self):
        r'''
            The accumulated reward value of the tree that setup with this grid as root node.
                That is, we will used as the "Real Q values" in training phrase.
        '''
        return self._data['accu_value']

    @accu_value.setter
    def accu_value(self, accumulated_rewards):
        r'''
            Set the accumulated rewards for this node.
        :param accumulated_rewards:
        '''
        self._data['accu_value'] = accumulated_rewards

    @property
    def clazz(self):
        return self._data['clazz']

    @property
    def is_fake(self):
        r'''
            Indicating this node is a "Fake" node. That is, the parent of this node is wrongly
                judge by the DQN in training phrase. This node is a "Should not exist" node.
        '''
        return self._data['is_fake']

    @is_fake.setter
    def is_fake(self, fake_flag):
        r'''
            Set the fake flag for this node.
        :param fake_flag: Fake flag.
        '''
        self._data['is_fake'] = fake_flag

    def is_leaf(self):
        r'''
            Judge whether this node is leaf or not.
        '''
        return len(self._children) == 0

    def add_child(self, child):
        r'''
            Add child to this node.

        Parameter
            child: The child node.
        '''

        # Check the validity.
        if not isinstance(child, GridNode):
            raise TypeError('The child must be GridNode !!!')

        # Add child.
        self._children.append(child)

        # Finish.
        return

    @property
    def children(self):
        return self._children

    def clear_children(self):
        r'''
            Clear the children list of this node.
        '''

        # Clear the children list.
        self._children.clear()

        # Finish.
        return

    @property
    def parent(self):
        return self._parent

    def spec_parent(self, parent):
        r'''
            Assign the parent for this node.

        Parameter
            parent: The parent node.
        '''

        # Check the validity.
        if not isinstance(parent, GridNode):
            raise TypeError('The parent must be GridNode !!!')

        # Specify the parent of this node.
        self._parent = parent

        # Finish.
        return


# Task-specific data structure.
class RegionTree:
    r'''
        The tree-structure for manage regions (grids) for given image (proposal region).
    '''

    def __init__(self,
                 proposal,
                 eval_dim,
                 split_enval,
                 split_func,
                 is_train,
                 clazz_func=None,
                 reward_func=None,
                 search_algo='BFS'):
        r'''
            The initialization method. Mainly to generate the root and construct (produce) the
                initial nodes from root. What's more pre-construct the whole tree in the
                training phrase.

        Parameters
            proposal: The region that contain the foreground object. Used as the initial bbox.
            eval_dim: This is actually the action dimension of DQN agent. Used to specify the
                dimension of rewards.
            split_enval: The numeric value of the action that indicating "Split Children".
            split_func: The "Split Bounding-box" function. Used to generate the children reigons.
            is_train: The flag indicating "Training" phrase or not.
            clazz_func: The "Class Assignment" function. Used to assign the true class for
                given node (grid).
            reward_func: The "Reward Calculation" function. Used to compute reward for given
                node (grid).
            search_algo: The search algorithm (mode) used in generating next nodes batch.
                Available algorithms including: "BFS" and "DFS".
        '''

        # Check validity of arguments.
        if not isinstance(proposal, tuple) or len(proposal) != 4:
            raise TypeError('The proposal should be a 4-D tuple !!!')
        if not isinstance(eval_dim, int):
            raise TypeError('The eval_dim must be a Python.int !!!')
        if not isinstance(split_enval, int):
            raise TypeError('The split_enval should be an int !!!')
        if not callable(split_func):
            raise TypeError('The split_func should be a function !!!')
        if is_train and not callable(clazz_func):
            raise TypeError('The clazz_func should be a function !!!')
        if is_train and not callable(reward_func):
            raise TypeError('The reward_func should be a function !!!')
        valid_SAs = ['BFS', 'DFS']  # The valid search algorithms.
        if search_algo not in valid_SAs:
            raise Exception('The search algorithm should be in ({})'.format(valid_SAs))

        # The action dimension. It is used to specify the dimension of
        #   Q values and rewards.
        self._eval_dim = eval_dim

        # The numeric value of the action should be splited.
        self._split_enval = split_enval

        # The sub-grid split function.
        self._split_func = split_func

        # The training flag.
        self._is_train = is_train

        # Declare the function according to the training flag.
        if is_train:
            # The class assignment function.
            self._clazz_func = clazz_func
            # The reward calculation function.
            self._reward_func = reward_func
        else:
            self._clazz_func = lambda x: None
            self._reward_func = lambda x, y: None

        # The search algorithm (mode) for traverse tree.
        self._search_algo = search_algo

        # The tree. That is, the root node in the very beginning.
        pp_shape = (proposal[1]-proposal[0], proposal[3]-proposal[2])   # The proposal region shape.
        self._tree = GridNode(bbox=proposal,
                              neighbour=proposal,
                              shape=pp_shape,
                              reward=self._reward_func(proposal, pp_shape),
                              clazz=self._split_enval)
        # Add "Q values" attribute for root node (grid).
        root_RQval = np.zeros(2, dtype=np.float32)
        root_RQval[0] = 1.0     # Normalized value.
        self._tree.Q_vals = root_RQval
        # The process deque. That is, the next nodes (batch) waiting for processing
        #   is pop from this deque.
        self._proc_deque = collections.deque()
        # Add the initial nodes (grids).
        self._add_children_2node(parent=self._tree,
                                 proc_deque=self._proc_deque,
                                 arg=True)

        # Pre-construct the tree and calculate the accumulation reward
        #   values for whole tree.
        if is_train:
            self._pre_cons_GTtree()
            self._calculate_accu_values()
            # What's more, Initialize the "Real Q values".
            self._Real_Qval = self._tree.accu_value

        # Declare the current process nodes (grids) list.
        self._cur_pnodes = []

        # The flags used to control the execution logic.
        self._logic_flags = {
        }

        # Finish initialization.
        return


    def _add_children_2node(self, parent, proc_deque, wrong_split=False, arg=None):
        r'''
            Add children to given node. That is, we will firstly split current grid
                region of given node into sub-grids according to given split size,
                and then add the sub-grids as children.

            Meanwhile, add the children of given node into the deque waiting for
                processing. Coz we only add the inference, so it will not cause OOM.

        Parameters:
            parent: The given node, that is, the node we want to add children.
            proc_deque: The deque for tree which used to containing the nodes
                waiting for processing.
            wrong_split: Indicating the split operation of parent node is wrong or not.
            arg: The additional arguments. Mainly used for the @Variable{Split function}.
        '''

        # The parent can not add children if the given grid (node) is actually a pixel.
        #   (The width and height are both 1.)
        if parent.shape[0] == 1 and parent.shape[1] == 1:
            return

        # Firstly split parent grid into sub-grids.
        sub_grids_data = self._split_func(parent.bbox, arg)

        # Then recursively add the child to the parent node.
        for sub_grid in sub_grids_data:
            bbox, neighbour, shape = sub_grid
            # Compute reward and clazz.
            clazz = self._clazz_func(bbox)
            reward = self._reward_func(bbox, shape)
            # Set rewards to zeros if wrong split in training phrase.
            if self._is_train and wrong_split:
                reward[:] = 0.
            # New the child node.
            child = GridNode(bbox=bbox,
                             neighbour=neighbour,
                             shape=shape,
                             reward=reward,
                             clazz=clazz)
            # The split of parent is a wrong judgement, so this child node
            #   is a fake node.
            if wrong_split:
                child.is_fake = True
            # Add relation between parent and children node.
            parent.add_child(child)
            child.spec_parent(parent)
            # Meanwhile add the child into the deque waiting for processing.
            proc_deque.append(child)

        # Finish add children to node.
        return


    def _post_traverse_cal(self, node):
        r'''
            Post (order) traverse the given sub-tree (node) to calculate the "Accumulation Reward" of
                each node (including children, but excluding the reward of itself) of the given sub-tree.
                Note that, this is a recursive process.

        Parameter:
            node: The given node.

        Return:
            The "Accumulation Reward" value of given node (excluding the reward of itself).
        '''

        # Check validity of calling.
        if not self._is_train:
            raise Exception('This function can only be called in the \"Training\" phrase !!!')

        # The accumulation reward is actually the reward of this node
        #   if it is a "Leaf".
        if node.is_leaf():
            node.accu_value = 0.    # Simple leaf do not have accumulation reward.
            return node.reward[node.clazz]

        # The accumulated rewards list.
        accu_values_list = []
        # Traverse the children to get accumulated rewards for each child, respectively.
        for child in node.children:
            child_accu_val = self._post_traverse_cal(child)
            accu_values_list.append(child_accu_val)
        # Then sum the all values to get total children accumulation reward.
        total_children_av = sum(accu_values_list)

        # Assign the "Accumulation Reward" for given node (excluding reward of itself).
        node.accu_value = total_children_av

        # The final accumulation reward for given node is the sum of two part:
        #   "Total accumulation reward", and "Reward of given node".
        accu_val_4node = node.reward[node.clazz] + total_children_av

        # Finish. The node accumulation reward.
        return accu_val_4node


    def _calculate_accu_values(self):
        r'''
            Calculate the accumulation reward for the whole tree.
        '''

        # Check validity of calling.
        if not self._is_train:
            raise Exception('This function can only be called in the \"Training\" phrase !!!')

        # Post traverse the root node is OK.
        self._post_traverse_cal(self._tree)

        # Finish.
        return


    def _pre_cons_GTtree(self):
        r'''
            Pre-construct the "Ground Truth" tree. This method can only be
                called in training phrase.
        '''

        # Check validity of calling.
        if not self._is_train:
            raise Exception('This function can only be called in the \"Training\" phrase !!!')

        # Duplicate the @Variable{self._proc_deque} as the temp deque which
        #   we use to pre-construct the "Ground Truth" tree.
        temp_deque = self._proc_deque.copy()
        # temp_deque = copy.deepcopy(self._proc_deque)

        # Recursively process the node (grid) from the deque.
        while len(temp_deque) != 0:
            # Get node.
            node = temp_deque.popleft()
            # Get bounding-box and pass through clazz function to get correct action.
            bbox = node.bbox
            act = self._clazz_func(bbox)
            # Split current node into sub-grids if it is a "Split Node".
            if act == self._split_enval:
                self._add_children_2node(parent=node,
                                         proc_deque=temp_deque)

        # Release the memory.
        del temp_deque

        # Finish the generation of "Ground Truth" tree.
        return


    def next_proc_batch(self, batch_size):
        r'''
            Get the next nodes (grids) batch waiting for process with the given
                batch size. Note that, it will return the batch whose size is
                smaller than the given batch size if it has not enough nodes.

        Parameter:
            batch_size: The given batch size.

        Return:
            The next nodes batch waiting for process with given batch size.
                It is possible to return an insufficient number of nodes.
        '''

        # Check the validity of execution logic.
        if len(self._cur_pnodes) != 0:
            raise Exception('The current nodes still not be processed, can not'
                            'switch to next batch !!!')

        # Clear the current process nodes list.
        self._cur_pnodes.clear()

        # Use different pop method according to the search algorithm.
        if self._search_algo == 'BFS':
            pop_func = self._proc_deque.popleft
        elif self._search_algo == 'DFS':
            pop_func = self._proc_deque.pop
        else:
            raise Exception('Invalid Situation !!! Unknown search algorithm !!!')

        # Return remnant when process deque has not enough nodes.
        batch_size = min(batch_size, len(self._proc_deque))
        # Add the current waiting for process node into list.
        for _ in range(batch_size):
            pnode = pop_func()
            self._cur_pnodes.append(pnode)

        # Finish.
        return self._cur_pnodes


    def apply_judgement(self, judge_list):
        r'''
            Apply the judgement by DQN (or Random) to the tree. The operation mainly
                includes: Set Q values, Split or Prune and Calculate rewards.

        Parameter:
            judge_list: The judgement list output by DQN or Random selection.
                The element of list is consisted of (action_index, Q_values).
                Note that, Q values may be None.

        Return:
            The reward list of given judgement list. The length is equal to judge list.
        '''

        # Check the validity of execution logic.
        if len(self._cur_pnodes) == 0:
            raise Exception('The current nodes is already processed, can not apply '
                            'judgement again. Should switch to next batch !!!')

        # Check the validity of judgement list.
        if not isinstance(judge_list, list) or len(judge_list) != len(self._cur_pnodes):
            raise Exception('The judgement list should be of Python.list !!! '
                            'What\' more, the length of judgement list ({}) must equal to current'
                            'process nodes ({})'.format(len(judge_list), len(self._cur_pnodes)))
        if not isinstance(judge_list[0], tuple) or len(judge_list[0]) != 2:
            raise Exception('The element in judgement list must be a Python.tuple containing '
                            'two elements !!!')

        # The reward list. (Respect to the judgement list)
        J_reward_list = []

        # Recursively apply the judgement to the node waiting for processing.
        for judge, node in zip(judge_list, self._cur_pnodes):
            # Get prediction of class and Q values of given nodes (grids).
            pred_cls, Q_vals = judge

            # Execute the action (class prediction).
            if self._is_train:
                # Indicates that this node is a TP "Split", so the judgement is totally right.
                #   But coz we do not change the @Variable{self._proc_deque} when pre-construct
                #   the "Region Tree", the children nodes are not push into the process deque
                #   at that time. So we need to push the children of current node into deque
                #   by ourselves. Otherwise will cause "Can not split into grids" error.
                if pred_cls == self._split_enval and node.clazz == self._split_enval:
                    for n_child in node.children:
                        self._proc_deque.append(n_child)
                # Indicates that this node do not need "Split", so it is a "Fake Node".
                #   And its all children is fake. Add children.
                if pred_cls == self._split_enval and node.clazz != self._split_enval:
                    self._add_children_2node(parent=node,
                                             proc_deque=self._proc_deque,
                                             wrong_split=True)
                # Indicate that this node need "Split". But the prediction do not
                #   split, so "Prune" the children of this node.
                elif pred_cls != self._split_enval and node.clazz == self._split_enval:
                    node.clear_children()

                # Use "Accumulation Rewards" and "Node Reward" to calculate
                #   the "Real Q Values" for given node (Exploration phrase).
                prune_loss = 0.     # The "Prune Loss".
                if pred_cls != self._split_enval and node.clazz == self._split_enval:
                    prune_loss = node.accu_value
                # The Real Q values for given node consists of:
                #   Successive_RQval + reward_4cur_node - Prune_loss
                node_real_Qval = self._Real_Qval - node.reward[node.clazz] + node.reward - prune_loss
                # Set the Q values for given node. (Generate the binary "Prune" Q values)
                acts_Q_vals = node_real_Qval / self._tree.accu_value  # Normalized values
                pbi_Q_vals = np.zeros(2, dtype=np.float32)
                pbi_Q_vals[0] = acts_Q_vals[self._split_enval]
                pbi_Q_vals[1] = np.max(acts_Q_vals[np.where(
                    np.arange(self._eval_dim) != self._split_enval)])   # pbi_Q_vals[1] = np.max(acts_Q_vals[:-1])
                node.Q_vals = pbi_Q_vals  # node.Q_vals = node_real_Qval / self._tree.accu_value  # Normalized values
                # Reset local variable.
                self._Real_Qval = node_real_Qval[pred_cls]

                # Debug
                if Q_vals is not None:
                    node.Q_vals = Q_vals
                # Debug

                # Calculate the reward for given judgement of given node. And then
                #   add into list.
                j_reward = node.reward[pred_cls] / self._tree.accu_value  # Normalized values
                J_reward_list.append(j_reward)

            else:
                # Validity check.
                if Q_vals is None:
                    raise Exception('Must give the Q values when in Inference phrase !!!')
                if len(Q_vals) != 2:
                    raise Exception('The predicted Q values dimension must be 2 !!!')
                # Set the Q values of given node if given the DQN Q values (Exploitation phrase).
                node.Q_vals = Q_vals

                # The "Inference" phrase. Only need to normally add the children
                #   to given node when the predicted class equal to "Split class".
                if pred_cls == self._split_enval:
                    self._add_children_2node(parent=node,
                                             proc_deque=self._proc_deque)

        # Clear the current process nodes list.
        self._cur_pnodes.clear()

        # Finish judgement apply. And return the reward list.
        return J_reward_list


    def finished(self):
        r'''
            Return whether this tree is finished processing or not.
        '''

        # Flag.
        flag = False

        # Finished when process deque and current pnodes are both empty.
        if len(self._proc_deque) == 0 and len(self._cur_pnodes) == 0:
            flag = True

        # Return the flag.
        return flag

