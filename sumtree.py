import numpy as np

class SumTree:
    """
    This class represents a sum tree that is used for sampling replay experiences over
    a probability distribution defined by the error observed for each experience.
    
    Attributes
    ----------
    capacity : int
        The maximum number of experiences that are stored. This also represents
        the number of leaf nodes in the sum tree.
        
    tree : np.array
        This is an array implementation of a tree. Each node in the tree contains the
        sum of its children. The leaf nodes contain the error observed from each replay
        experience.
        
    data : np.array
        This array contains the replay experiences observed by the agent. The 0th element
        represents the leftmost leaf node in the tree, the 1st element element represents
        the second leaf node, and so on.
        
    size : int
        The number of experiences currently stored in the sum tree.
        0 <= size < capacity
        
    data_ptr : int
        The data pointer points to the ith index of the data array that will be overwritten
        on the next push to memory.
        0 <= data_ptr < capacity
    """
    
    def __init__(self, capacity):
        """
        Initializes the sum tree object.
        
        Parameters
        ----------
        capacity : int
            The maximum number of experiences that can be stored in the sum tree.
            capacity > 0
        """
        assert capacity > 0 and isinstance(capacity, int), 'capacity should be a positive integer'
        
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.data_ptr = 0

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Returns the number of experiences currently stored in the sum tree.
        """
        return self.size
        
    def retrieve(self, idx, s) -> int:
        """
        A recursive function that retrives the index of the experience in the sum tree
        based on the probability distrubtion.
        
        Parameters
        ----------
        idx : int
            The index of the current node being observed in the sum tree.
        
        s : float
            A randomly generated float used to sample an experience from the sum tree.
            
        Returns
        -------
        int
            Returns the index of the experience in the sum tree such that based on the
            probability distribution and a randomly generated number s.
        """
        left_child = 2 * idx + 1
        right_child = left_child + 1
        
        if left_child >= len(self.tree):
            return idx
        elif s <= self.tree[left_child]:
            return self.retrieve(left_child, s)
        else:
            return self.retrieve(right_child, s - self.tree[left_child])
        
    def get_total_priority(self) -> float:
        """
        Returns
        -------
        float
            Returns the sum of priorities for all the experiences stored in the sum tree.
        """
        return self.tree[0]
    
    def add(self, priority, experience):
        """
        Adds an experience and its priority to the sum tree and updates the sums in the
        sum tree.
        
        Parameters
        ----------
        priority : float
            The priority of the experience.
            
        experience : namedtuple
            The experience to be inserted to the sum tree
        """
        idx = self.data_ptr + self.capacity - 1

        self.data[self.data_ptr] = experience
        self.update(idx, priority)

        self.data_ptr += 1
        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

        if self.size < self.capacity:
            self.size += 1
            
    def update(self, idx, priority):
        """
        Updates the nodes in the sum tree to have the correct sums as a result
        of adding the new experience.
        
        Parameters
        ----------
        idx : int
            The index where the new experience will be stored in the sum tree.
            
        priority : float
            The priority of the experience.
        """
        change = priority - self.tree[idx]
        self.propagate(idx, change)
        
    def propagate(self, idx, change):
        """
        A recursive helper function for the update function. Recursively updates
        the sums in the sum tree from the leaves to the root.
        
        Parameters
        ----------
        idx : int
            The index of the current node subject to a change in its sum.
            
        change : float
            The change applied to the sum for the current node subject to change.
        """
        self.tree[idx] += change
        if idx > 0:
            parent = (idx - 1) // 2
            self.propagate(parent, change)
            
    def get(self, s) -> tuple:
        """
        Retrieves the index of an experience based on the probability distribution
        and a randomly generated number s.
        
        Parameters
        ----------
        s : float
            A randomly generated number.
            0 <= s < sum(experience priorities)
            
        Returns
        -------
        tuple
            A tuple containing the index of the experience, its priority, and
            the experience tuple containing the state, action, reward, and next state.
        """
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])