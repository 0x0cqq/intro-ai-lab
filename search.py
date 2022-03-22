"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from time import sleep
import util

#######################################################
#            This portion is written for you          #
#######################################################

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A example of heuristic function which estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial. You don't need to edit this function
    """
    return 0

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # get the initial state
    initial_state = problem.getStartState()

    def dfs(now_state, visited):
        visited = visited + [now_state]
        # now_state: current state
        # last_state: last state, we need to avoid this state when searching for next state
        if(problem.isGoalState(now_state)):
            # if the current state is the goal state, return the path
            from game import Actions
            return [Actions.vectorToDirection((0,0))]
        else:
            # if the current state is not the goal state,
            # expand the current state and get the next state
            next_states = problem.expand(now_state)
            # recursively call dfs to get the path
            for next_state in next_states:
                if(next_state[0] in visited):
                    continue
                path = dfs(next_state[0], visited)
                # vaild path
                if path != None:
                    return [next_state[1]] + path
            return None
    # call dfs on the initial state
    return dfs(initial_state, [])

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    def bfs():
        # queue: (state, path, distance)
        from util import Queue
        q = Queue()
        # visited: states
        visited = []
        # add initial state into queue
        q.push((problem.getStartState(), [], 0))
        while(not q.isEmpty()):
            now_state = q.pop()
            # if the current state is the goal state, return the path
            if(problem.isGoalState(now_state[0])):
                return now_state[1]
            if(now_state[0] in visited):
                continue
            visited.append(now_state[0])
            # if the current state is not the goal state,
            # expand the current state and get the next state
            next_states = problem.expand(now_state[0])
            # add next state into queue
            for next_state in next_states:
                q.push((next_state[0], now_state[1] + [next_state[1]], now_state[2] + 1))
        return None
    
    return bfs()

def uniformCostSearch(problem):
    """Search the node of least cost from the root."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue


    def ucs():
        q = PriorityQueue()
        # postion, actions, total cost, visited positions
        q.update((problem.getStartState(), [], 0, []), 0)
        visited = {}
        while(not q.isEmpty()):
            now_state = q.pop()
            # print(now_state)
            # sleep(1)
            if(problem.isGoalState(now_state[0])):
                return now_state[1]
            if(now_state[0] in visited and visited[now_state[0]] <= now_state[2]):
                continue
            visited[now_state[0]] = now_state[2]
            next_states = problem.expand(now_state[0])
            for next_state in next_states:
                # add next state into queue
                q.update(
                    (next_state[0], 
                     now_state[1] + [next_state[1]], 
                     now_state[2] + problem.getActionCost(now_state[0], next_state[1], next_state[0])),
                     now_state[2] + problem.getActionCost(now_state[0], next_state[1], next_state[0]))
        return None
    
    return ucs()

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    def astar():
        q = PriorityQueue()
        q.update((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
        visited = []
        while(not q.isEmpty()):
            now_state = q.pop()
            if(problem.isGoalState(now_state[0])):
                return now_state[1]
            if(now_state[0] in visited):
                continue
            visited.append(now_state[0])
            next_states = problem.expand(now_state[0])
            for next_state in next_states:
                if(next_state[0] in visited):
                    continue
                # add next state into queue
                q.update(
                    (next_state[0], 
                     now_state[1] + [next_state[1]], 
                     now_state[2] + problem.getActionCost(now_state[0], next_state[1], next_state[0])),
                     now_state[2] + problem.getActionCost(now_state[0], next_state[1], next_state[0]) + heuristic(next_state[0], problem))
        return None

    return astar()