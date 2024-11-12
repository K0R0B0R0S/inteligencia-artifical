# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

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

    def getStartNode(self):
        """Inicializa o nó inicial para o problema de busca."""
        return {'STATE': self.getStartState(), 'PATH-COST': 0}

    def getActionSequence(self, node):
        """Obtém a sequência de ações do estado inicial até o nó atual."""
        actions = []
        while node['PATH-COST'] > 0:
            actions.insert(0, node['ACTION'])
            node = node['PARENT']
        return actions

    def getChildNode(self, successor, parent_node):
        """Gera um nó filho a partir de um sucessor e do nó pai."""
        return {
            'STATE': successor[0],
            'PARENT': parent_node,
            'ACTION': successor[1],
            'PATH-COST': parent_node['PATH-COST'] + successor[2]
        }

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    node = problem.getStartNode()
    frontier = util.Stack()
    frontier.push(node)

    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()
        
        if node['STATE'] in explored:
            continue

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']):
            return problem.getActionSequence(node)
        
        for successor in problem.expand(node['STATE']):
            child_node = problem.getChildNode(successor, node)
            frontier.push(child_node)

    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    node = problem.getStartNode()
    frontier = util.Queue()
    frontier.push(node)
    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()

        if node['STATE'] in explored:
            continue

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']):
            return problem.getActionSequence(node)
        
        for sucessor in problem.expand(node['STATE']):
            child_node = problem.getChildNode(sucessor, node)
            frontier.push(child_node)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    node = problem.getStartNode()
    fn_total_cost_for_node = lambda a_node: a_node['PATH-COST'] + heuristic(a_node['STATE'], problem=problem)
    frontier = util.PriorityQueueWithFunction(fn_total_cost_for_node)
    frontier.push(node)

    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()       

        if node['STATE'] in explored:
            continue        

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']): 
          return problem.getActionSequence(node)

        successors = problem.expand(node['STATE'])

        for sucessor in successors:
            child_node = problem.getChildNode(sucessor, node)
            frontier.push(child_node)

    return []

def greedySearch(problem: SearchProblem, heuristic=nullHeuristic):
    node = problem.getStartNode()
    fn_total_cost_for_node = lambda a_node: heuristic(a_node['STATE'], problem=problem)
    frontier = util.PriorityQueueWithFunction(fn_total_cost_for_node)
    frontier.push(node)

    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()       

        if node['STATE'] in explored:
            continue        

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']): 
          return problem.getActionSequence(node)

        successors = problem.expand(node['STATE'])

        for sucessor in successors:
            child_node = problem.getChildNode(sucessor, node)
            frontier.push(child_node)

    return []

def depthLimitedSearch(problem: SearchProblem, limit):
    node = problem.getStartNode()
    frontier = util.Stack()
    frontier.push(node)
    
    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()
        if node['STATE'] in explored:
            continue

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']):
            return problem.getActionSequence(node)

        if node['PATH-COST'] < limit:
            for successor in problem.expand(node['STATE']):
                child_node = problem.getChildNode(successor, node)
                frontier.push(child_node)
    
    return None

def iterativeDeepeningSearch(problem: SearchProblem, max_depth = 100):
    depth = 0
    while True:
        if max_depth is not None and depth > max_depth:
            return []
        
        result = depthLimitedSearch(problem, depth)
        if result is not None:
            return result
        depth += 1

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
gs = greedySearch
ids = iterativeDeepeningSearch
