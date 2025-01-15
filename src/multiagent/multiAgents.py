# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.


        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        #Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        ghostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print('Successor game state:\n', successorGameState)
        # print('Pacman current position: ', newPos)
        # print('oldFood:\n', oldFood)
        # print('newFood:\n', newFood)
        # print('ghostPositions: ', ghostPositions)
        # print('successorGameState.score: ', successorGameState.getScore())
        # print('newScaredTimes: ', newScaredTimes)

        # computa distância para o fantasma mais próximo.
        minDistanceGhost = float("+inf")
        for ghostPos in ghostPositions:
            minDistanceGhost = min(minDistanceGhost, util.manhattanDistance(newPos, ghostPos))

        # se a acao selecionada leva à colisão com o ghost, pontuação é mínima
        if minDistanceGhost == 0:
            return float("-inf")

        # se a acao conduzir para a vitoria, pontuação é máxima
        if successorGameState.isWin():
            return float("+inf")

        score = successorGameState.getScore()

        # incentiva acao que conduz o agente para mais longe do fantasma mais próximo
        score += 2 * minDistanceGhost

        minDistanceFood = float("+inf")
        for foodPos in newFoodList:
            minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

        # incentiva acao que conduz o agente para mais perto da comida mais próxima
        score -= 2 * minDistanceFood

        # incentiva acao que leva a uma comida
        if(successorGameState.getNumFood() < currentGameState.getNumFood()):
            score += 5

        # penaliza as acoes de parada
        if action == Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.    

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        minimax = self.minimax(gameState, agentIndex=0, depth=self.depth)
        return minimax['action']

    def minimax(self, gameState, agentIndex=0, depth='2', action=Directions.STOP):
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0: 
            depth = depth-1

        if gameState.isWin() or gameState.isLose() or depth == -1:
            return {'value':self.evaluationFunction(gameState), 'action':action}
        else:
            if agentIndex == 0: 
                return self.maxValue(gameState,agentIndex,depth)
            else: 
                return self.minValue(gameState,agentIndex,depth)

    def maxValue(self, gameState, agentIndex, depth):
        v = {'value':float('-inf'), 'action':Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            if action == Directions.STOP: continue
            successorGameState = gameState.generateSuccessor(agentIndex, action) 
            successorMinMax = self.minimax(successorGameState, agentIndex+1, depth, action)
            if v['value'] < successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action
        return v

    def minValue(self, gameState, agentIndex, depth):
        v = {'value': float('inf'), 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            if action == Directions.STOP: continue
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorMinMax = self.minimax(successorGameState, agentIndex + 1, depth, action)
            if v['value'] > successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minimax = self.minimax(gameState, agentIndex=0, depth=self.depth, alpha=float('-inf'), beta=float('inf'))
        return minimax['action']
    
    def minimax (self, gameState, agentIndex=0, depth='0', action=Directions.STOP, alpha=float('-inf'), beta=float('inf')):
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0: 
            depth = depth-1

        if gameState.isWin() or gameState.isLose() or depth == -1:
            return {'value': self.evaluationFunction(gameState), 'action': action}
        else:
            if agentIndex == 0: 
                return self.maxValue(gameState, agentIndex, depth, alpha, beta)
            else: 
                return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        v = {'value':-float('inf'), 'action':Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:

            if action == Directions.STOP: continue

            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorMinMax = self.minimax(successorGameState, agentIndex + 1, depth, action, alpha, beta)

            if v['value'] <= successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action

            if v['value'] > beta: return v
            
            alpha = max(alpha, v['value'])

        return v
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        v = {'value':float('inf'), 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:

            if action == Directions.STOP: continue

            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorMinMax = self.minimax(successorGameState, agentIndex + 1, depth, action, alpha, beta)

            if v['value'] >= successorMinMax['value']:
                v['value'] = successorMinMax['value']
                v['action'] = action

            if v['value'] < alpha: return v

            beta = min(beta, v['value'])

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        expectimax = self.expectimax(gameState, agentIndex=0, depth=self.depth)
        return expectimax['action']

    def expectimax(self, gameState, agentIndex = 0, depth = '2', action = Directions.STOP):
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0: 
            depth = depth-1

        if gameState.isWin() or gameState.isLose() or depth == -1:
            return {'value': self.evaluationFunction(gameState), 'action': action}
        else:
            if agentIndex == 0: 
                return self.maxValue(gameState, agentIndex, depth)
            else: 
                return self.expValue(gameState, agentIndex, depth)
            
    def expValue(self, gameState, agentIndex, depth):
        v = {'value': 0, 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)
        probability = 1.0 / len(legalMoves)
        for action in legalMoves:
            if action == Directions.STOP: continue
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorExpectiMax = self.expectimax(successorGameState, agentIndex + 1, depth, action)
            v['value'] += probability * successorExpectiMax['value']
            v['action'] = action
        return v

    def maxValue(self, gameState, agentIndex, depth):
        v = {'value': float('-inf'), 'action': Directions.STOP}
        legalMoves = gameState.getLegalActions(agentIndex)        

        for action in legalMoves:
            if action == Directions.STOP: continue

            successorGameState = gameState.generateSuccessor(agentIndex, action) 
            successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)

            if v['value'] < successorExpectiMax['value']:
                v['value'] = successorExpectiMax['value']
                v['action'] = action

        return v

def betterEvaluationFunction(currentGameState):
    # Prioriza estados de vitória e derrota
    if currentGameState.isWin():
        return float("+inf")
    if currentGameState.isLose():
        return float("-inf")

    # Pontuação base do estado
    score = scoreEvaluationFunction(currentGameState)

    # Informações sobre o estado atual
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleLocations = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Incentiva proximidade à comida mais próxima
    if foodList:
        minFoodDistance = min(util.manhattanDistance(pacmanPos, foodPos) for foodPos in foodList)
        score += 0.5 / (1 + minFoodDistance)  # Aumentado para priorizar comida

    # Incentiva proximidade às cápsulas e penaliza cápsulas restantes
    if capsuleLocations:
        minCapsuleDistance = min(util.manhattanDistance(pacmanPos, capsule) for capsule in capsuleLocations)
        score += 5 / (1 + minCapsuleDistance)
    
    # Interação com fantasmas
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        distanceToGhost = util.manhattanDistance(pacmanPos, ghostPos)

        # Fantasmas assustados
        if scaredTimes[i] > 0:
            # Incentiva perseguir fantasmas assustados
            score += 25.0 / (1 + distanceToGhost)
        else:
            # Penaliza proximidade a fantasmas ativos
            if distanceToGhost > 0:
                score -= 10.0 / distanceToGhost

    return score



# Abbreviation
better = betterEvaluationFunction
