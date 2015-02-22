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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        def getClosestDistance(locationList, myLocation):
            distance = float("inf")
            for location in locationList:
                newDistance = util.manhattanDistance(location, myLocation)
                distance = newDistance if newDistance < distance else distance

            return distance

        def weightFunction(foodDistance, ghostDistance):
            foodWeight = (1 / float(foodDistance + 1))
            ghostWeight = ((1 / float(ghostDistance + 1)) + 1) if ghostDistance < 4 else 0 
            weightedSum = foodWeight - ghostWeight
            return weightedSum

        foodDistance = getClosestDistance(currFood.asList(), newPos)
        ghostDistance = getClosestDistance([ghostState.getPosition() for ghostState in newGhostStates], newPos)
        returnVal = weightFunction(foodDistance, ghostDistance)

        return returnVal

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.firstGhost = 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, agent, state, depth):

        if depth <= 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        value = float("-inf") if agent == self.index else float("inf") 
        actions = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, action) for action in actions]        

        if agent == self.index:
            minmax = lambda x, y: max(x, self.minimax( agent + 1, y, depth ))
        elif agent == state.getNumAgents() - 1:
            minmax = lambda x, y: min(x, self.minimax( self.index, y, depth - 1 ))
        else:
            minmax = lambda x, y: min(x, self.minimax( agent + 1, y, depth )) 

        for successor in successors:
            value = minmax( value, successor )

        return value;

    def getAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        successors = [gameState.generateSuccessor(self.index, action) for action in actions]

        options = {}
        for index, successor in enumerate(successors):
            options[ self.minimax(self.firstGhost, successor, self.depth) ] = actions[index]

        return options[max(options.keys())]

class AlphaBetaAgent(MultiAgentSearchAgent):

    def minimax(self, agent, state, depth, alpha, beta):

        if depth <= 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        value = float("-inf") if agent == self.index else float("inf") 
        actions = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, action) for action in actions]        

        if agent == self.index:
            minmax = lambda x, y: max(x, self.minimax( agent + 1, y, depth, alpha, beta ))
            alphabeta = lambda a, b, v: (max(a, v), b)
        elif agent == state.getNumAgents() - 1:
            minmax = lambda x, y: min(x, self.minimax( self.index, y, depth - 1, alpha, beta ))
            alphabeta = lambda a, b, v: (a, min(b, v))
        else:
            minmax = lambda x, y: min(x, self.minimax( agent + 1, y, depth, alpha, beta ))
            alphabeta = lambda a, b, v: (a, min(b, v))

        for successor in successors:
            value = minmax( value, successor )

            if agent == self.index and value > beta:
                return value
            elif value < alpha:
                return value

            alpha, beta = alphabeta( alpha, beta, value )

        return value;

    def getAction(self, gameState):
        alpha = float("-inf")
        beta = float("inf")
        actions = gameState.getLegalActions(self.index)
        successors = [gameState.generateSuccessor(self.index, action) for action in actions]

        best, current = 0, float("-inf")
        for index, successor in enumerate(successors):
            tmp = self.minimax(self.firstGhost, successor, self.depth, alpha, beta)

            if tmp > current:
                current = tmp
                best = actions[index]
            
            if current > beta:
                return best

            alpha = max(alpha, current)

        return best

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

