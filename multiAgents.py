from game import Actions, Directions
from typing import Tuple
import random
import util

from pacman import GameState
from game import Agent

#######################################################
#            This portion is written for you          #
#######################################################


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()


def myScoreEvaluationFunction(currentGameState: GameState):
    # considering the food and the ghost's relative position
    ans = 0
    for food in currentGameState.getFood().asList():
        ans += 50 / (abs(food[0] - currentGameState.getPacmanPosition()[0]) +
                     abs(food[1] - currentGameState.getPacmanPosition()[1]) + 10)
    ans += currentGameState.getScore()
    return ans


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 3)
    """

    from pacman import GameState

    def searchForAgent(self, gameState: GameState, agentIndex: int, depth: int) -> Tuple[Actions, float]:
        # search the next step for the agent[agentIndex]
        if(gameState.isLose() or gameState.isWin() or depth == 0):
            # lose or win or reach maxium depth
            return (None, self.evaluationFunction(gameState))
        # the second is the next_state value, we see shorter to avoid trapped
        best_score = float("-inf") if agentIndex == 0 else float("inf")
        best_action = None
        best_next_state_score = None
        # iterate all the actions
        for action in gameState.getLegalActions(agentIndex):
            # get the next state
            nextState = gameState.generateChild(agentIndex, action)
            next_state_score = self.evaluationFunction(nextState)
            (next_agent_action, next_agent_score) = self.searchForAgent(nextState, (0 if agentIndex +
                                                                                    1 == gameState.getNumAgents() else agentIndex + 1), depth - (1 if agentIndex == 0 else 0))
            if((agentIndex == 0 and (next_agent_score > best_score or (next_agent_score == best_score and next_state_score > best_next_state_score))) or (agentIndex != 0 and next_agent_score < best_score)):
                best_score = next_agent_score
                best_action = action
                best_next_state_score = next_state_score

        return (best_action, best_score)

    def getAction(self, gameState):
        return self.searchForAgent(gameState, 0, self.depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # search the next step for the agent[agentIndex]
    # return the action and the score
    # alpha is the highest score in the max node
    # beta is the lowest score in the min node
    def searchForAgent(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float) -> Tuple[Actions, float]:
        # search the next step for the agent[agentIndex]
        if(gameState.isLose() or gameState.isWin() or depth == 0):
            # lose or win or reach maxium depth
            return (None, self.evaluationFunction(gameState))
        # the second is the next_state value, we see shorter to avoid trapped
        best_score = float("-inf") if agentIndex == 0 else float("inf")
        best_action = None
        best_next_state_score = None
        # iterate all the actions
        for action in gameState.getLegalActions(agentIndex):
            # get the next state
            nextState = gameState.generateChild(agentIndex, action)
            next_state_score = self.evaluationFunction(nextState)
            # generate new alpha and beta
            (next_agent_action, next_agent_score) = self.searchForAgent(nextState, (0 if agentIndex + 1 ==
                                                                                    gameState.getNumAgents() else agentIndex + 1), depth - (1 if agentIndex == 0 else 0), alpha, beta)
            if((agentIndex == 0 and (next_agent_score > best_score or (next_agent_score == best_score and next_state_score > best_next_state_score))) or (agentIndex != 0 and next_agent_score < best_score)):
                best_score = next_agent_score
                best_action = action
                best_next_state_score = next_state_score
            # pruning, then update alpha and beta
            if(agentIndex == 0):
                if(best_score >= beta):
                    return (best_action, best_score)
                alpha = max(alpha, best_score)
            else:
                if(best_score <= alpha):
                    return (best_action, best_score)
                beta = min(beta, best_score)

        return (best_action, best_score)

    def getAction(self, gameState):
        return self.searchForAgent(gameState, 0, self.depth, float("-inf"), float("inf"))[0]
