# inference.py
# ------------
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


import itertools
import util
import random
import busters
import game

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        belief = util.Counter()

        if noisyDistance is None:
            belief[self.getJailPosition()] = 1.0
        else:
            for loc in self.legalPositions:
                distance = util.manhattanDistance(loc, pacmanPosition)
                belief[loc] = emissionModel[distance] * self.beliefs[loc]

        belief.normalize()
        self.beliefs = belief

    def elapseTime(self, gameState):
        allPossible = util.Counter()

        for oldPos in self.legalPositions:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            for newPos, prob in newPosDist.items():
                allPossible[newPos] += prob * self.beliefs[oldPos]

        self.beliefs = allPossible

    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)
        self.particles = []

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        def uniformlyIndexedElements(n, domain):
            if not 0 <= n <= len(domain):
                return list(domain)

            step = int((len(domain) - 1.0)/(n - 1.0))

            elements = []
            for i in range(n):
                elements.append(domain[i * step])

            return elements

        self.particles = []
        unassigned = self.numParticles

        self.particles.extend( uniformlyIndexedElements(unassigned, self.legalPositions) )
        while unassigned > 0:
            selectedElements = uniformlyIndexedElements(unassigned, self.legalPositions)
            self.particles.extend( selectedElements )
            unassigned -= len(selectedElements)

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        if noisyDistance is None:
            self.particles = [self.getJailPosition()] * self.numParticles
        else:
            newBelief = util.Counter()
            belief = self.getBeliefDistribution()

            for position in self.legalPositions:
                distance = util.manhattanDistance(position, pacmanPosition)
                newBelief[position] += emissionModel[distance] * belief[position]

            if newBelief.totalCount() == 0:
                self.initializeUniformly(gameState)
            else:
                self.particles = []
                for _ in range(self.numParticles):
                    self.particles.append(util.sample(newBelief))

    def elapseTime(self, gameState):
        tmp = []
        for particle in self.particles:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, particle))
            tmp.append(util.sample(newPosDist))

        self.particles = tmp

    def getBeliefDistribution(self):
        beliefDistribution = util.Counter()
        for particle in self.particles:
            beliefDistribution[particle] += 1.0

        beliefDistribution.normalize()
        return beliefDistribution

class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1:
            jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.particles = []
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        self.particles = []
        permutations = list(itertools.product(self.legalPositions, repeat = self.numGhosts))

        if self.numParticles <= len(permutations):
            random.shuffle(permutations)
            self.particles.extend(permutations[:self.numParticles])
        else:
            unassigned = self.numParticles
            while True:
                if unassigned >= len(permutations):
                    self.particles.extend(permutations)
                    unassigned -= len(permutations)
                else:
                    random.shuffle(permutations)
                    self.particles.extend(permutations[:unassigned])
                    break

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts: return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

        newBelief = util.Counter()
        for particle in self.particles:
            weight = 1.0
            for index in range(self.numGhosts):
                if noisyDistances[index] is None:
                    particle = self.getParticleWithGhostInJail(particle, index)
                else:
                    distance = util.manhattanDistance(particle[index], pacmanPosition)
                    weight *= emissionModels[index][distance]

            newBelief[particle] += weight

        if newBelief.totalCount() == 0:
            self.initializeParticles()
            for particle in self.particles:
                for i in range(self.numGhosts):
                    if noisyDistances[i] is None:
                        particle = self.getParticleWithGhostInJail(particle, i)
        else:
            self.particles = []
            newBelief.normalize()
            for _ in range(self.numParticles):
                self.particles.append(util.sample(newBelief))

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghostIndex'th ghost in jail.
        """
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)
            for i in range(self.numGhosts):
                newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, newParticle), i, self.ghostAgents[i])
                newParticle[i] = util.sample(newPosDist)

            newParticles.append(tuple(newParticle))

        self.particles = newParticles

    def getBeliefDistribution(self):
        beliefDistribution = util.Counter()
        for particle in self.particles:
            beliefDistribution[particle] += 1.0

        beliefDistribution.normalize()
        return beliefDistribution

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState

