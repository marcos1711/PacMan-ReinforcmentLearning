# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # impleminatation of value itteration (running it)

        # first we loop for the iterations
        for i in range(self.iterations):
          states = self.mdp.getStates() #get states
          temp_counter = util.Counter() #inilize coutner
          for state in states: #iterate through states
            maxV = float("-inf") #iniliaize max value
            for action in self.mdp.getPossibleActions(state): #check actions based on state
              q_value = self.computeQValueFromValues(state, action)
              if q_value > maxV: #compare q val to choose the max
                max_val = q_value
              temp_counter[state] = max_val
          self.values = temp_counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Value has an initial value of 0
        action_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action) #set action probability pairs to states and probability
        total = 0 #initialize total to 0
        for next_state, prob in action_prob_pairs: #check next state and iterate through probability in action probability pairs 
            reward = self.mdp.getReward(state, action, next_state) #set reward
            total += prob * (reward + self.discount * self.values[next_state]) #update total
        return total #return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # check if there are any legal actions avaibale
        best_action = None
        max_val = float("-inf")
        for action in self.mdp.getPossibleActions(state): #iterate through possible actions 
          q_value = self.computeQValueFromValues(state, action) #set qvalue to currrent state and action 
          if q_value > max_val: #check if qvalue is greater than maxvalue 
            max_val = q_value #update max value 
            best_action = action #update best action
        return best_action #return best action 


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        numberOfStates = len(states)

        for i in range(self.iterations):
            state = states[i % numberOfStates]
            # check if current state is terminal if not continue 
            if not self.mdp.isTerminal(state):
                values = []  
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    values.append(q_value)  
                    # append q value to array of values
                self.values[state] = max(values)  
                # assign max value 


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pq = util.PriorityQueue()
        predecessors = {}
        for state in self.mdp.getStates(): #iterate throug states
            if not self.mdp.isTerminal(state): #check if state is terminal if not proceede
                for action in self.mdp.getPossibleActions(state): #iterate through possible actions for that state
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action): #iterate through next possible states
                        if nextState in predecessors: #check if next has been a predecessor
                            predecessors[nextState].add(state) #if so add state
                        else:
                            predecessors[nextState] = {state} #else add state to predecessors

        for state in self.mdp.getStates():  #iterate trhough states
            if not self.mdp.isTerminal(state): #check if state is terminal 
                values = [] #intialize array 
                for action in self.mdp.getPossibleActions(state): #iterate through actions 
                    q_value = self.computeQValueFromValues(state, action) #set Qvalue
                    values.append(q_value) #append qvalue 
                diff = abs(max(values) - self.values[state]) #inzialize diff with absolute value of values and state values 
                pq.update(state, - diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            temp_state = pq.pop()
            if not self.mdp.isTerminal(temp_state):
                values = []
                for action in self.mdp.getPossibleActions(temp_state):
                    q_value = self.computeQValueFromValues(temp_state, action)
                    values.append(q_value)
                self.values[temp_state] = max(values)

            for p in predecessors[temp_state]:
                if not self.mdp.isTerminal(p):
                    values = []
                    for action in self.mdp.getPossibleActions(p):
                        q_value = self.computeQValueFromValues(p, action)
                        values.append(q_value)
                    diff = abs(max(values) - self.values[p])
                    if diff > self.theta:
                        pq.update(p, -diff)
