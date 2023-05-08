"""
RockSample(n,k) problem
Origin: Heuristic Search Value Iteration for POMDPs (UAI 2004)
Description:
State space:
    Position {(1,1),(1,2),...(n,n)}
    :math:`\\times` RockType_1 :math:`\\times` RockType_2, ..., :math:`\\times` RockType_k
    where RockType_i = {Good, Bad}
    :math:`\\times` TerminalState
    (basically, the positions of rocks are known to the robot,
     but not represented explicitly in the state space. Check_i
     will smartly check the rock i at its location.)
Action space:
    North, South, East, West, Sample, Check_1, ..., Check_k
    The first four moves the agent deterministically
    Sample: samples the rock at agent's current location
    Check_i: receives a noisy observation about RockType_i
    (noise determined by eta (:math:`\eta`). eta=1 -> perfect sensor; eta=0 -> uniform)
Observation: observes the property of rock i when taking Check_i.
Reward: +10 for Sample a good rock. -10 for Sampling a bad rock.
        Move to exit area +10. Other actions have no cost or reward.
Initial belief: every rock has equal probability of being Good or Bad.
"""

import ast
import math
import re
import numpy as np
import sys
import copy
import os
import time
from example import State, Action, Observation, TransitionModel, ObservationModel, RewardModel, RolloutPolicy, Agent, Environment, Histogram, Particles, POUCT
import random

global O, R
EPSILON = 1e-9


# check_pattern = re.compile(r'check-(\d+)')
def is_check_action(s):
    if 'c' in s:
        return True
    else:
        return False

def extract_number_check_action(s):
    return int(s.split('-')[1])

def is_move_action(s):
    if 'mo' in s:
        return True
    else:
        return False
    
map_dic = {
    'EAST': (1, 0),
    'WEST' : (-1, 0),
    'NORTH' : (0, -1),
    'SOUTH' : (0, 1)
    }

def extract_move_action(s):
    return map_dic[s.split('-')[1]]
# def extract_move_action(s):

#     if not s.startswith('move-'):
#         return None
#     elif s == 'move-EAST':
#         return (1, 0)
#     elif s == 'move-WEST':
#         return (-1, 0)
#     elif s == 'move-NORTH':
#         return (0, -1)
#     elif s == 'move-SOUTH':
#         return (0, 1)
#     else:
#         return None
    
def is_sample_action(s):
    return s == 'sample'

# def get_position_from_name(name):
#     state_values = name.split("|")
#     first_value = state_values[0].strip()
#     return tuple(ast.literal_eval(first_value))

# def get_rocktypes_from_name(name):
#     state_values = name.split("|")
#     second_value = state_values[1].strip()
#     return tuple(ast.literal_eval(second_value))

# def get_terminal_from_name(name):
#     state_values = name.split("|")
#     third_value = state_values[2].strip()
#     return ast.literal_eval(third_value)

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class RockType:
    GOOD = 'good'
    BAD = 'bad'
    @staticmethod
    def invert(rocktype):
        if rocktype == 'good':
            return 'bad'
        else:
            return 'good'
    @staticmethod
    def random(p=0.5):
        if random.uniform(0,1) >= p:
            return RockType.GOOD
        else:
            return RockType.BAD

class RockState(State):
    def __init__(self, position, rocktypes, terminal=False):
        """
        position (tuple): (x,y) position of the rover on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robot is at the terminal state.
        (It is so true that the agent's state doesn't need to involve the map!)
        x axis is horizontal. y axis is vertical.
        """
        # initialize State.init with position rocktype and terminals string values - add spaces between them
        State.__init__(self, str(position) + " | " + str(rocktypes) + " | " + str(terminal), tuple(position), tuple(rocktypes), terminal)

        self.name = str(position) + " | " + str(rocktypes) + " | " + str(terminal)
        self.position = tuple(position)
        self.rocktypes = tuple(rocktypes)
        self.terminal = terminal


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "RockState(%s)" % (self.name)

class RockAction(Action):
    def __init__(self, name):
        Action.__init__(self,name)
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, RockAction):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "RockAction(%s)" % self.name


class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, -1)
    SOUTH = (0, 1)
    def __init__(self, motion, name):
        # Action.__init__(self,name)
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        Action.__init__(self, "move-%s" % str(name))
MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")

class SampleAction(Action):
    def __init__(self):
        Action.__init__(self,"sample")

class CheckAction(Action):
    def __init__(self, rock_id):
        self.rock_id = rock_id
        Action.__init__(self,"check-%d" % rock_id)

class RockObservation(Observation):
    def __init__(self, quality):
        Observation.__init__(self,quality)
        self.quality = quality
    def __hash__(self):
        return hash(self.quality)
    def __eq__(self, other):
        if isinstance(other, RockObservation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other
    def __str__(self):
        return str(self.quality)
    def __repr__(self):
        return "RockObservation(%s)" % str(self.quality)

class RSTransitionModel(TransitionModel):

    """ The model is deterministic """

    def __init__(self, n, rock_locs, in_exit_area):
        """
        rock_locs: a map from (x,y) location to rock_id
        in_exit_area: a function (x,y) -> Bool that returns True if (x,y) is in exit area"""
        TransitionModel.__init__(self)
        self._n = n
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area
    
    def _move_or_exit(self, position, action, action_motion):
        expected = (position[0] + action_motion[0],
                    position[1] + action_motion[1])
        if self._in_exit_area(expected):
            return expected, True
        else:
            return (max(0, min(position[0] + action_motion[0], self._n-1)),
                    max(0, min(position[1] + action_motion[1], self._n-1))), False

    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        
        next_position = state.position
        rocktypes = state.rocktypes
        next_rocktypes = rocktypes
        next_terminal = state.terminal
        if next_terminal:
            next_terminal = True  # already terminated. So no state transition happens
        else:
            # action_motion = extract_move_action(action.name)
            if is_move_action(action.name):
            # if action_motion is not None:
                
                next_position, exiting = self._move_or_exit(next_position, action, extract_move_action(action.name))
                if exiting:
                    next_terminal = True
              
            elif is_sample_action(action.name):
                if next_position in self._rock_locs:
                    rock_id = self._rock_locs[next_position]
                    _rocktypes = list(rocktypes)
                    _rocktypes[rock_id] = RockType.BAD
                    next_rocktypes = tuple(_rocktypes)
      
        return RockState(tuple(next_position), next_rocktypes, next_terminal)

    def full_sample(self, state, action):
        next_state = self.sample(state, action)
        obs = O.sample(next_state, action)
        reward = R.sample(state, action, next_state)
        return(tuple([next_state, obs, reward]))
    
    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)


class RSObservationModel(ObservationModel):
    def __init__(self, rock_locs, half_efficiency_dist=20):
        ObservationModel.__init__(self)
        self._half_efficiency_dist = half_efficiency_dist
        self._rocks = {rock_locs[pos]:pos for pos in rock_locs}

    def probability(self, observation, next_state, action):
        # check_action_rock_id = extract_number_check_action(action.name)
        # if check_action_rock_id is not None:
        if is_check_action(action.name):
            rock_pos = self._rocks[extract_number_check_action(action.name)]
            dist = euclidean_dist(rock_pos, next_state.position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5

            rocktypes = next_state.rocktypes
            actual_rocktype = rocktypes[extract_number_check_action(action.name)]
            if actual_rocktype == observation:
                return eta
            else:
                return 1.0 - eta
        else:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action, argmax=False):
     
        # check_action_rock_id = extract_number_check_action(action.name)
        if not next_state.terminal and is_check_action(action.name):

            rock_pos = self._rocks[extract_number_check_action(action.name)]           
            dist = euclidean_dist(rock_pos, next_state.position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5

            if argmax:
                keep = eta > 0.5
            else:
                keep = eta > random.uniform(0, 1)
            rocktypes = next_state.rocktypes
            actual_rocktype = rocktypes[extract_number_check_action(action.name)]
            if not keep:
                observed_rocktype = RockType.invert(actual_rocktype)
                return RockObservation(observed_rocktype)
            else:
                return RockObservation(actual_rocktype)
        else:
            return RockObservation("None")

        return self._probs[next_state][action][observation]

    def argmax(self, next_state, action):
        """Returns the most likely observation"""
        return self.sample(next_state, action, argmax=True)


class RSRewardModel(RewardModel):
    def __init__(self, rock_locs, in_exit_area):
        RewardModel.__init__(self)
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area
        
    def sample(self, state, action, next_state, normalized=False, **kwargs):
        if state.terminal:
            return 0  
        if is_sample_action(action.name):
            if state.position in self._rock_locs:
                rocktypes = state.rocktypes
                if rocktypes[self._rock_locs[state.position]] == RockType.GOOD:
                    return 20
                else:
                    return -20
            else:
                return 0

        
        # elif extract_move_action(action.name) is not None:
        elif is_move_action(action.name):
            pos = state.position
            action_motion = extract_move_action(action.name)
            new_position = (pos[0] + action_motion[0],
                    pos[1] + action_motion[1])
            if self._in_exit_area(new_position):
                
                return 20
        return 0

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError


class RSPolicyModel(RolloutPolicy):
    """Simple policy model according to problem description."""
    def __init__(self, n, k):
        RolloutPolicy.__init__(self)
        check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {SampleAction()} | check_actions
        self._all_actions = self._move_actions | self._other_actions
        self._n = n
    def get_all_actions(self, state, history=None):
                
        if state is None:
            return self._all_actions
        else:
            motions = set(self._all_actions)
            rover_x, rover_y = state.position
            if rover_x == 0:
                motions.remove(MoveWest)
            if rover_y == 0:
                motions.remove(MoveNorth)
            if rover_y == self._n - 1:              
                motions.remove(MoveSouth)
          
            
            combined_actions= motions | self._other_actions
            return list(combined_actions)
    
    def sample(self, state, normalized=False, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]
    
    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError
    
    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state,), 1)[0]


class RockSampleProblem():

    @staticmethod
    def random_free_location(n, not_free_locs):
        """returns a random (x,y) location in nxn grid that is free."""
        while True:
            loc = (random.randint(0, n-1),
                   random.randint(0, n-1))
            if loc not in not_free_locs:
                return loc

    def in_exit_area(self, pos):
        return pos[0] == self._n

    @staticmethod
    def generate_instance(n, k):
        """Returns init_state and rock locations for an instance of RockSample(n,k)"""

        rover_position = (0, random.randint(0, n-1))
        rock_locs = {}  # map from rock location to rock id
        for i in range(k):
            loc = RockSampleProblem.random_free_location(n, set(rock_locs.keys()) | set({rover_position}))
            rock_locs[loc] = i

        # random rocktypes
        rocktypes = tuple(RockType.random() for i in range(k))

        # Ground truth state
        init_state = RockState(tuple(rover_position), rocktypes, False)

        return init_state, rock_locs

    def print_state(self):
        string = "\n______ID______\n"        
        rover_position = self.env.getstate().position
        rocktypes = self.env.getstate().rocktypes
        # Rock id map
        for y in range(self._n):
            for x in range(self._n+1):
                char = "."
                if x == self._n:
                    char = ">"
                if (x,y) in self._rock_locs:
                    char = str(self._rock_locs[(x,y)])
                if (x,y) == rover_position:
                    char = "R"
                string += char
            string += "\n"
        string += "_____G/B_____\n"
        # Good/bad map
        for y in range(self._n):
            for x in range(self._n+1):
                char = "."
                if x == self._n:
                    char = ">"
                if (x,y) in self._rock_locs:
                    if rocktypes[self._rock_locs[(x,y)]] == RockType.GOOD:
                        char = "$"
                    else:
                        char = "x"
                if (x,y) == rover_position:
                    char = "R"
                string += char
            string += "\n"
        print(string)

    def __init__(self, n, k, init_state, rock_locs, init_belief):
        self._n, self._k = n, k
        P = RSPolicyModel(n,k)
        T = RSTransitionModel(n, rock_locs, self.in_exit_area)
        global O, R
        O = RSObservationModel(rock_locs)
        R = RSRewardModel(rock_locs, self.in_exit_area)
        agent = Agent(init_belief,P, T, O,R)
        env = Environment(init_state,
                                   RSTransitionModel(n, rock_locs, self.in_exit_area),
                                   RSRewardModel(rock_locs, self.in_exit_area))
        self._rock_locs = rock_locs
        self.agent = agent
        self.env = env
        self.particles = init_belief.particles()

    

def test_planner(rocksample, planner, nsteps=1, discount=0.95):
    
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    total_time = 0
    for i in range(nsteps):
        
        start_time = time.time()
        print("==== Step %d ====" % (i+1))
        action = planner.plan()
        print("time took for planning: " + str(time.time() - start_time))
        total_time += time.time() - start_time
        env_reward = rocksample.env.state_transition(action, discount)
        real_observation = rocksample.agent.getObsModel().sample(rocksample.env.getstate(), action)
        
        planner.getAgent().update_hist(action, real_observation)

        planner.update(action, real_observation)
        
        #UPDATE BELIEF ADDED
        new_belief = planner.getAgent().cur_belief().update_particle_belief(
            rocksample.particles,
                action, real_observation,
                planner.getAgent().getObsModel(),
                planner.getAgent().getTransModel())
     
        rocksample.agent.setbelief(new_belief, True)
        
        rocksample.particles = new_belief.particles()
      
        
    
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount

        print("Time took for step: " + str(time.time() - start_time))
        print("True state: %s" % rocksample.env.getstate())
        print("RockAction: %s" % str(action.name))
        print("RockObservation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
     
        print("World:")
        rocksample.print_state()
        
        if rocksample.in_exit_area(rocksample.env.getstate().position):
            break
    print("Total time: " + str(total_time))
    return total_reward, total_discounted_reward


def init_particles_belief(k, num_particles, init_state, belief="uniform"):
    num_particles = 200
    particles = []
    for _ in range(num_particles):
        if belief == "uniform":
            rocktypes = []
            for i in range(k):
                rocktypes.append(RockType.random())
            rocktypes = tuple(rocktypes)
        elif belief == "groundtruth":
            rocktypes = copy.deepcopy(init_state.rocktypes)
        particles.append(RockState(init_state.position, rocktypes, False))
    init_belief = Particles(particles)
    
    return init_belief

def main():
    n, k = 5, 5
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
   
    belief = "uniform"

    init_belief = init_particles_belief(k, 200, init_state, belief=belief)
    
    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief)
    rocksample.print_state()

#     POUCT(int max_depth, float plan_time, int num_sims, float discount_factor, float exp_const, int num_visits_init, float val_init, std::shared_ptr<RolloutPolicy> rollout_pol,bool show_prog, int pbar_upd_int, std::shared_ptr<Agent> ag) :  _max_depth(max_depth), _planning_time(plan_time), _num_sims(num_sims), _discount_factor(discount_factor), _exploration_const(exp_const), _num_visits_init(num_visits_init), _rollout_policy(rollout_pol), _show_progress(show_prog), _pbar_update_interval(pbar_upd_int), _agent(ag)
    pouct = POUCT(9,1,6000,0.95,10,1,0,rocksample.agent.getPolicyModel(),True,5,rocksample.agent)
    print("pouct initialized")
 
    tt, ttd = test_planner(rocksample, pouct, nsteps=10, discount=0.95)
    print("Total reward: %s" % str(tt))

if __name__ == '__main__':
    main()
