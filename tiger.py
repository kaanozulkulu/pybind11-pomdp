import os
import time
from example import State, Action, Observation, TransitionModel, ObservationModel, RewardModel, RolloutPolicy, Agent, Environment, Histogram, POUCT
import random


class TigerState(State):
    def __init__(self, name):
        State.__init__(self,name)
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerState(%s)" % self.name
    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")

class TigerAction(Action):
    def __init__(self, name):
        Action.__init__(self,name)
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerAction(%s)" % self.name
class TigerObservation(Observation):
    def __init__(self, name):
        Observation.__init__(self,name)
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerObservation(%s)" % self.name

class TObservationModel(ObservationModel):
    def __init__(self, noise=0.15):
        ObservationModel.__init__(self)
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5
        if random.uniform(0,1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [TigerObservation(s)
                for s in {"tiger-left", "tiger-right"}]
class TTransitionModel(TransitionModel):
    
    def __init__(self):
        
        TransitionModel.__init__(self)
        
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            
            return TigerState(str(state))

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerState(s) for s in {"tiger-left", "tiger-right"}]

class TRewardModel(RewardModel):
    def __init__(self):
        #pass
        RewardModel.__init__(self)
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, state, action, next_state):
       
        return self._reward_func(state, action)

class PPolicyModel(RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    def __init__(self):
        #pass
        RolloutPolicy.__init__(self)

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        a =  [TigerAction(s)
              for s in {"open-left", "open-right", "listen"}]
        return a
        



if __name__ == "__main__":

    

    # print(os.getpid())
    # # wait for 10 seconds before continuing
    # # time.sleep(1)
    print("\n** Tiger Problem Started **")
    st1 = TigerState("tiger-left")
    st2 = TigerState("tiger-right")
    # print ("st1", st1)
    init_belief = Histogram({st1: 0.5, st2: 0.5})
    P = PPolicyModel()
    O = TObservationModel(0.15)
    T = TTransitionModel()
    R = TRewardModel()
    newenv = Environment(st2,T,R)
    # print("len bel ", init_belief.lenHist())
    # print("init bel ", init_belief.getitem(st1))
    agent = Agent(init_belief,P,T,O,R)
    # print("agents belief" , agent.belief().getitem(st1))
    # print(init_belief.getitem(st1))
    
    pouct = POUCT(3,1,7500,0.9,1.4,0,0,P,True,5,agent)
   
    val = 0
    avg_val = 0
    for i in range(10):
        start_time1 = time.time()
        action = pouct.plan()
        
        val = (time.time() - start_time1)
        avg_val += val
        print("Val: ", val)
        print("==== Step %d ====" % (i+1))
        print("Action:", action.name)
       
        reward = newenv.reward_model().sample(newenv.getstate(), action, None)
        print("Reward:", reward)

        real_observation = TigerObservation(newenv.getstate().name)
        print(">> Observation:",  real_observation)
        pouct.getAgent().update_hist(action, real_observation)
        pouct.update(action, real_observation)
        new_belief = pouct.getAgent().cur_belief().update_hist_belief(
                action, real_observation,
                pouct.getAgent().getObsModel(),
                pouct.getAgent().getTransModel(),True,False)
        agent.setbelief(new_belief, True)
        pouct.setAgent(agent)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")

    print("Average Plan time:" , avg_val/10)
        

