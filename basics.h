#include <string>
using namespace std;
using std::string;
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <list>
#include <map>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>
#include <functional>
using namespace std;
namespace py = pybind11;




class VNode;
class State {
public:
    State(string name_, std::tuple<int, int> position_ = std::make_tuple(0, 0), std::vector<string> rocktypes_=std::vector<std::string>(), bool terminal_=false) :name(name_), position(position_), rocktypes(rocktypes_), terminal(terminal_) {}

    State() {}
    virtual string getname() { return name; }
   

//protected:
    string name;
    std::tuple<int, int> position;
    std::vector<string> rocktypes;
    bool terminal;
    virtual ~State() {}
};

class Observation {
public:
    Observation(string n) :name(n) {}
    string name;
    bool equals(Observation& other) { if (name == other.name) return true; else return false; }
    Observation(const Observation& obs)
    {
        name = obs.name;
       
    }
    virtual ~Observation() {}

};
class Belief {

public:
    Belief() {}
   virtual std::shared_ptr<State> random() = 0;
   virtual ~Belief() {}
};


class Action {
public:
    Action(string n) :name(n) {}
    Action() {}
    string name;
    Action(const Action& act)
    {
        name = act.name;

    }
    bool equals(Action& other) { if (name == other.name) return true; else return false; }
    
    virtual string getname() { return name; }
};
class History {

public:
    
    void add(std::shared_ptr<Action> act , std::shared_ptr<Observation> obs){
        history.push_back(make_tuple(act, obs));
    }
    std::vector<tuple<std::shared_ptr<Action>, std::shared_ptr<Observation>>> getHist() { return history; }
    virtual ~History(){}

   std::vector<tuple<std::shared_ptr<Action>, std::shared_ptr<Observation>>> history;


};

class ObservationModel {
public:
    virtual double probability(Observation* observation,
        State* next_state,
        Action* action) = 0;

    virtual std::shared_ptr<Observation> sample(State* next_state,
        Action* action) = 0;

    State argmax(const State& next_state,
        const Action& action) {};
};

class TransitionModel {
public:
    //TransitionModel() {};
    virtual double probability(State* next_state,
         State* state,
        Action* action) = 0;


    virtual std::shared_ptr<State> sample(State* state,
        Action* action) = 0;
    virtual std::tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double> full_sample(State* state,
        Action* action) = 0;
    virtual int increment_dummy_test(int num) = 0;

    State argmax(const State& state,
        const Action& action) {};
    virtual ~TransitionModel() { }
};

class RewardModel {
public:
    
    
    virtual double sample(State* state,
       Action* action, State* next_state) = 0;

    virtual ~RewardModel() { }
};

class PolicyModel {
public:
    PolicyModel() {}
    virtual double probability(std::shared_ptr<Action> action,
        std::shared_ptr<State> state
    ) {};

    virtual std::shared_ptr<Action> sample(std::shared_ptr<State> state) {};
    double argmax(const std::shared_ptr<State> state) {};
   
    virtual vector<std::shared_ptr<Action>> get_all_actions(State* state, History history) {};
    
    virtual ~PolicyModel() { }

};

class Agent {
public:
   
    Agent( std::shared_ptr<Belief> init_belief, std::shared_ptr<PolicyModel> pi, std::shared_ptr<TransitionModel> T, std::shared_ptr<ObservationModel> O,
         std::shared_ptr<RewardModel> R)
        : pi_(pi), T_(T), O_(O), R_(R) {
        belief_ = init_belief,
        _cur_belief = init_belief;
        
    }
  
    History gethistory();
    void update_hist(std::shared_ptr<Action> act, std::shared_ptr<Observation> obs);
    std::shared_ptr<Belief> init_belief();
    std::shared_ptr<Belief> belief();
    Belief* cur_belief();
    void setbelief(std::shared_ptr<Belief> bel, bool prior);
    std::shared_ptr<State> sample_belief();
    std::shared_ptr<ObservationModel> getObsModel();
    std::shared_ptr<TransitionModel> getTransModel();
    std::shared_ptr<RewardModel> getRewardModel();
    std::shared_ptr<PolicyModel> getPolicyModel();
    std::vector<std::shared_ptr<Action>> validActions(std::shared_ptr<State> state,History history);
    virtual ~Agent() {}
private:
    std::shared_ptr<Belief> belief_;
    std::shared_ptr<TransitionModel> T_;
    std::shared_ptr<ObservationModel> O_;
    std::shared_ptr<RewardModel> R_;
    std::shared_ptr<PolicyModel> pi_;
    History hist;
    std::shared_ptr<Belief> _cur_belief;
};

class Environment {
public:
    Environment(std::shared_ptr<State> init_state, std::shared_ptr<TransitionModel> T, std::shared_ptr<RewardModel> R)
        : state_(init_state), T_(T), R_(R) {}
  
    std::shared_ptr<State> getstate();
    
    std::shared_ptr<TransitionModel> transitionmodel();
    std::shared_ptr<RewardModel> reward_model();
    double state_transition(std::shared_ptr<Action> action, float discount_factor = 1.0);
    tuple<std::shared_ptr<State>, double> state_transition_sim(std::shared_ptr <Action> action, float discount_factor = 1.0);
    void apply_transition(std::shared_ptr<State> next_st);
    tuple< std::shared_ptr<Observation>, double> execute(std::shared_ptr<Action> act, std::shared_ptr<ObservationModel> Omodel);
    std::shared_ptr<Observation> provide_observation(std::shared_ptr<ObservationModel> Omodel, std::shared_ptr<Action> act);
    virtual ~Environment() { }

private:
    std::shared_ptr<State> state_;
    std::shared_ptr<TransitionModel> T_;
    std::shared_ptr<RewardModel> R_;
};

class Histogram : public Belief {
public:
    std::map<std::shared_ptr<State>, float> _histogram;

public:
    Histogram(std::map<std::shared_ptr<State>, float> histogram) : _histogram(std::move(histogram)) {}
    Histogram(Histogram& h1)
    {
        h1._histogram = _histogram;
    }
    std::map<std::shared_ptr<State>, float> getHist();
    int lenHist();
    float getitem(std::shared_ptr<State> st);
    void setitem(std::shared_ptr<State> st, float prob);
    bool isEq(std::shared_ptr<Histogram> b);
    std::shared_ptr<State> mpe();
    std::shared_ptr<State> random();
    bool isNormalized(double eps);
    std::shared_ptr<Histogram> update_hist_belief(std::shared_ptr<Action> real_act, std::shared_ptr<Observation> real_obs, std::shared_ptr<ObservationModel> O, std::shared_ptr<TransitionModel> T, bool normalize, bool static_transition);
    virtual ~Histogram() { }

};

class WeightedParticles : public Belief {
public:
    // Constructor
     WeightedParticles(std::vector<std::pair<std::shared_ptr<State>, float>> particles);

    virtual void add(std::pair<std::shared_ptr<State>, float> particle) = 0;
    //getters
    std::vector<std::pair<std::shared_ptr<State>, float>> particles() const;
    std::vector<std::shared_ptr<State>> values() const;
    std::vector<float> weights() const;
    std::shared_ptr<State> random();
    virtual ~WeightedParticles() {}
   
protected:
    std::vector<std::pair<std::shared_ptr<State>, float>> _particles;
    std::vector<float> _weights;
    std::vector<std::shared_ptr<State>> _values;
    // std::string _approx_method;
    // std::function<float(std::shared_ptr<State>, std::shared_ptr<State>)> _distance_func;  
};

class Particles : public Belief {

public:
    Particles(std::vector<std::shared_ptr<State>> particles) : _particles(particles) {};

    void add(std::shared_ptr<State> particle);    
    std::shared_ptr<State> random();
    std::vector<std::shared_ptr<State>> particles() const { return _particles;}
                                                                   
    
    std::shared_ptr<Particles> update_particle_belief(std::vector<std::shared_ptr<State>> current_particles, std::shared_ptr<Action> real_act, std::shared_ptr<Observation> real_obs, std::shared_ptr<ObservationModel> O, std::shared_ptr<TransitionModel> T);
    virtual ~Particles() { }
private:
    std::vector<std::shared_ptr<State>> _particles;
};


class PyState : public py::wrapper<State> {
public:
    /* Inherit the constructors */
    using py::wrapper<State>::wrapper;
    std::string getname() override {
        PYBIND11_OVERLOAD(
        std::string, /* Return type */
        State,  /* Parent class */
        getname); /* Name of function in C++ (must match Python name) */
    }
};
class PyHistory : public py::wrapper<History> {
public:
    /* Inherit the constructors */
    using py::wrapper<History>::wrapper;
   
};
class PyAction : public py::wrapper<Action> {
public:
    /* Inherit the constructors */
    using py::wrapper<Action>::wrapper;
};
class PyObservationModel : public py::wrapper<ObservationModel> {
public:

    // inherit the constructors
    using py::wrapper<ObservationModel>::wrapper;

    // trampoline (one for each virtual function)
    double probability(Observation* observation,
        State* next_state,
        Action* action) override {
        PYBIND11_OVERLOAD_PURE(
            double, /* Return type */
            ObservationModel,      /* Parent class */
            probability,        /* Name of function in C++ (must match Python name) */
            observation,      /* Argument(s) */
            next_state,
            action
        );
    }

    std::shared_ptr<Observation> sample(State* next_state,
        Action* action) override {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<Observation>,  /*Return type */
            ObservationModel,      /* Parent class */
            sample,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            next_state,
            action
        );
    }

};

class PyTransitionModel : public py::wrapper<TransitionModel> {
public:

    // inherit the constructors
    using py::wrapper<TransitionModel>::wrapper;
    // trampoline (one for each virtual function)
    double probability(State* next_state,
        State* state,
        Action* action) override {
        PYBIND11_OVERLOAD_PURE(
            double, /* Return type */
            TransitionModel,      /* Parent class */
            probability,        /* Name of function in C++ (must match Python name) */
            next_state,      /* Argument(s) */
            state,
            action
        );
    }
    std::shared_ptr<State> sample(State* state,
        Action* action) override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<State>, TransitionModel, sample, state,action);
    }
    typedef std::tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double> full_sample_return_type;
    std::tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double> full_sample(State* state,
        Action* action) override {
        PYBIND11_OVERLOAD_PURE(full_sample_return_type, TransitionModel, full_sample, state,action);
    }
    int increment_dummy_test(int a) override {
        PYBIND11_OVERLOAD_PURE(int, TransitionModel, increment_dummy_test, a);
    }
  
};

class PyRewardModel : public py::wrapper<RewardModel>
{
public:

    // inherit the constructors
    //using RewardModel::RewardModel;
    using py::wrapper<RewardModel>::wrapper;

   

    double sample(State* state,
        Action* action, State* next_state) override {
        PYBIND11_OVERLOAD_PURE(double, RewardModel, sample, state, action, next_state);
    }

};

class PyPolicyModel : public py::wrapper<PolicyModel>
{
public:

    // inherit the constructors
    using py::wrapper<PolicyModel>::wrapper;


    // trampoline (one for each virtual function)
    double probability(std::shared_ptr<Action> action,
        std::shared_ptr<State> state
        ) override {
        PYBIND11_OVERLOAD(
            double, /* Return type */
            PolicyModel,      /* Parent class */
            probability,        /* Name of function in C++ (must match Python name) */
            action,      /* Argument(s) */
            state
            
        );
    }

    std::shared_ptr<Action> sample(std::shared_ptr<State> state
        ) override {
        PYBIND11_OVERLOAD(
            std::shared_ptr<Action>,  /*Return type */
            PolicyModel,      /* Parent class */
            sample,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            state
            
        );
    }

    vector<std::shared_ptr<Action>> get_all_actions(State* state, History history
    ) override {
        PYBIND11_OVERLOAD(
            vector<std::shared_ptr<Action>>,  /*Return type */
            PolicyModel,      /* Parent class */
            get_all_actions,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            state,
            history

        );
    }
};


/*class PyEnvironment : public Environment

{
public:

    // inherit the constructors
    using Environment::Environment;
};*/
class PyEnvironment : public py::wrapper<Environment> {
public:
    /* Inherit the constructors */
    using py::wrapper<Environment>::wrapper;
   
};
class PyAgent : public py::wrapper<Agent> {
    public:

    // inherit the constructors
    using py::wrapper<Agent>::wrapper;
};
class PyBelief : public py::wrapper<Belief> {
public:

    using py::wrapper<Belief>::wrapper;
    std::shared_ptr<State> random() override {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<State>, /* Return type*/
            Belief,      /* Parent class*/
            random,      /* Name of function in C++ (must match Python name) */
            /* Argument(s)*/


            );
    }
};
class PyHistogram : public py::wrapper<Histogram> {
    public:
    /* Inherit the constructors */
    using py::wrapper<Histogram>::wrapper;
    std::shared_ptr<State> random() override{
        PYBIND11_OVERLOAD_PURE(
        std::shared_ptr<State>, /* Return type*/
        Histogram,      /* Parent class*/
        random,      /* Name of function in C++ (must match Python name) */
        /* Argument(s)*/


        );
  
    }
};

class PyParticles : public py::wrapper<Particles> {
    public:
    /* Inherit the constructors */
    using py::wrapper<Particles>::wrapper;
    std::shared_ptr<State> random() override {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<State>, /* Return type*/
            Particles,      /* Parent class*/
            random,      /* Name of function in C++ (must match Python name) */
            /* Argument(s)*/
        )
    }
};


 class PyObservation : public py::wrapper<Observation> {
     public:
      /* Inherit the constructors */
      using py::wrapper<Observation>::wrapper;
 };

tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_generative_model(std::shared_ptr<Agent> agent, std::shared_ptr<State> state, std::shared_ptr<Action> action, float discount_factor=1);
// tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_generative_model(Agent agent, std::shared_ptr<State> state, std::shared_ptr<Action> action, float discount_factor = 1);
tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_explict_models1(std::shared_ptr<TransitionModel> T, std::shared_ptr<ObservationModel> O, std::shared_ptr<RewardModel> R, std::shared_ptr<State> state, std::shared_ptr<Action> a, float discount_factor);
//tuple<State*, Observation*, double, int> sample_explict_models1(TransitionModel* T, ObservationModel* O, RewardModel* R, State* state, Action* a, float discount_factor=1);
tuple<std::shared_ptr<State>, double, int> sample_explict_models(std::shared_ptr<TransitionModel> T, std::shared_ptr<RewardModel> R, std::shared_ptr<State> state, std::shared_ptr<Action> a, float discount_factor);
// tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_explicit_models2(std::shared_ptr<Agent> agent, std::shared_ptr<State> state, std::shared_ptr<Action> action);
