//#include "basics.h"
#include "planner.h"
#include <map>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <list>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
using namespace std;
class QNode {
public:
    std::unordered_map<string, std::shared_ptr<VNode>> children;
    int num_visits;
    float value;
    QNode(int nvisit, float val) :num_visits(nvisit), value(val) {}
    QNode(const QNode& qn)
    {
        num_visits = qn.num_visits;
        value = qn.value;
        children = qn.children;
    }
    virtual ~QNode() {}
    friend class VNode;
};
class VNode {
public:
    
    VNode(int nvisit) :num_visits(nvisit) {}
    std::unordered_map<string, std::shared_ptr<QNode>> children;
    int num_visits;
    float value;
    std::shared_ptr<Action> argmax();
    int getnumvis() { return num_visits; };
    VNode(const VNode& vn)
    {
        num_visits = vn.num_visits;
        value = vn.value;
        children = vn.children;
    }
    virtual ~VNode() {}
    friend class QNode;
};

class RootVNode :public VNode {
public:
    RootVNode(int nvis, History hist) :VNode(nvis),history(hist)
    {

    }
    History history;
    static std::shared_ptr<RootVNode> from_vnode(std::shared_ptr<VNode> vnode, History hist);
    
};

class RolloutPolicy : public PolicyModel {
public:
    RolloutPolicy() {}
    virtual std::shared_ptr<Action> rollout(State* state, History history) {};
};

class RandomRollout : public RolloutPolicy {
    std::shared_ptr<Action> rollout(std::shared_ptr<State> state, History history)
    {
        vector<std::shared_ptr<Action>> all_actions = PolicyModel::get_all_actions(state.get(), history);
        srand(time(NULL));
        int randnum = rand() % (all_actions).size();
        auto ind = std::next(all_actions.begin(), randnum);
        return *ind;
    };
}; 

class ActionPrior {
public:
    virtual set<tuple<std::shared_ptr<Action>, int, float>> get_preferred_actions(std::shared_ptr<State> state, History history) =0;
};




class POUCT : public Planner {
public:

    int _max_depth;
    float _planning_time;
    int _num_sims;
    int _num_visits_init;
    float _value_init;
    float _discount_factor;
    float _exploration_const;
    std::shared_ptr<ActionPrior> _action_prior;
    std::shared_ptr<RolloutPolicy> _rollout_policy;
    std::shared_ptr<Agent> _agent;
    int _last_num_sims;
    float _last_planning_time;
    std::shared_ptr<VNode> tree;
    bool _show_progress;
    int _pbar_update_interval;
    POUCT(int max_depth, float plan_time, int num_sims, float discount_factor, float exp_const, int num_visits_init, float val_init, std::shared_ptr<RolloutPolicy> rollout_pol,bool show_prog, int pbar_upd_int, std::shared_ptr<Agent> ag) :  _max_depth(max_depth), _planning_time(plan_time), _num_sims(num_sims), _discount_factor(discount_factor), _exploration_const(exp_const), _num_visits_init(num_visits_init), _rollout_policy(rollout_pol), _show_progress(show_prog), _pbar_update_interval(pbar_upd_int), _agent(ag)
    {
        tree = nullptr;
    }


    std::shared_ptr<Action> plan();
    std::shared_ptr<Agent> getAgent() { return _agent; };
    void setAgent(shared_ptr<Agent> a) { _agent = a; }
    // void threaded_simulate(std::shared_ptr<State> state, History history, std::shared_ptr<VNode> root, std::shared_ptr<QNode> parent, std::shared_ptr<Observation> observation, int depth);
    tuple<shared_ptr<Action>, double, int> _search();
    double _simulate(std::shared_ptr<State> state, History history, std::shared_ptr<VNode> root, std::shared_ptr<QNode> parent, std::shared_ptr<Observation> observation, int depth);
    void _expand_vnode(std::shared_ptr<VNode> vnode, History history, std::shared_ptr<State> state);
    double _rollout(std::shared_ptr<State> state, History history, std::shared_ptr<VNode> root, int depth);
    std::shared_ptr<Action> _ucb(std::shared_ptr<VNode> root);
    /*tuple<State, Observation, double> _sample_generative_model(State state, Action action); */
    std::shared_ptr<VNode> _VNode(bool root = false);
    void update(std::shared_ptr<Action> real_action, std::shared_ptr<Observation> real_observation);
    virtual ~POUCT() {}
};


class PyPOUCT : public py::wrapper<POUCT>
{
public:

    // inherit the constructors
    using py::wrapper<POUCT>::wrapper;

    // trampoline (one for each virtual function)
        std::shared_ptr<Action> plan(
    ) override {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<Action>, /* Return type */
            POUCT,      /* Parent class */
            plan        /* Name of function in C++ (must match Python name) */
                  /* Argument(s) */

        );
    }



};
class PyRolloutPolicy : public py::wrapper<RolloutPolicy>
{
public:

    // inherit the constructors
    using py::wrapper<RolloutPolicy>::wrapper;


    // trampoline (one for each virtual function)
    double probability(std::shared_ptr<Action> action,
        std::shared_ptr<State> state
    ) override {
        PYBIND11_OVERLOAD(
            double, /* Return type */
            RolloutPolicy,      /* Parent class */
            probability,        /* Name of function in C++ (must match Python name) */
            action,      /* Argument(s) */
            state

        );
    }

    std::shared_ptr<Action> sample(std::shared_ptr<State> state
    ) override {
        PYBIND11_OVERLOAD(
            std::shared_ptr<Action>,  /*Return type */
            RolloutPolicy,      /* Parent class */
            sample,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            state

        );
    }
   

    vector<std::shared_ptr<Action>> get_all_actions(State* state, History history
    ) override {

        PYBIND11_OVERLOAD(
            vector<std::shared_ptr<Action>>,  /*Return type */
            RolloutPolicy,      /* Parent class */
            get_all_actions,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            state,
            history

        );
    }

    std::shared_ptr<Action> rollout(State* state, History history
    ) override {

        PYBIND11_OVERLOAD(
            std::shared_ptr<Action>,  /*Return type */
            RolloutPolicy,      /* Parent class */
            rollout,        /* Name of function in C++ (must match Python name) */
                           /* Argument(s) */
            state,
            history

        );
    }
};