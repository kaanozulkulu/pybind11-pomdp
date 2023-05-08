#include <string>
using namespace std;
using std::string;
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <list>
#include <tuple>
#include "pouct.h"

namespace py = pybind11;

PYBIND11_MODULE(example, m)
{
    py::class_<State, PyState, std::shared_ptr<State>> state(m, "State", py::dynamic_attr());
    state
        .def(py::init<string, std::tuple<int, int>, std::vector<std::string>, bool>(),
             py::arg("name_"),
             py::arg("position_") = std::make_tuple(0, 0),
             py::arg("rocktypes_") = std::vector<std::string>(),
             py::arg("terminal_") = false)

        .def(py::init<>())

        .def("getname", &State::getname)
        .def_readwrite("name", &State::name)
        .def_readwrite("position", &State::position)
        .def_readwrite("rocktypes", &State::rocktypes)
        .def_readwrite("terminal", &State::terminal);

    py::class_<Belief, PyBelief, std::shared_ptr<Belief>> belief_model(m, "Belief");
    belief_model
        .def(py::init<>())
        .def("random", &Belief::random);
    

    
    py::class_<Action, PyAction, std::shared_ptr<Action>> action(m, "Action", py::dynamic_attr());
    action
        .def(py::init<string>())
        .def(py::init<>())
        .def_readwrite("name", &Action::name);


    py::class_<Observation,PyObservation, std::shared_ptr<Observation>>(m, "Observation")
        .def(py::init<string>());

    py::class_<History, PyHistory, std::shared_ptr<History>> history(m, "History", py::dynamic_attr());
    history
        
        .def(py::init<>())
        .def_readwrite("history", &History::history);
    py::class_<Planner, PyPlanner, std::shared_ptr<Planner>> pln(m, "Planner");
    pln
        .def("plan", &Planner::plan);

    py::class_<ObservationModel, PyObservationModel, std::shared_ptr<ObservationModel>> omodel(m, "ObservationModel");
    omodel
        .def(py::init<>())
        .def("probability", &ObservationModel::probability)
        .def("sample", &ObservationModel::sample)
        .def("argmax", &ObservationModel::argmax);

    
    py::class_<TransitionModel, PyTransitionModel, std::shared_ptr<TransitionModel>> tmodel(m, "TransitionModel");
    tmodel
        .def(py::init<>())
        .def("probability", &TransitionModel::probability)
        .def("sample", &TransitionModel::sample)
        .def("argmax", &TransitionModel::argmax)
        .def("full_sample", &TransitionModel::full_sample);
        // .def("increment_dummy_test", &TransitionModel::increment_dummy_test, py::call_guard<py::gil_scoped_release>());

    py::class_<RewardModel, PyRewardModel, std::shared_ptr<RewardModel>> rmodel(m, "RewardModel");
    rmodel
        .def(py::init<>())
        .def("sample", &RewardModel::sample);

    py::class_<PolicyModel, PyPolicyModel, std::shared_ptr<PolicyModel>> pmodel(m, "PolicyModel");
    pmodel
        .def(py::init_alias<>())
        .def("probability", &PolicyModel::probability)
        .def("sample", &PolicyModel::sample)
        .def("argmax", &PolicyModel::argmax)
        .def("get_all_actions", &PolicyModel::get_all_actions);

     py::class_<RolloutPolicy, PyRolloutPolicy, std::shared_ptr<RolloutPolicy>> rollmodel(m, "RolloutPolicy", pmodel);
     rollmodel
         .def(py::init_alias<>())
         .def("probability", &RolloutPolicy::probability)
         .def("sample", &RolloutPolicy::sample)
         .def("argmax", &RolloutPolicy::argmax)
         .def("get_all_actions", &RolloutPolicy::get_all_actions)
         .def("rollout", &RolloutPolicy::rollout);
 

    py::class_<Environment, PyEnvironment, std::shared_ptr<Environment>> environment(m, "Environment");
    environment
    
        .def(py::init<std::shared_ptr<State>, std::shared_ptr<TransitionModel>, std::shared_ptr<RewardModel> >())
        //.def(py::init<State&>())
        .def("getstate", &Environment::getstate)
        // .def("getstate_helpers", &Environment::getstate_helpers)
        .def("transitionmodel", &Environment::transitionmodel)
        .def("reward_model", &Environment::reward_model)
        .def("state_transition", &Environment::state_transition)
        .def("state_transition_sim", &Environment::state_transition_sim)
        .def("apply_transition", &Environment::apply_transition)
        .def("execute", &Environment::execute)
        .def("provide_observation", &Environment::provide_observation); 

    // py::class_<Histogram, Belief>(m, "Histogram")
    py::class_<Histogram, PyHistogram, std::shared_ptr<Histogram>> histogram(m, "Histogram", belief_model);
    histogram

        .def(py::init<std::map<std::shared_ptr<State>, float>>())
        .def("getHist", &Histogram::getHist)
        .def("lenHist", &Histogram::lenHist)
        .def("getitem", &Histogram::getitem)
        .def("setitem", &Histogram::setitem)
        .def("isEq", &Histogram::isEq)
        .def("mpe", &Histogram::mpe)
        .def("random", &Histogram::random)
        .def("isNormalized", &Histogram::isNormalized)
        .def("update_hist_belief", &Histogram::update_hist_belief);


    py::class_<Agent, PyAgent, std::shared_ptr<Agent>> amodel(m, "Agent");
    amodel
        
        .def(py::init_alias<std::shared_ptr<Belief>, std::shared_ptr<PolicyModel>, std::shared_ptr<TransitionModel>,
            std::shared_ptr<ObservationModel>, std::shared_ptr<RewardModel> >())
        //.def(py::init<std::shared_ptr<Belief>, std::shared_ptr<TransitionModel>, std::shared_ptr<ObservationModel>,
         //   std::shared_ptr<RewardModel> >())
        .def("gethistory", &Agent::gethistory)
        .def("update_hist", &Agent::update_hist)
        .def("init_belief", &Agent::init_belief)
        .def("belief", &Agent::belief)
        .def("cur_belief", &Agent::cur_belief)
        .def("setbelief", &Agent::setbelief)        
        .def("sample_belief", &Agent::sample_belief)
        .def("getObsModel", &Agent::getObsModel)
        .def("getTransModel", &Agent::getTransModel)
        .def("getRewardModel", &Agent::getRewardModel)
        .def("getPolicyModel", &Agent::getPolicyModel)
        
        .def("validActions", &Agent::validActions);
        //.def("update", &Agent::update)
       // .def("getRewardModel", &Agent::getRewardModel);

    py::class_<ActionPrior>(m, "ActionPrior")
        //.def(py::init<>())
        .def("get_preferred_actions", &ActionPrior::get_preferred_actions);

    py::class_<POUCT, PyPOUCT, std::shared_ptr<POUCT> >(m, "POUCT", pln)
        .def(py::init<int, float, int, float, float, int, float, std::shared_ptr<RolloutPolicy>, bool, int, std::shared_ptr<Agent>>())
        .def("getAgent", &POUCT::getAgent)
        .def("setAgent", &POUCT::setAgent)
        .def("update", &POUCT::update)
        .def("plan", &POUCT::plan, py::call_guard<py::gil_scoped_release>());
       


    py::class_<Particles, PyParticles, std::shared_ptr<Particles>> particles(m, "Particles", belief_model);
    particles

        // initialize with particles 
        .def(py::init<std::vector<std::shared_ptr<State>>>())
        .def("particles", &Particles::particles)
        // .def(py::init<std::vector<std::pair<std::shared_ptr<State>, float>>, std::string, std::function<float(std::shared_ptr<State>, std::shared_ptr<State>)>>())
        .def("add", &Particles::add)
        .def("random", &Particles::random)
        .def("update_particle_belief", &Particles::update_particle_belief);
    
    m.def("sample_generative_model", &sample_generative_model, py::arg("agent"), py::arg("state"),py::arg("action"),py::arg("discount_factor")=1);
    //m.def("sample_explict_models1", &sample_explict_models1, py::arg("T"), py::arg("O"), py::arg("R"), py::arg("state"), py::arg("a"),py::arg("discount_factor")=1);
    m.def("sample_explict_models", &sample_explict_models, py::arg("T"), py::arg("R"), py::arg("state"), py::arg("a"), py::arg("discount_factor")=1);


}