#include "basics.h"
#include <iostream>
#include <typeinfo>
#include <set>
#include <algorithm>
#include <random>
#include <map>
#include <vector>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
using namespace std;


void Particles::add(std::shared_ptr<State> particle)
{
	_particles.push_back(particle);
}


std::shared_ptr<State> Particles::random()
{

	int randIndex = rand() % _particles.size();
	return _particles[randIndex];
}

std::shared_ptr<Particles> Particles::update_particle_belief(std::vector<std::shared_ptr<State>> old_particles, std::shared_ptr<Action> real_act, std::shared_ptr<Observation> real_obs, std::shared_ptr<ObservationModel> O, std::shared_ptr<TransitionModel> T)
{
	std::vector<std::shared_ptr<State>> filtered_particles;
	for (auto const& old_particle : old_particles)
	{
		std::shared_ptr<State> new_state = T->sample(old_particle.get(), real_act.get());
		std::shared_ptr<Observation> obs = O->sample(new_state.get(), real_act.get());

		
		if (obs->name == real_obs->name){
			filtered_particles.push_back(old_particle);
		}
	
	}
	
	if (filtered_particles.size() == 0){
		cout << "Empty filtered particles" << endl;
	}
	while (filtered_particles.size() < old_particles.size()){
		int randIndex = rand() % filtered_particles.size();
		filtered_particles.push_back(filtered_particles[randIndex]);
	}

	
	
	std::shared_ptr<Particles> updatedParticles(new Particles(filtered_particles));
	return updatedParticles;
}

//PYBIND11_MAKE_OPAQUE(std::map<std::shared_ptr<State>, float>);
std::map<std::shared_ptr<State>, float> Histogram::getHist()
{
	return _histogram;
}
int Histogram::lenHist()
{
	return _histogram.size();
}
float Histogram::getitem(std::shared_ptr<State> st)
{
	return _histogram[st];
}
void Histogram::setitem(std::shared_ptr<State> st, float prob)
{
	_histogram[st] = prob;
}
bool Histogram::isEq(std::shared_ptr<Histogram> b)
{
	return (_histogram == ( b->getHist()));
}
std::shared_ptr<State> Histogram::mpe()
{
	//From here: https://stackoverflow.com/questions/30611709/find-element-with-max-value-from-stdmap

	auto x = std::max_element((_histogram).begin(), (_histogram).end(),
		[](const pair<std::shared_ptr<State>, float>& p1, const pair<std::shared_ptr<State>, float>& p2) {
			return p1.second < p2.second; });
	return x->first;
}
std::shared_ptr<State> Histogram::random()
{
	auto sel = _histogram.begin();
	std::advance(sel, rand() % _histogram.size());
	std::shared_ptr<State> random_key = sel->first;
	return random_key;
}
bool Histogram::isNormalized(double eps = 1e-9)
{
	float prob_sum = 0;
	for (auto const& x : _histogram)
	{
		prob_sum += x.second;
	}
	if (abs(1 - prob_sum) < eps)
		return true;
	else
		return false;
}

std::shared_ptr<Histogram> Histogram::update_hist_belief(std::shared_ptr<Action> real_act, std::shared_ptr<Observation> real_obs, std::shared_ptr<ObservationModel> O, std::shared_ptr<TransitionModel> T, bool normalize = true, bool static_transition = false)
{
	std::map<std::shared_ptr<State>, float> new_histogram;
	double total_prob = 0;
	for (auto const& next_state : _histogram)
	{
		double obs_prob = O->probability(real_obs.get(), next_state.first.get(), real_act.get());
		double trans_prob = 0;
		if (!static_transition)
		{

			for (auto const& state : _histogram)
			{
				trans_prob += T->probability(next_state.first.get(), state.first.get(), real_act.get()) * getitem(state.first);
			}
		}
		else
		{
			trans_prob = getitem(next_state.first);
		}
		new_histogram[next_state.first] = obs_prob * trans_prob;
		total_prob += new_histogram[next_state.first];
	}
	if (normalize)
	{
		for (auto const& state : new_histogram)
		{
			if (total_prob > 0)
			{
				new_histogram[state.first] /= total_prob;
			}
		}
	}
	std::shared_ptr<Histogram> updatedhist(new Histogram(new_histogram));
	return updatedhist;
}


History Agent::gethistory()
{
	return hist;
}

void Agent::update_hist(std::shared_ptr<Action> act, std::shared_ptr<Observation> obs)
{
	hist.add(act,obs);
}

std::shared_ptr<Belief> Agent::init_belief()
{
	return belief_;
}

std::shared_ptr<Belief> Agent::belief()
{
	return _cur_belief;
}

Belief* Agent::cur_belief()
{
	return 	_cur_belief.get();
	
}

void Agent::setbelief(std::shared_ptr<Belief> bel, bool prior)
{
	_cur_belief = bel;
	{
		belief_ = bel;
	}

}

std::shared_ptr<State> Agent::sample_belief()
{
	return _cur_belief->random();
}

std::shared_ptr<ObservationModel> Agent::getObsModel()
{
	return O_;
}

std::shared_ptr<TransitionModel> Agent::getTransModel()
{
	return T_;
}

std::shared_ptr<RewardModel> Agent::getRewardModel()
{
	return R_;
}

std::shared_ptr<PolicyModel> Agent::getPolicyModel()
{
	return pi_;
}

/*void Agent::update(Action* act, Observation* obs)
{
}*/

vector<std::shared_ptr<Action>> Agent::validActions(std::shared_ptr<State> state, History history)
{
	std::shared_ptr<PolicyModel> pm = getPolicyModel();
	vector<std::shared_ptr<Action>> actlist = (pi_->get_all_actions(state.get(), history));
	
	return actlist;
}

std::shared_ptr<State> Environment::getstate() {
	return state_;
}

std::shared_ptr<TransitionModel> Environment::transitionmodel()
{
	return T_;
}

std::shared_ptr<RewardModel> Environment::reward_model()
{
	return R_;
}

double Environment::state_transition(std::shared_ptr<Action> action, float discount_factor)
{
	tuple<std::shared_ptr<State>, double, int> result;
	std::shared_ptr<TransitionModel> Tr = transitionmodel();
	std::shared_ptr<RewardModel> Re = reward_model();
	std::shared_ptr<State> st = getstate();
	
	result = sample_explict_models(Tr,Re, st, action, discount_factor);
	
	apply_transition(get<0>(result));
	
	return get<1>(result);
}

tuple<std::shared_ptr<State>, double> Environment::state_transition_sim(std::shared_ptr<Action> action, float discount_factor)
{
	tuple<std::shared_ptr<State>, double, int> result;
	result = sample_explict_models(transitionmodel(), reward_model(), state_, action, discount_factor);
	tuple <std::shared_ptr<State>, double> retRes(get<0>(result), get<1>(result));
	return retRes;
}

void Environment::apply_transition(std::shared_ptr<State> next_st)
{
	
	state_ = next_st;

}

tuple<std::shared_ptr<Observation>, double> Environment::execute(std::shared_ptr<Action> act, std::shared_ptr<ObservationModel> Omodel)
{
	double reward = state_transition(act);
	std::shared_ptr<Observation> obs = provide_observation(Omodel, act);
	tuple<std::shared_ptr<Observation>, double> result(obs, reward);
	return result;
}

std::shared_ptr<Observation> Environment::provide_observation(std::shared_ptr<ObservationModel> Omodel, std::shared_ptr<Action> act)
{
	return Omodel->sample(getstate().get(), act.get());
}

tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_generative_model(std::shared_ptr<Agent> agent, std::shared_ptr<State> state, std::shared_ptr<Action> action, float discount_factor)

{
	tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> result;
	result = sample_explict_models1(agent->getTransModel(), agent->getObsModel(), agent->getRewardModel(), state, action, discount_factor);
	return result;
}

tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_explict_models1(std::shared_ptr<TransitionModel> T, std::shared_ptr<ObservationModel> O, std::shared_ptr<RewardModel> R, std::shared_ptr<State> state, std::shared_ptr<Action> a, float discount_factor)
{
	int nsteps = 0;
	// cout << "action name" << endl;
	
	tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double> tups;
	// tups = T->full_sample(state.get(), a.get());
	// // 1 helper to to call all 3 samples, return list of next state reward and obs
    // // print first element in tuple
	// // cout << "first element in tuple: " << get<0>(tups)->name << endl;
	// // cout << "second element in tuple: " << get<1>(tups)->name << endl;
	// // cout << "third element in tuple: " << get<2>(tups) << endl;

	// std::shared_ptr<State> next_st = get<0>(tups);
	// std::shared_ptr<Observation> obs = get<1>(tups);
	// double reward = get<2>(tups);
	nsteps += 1;
	// tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> res(get<0>(tups),  get<1>(tups), get<2>(tups), nsteps);


	std::shared_ptr<State> next_st = T->sample(state.get(), a.get());
	// // //cout << "Done transitioning to ns" << endl;
	double reward = R->sample(state.get(), a.get(), next_st.get());
	// nsteps += 1;
	// // cout << " sampling" << endl;
	std::shared_ptr<Observation> obs = O->sample(next_st.get(), a.get());
	// cout << "Obs sampling next" << obs->name << endl;
	tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> res(next_st, obs, reward, nsteps);
	// cout << "Obs sampling done" << endl;
	return res;
}
// tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> sample_explict_models2(std::shared_ptr<Agent> agent, std::shared_ptr<State> state, std::shared_ptr<Action> a){
// 	int nsteps = 0;
// 	std::tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double> result;
// 	result = full_sample(agent, state.get(), a.get());
// 	nsteps += 1;
// 	tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> res(get<0>(result), get<1>(result), get<2>(result), nsteps);
// 	return res;
// }
tuple<std::shared_ptr<State>, double, int> sample_explict_models(std::shared_ptr<TransitionModel> T, std::shared_ptr<RewardModel> R, std::shared_ptr<State> state, std::shared_ptr<Action> a, float discount_factor)
{
	int nsteps = 0;
	
	std::shared_ptr<State> next_st = (T->sample(state.get(), a.get()));
	
	double reward = R->sample(state.get(), a.get(), next_st.get());
	
	nsteps += 1;
	tuple<std::shared_ptr<State>, double, int> res(next_st, reward, nsteps);
	return res;
}
