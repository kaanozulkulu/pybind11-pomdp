#include "pouct.h"
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>
#include <omp.h>

using namespace std;


std::shared_ptr<Action> VNode::argmax()
{
	std::shared_ptr<Action> action;
	float best_val =  -INFINITY;
	bool it = children.empty();
	string ba;
	for (auto const& x : children)
	{
		if (x.second->value > best_val)
		{
			ba = x.first;
			best_val = x.second->value;
		}
	}
	auto best_action = std::make_shared<Action>(ba);
	return best_action;
}

std::shared_ptr<RootVNode> RootVNode::from_vnode(std::shared_ptr<VNode> vnode, History hist)
{
	std::shared_ptr<RootVNode> rootnode = make_shared<RootVNode>(vnode->num_visits, hist);
	rootnode->children = vnode->children;
	return rootnode;
}


std::shared_ptr<Action> POUCT::plan()
{

	
	tuple<std::shared_ptr<Action>, float, int> planned_act = _search();
	_last_num_sims = get<2>(planned_act);
	_last_planning_time = get<1>(planned_act);

	return get<0>(planned_act);
}


void simulate_threaded(POUCT* p)
{
	int sims_count = 0;
	
	while (sims_count < p->_num_sims/1)
	{
		std::shared_ptr<State> state;
		state = p->_agent->sample_belief();
		p->_simulate(state, p->_agent->gethistory(), p->tree, NULL, NULL, 0);
		sims_count += 1;
	}

}
tuple<std::shared_ptr<Action>, double, int> POUCT::_search()
{
	std::chrono::time_point<std::chrono::system_clock> start_time;
	std::shared_ptr<State> state;
	std::shared_ptr<Action> best_action;
	
	int sims_count = 0;
	double time_taken = 0;
	bool stop_by_sims = _num_sims > 0 ? true : false;
	start_time = std::chrono::system_clock::now();
	History hist = _agent->gethistory();

	#pragma omp parallel num_threads(1)
    {
        simulate_threaded(this);
    }
	
	std::unordered_map<string, std::shared_ptr<QNode>> a = tree->children;
	
	best_action = tree->argmax();
	// cout << "number of children" << a.size() << endl;
	// cout << "best action " << best_action->name << endl;
	tuple<std::shared_ptr<Action>, double, int> result(best_action, time_taken, sims_count);
	return result;
}

double POUCT::_simulate(std::shared_ptr<State> state, History history, std::shared_ptr<VNode> root, std::shared_ptr<QNode> parent, std::shared_ptr<Observation> observation, int depth)
{
	
	if (depth > _max_depth)
	{
		return 0;
	}
	
	if (root == nullptr)
	{
		#pragma omp critical
	{

		if (tree == nullptr)
		{
			root = _VNode(true);
			tree = root;
		}
		else
		{
			root = _VNode(false);
		}
	}

			if (parent != NULL)
		{
			{
				auto it = parent->children.find(std::move(observation->name));
					if (it != parent->children.end())
						it->second = root;	
			}
		}
		_expand_vnode(root, history, state);
		double rollout_reward;
		rollout_reward = _rollout(state, history, root, depth);
		return rollout_reward;
	}
	
	int nsteps;
	
	std::shared_ptr<Action> action;
	action = _ucb(root);
	tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> generated = sample_generative_model(_agent, state, action);
	std::shared_ptr<State> next_state = get<0>(generated);
	std::shared_ptr<Observation> obs = get<1>(generated);
	double rd = get<2>(generated);
	nsteps = get<3>(generated);
	if (nsteps == 0)
		return rd;

	history.add(action, obs);
	
	double total_reward = rd + pow(_discount_factor,nsteps) * _simulate(next_state, history, root->children[action->name]->children[obs->name], root->children[action->name], obs, depth + nsteps);
	#pragma omp critical
	{
	root->num_visits += 1;
	root->children[action->name]->num_visits += 1;
	root->children[action->name]->value = root->children[action->name]->value + (total_reward - (root->children[action->name]->value)) / (root->children[action->name]->num_visits);
	}
	return total_reward;
}

std::shared_ptr<VNode> POUCT::_VNode(bool root)
{
	if (root)
	{
		return std::move(std::make_shared<RootVNode>(_num_visits_init, _agent->gethistory()));
	}
	else
	{
		return std::move(std::make_shared<VNode>(_num_visits_init));
	}
}
void POUCT::_expand_vnode(std::shared_ptr<VNode> vnode, History history, std::shared_ptr<State> state)
{
		#pragma omp critical
	{
	vector<std::shared_ptr<Action>> validActionsList;
	validActionsList = _agent->validActions(state, history);
	int ct = 0;
	
	for (auto const& ptr : (validActionsList))
	{
		ct = ct + 1;
		if (!vnode->children.count(ptr->name))
		{
			 vnode->children[ptr->name] = std::move(std::make_shared<QNode>(_num_visits_init, _value_init));
		}
	}
	}
}

double POUCT::_rollout(std::shared_ptr<State> state, History history, std::shared_ptr<VNode> root, int depth)
{
	std::shared_ptr<Action> action;
	float discount = 1.0;
	float total_discounted_reward = 0;
	std::shared_ptr<State> next_state;
	std::shared_ptr<Observation> observation;
	float reward;

	while (depth < _max_depth)
	{
		
		action = _rollout_policy->rollout(state.get(), history);

		tuple<std::shared_ptr<State>, std::shared_ptr<Observation>, double, int> generated = sample_generative_model(_agent, state, action);
		next_state = get<0>(generated);
		observation = get<1>(generated);
		reward = get<2>(generated);
		int nsteps = get<3>(generated);
		history.add(action, observation);
		depth += nsteps;
		total_discounted_reward += reward * discount;
		discount *= pow(_discount_factor , nsteps);
		state = next_state;
	}
	return total_discounted_reward;
}

std::shared_ptr<Action> POUCT::_ucb(std::shared_ptr<VNode> root)
{
	std::shared_ptr<Action> best_action;
	float best_value = -1000000000;
	double val;
	std::unordered_map<string, std::shared_ptr<QNode>> rtchld;
	#pragma omp critical
	{
	rtchld = root->children;
	}
	for (const auto& i : rtchld)
	{
		auto act = std::make_shared<Action>(i.first);
		if (root->children[act->name]->num_visits == 0)
		{
			val = 1000000000;
		}
		else
		{
			val = (root->children[act->name]->value) + _exploration_const * sqrt(log(root->num_visits + 1) / root->children[act->name]->num_visits);
		}
		if (val > best_value)
		{
			best_action = act;
			best_value = val;
		}
	}
	return best_action;
}

void POUCT::update(std::shared_ptr<Action> real_action, std::shared_ptr<Observation> real_observation)
{
	if (tree == nullptr)
	{
		cout << "No tree. Have you planned yet?" << endl;
	}
	else
	{
		int ct = 0;
		auto test = tree->children;
		
		if (test[real_action->name]->children[real_observation->name])
		{
			tree = RootVNode::from_vnode(
				tree->children[real_action->name]->children[real_observation->name],
				_agent->gethistory());
		}
		else
		{
			cout << "In pouct update: Unexpected state; child should not be None" << endl;
		}
	}
}
