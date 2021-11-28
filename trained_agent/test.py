import rlcard
env = rlcard.make('no-limit-holdem')
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
agent = DQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], mlp_layers=[32])
agents = [agent, RandomAgent(num_actions=env.num_actions)]
# agents = [RandomAgent(num_actions=env.num_actions), RandomAgent(num_actions=env.num_actions)]
env.set_agents(agents)
trajectories, payoffs = env.run(is_training=False)
# exec(open('test.py').read())