import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger

from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from fish_agent import FishAgent
from rlcard.utils import print_card
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy

def train(args):
    
    # Check whether gpu is available
    device = get_device()
    args.log_dir += datetime.now().strftime('-%d-%H_%M_%S')
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})
    save_path = os.path.join(args.log_dir, 'model.pth')
    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        if(os.path.exists(save_path) and not args.init):
            print(f"Loading existing model from {save_path}")
            agent = torch.load(save_path)
        else:
            from rlcard.agents import DQNAgent
            agent = DQNAgent(num_actions=env.num_actions,
                            state_shape=env.state_shape[0],
                            mlp_layers=args.mlp_shape,
                            discount_factor = 0.7,
                            epsilon_decay_steps = int(float(args.num_episodes) * args.epsilon_decay_steps), # Set decay schedule same as episodes
                            update_target_estimator_every = args.update_target_estimator_every,
                            device=device)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          hidden_layers_sizes=[64,64],
                          q_mlp_layers=[64,64],
                          device=device)
    # agents = [agent]
    # for _ in range(env.num_players):
    #     agents.append(RandomAgent(num_actions=env.num_actions))
    if(args.villan == 'fish'):
        agents = [agent, FishAgent(num_actions=env.num_actions)]
    elif(args.villan == 'random'):
        agents = [agent, RandomAgent(num_actions=env.num_actions)]
    elif(args.villan == 'human'):
        agents = [agent, HumanAgent(num_actions=env.num_actions)]
    env.set_agents(agents)
    rounds, switched = [], [0]
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                reward, ev = tournament(env, args.num_games)
                reward = reward[0]
                ev = ev[0]
                logger.log_performance(episode, reward, ev)
                if(args.progressive_train):
                    rounds.append(reward)
                    score_queue_size, stable_range = 5, 5.0
                    if(len(rounds) > score_queue_size):
                        rounds.pop(0)
                    if(max(rounds) - min(rounds) < stable_range and len(rounds) == score_queue_size and episode - switched[-1] > args.evaluate_every * 10):
                        env.set_agents([agent, deepcopy(agent)])
                        switched.append(episode)

    switched.pop(0)
    # Plot the learning curve
    plt.figure(figsize = (20, 10))
    df = pd.read_csv(os.path.join(args.log_dir, 'performance.csv'))
    plt.plot(df['timestep'], df['reward'], label = 'reward', color = 'g')
    plt.plot(df['timestep'], df['ev'], label = 'ev', color = 'r')
    switch_plot_x, switch_plot_y = switched, []
    for ep in switch_plot_x:
        switch_plot_y.append(df[df['timestep'] == ep]['reward'].values[0])
    plt.scatter(switch_plot_x, switch_plot_y)
    plt.xlabel('Episode')
    plt.ylabel('Chips')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig(os.path.join(args.log_dir, 'fig-{}.png'.format(datetime.now().strftime("%d-%H_%M_%S"))))

    # Save model
    torch.save(agent, save_path)
    print('Model saved in', save_path)
    with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('switched model {} times:\n{}\n'.format(len(switched), switched))

    if(args.play):
        test_env = rlcard.make(args.env, config={'seed': args.seed})
        agents = [HumanAgent(num_actions=env.num_actions), agent]
        test_env.set_agents(agents)
        evaluate_my_model(test_env)
    

def evaluate_my_model(env):
    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

        print('\n=============== Community Card ===============')
        print_card(state['public_cards'])
        # Let's take a look at what the agent card is
        print('===============     Cards all Players    ===============')
        for hands in env.get_perfect_information()['hand_cards']:
            print_card(hands)

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        if(input("Press any key to continue...") == 'q'):
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--env', type=str, default='no-limit-holdem')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_games', type=int, default=2000) # during evaluation (tournament function), how many times to simulate
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='log/nl_holdem_dqn')
    parser.add_argument('--epsilon_decay_steps', type=float, default=0.5)
    parser.add_argument('--update_target_estimator_every', type=int, default=5000)
    parser.add_argument('--mlp_shape', nargs = '+', type = int, default=[128])
    parser.add_argument('--villan', type=str, default='fish')
    parser.add_argument('--progressive_train', type=int, default=0)
    parser.add_argument('--init', type=int, default=1)
    parser.add_argument('--play', type=int, default=0)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
