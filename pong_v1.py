import sys
import gym
from DqnAgent import DqnAgent

env_name = sys.argv[1] if len(sys.argv) > 1 else "Pong-v0"
env = gym.make(env_name)

agent = DqnAgent(state_size=env.observation_space.shape,
              number_of_actions=env.action_space.n,
              save_name=env_name)

num_episodes = 200

for episode in xrange(num_episodes):

    print("Episode Id:\t{}".format(episode))
    next_state = env.reset()

    agent.new_episode()

    done = False
    total_cost = 0.0
    total_reward = 0.0
    frame = 0

    while not done:
        frame += 1
        # env.render()

        action, values = agent.act(next_state)
        #action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        if reward!=0:
            print "Reward:", reward
        total_cost += agent.observe(reward)
        total_reward += reward

    print "total reward", total_reward
    print "mean cost", total_cost/frame
