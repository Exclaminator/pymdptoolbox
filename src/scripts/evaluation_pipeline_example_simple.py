import mdptoolbox.example
import numpy as np
import mdp_base
import mdptoolbox.Robust


def simulate_forest_problem():
    # for problem details, look up https://pymdptoolbox.readthedocs.io/en/latest/api/example.html
    # An action is decided each year
    # first objective is to maintain an old forest for wildlife and
    # second objective is to make money selling cut wood.
    # after a fire, the forest returns in its oldest state
    # after cutting, the forest is in the oldest state
    S = 10  # number of states
    r1 = 40  # reward when 'wait' is performed in its oldest state
    r2 = 1  # reward when 'cut' is performed in its oldest state
    p = 0.05  # probability of fire
    P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

    ambiguous_ratio = 2

    p_up = P*ambiguous_ratio    #np.full(S, p*ambiguous_ratio)
    p_low = P/ambiguous_ratio   #np.full(S, p/ambiguous_ratio)

    discount_factor = 0.9
    # epsilon = 0.01
    # max_iter = 100

    # run mdp 1
    v1 = mdptoolbox.mdp.ValueIteration(P, R, discount=discount_factor)
    v1.run()

    # run mdp 2
    # note that the q-learner has a random factor involved on creation,
    # and therefore policies can differ when the same parameters are used as an input to train it
    v2 = mdptoolbox.mdp.QLearning(P, R, discount_factor)
    v2.run()

    # v3 is our interval model
    v3 = mdptoolbox.Robust.RobustIntervalModel(P, R, discount=discount_factor, p_lower=p_low, p_upper=p_up)
    v3.setVerbose()
    v3.run()

    # agent can either wait (0) or cut (1)
    # print the best action to take for each state
    print("value iteration policy " + str(v1.policy))
    print("q learning policy " + str(v2.policy))
    print("Robust policy " + str(v3.policy))

    # a robust policy cuts as early as possible
    # a too conservative policy would cut even when p is very low,
    # then a low average expected reward compared to the minimum loss gained
    t_max = 20
    simulation_runs = 1000

    rewards_v1 = np.zeros(simulation_runs)
    rewards_v2 = np.zeros(simulation_runs)
    rewards_v3 = np.zeros(simulation_runs)
    for i in range(simulation_runs):
        rewards_v1[i] = run_policy_on_robust_problem(v1.policy, t_max, P, R, p_up, p_low)
        rewards_v2[i] = run_policy_on_robust_problem(v2.policy, t_max, P, R, p_up, p_low)
        rewards_v3[i] = run_policy_on_robust_problem(v3.policy, t_max, P, R, p_up, p_low)

    print("v1, mean, variance, min_reward: "
          + str(np.mean(rewards_v1))+", "
          + str(np.var(rewards_v1))+", "
          + str(np.min(rewards_v1)))
    print("v2, mean, variance, min_reward: "
          + str(np.mean(rewards_v2))+", "
          + str(np.var(rewards_v2))+", "
          + str(np.min(rewards_v2)))
    print("v3, mean, variance, min_reward: "
          + str(np.mean(rewards_v3))+", "
          + str(np.var(rewards_v3))+", "
          + str(np.min(rewards_v3)))
    # todo: make plots that show the distribution of results


def run_policy_on_robust_problem(policy, t_max, P, R, p_up, p_low):
    s = 0
    total_reward = 0

    # P = np.random.uniform(p_low, p_up)

    for t in range(t_max):
        action = policy[s]
        # simulate ambiguity: get transition probability based on the interval
        PP = np.random.uniform(p_low[action, s], p_up[action, s])

        # normalize to make sum = 1
        PP = PP/sum(PP)

        s_new = np.random.choice(a=len(PP), p=PP)
        RR = R[s]
        total_reward += RR[action]
        s = s_new

    return total_reward


def test1():
    P, R = mdptoolbox.example.forest()
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi.run()
    print(vi.policy)
    # result is (0, 0, 0)


# test1()
simulate_forest_problem()
