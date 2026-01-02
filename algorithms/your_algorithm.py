import numpy as np 
import torch
import time
import tqdm   # kept as template import (unused, but allowed)

from common import PolicyNetwork
from utils import setup_model_saving


def your_optimization_alg(env, policy_net=None, *,
                          max_episodes=200,
                          max_time=60.,
                          ):    

    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm = "Your algorithm")
    
    # Initialize buffers to store data for plotting
    plot_data = {'reward_history': [],
                 'timesteps':[]
                 }
    
    # Initialize policies (use provided network if passed from notebook)
    if policy_net is None:
        policy_net = PolicyNetwork(input_size=env.observation_space.shape[0],
                                   output_size=env.action_space.shape[0],
                                   )
    
    start_time = time.time()
    best_reward = -np.inf
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------  

    # Ensure notebook plotting keys exist (non-invasive)
    plot_data.setdefault('best_reward_history', [])
    plot_data.setdefault('std_history', [])

    # ---------- One-time initialisation ----------
    device = torch.device("cpu")
    policy_net.to(device)

    class ValueNetwork(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    value_net = ValueNetwork(env.observation_space.shape[0]).to(device)

    # PPO hyperparameters
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    entropy_coef = 0.001
    value_coef = 0.5

    lr_pi = 3e-4
    lr_v = 1e-3

    ppo_epochs = 6
    minibatch_size = 128
    target_batch_steps = 512   # small enough to guarantee updates

    # Exploration std (buffer-frozen)
    action_std = 0.6
    action_std_min = 0.15
    action_std_decay = 0.995

    pi_optim = torch.optim.Adam(policy_net.parameters(), lr=lr_pi)
    v_optim = torch.optim.Adam(value_net.parameters(), lr=lr_v)

    act_high = env.action_space.high.astype(np.float32)
    act_high_t = torch.tensor(act_high, dtype=torch.float32, device=device)
    act_low_t = torch.zeros_like(act_high_t)

    def make_dist(mean, std):
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            1
        )

    def env_action(a_sample):
        # safer: round then clip, return int32
        a = a_sample.detach().cpu().numpy()
        a = np.rint(a)
        a = np.clip(a, 0.0, act_high)
        return a.astype(np.int32)

    def compute_gae(rewards, values, dones):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + values[:-1]

    # Rollout buffer
    buf_s, buf_a, buf_r, buf_d, buf_lp, buf_v = [], [], [], [], [], []
    buf_steps = 0
    total_steps = 0
    last_state = None
    last_done = True

    std_buffer = action_std  # std frozen per buffer

    def ppo_update():
        nonlocal buf_steps, std_buffer, action_std

        if len(buf_s) < minibatch_size:
            return

        # Terminal bootstrap handling
        if last_done:
            v_last = 0.0
        else:
            with torch.no_grad():
                s_last = torch.tensor(last_state, dtype=torch.float32, device=device)
                v_last = value_net(s_last).item()

        rewards = np.asarray(buf_r, dtype=np.float32)
        dones = np.asarray(buf_d, dtype=np.float32)
        values = np.asarray(buf_v + [v_last], dtype=np.float32)

        adv, ret = compute_gae(rewards, values, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = np.clip(adv, -5.0, 5.0)

        S = torch.tensor(np.asarray(buf_s), dtype=torch.float32, device=device)
        A = torch.tensor(np.asarray(buf_a), dtype=torch.float32, device=device)
        OLD_LP = torch.tensor(np.asarray(buf_lp), dtype=torch.float32, device=device)
        ADV = torch.tensor(adv, dtype=torch.float32, device=device)
        RET = torch.tensor(ret, dtype=torch.float32, device=device)

        idx = np.arange(S.shape[0])

        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), minibatch_size):
                mb = idx[start:start+minibatch_size]

                # means = torch.clamp(policy_net(S[mb]), 0.0, act_high_t)
                means = torch.clamp(policy_net(S[mb]), act_low_t, act_high_t)
                dist = make_dist(means, std_buffer)

                new_lp = dist.log_prob(A[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - OLD_LP[mb])
                surr1 = ratio * ADV[mb]
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * ADV[mb]

                pi_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                v_loss = (value_net(S[mb]) - RET[mb]).pow(2).mean()

                pi_optim.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                pi_optim.step()

                v_optim.zero_grad()
                (value_coef * v_loss).backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                v_optim.step()

        # Clear buffer and decay std safely
        buf_s.clear(); buf_a.clear(); buf_r.clear()
        buf_d.clear(); buf_lp.clear(); buf_v.clear()
        buf_steps = 0

        action_std = max(action_std_min, action_std * action_std_decay)
        std_buffer = action_std

    # -------------------------------
    # Main training loop (NO tqdm)
    # -------------------------------
    for episode in range(int(max_episodes)):

        state = env.reset()
        # Doc: some env implementations return None from reset(); fall back to env.state.
        # If neither is available, raise a clear error to help debugging.
        if state is None:
            state = getattr(env, "state", None)
        if state is None:
            raise RuntimeError("Environment.reset() returned None and 'env.state' not found. Ensure the environment returns the initial state from reset().")

        done = False
        ep_return = 0.0

        std_buffer = action_std   # freeze std for this buffer

        while not done:
            s = torch.tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                # mean = torch.clamp(policy_net(s), 0.0, act_high_t)
                mean = torch.clamp(policy_net(s), act_low_t, act_high_t)
                dist = make_dist(mean, std_buffer)
                a_sample = dist.sample()
                logp = dist.log_prob(a_sample)
                v = value_net(s)

            a_exec = env_action(a_sample)
            next_state, reward, done, _ = env.step(a_exec)

            buf_s.append(s.cpu().numpy())
            buf_a.append(a_sample.cpu().numpy())
            buf_r.append(float(reward))
            buf_d.append(1.0 if done else 0.0)
            buf_lp.append(float(logp.cpu().item()))
            buf_v.append(float(v.cpu().item()))

            ep_return += reward
            state = next_state
            buf_steps += 1
            total_steps += 1
            last_state = state
            last_done = done

            if buf_steps >= target_batch_steps:
                ppo_update()

            if (time.time()-start_time) > max_time:
                break

        plot_data['reward_history'].append(ep_return)
        plot_data['timesteps'].append(total_steps)

        # Keep best/std histories for plotting cells (non-invasive)
        plot_data['best_reward_history'].append(best_reward)
        if len(plot_data['reward_history']) > 1:
            plot_data['std_history'].append(float(np.std(plot_data['reward_history'][-20:])))
        else:
            plot_data['std_history'].append(0.0)

        if ep_return > best_reward:
            best_reward = ep_return
            try:
                # ensure save directory exists without changing top imports
                os_mod = __import__('os')
                os_mod.makedirs(os_mod.path.dirname(save_f_path) or '.', exist_ok=True)
                torch.save(policy_net.state_dict(), save_f_path)
            except Exception as e:
                print("Error saving model:", e)

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}") 
    print(f"Best reward: {best_reward}")

    team_names = ["Student1","Student2"]
    cids = ["1234", "5678"]
    question = [0,0] # 1 if RL 0 else
    
    # Return model parameters (state_dict) and plot data to match notebook expectations

    best_policy = policy_net.state_dict()
    
    return best_policy, plot_data
    #return policy_net.state_dict(), plot_data
    # return save_f_path , plot_data

# Compatibility wrapper for old notebook calls — safe, non-invasive
def your_algorithm(env, *args, **kwargs):
    """
    Backwards-compatible wrapper expected by notebooks:
    - Accepts arbitrary kwargs (e.g. param_min from SA) and forwards only supported keys.
    """
    allowed = ("max_episodes", "max_time", "seed")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return your_optimization_alg(env, **filtered)

