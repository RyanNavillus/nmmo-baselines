import multiprocessing as mp


def worker(remote, env_fn):
    """
    The worker function run in each subprocess.

    Args:
        remote (Connection): The pipe connection used to receive commands and send results.
        env_fn (callable): A function that creates and returns a new PettingZoo environment.
    """
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break  # Pipe was closed

        if cmd == 'reset':
            observation, info = env.reset()
            term = {agent: False for agent in env.agents}
            trunc = {agent: False for agent in env.agents}
            remote.send((observation, info))
        elif cmd == 'step':
            observation, reward, term, trunc, info = env.step(data)
            if all([a or b for a, b in zip(term.values(), trunc.values())]):
                observation, _ = env.reset()
            remote.send((observation, reward, term, trunc, info))
        elif cmd == 'render':
            # Here, data can be the render mode (e.g., 'human').
            result = env.render(mode=data)
            remote.send(result)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command '{cmd}'")


class PettingZooAsyncVectorEnv:
    """
    An asynchronous vectorized wrapper for PettingZoo parallel environments using multiprocessing.

    Each environment is run in its own process, and communication is handled via Pipes.
    The class supports asynchronous stepping with step_async() and step_wait() methods.

    Example:
        from pettingzoo.butterfly import pistonball_v4

        def make_env():
            # Create a new instance of a PettingZoo parallel environment.
            return pistonball_v4.env()

        # Create a list of environment constructor functions.
        env_fns = [make_env for _ in range(4)]
        vec_env = PettingZooAsyncVectorEnv(env_fns)

        # Reset all environments.
        observations = vec_env.reset()

        # Compute actions for each environment (each is a dict mapping agent names to actions).
        actions = []
        for obs in observations:
            act = {agent: my_policy(obs[agent]) for agent in vec_env.agents}
            actions.append(act)

        # Asynchronously step all environments.
        vec_env.step_async(actions)
        obs, rewards, dones, infos = vec_env.step_wait()

        # (Optionally) render the first environment.
        vec_env.render(mode='human')

        # Close all environments.
        vec_env.close()
    """

    def __init__(self, env_fns):
        """
        Args:
            env_fns (list of callables): A list of functions, each returning a new PettingZoo environment.
        """
        self.num_envs = len(env_fns)
        # Create a pair of connection objects for each environment.
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        # Create a process for each environment.
        self.processes = [
            mp.Process(target=worker, args=(work_remote, env_fn))
            for work_remote, env_fn in zip(self.work_remotes, env_fns)
        ]
        # Set processes as daemonic and start them.
        for p in self.processes:
            p.daemon = True
            p.start()
        # In the main process, we only use one end of the pipe.
        for work_remote in self.work_remotes:
            work_remote.close()

        self.waiting = False  # Flag to ensure proper asynchronous behavior.

        # To retrieve the agent names (assumed to be the same across environments),
        # we do a synchronous reset on the first environment.
        observations, infos = self.reset()
        # observations is a list (one per environment); here we use the first.
        self.agents = list(observations[0].keys())

    def reset(self):
        """
        Resets all environments synchronously.

        Returns:
            List of observation dicts, one per environment.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        outputs = [remote.recv() for remote in self.remotes]
        return [o[0] for o in outputs], [o[1] for o in outputs]

    def step_async(self, actions):
        """
        Asynchronously sends the step command to all environments.

        Args:
            actions (list of dict): A list of action dictionaries, one per environment.
                Each dictionary maps agent names to actions.
        """
        if self.waiting:
            raise RuntimeError("step_async() has already been called and results are pending.")
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        """
        Waits for the results of the asynchronous step commands.

        Returns:
            A tuple of lists: (observations, rewards, dones, infos), each a list with one
            entry per environment.
        """
        if not self.waiting:
            raise RuntimeError("step_wait() called without a pending step_async() call.")
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # Unpack the tuple (observation, reward, done, info) from each environment.
        all_obs, all_rewards, all_terms, all_truncs, all_infos = zip(*results)
        return list(all_obs), list(all_rewards), list(all_terms), list(all_truncs), list(all_infos)

    def step(self, actions):
        """
        A synchronous step that combines step_async() and step_wait().

        Args:
            actions (list of dict): A list of action dictionaries, one per environment.

        Returns:
            (observations, rewards, dones, infos): The results from stepping each environment.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        """
        Renders the first environment.

        Args:
            mode (str): The mode to pass to the environment's render() method.

        Returns:
            The result of the environment's render() method.
        """
        self.remotes[0].send(('render', mode))
        return self.remotes[0].recv()

    def close(self):
        """
        Closes all environments and terminates their processes.
        """
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception:
                pass  # In case the remote is already closed.
        for p in self.processes:
            p.join()
