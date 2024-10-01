from multiprocessing import Process, Pipe
import gym


def worker(conn, env, use_compositional_split, compositional_test_splits):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
                if use_compositional_split:
                    while any([j in obs['mission'] for j in compositional_test_splits]):
                        obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()  
            if use_compositional_split:
                while any([j in obs['mission'] for j in compositional_test_splits]):
                    obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, use_compositional_split=False, compositional_test_splits=None):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.use_compositional_split = use_compositional_split
        self.compositional_test_splits = compositional_test_splits

        # configure test seeds
        self.leftout_seeds = {}
        if self.use_compositional_split:
            self.leftout_seeds = {k: [] for k in self.compositional_test_splits}
            env_id = self.envs[0].spec.id
            assert all([e.spec.id == env_id for e in self.envs[1:]]) # our assumption
            for i in range(15000):
                e = gym.make(env_id)
                e.seed(i)
                m = e.reset()['mission']
                for j in self.compositional_test_splits:
                    if j in m:
                        self.leftout_seeds[j].append(i)
            print('Left out seeds: ', {k: len(v) for k, v in self.leftout_seeds.items()})

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, self.use_compositional_split, self.compositional_test_splits))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        obs = self.envs[0].reset()
        if self.use_compositional_split:
            while any([j in obs['mission'] for j in self.compositional_test_splits]):
                obs = self.envs[0].reset()
        results = [obs] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
            if self.use_compositional_split:
                while any([j in obs['mission'] for j in self.compositional_test_splits]):
                    obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
