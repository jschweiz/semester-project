import numpy as np
from gym.spaces import Dict

from .simple_replay_pool import SimpleReplayPool

EXP_HORIZON = 50 # point mass and speaker listener
#EXP_HORIZON = 10 # DRIVING



class MultitaskReplayPool(object):
    def __init__(self,
                 environment,
                 total_tasks=4000,
                 episode_length=50,
                 per_task_batch_size=8,
                 *args,
                 extra_fields=None,
                 **kwargs):
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space
    
        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space
        self._total_tasks = total_tasks
        self._current_task = 0
        self._task_pools = dict([(idx, SimpleReplayPool(
            environment=environment,
            *args,
            **kwargs
        )) for idx in range(self._total_tasks)])

        self._episode_length = episode_length
        self._per_task_batch_size = per_task_batch_size

    def add_sample(self, sample):
        self._task_pools[self._current_task].add_sample(sample)

    def terminate_episode(self):
        self._task_pools[self._current_task].terminate_episode()
        self._current_task += 1
        if self._current_task >= self._total_tasks:

            self._environment.save_gif_interactions()
            self._environment.PCA_plot_predicted_latents()
            1/0 # terminate training

    @property
    def size(self):
        total_size = 0
        for idx in range(self._total_tasks):
            total_size += self._task_pools[idx].size
        return total_size

    def add_path(self, path):
        self._task_pools[self._current_task].add_path(path)

    def random_task_indices(self, task_batch_size):
        return np.random.randint(1, self._current_task, task_batch_size)

    def random_batch(self, batch_size):
        batch = {
            'observations': {'observations': []},
            'next_observations': {'observations': []},
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timesteps': [],

            'prev_observations': {'observations': []},
            'prev_next_observations': {'observations': []},
            'prev_actions': [],
            'prev_rewards': [],
            'prev_terminals': [],
        }

        tasks = self.random_task_indices(batch_size // self._per_task_batch_size)
        for idx in tasks:
            episode_length = self._task_pools[idx - 1].size
            #prev_indices = np.random.randint(0, episode_length, self._per_task_batch_size)

            prev_indices = None
            if EXP_HORIZON == 50:
                prev_indices = (np.arange(self._per_task_batch_size) + 1) * 3
            else:
                prev_indices = np.arange(self._per_task_batch_size)
            prev_indices[-1] = episode_length - 1


            episode_length = self._task_pools[idx].size
            #indices = np.random.randint(0, episode_length, self._per_task_batch_size)
            #indices = np.arange(self._per_task_batch_size) + 1
            indices = None
            if EXP_HORIZON == 50:
                indices = (np.arange(self._per_task_batch_size) + 1) * 3
            else:
                indices = np.arange(self._per_task_batch_size)
            indices[-1] = episode_length - 1

            #print(prev_indices, indices)
            

            prev_task_batch = self._task_pools[idx - 1].batch_by_indices(prev_indices)
            task_batch = self._task_pools[idx].batch_by_indices(indices)
            batch['timesteps'].append(np.expand_dims(indices, axis=-1))

            for key, value in prev_task_batch.items():
                if key in ['observations', 'next_observations']:
                    batch['prev_' + key]['observations'].append(prev_task_batch[key]['observations'])
                elif key in ['actions', 'rewards', 'terminals']:
                    batch['prev_' + key].append(prev_task_batch[key])

            for key, value in task_batch.items():
                if key in ['observations', 'next_observations']:
                    batch[key]['observations'].append(task_batch[key]['observations'])
                elif key in ['actions', 'rewards', 'terminals']:
                    batch[key].append(task_batch[key])

        for key in batch:
            if key in ['observations', 'next_observations', 'prev_observations', 'prev_next_observations']:
                batch[key]['observations'] = np.concatenate(batch[key]['observations'], 0)
            else:
                batch[key] = np.concatenate(batch[key], 0)
        # print(batch['prev_next_observations']['observations'][:, [9, 10]])
        # exit()
        return batch

    def batch_from_index(self, task_index):
        episode_length = self._task_pools[task_index].size
        #indices = np.random.randint(0, episode_length, self._per_task_batch_size)
        indices = None
        if EXP_HORIZON == 50:
            indices = (np.arange(self._per_task_batch_size) + 1) * 3
        else:
            indices = np.arange(self._per_task_batch_size)
        indices[-1] = episode_length - 1
        return self._task_pools[task_index].batch_by_indices(indices)
