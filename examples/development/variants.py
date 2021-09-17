from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update


stabilize = True # Set to False for vanilla LILI
stabilize_weight = 1.0
end_stabilize_weight = 1.0
num_anneal_episodes = 300
pretrain_iters = 0            # number of pretraining steps for encoder & decoder
min_episode_log = 1100 # saves images of episodes after this number 
gif_index_from_end = 100 # saves images of episodes into a gif with the last gif_index_from_end episodes
#exp_name = "changing_strategy_random_goal_speaker_listener_0"
exp_name = "circle_3_goals_beta_10"
save_path = "/Users/Documents/"

DEFAULT_KEY = "__DEFAULT_KEY__"

M = 128
REPARAMETERIZE = True
NUM_CHECKPOINTS = 10
NUM_COUPLING_LAYERS = 2

#Point2d Suboptimal
#Hyperparameters for our algorithm
# latent_dim = 3                  # dimension of latent space
# mean_only = True                # True: infer mean of latent; False: infer mean and variance 
# recon_loss = True               # True: include reconstruction loss
# encoder_size = (128,128)
# decoder_size = (128,128)

# clip_grad = True                # clip gradient (clipping value set in sac.py
# task_batch_size = 16            # number of tasks to train on in each batch


# per_task_batch_size = 16         # number of (s,a,s',r) tuples to sample from each task


# # Changed to 5 to include step time
# state_dim = 2 
# episode_length = 50 
# total_tasks = 5000

# For Driving (2D)
# latent_dim = 2                  # dimension of latent space
# mean_only = True                # True: infer mean of latent; False: infer mean and variance 
# recon_loss = True               # True: include reconstruction loss, False to remove representation learning
# encoder_size = (256,256)
# decoder_size = (256,256)
# clip_grad = True                # clip gradient (clipping value set in sac.py
# task_batch_size = 16            # number of tasks to train on in each batch
# per_task_batch_size = 9         # number of (s,a,s',r) tuples to sample from each task
# state_dim = 6 #5 6 when using ORACLE_STRATEGIES in driving.py
# episode_length = 10 
# total_tasks = 2500

#For SpeakerListener
latent_dim = 3                  # dimension of latent space
mean_only = True                # True: infer mean of latent; False: infer mean and variance 
recon_loss = True               # True: include reconstruction loss, False to remove representation learning
encoder_size = (256,256)
decoder_size = (256,256)
clip_grad = True                # clip gradient (clipping value set in sac.py
task_batch_size = 16            # number of tasks to train on in each batch
per_task_batch_size = 16         # number of (s,a,s',r) tuples to sample from each task
state_dim = 10 # 11 for oracle observation, 9 otherwise
episode_length = 50 
total_tasks = 1200


## for the CARLA
# latent_dim = 8                  # dimension of latent space
# mean_only = True                # True: infer mean of latent; False: infer mean and variance 
# recon_loss = True               # True: include reconstruction loss
# encoder_size = (128,128)
# decoder_size = (128,128)
# pretrain_iters = 0              # number of pretraining steps for encoder & decoder
# clip_grad = True                # clip gradient (clipping value set in sac.py
# task_batch_size = 16            # number of tasks to train on in each batch
# per_task_batch_size = 20        # number of (s,a,s',r) tuples to sample from each task

# state_dim = 12
# episode_length = 200
# total_tasks = 1000

assert per_task_batch_size < episode_length

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'latent_dim': latent_dim,
        'observation_keys': None,
        'observation_preprocessors_params': {}
    }
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 150,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 0,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'latent_dim': latent_dim,
        'mean_only': mean_only,
        'recon_loss': recon_loss,
        'encoder_size': encoder_size,
        'decoder_size': decoder_size,
        'pretrain_iters': pretrain_iters,
        'clip_grad': clip_grad,
        'per_task_batch_size': per_task_batch_size,

        'state_dim': state_dim,
        'episode_length': episode_length,

        'stabilize': stabilize,
        'stabilize_weight': stabilize_weight,
        'end_stabilize_weight': end_stabilize_weight,
        'num_anneal_episodes': num_anneal_episodes,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Point2DEnv': {
            DEFAULT_KEY: episode_length,
        },
        'LunarReacherContinuous': {
            DEFAULT_KEY: episode_length,
        },
        'CarlaEnv': {
            DEFAULT_KEY: episode_length,
        },
        'SpeakerListenerEnv': {
            DEFAULT_KEY: episode_length,
        },
        'Pendulum': {
            DEFAULT_KEY: 200,
        },
        'HalfCheetah': {
            DEFAULT_KEY: 50,
        },
        'SawyerReachPushPickPlaceEnv': {
            DEFAULT_KEY: 150,
        },
        'HalfCheetah-Wind': {
            DEFAULT_KEY: 50,
        },
        'HalfCheetah-Vel': {
            DEFAULT_KEY: 50,
        },
        'HalfCheetah-WindVel': {
            DEFAULT_KEY: 50,
        },
        'MinitaurGoalVelEnv': {
            DEFAULT_KEY: 100,
        },
        'SawyerReachXYZEnv': {
            DEFAULT_KEY: episode_length,
        },
    },
}

NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: int(5e5),
    'gym': {
        DEFAULT_KEY: 200,
        'Swimmer': {
            DEFAULT_KEY: int(3e2),
        },
        'Hopper': {
            DEFAULT_KEY: int(1e3),
        },
        'HalfCheetah': {
            DEFAULT_KEY: int(3e3),
        },
        'Walker2d': {
            DEFAULT_KEY: int(3e3),
        },
        'Ant': {
            DEFAULT_KEY: int(3e3),
        },
        'Humanoid': {
            DEFAULT_KEY: int(1e4),
        },
        'Pusher2d': {
            DEFAULT_KEY: int(2e3),
        },
        'HandManipulatePen': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateEgg': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateBlock': {
            DEFAULT_KEY: int(1e4),
        },
        'HandReach': {
            DEFAULT_KEY: int(1e4),
        },
        'Point2DEnv': {
            DEFAULT_KEY: int(10000),
        },
        'Reacher': {
            DEFAULT_KEY: int(200),
        },
        'Pendulum': {
            DEFAULT_KEY: 10,
        },
        'SawyerReachPushPickPlaceEnv': {
            DEFAULT_KEY: int(3e3),
        },
        'HalfCheetah-Wind': {
            DEFAULT_KEY: int(3e3),
        },
        'HalfCheetah-Vel': {
            DEFAULT_KEY: int(3e3),
        },
        'HalfCheetah-WindVel': {
            DEFAULT_KEY: int(3e3),
        },
        'MinitaurGoalVelEnv': {
            DEFAULT_KEY: int(3e3),
        },
        'LunarReacherContinuous': {
            DEFAULT_KEY: int(3e3),
        },
        'CarlaEnv':{ 
            DEFAULT_KEY: int(5e6)
        },
        'SpeakerListenerEnv':{ 
            DEFAULT_KEY: int(100000)
        },
        'Driving':{ 
            DEFAULT_KEY: int(100000)
        },
        'SawyerReachXYZEnv': {
            DEFAULT_KEY: int(100000), 
        },
    },
    'dm_control': {
        DEFAULT_KEY: 200,
        'ball_in_cup': {
            DEFAULT_KEY: int(2e4),
        },
        'cheetah': {
            DEFAULT_KEY: int(2e4),
        },
        'finger': {
            DEFAULT_KEY: int(2e4),
        },
    },
    'robosuite': {
        DEFAULT_KEY: 200,
    }
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Swimmer': {  # 2 DoF
        },
        'Hopper': {  # 3 DoF
        },
        'HalfCheetah': {  # 6 DoF
        },
        'Walker2d': {  # 6 DoF
        },
        'Ant': {  # 8 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Humanoid': {  # 17 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Pusher2d': {  # 3 DoF
            'Default-v3': {
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 1.0,
                'goal': (0, -1),
            },
            'DefaultReach-v0': {
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'ImageDefault-v0': {
                'image_shape': (32, 32, 3),
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 3.0,
            },
            'ImageReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'BlindReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            }
        },
        'Point2DEnv': {
            'Default-v0': {
                
                #'observation_keys': ('observations', 'desired_goal'),
                'observation_keys': ('observations',),
                'stabilize': stabilize, 
                'stabilize_weight': stabilize_weight,
                'episode_length': episode_length,
                'min_episode_log': min_episode_log,
                'gif_index_from_end': gif_index_from_end,
                'exp_name': exp_name,
                'save_path': save_path,
            },
            'Wall-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
        },
        'SpeakerListenerEnv': {
            'Default-v0': {
                
                #'observation_keys': ('observations', 'desired_goal'),
                'observation_keys': ('observations',),
                'stabilize': stabilize, 
                'stabilize_weight': stabilize_weight,
                'episode_length': episode_length,
                'min_episode_log': min_episode_log,
                'gif_index_from_end': gif_index_from_end,
                'exp_name': exp_name,
                'save_path': save_path,
            },
        },
        'SawyerReachXYZEnv': {
            'Default-v0': {
                
                #'observation_keys': ('observations', 'desired_goal'),
                'observation_keys': ('observations',),
                'episode_length': episode_length,
                'min_episode_log': min_episode_log,
                'gif_index_from_end': gif_index_from_end,
                'exp_name': exp_name,
                'save_path': save_path,
            },
        },
        'Driving': {
            'v0': {
                
                #'observation_keys': ('observations', 'desired_goal'),
                'min_episode_log': min_episode_log,
                'gif_index_from_end': gif_index_from_end,
                'exp_name': exp_name,
                'save_path': save_path,
            },
        },
        'LunarReacherContinuous': {
            'v2': {
                'observation_keys': ('observations',),
            },
        },
        'Sawyer': {
            task_name: {
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                'reward_shaping': tune.grid_search([True, False]),
            }
            for task_name in (
                    'Lift',
                    'NutAssembly',
                    'NutAssemblyRound',
                    'NutAssemblySingle',
                    'NutAssemblySquare',
                    'PickPlace',
                    'PickPlaceBread',
                    'PickPlaceCan',
                    'PickPlaceCereal',
                    'PickPlaceMilk',
                    'PickPlaceSingle',
                    'Stack',
            )
        },
    },
    'dm_control': {
        'ball_in_cup': {
            'catch': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'cheetah': {
            'run': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'finger': {
            'spin': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
    },
}


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 10 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_algorithm_params(universe, domain, task):
    algorithm_params = {
        'kwargs': {
            'n_epochs': get_num_epochs(universe, domain, task),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        get_algorithm_params(universe, domain, task),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            #'evaluation': tune.sample_from(lambda spec: (
            #    spec.get('config', spec)
            #    ['environment_params']
            #    ['training']
            #)),
        },
        'policy_params': get_policy_params(universe, domain, task),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'latent_dim': latent_dim,
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {},
                'latent_dim': latent_dim,
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'MultitaskReplayPool',
            'kwargs': {
                'max_size': int(episode_length),
                'episode_length': episode_length,
                'total_tasks': total_tasks,
                'per_task_batch_size': per_task_batch_size,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': get_max_path_length(universe, domain, task),
                'batch_size': task_batch_size * per_task_batch_size,
                'latent_dim': latent_dim,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def is_image_env(universe, domain, task, variant_spec):
    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs']))


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(universe, domain, task, variant_spec):
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
