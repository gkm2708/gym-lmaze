from gym.envs.registration import register

register(
    id='lmaze-v0',
    entry_point='gym_lmaze.envs:LmazeEnv',
)

register(
    id='lmaze-v1',
    entry_point='gym_lmaze.envs:LmazeEnv_v1',
)

register(
    id='lmaze-v2',
    entry_point='gym_lmaze.envs:LmazeEnv_v2',
)
