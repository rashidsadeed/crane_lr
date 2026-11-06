import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env import CraneEnv, N_ACTIONS   # noqa: E402

env = CraneEnv()
print(env.action_space, env.observation_space.shape)        # Discrete(15) (13,)
print(env.observation_space.low)
print(env.observation_space.high)
for a in (0, 7, 8, 14):
    print(a, CraneEnv._decode_action(a))
decoded = {CraneEnv._decode_action(a) for a in range(N_ACTIONS)}
print("all 15 actions distinct:", len(decoded) == 15)