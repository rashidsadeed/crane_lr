import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env import CraneEnv, N_ACTIONS, VX_LEVELS, VL_LEVELS   # noqa: E402

env = CraneEnv()
print(env.action_space, env.observation_space.shape)        # Discrete(28) (13,)
print(env.observation_space.low)
print(env.observation_space.high)
for a in (0, len(VL_LEVELS) - 1, N_ACTIONS // 2, N_ACTIONS - 1):
    print(a, CraneEnv._decode_action(a))
decoded = {CraneEnv._decode_action(a) for a in range(N_ACTIONS)}
ok = (env.action_space.n == N_ACTIONS == len(VX_LEVELS) * len(VL_LEVELS)
      and len(decoded) == N_ACTIONS and env.observation_space.shape == (13,))
print(f"all {N_ACTIONS} actions distinct, spaces consistent:", ok)
print("ALL PASSED" if ok else "1 CHECK(S) FAILED")
