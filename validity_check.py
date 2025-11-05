from gymnasium.utils.env_checker import check_env
from ev_routing_env import EVRoutingEnv

env = EVRoutingEnv()



try:
    check_env(env)
    print("Environment passes all checks")

except Exception as e:
    print(f"Environment has issues: {e}")
