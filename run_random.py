from ev_routing_env import EVRoutingEnv, make_tiny_instance
from ev_render import EVRenderer

env = EVRoutingEnv(make_tiny_instance(1))
renderer = EVRenderer(env)

obs, info = env.reset()
for _ in range(100):
    # ... pick an action ...
    obs, reward, term, trunc, info = env.step(action)
    renderer.draw(block=False, title="EV Routing")
    if term or trunc:
        break

renderer.draw(block=True)
