# visualize_env.py
import argparse
import numpy as np

from ev_routing_env import EVRoutingEnv, make_tiny_instance
from ev_render import EVRenderer


def sample_masked_random(env):
    """Pick a random VEHICLE, then a random TARGET from its vehicle-specific mask."""
    veh_mask = env._compute_action_mask()["vehicle"]
    veh_choices = np.flatnonzero(veh_mask)
    if len(veh_choices) == 0:
        return {
            "vehicle": 0,
            "target": env.maxC + env.maxS + 1,  # WAIT
            "discharge_frac": np.array([0.0], dtype=np.float32),
        }
    v = int(np.random.choice(veh_choices))
    tgt_mask = env._compute_action_mask_for_vehicle(v)["target"]
    tgt_choices = np.flatnonzero(tgt_mask)
    if len(tgt_choices) == 0:
        return {
            "vehicle": v,
            "target": env.maxC + env.maxS + 1,  # WAIT
            "discharge_frac": np.array([0.0], dtype=np.float32),
        }
    target = int(np.random.choice(tgt_choices))
    return {
        "vehicle": v,
        "target": target,
        "discharge_frac": np.array([np.random.rand()], dtype=np.float32),
    }


def heuristic_action(env):
    """
    Heuristic:
    1) Pick any idle, usable vehicle.
    2) Among reachable & feasible customers, choose earliest deadline b (tie -> nearest).
    3) If no customer is reachable, try stations (prefer in-peak w/ higher price).
    4) Else return if possible; else wait.
    """
    veh_mask = env._compute_action_mask()["vehicle"]
    veh_choices = np.flatnonzero(veh_mask)
    if len(veh_choices) == 0:
        return {
            "vehicle": 0,
            "target": env.maxC + env.maxS + 1,  # WAIT
            "discharge_frac": np.array([0.0], dtype=np.float32),
        }

    v = int(np.random.choice(veh_choices))
    mask_v = env._compute_action_mask_for_vehicle(v)
    tgt_mask = mask_v["target"]

    C, S = env.maxC, env.maxS
    IDX_RETURN = C + S
    IDX_WAIT = C + S + 1

    node_u = env.v_loc[v]

    # Customers: earliest deadline b, then nearest
    candidates = []
    for j in range(C):
        if not tgt_mask[j]:
            continue
        if env.cust_demand[j] <= 0.0 or env.c_served[j]:
            continue
        a, b = env.cust_tw[j]
        node_v = 1 + j
        dist = float(env.dist_km[node_u, node_v])
        candidates.append((j, b, dist))
    if candidates:
        candidates.sort(key=lambda t: (t[1], t[2]))
        best_j = candidates[0][0]
        return {
            "vehicle": v,
            "target": best_j,
            "discharge_frac": np.array([0.0], dtype=np.float32),
        }

    # Stations: prefer in-peak, higher price, then nearer
    station_choices = []
    for k in range(S):
        idx = C + k
        if tgt_mask[idx]:
            node_v = 1 + C + k
            in_peak = 1.0 if env._is_peak_time(k, env.t) else 0.0
            price = env.st_price[k]
            dist = float(env.dist_km[node_u, node_v])
            station_choices.append((idx, -in_peak, -price, dist))
    if station_choices:
        station_choices.sort(key=lambda x: (x[1], x[2], x[3]))
        best_idx = station_choices[0][0]
        k = best_idx - C
        frac = 0.5 if env._is_peak_time(k, env.t) else 0.0
        return {
            "vehicle": v,
            "target": best_idx,
            "discharge_frac": np.array([frac], dtype=np.float32),
        }

    # Return if possible; otherwise wait
    if tgt_mask[IDX_RETURN]:
        return {"vehicle": v, "target": IDX_RETURN,
                "discharge_frac": np.array([0.0], dtype=np.float32)}
    return {"vehicle": v, "target": IDX_WAIT,
            "discharge_frac": np.array([0.0], dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "heuristic"], default="random",
                        help="Policy to drive the env.")
    parser.add_argument("--seed", type=int, default=1, help="Instance seed")
    parser.add_argument("--steps", type=int, default=400, help="Max steps to simulate")
    args = parser.parse_args()

    # Build env and renderer
    inst = make_tiny_instance(args.seed)
    env = EVRoutingEnv(inst)
    renderer = EVRenderer(env)

    obs, info = env.reset()
    print("Initial mask:", info["action_mask"])

    for step in range(args.steps):
        if args.mode == "random":
            action = sample_masked_random(env)
        else:
            action = heuristic_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        tag = f" [{info.get('safety_override')}]" if "safety_override" in info else ""
        print(f"t={info['time']:.1f}, r={reward:.2f}, term={terminated}, trunc={truncated}{tag}")

        renderer.draw(block=False, title=f"EV Routing ({args.mode})")

        if terminated or truncated:
            break

    # keep the final plot on screen
    renderer.draw(block=True, title=f"EV Routing ({args.mode})")


if __name__ == "__main__":
    main()
