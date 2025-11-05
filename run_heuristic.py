# run_heuristic.py
import numpy as np
from ev_routing_env import EVRoutingEnv, make_tiny_instance

def pick_action(env):
    """
    Heuristic:
    1) Pick any idle, usable vehicle.
    2) Among unserved customers that vehicle can reach (mask), choose the one
       with earliest deadline (b), tie-break by nearest distance.
    3) If no customer is reachable, try a station; else wait.
    """
    # choose a vehicle
    veh_mask = env._compute_action_mask()['vehicle']
    veh_choices = np.flatnonzero(veh_mask)
    if len(veh_choices) == 0:
        return {'vehicle': 0, 'target': env.maxC + env.maxS + 1,
                'discharge_frac': np.array([0.0], dtype=np.float32)}

    veh = int(np.random.choice(veh_choices))
    mask_v = env._compute_action_mask_for_vehicle(veh)
    tgt_mask = mask_v['target']

    # indices
    C = env.maxC
    S = env.maxS
    IDX_RETURN = C + S
    IDX_WAIT = C + S + 1

    node_u = env.v_loc[veh]

    # candidate customers (reachable by this vehicle)
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
        # earliest deadline first, then nearest
        candidates.sort(key=lambda t: (t[1], t[2]))
        best_j = candidates[0][0]
        return {'vehicle': veh, 'target': best_j,
                'discharge_frac': np.array([0.0], dtype=np.float32)}

    # else: try a station if reachable, discharge half of available above reserve
    station_choices = []
    for k in range(S):
        idx = C + k
        if tgt_mask[idx]:
            station_choices.append(idx)
    if station_choices:
        # prefer stations with higher price if in peak
        scored = []
        for idx in station_choices:
            k = idx - C
            price = env.st_price[k]
            in_peak = 1.0 if env._is_peak_time(k, env.t) else 0.0
            # score peak higher, then price, then proximity
            node_v = 1 + C + k
            dist = float(env.dist_km[node_u, node_v])
            scored.append((idx, -in_peak, -price, dist))
        scored.sort(key=lambda t: (t[1], t[2], t[3]))
        best_idx = scored[0][0]
        # discharge fraction: try 0.5 if in peak else 0
        k = best_idx - C
        frac = 0.5 if env._is_peak_time(k, env.t) else 0.0
        return {'vehicle': veh, 'target': best_idx,
                'discharge_frac': np.array([frac], dtype=np.float32)}

    # else: if at depot and nothing to do, wait; otherwise try return
    if env.v_loc[veh] != 0 and tgt_mask[IDX_RETURN]:
        return {'vehicle': veh, 'target': IDX_RETURN,
                'discharge_frac': np.array([0.0], dtype=np.float32)}

    return {'vehicle': veh, 'target': IDX_WAIT,
            'discharge_frac': np.array([0.0], dtype=np.float32)}


if __name__ == "__main__":
    inst = make_tiny_instance(1)
    env = EVRoutingEnv(inst)

    obs, info = env.reset()
    print("Initial mask:", info['action_mask'])

    # Optional: lazy inline plot using your ev_render.py if present
    renderer = None
    try:
        from ev_render import EVRenderer
        renderer = EVRenderer(env)
    except Exception:
        pass

    for step in range(200):
        action = pick_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        tag = f" [{info.get('safety_override')}]" if 'safety_override' in info else ""
        print(f"t={info['time']:.1f}, r={reward:.2f}, term={terminated}, trunc={truncated}{tag}")
        if renderer:
            renderer.draw(block=False, title="EV Routing (heuristic)")
        if terminated or truncated:
            break

    if renderer:
        renderer.draw(block=True)
