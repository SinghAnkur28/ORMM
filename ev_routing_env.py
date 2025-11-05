import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any

# -------------------------------------------------------------
# EV Routing with Optional V2G Discharge — Gymnasium Environment
# -------------------------------------------------------------
#
# Formulation summary
# - Centralized single agent selects (vehicle_id, target) at each step.
# - Targets include: unserved customers, discharge stations, return to depot, wait.
# - Event-driven time advance: after an action, we advance by the travel time to the target
#   and apply service/discharge durations.
# - Observation is a fixed-size padded vector composed of global, per-vehicle, per-customer,
#   and per-station features.
# - Action masking (generic + vehicle-specific) is provided in info["action_mask"].
# - Safety layer: invalid actions are converted to a 1.0 time-unit WAIT (no huge penalty).
# - Rewards combine travel/time costs, penalties, vehicle usage, and V2G revenue.
#
# This is a self-contained environment ready to plug into Stable-Baselines3 PPO/A2C.
# For multi-instance experiments, instantiate with different problem dictionaries.
# -------------------------------------------------------------


# -----------------------
# Utility helper functions
# -----------------------

def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _safe_norm(x: float, denom: float) -> float:
    return float(x / denom) if denom > 0 else 0.0


# -----------------------
# Problem/Instance schema
# -----------------------
# instance = {
#   'seed': int,
#   'max_vehicles': int,
#   'max_customers': int,
#   'max_stations': int,
#   'depot': {'xy': (x,y)},
#   'vehicles': [
#       {'soc_init_kwh': 18.0, 'soc_max_kwh': 18.0, 'capacity': 120.0},
#       ... (length <= max_vehicles)
#   ],
#   'customers': [
#       {'xy': (x,y), 'demand': 10.0, 'tw': (a,b), 'service_time': 3.0},
#       ... (length <= max_customers)
#   ],
#   'stations': [
#       {
#         'xy': (x,y),
#         'peak_windows': [(start_t, end_t), ...],  # time units consistent with travel times
#         'price': 2.0,         # revenue per kWh during peak
#         'discharge_rate_kw': 10.0, # kW; discharge_time = e_kwh / rate
#       },
#       ... (length <= max_stations)
#   ],
#   'speed': 1.0,            # units distance per time
#   'consumption_kwh_per_km': 0.2,  # simple linear consumption
#   'horizon_T': 500.0,
#   'costs': {
#       'per_km': 1.0,
#       'per_time': 0.01,
#       'vehicle_fixed': 50.0,
#       'late_penalty': 50.0,
#       'miss_penalty': 1000.0,
#       'infeas_penalty': 1000.0,
#   },
#   'soc_reserve_kwh': 2.0,
# }


class EVRoutingEnv(gym.Env):
    """Centralized EV routing with optional V2G discharge.

    Action space:
      Dict({
        'vehicle': Discrete(max_vehicles),
        'target': Discrete(max_customers + max_stations + 2),  # customers | stations | return | wait
        'discharge_frac': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) # only used at stations
      })

    Observation: 1D float vector with the following concatenation (all normalized/padded):
      [global, vehicles[maxV*Vdim], customers[maxC*Cdim], stations[maxS*Sdim], depot[Ddim]]

    Masking:
      info['action_mask'] is a dict with boolean arrays for vehicles and targets.
      Use _compute_action_mask_for_vehicle(veh) for vehicle-specific target feasibility.
    """

    metadata = {"render.modes": []}

    def __init__(self, instance: Dict[str, Any]):
        super().__init__()
        self.instance = instance
        self.rng = np.random.default_rng(instance.get('seed', 0))

        # Settings
        self.maxV = int(instance['max_vehicles'])
        self.maxC = int(instance['max_customers'])
        self.maxS = int(instance['max_stations'])
        self.speed = float(instance.get('speed', 1.0))
        self.cons_kwh_per_km = float(instance.get('consumption_kwh_per_km', 0.2))
        self.horizon_T = float(instance.get('horizon_T', 500.0))
        self.soc_reserve_kwh = float(instance.get('soc_reserve_kwh', 2.0))
        self.costs = instance.get('costs', {})
        self.c_km = float(self.costs.get('per_km', 1.0))
        self.c_time = float(self.costs.get('per_time', 0.01))
        self.c_vehicle = float(self.costs.get('vehicle_fixed', 50.0))
        self.c_late = float(self.costs.get('late_penalty', 50.0))
        self.c_miss = float(self.costs.get('miss_penalty', 1000.0))
        self.c_infeas = float(self.costs.get('infeas_penalty', 1000.0))

        # Shapes (feature dims)
        self.Gdim = 2  # [t_norm, unserved_norm]
        self.Vdim = 6  # [x,y,q_norm,cap_norm,is_idle,tau_norm]
        self.Cdim = 7  # [x,y,demand_norm,a_norm,b_norm,served_flag,dist_to_depot_norm]
        self.Sdim = 5  # [x,y,price_norm,is_peak,dist_to_depot_norm]
        self.Ddim = 2  # [x_depot, y_depot]

        # Build action/observation spaces
        self.action_space = spaces.Dict({
            'vehicle': spaces.Discrete(self.maxV),
            'target': spaces.Discrete(self.maxC + self.maxS + 2),  # + return + wait
            'discharge_frac': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        obs_len = self.Gdim + self.maxV * self.Vdim + self.maxC * self.Cdim + \
                  self.maxS * self.Sdim + self.Ddim
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # Internal dynamic state (set in reset)
        self._reset_static()
        self.reset()

    # -------------------------
    # Static/problem setup bits
    # -------------------------
    def _reset_static(self):
        inst = self.instance
        self.depot_xy: Tuple[float, float] = tuple(inst['depot']['xy'])  # type: ignore

        # Vehicles (pad to maxV)
        self.veh_spec: List[Dict[str, float]] = []
        for i in range(self.maxV):
            if i < len(inst['vehicles']):
                v = inst['vehicles'][i]
            else:
                # pad empty vehicle with zero capacity (won't be used)
                v = {'soc_init_kwh': 0.0, 'soc_max_kwh': 0.0, 'capacity': 0.0}
            self.veh_spec.append({
                'soc_init_kwh': float(v.get('soc_init_kwh', 0.0)),
                'soc_max_kwh': float(v.get('soc_max_kwh', 0.0)),
                'capacity': float(v.get('capacity', 0.0)),
            })

        # Customers (pad to maxC)
        self.cust_xy: List[Tuple[float, float]] = []
        self.cust_demand: List[float] = []
        self.cust_tw: List[Tuple[float, float]] = []
        self.cust_service: List[float] = []
        for i in range(self.maxC):
            if i < len(inst['customers']):
                c = inst['customers'][i]
            else:
                c = {'xy': self.depot_xy, 'demand': 0.0, 'tw': (0.0, 0.0), 'service_time': 0.0}
            self.cust_xy.append(tuple(c['xy']))
            self.cust_demand.append(float(c.get('demand', 0.0)))
            self.cust_tw.append(tuple(c.get('tw', (0.0, 0.0))))
            self.cust_service.append(float(c.get('service_time', 0.0)))

        # Stations (pad to maxS)
        self.st_xy: List[Tuple[float, float]] = []
        self.st_price: List[float] = []
        self.st_peak: List[List[Tuple[float, float]]] = []
        self.st_rate: List[float] = []
        for i in range(self.maxS):
            if i < len(inst['stations']):
                s = inst['stations'][i]
            else:
                s = {'xy': self.depot_xy, 'peak_windows': [], 'price': 0.0, 'discharge_rate_kw': 1.0}
            self.st_xy.append(tuple(s['xy']))
            self.st_peak.append([tuple(w) for w in s.get('peak_windows', [])])
            self.st_price.append(float(s.get('price', 0.0)))
            self.st_rate.append(float(s.get('discharge_rate_kw', 1.0)))

        # Precompute distances (km) and travel times
        # All nodes: depot + customers + stations
        self.node_coords: List[Tuple[float, float]] = [self.depot_xy] + self.cust_xy + self.st_xy
        n_nodes = len(self.node_coords)
        self.dist_km = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for u in range(n_nodes):
            for v in range(n_nodes):
                self.dist_km[u, v] = _euclid(self.node_coords[u], self.node_coords[v])
        self.time_mat = self.dist_km / max(self.speed, 1e-6)

    # -------------
    # Gym API: reset
    # -------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t: float = 0.0
        self.step_count: int = 0

        # Vehicles dynamic state
        self.v_loc: List[int] = [0 for _ in range(self.maxV)]  # node index (0=depot)
        self.v_soc: List[float] = [v['soc_init_kwh'] for v in self.veh_spec]
        self.v_soc_max: List[float] = [v['soc_max_kwh'] for v in self.veh_spec]
        self.v_cap: List[float] = [v['capacity'] for v in self.veh_spec]
        self.v_idle: List[bool] = [True if self.v_cap[i] > 0 and self.v_soc_max[i] > 0 else False for i in range(self.maxV)]
        self.v_tau: List[float] = [0.0 for _ in range(self.maxV)]  # reserved for extensions
        self.v_used_flag: List[bool] = [False for _ in range(self.maxV)]

        # Customers served flags
        self.c_served: List[bool] = [(self.cust_demand[i] <= 0.0) for i in range(self.maxC)]

        # Accounting
        self.total_distance = 0.0
        self.total_travel_time = 0.0
        self.total_revenue = 0.0
        self.total_penalty = 0.0

        obs = self._make_obs()
        info = {"action_mask": self._compute_action_mask()}
        return obs, info

    # -----------
    # Gym API: step
    # -----------
    def step(self, action: Dict[str, Any]):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        veh = int(action['vehicle'])
        tgt = int(action['target'])
        discharge_frac = float(action.get('discharge_frac', [0.0])[0])

        # Vehicle-specific feasibility mask
        mask = self._compute_action_mask_for_vehicle(veh)
        info['action_mask'] = mask
        if (mask is None) or (not mask['vehicle'][veh]) or (not mask['target'][tgt]):
            # Safety layer: convert invalid action to WAIT (advance small tick)
            self.v_idle[veh] = True
            tick = 1.0
            self.t += tick
            reward -= self.c_time * tick
            obs = self._make_obs()
            info.update({
                'time': self.t,
                'total_distance': self.total_distance,
                'total_travel_time': self.total_travel_time,
                'total_revenue': self.total_revenue,
                'total_penalty': self.total_penalty,
                'safety_override': 'invalid_action->wait'
            })
            return obs, reward, terminated, truncated, info

        # Map target index to node / special
        # customers: 0..maxC-1, stations: maxC..maxC+maxS-1, return: -2, wait: -1
        if tgt < self.maxC:
            kind = 'customer'
            cust_idx = tgt
            node_v = 1 + cust_idx  # node index in node_coords matrix (0=depot)
        elif tgt < self.maxC + self.maxS:
            kind = 'station'
            st_idx = tgt - self.maxC
            node_v = 1 + self.maxC + st_idx
        elif tgt == self.maxC + self.maxS:
            kind = 'return'
            node_v = 0
        else:
            kind = 'wait'
            node_v = self.v_loc[veh]

        # Travel (or wait)
        node_u = self.v_loc[veh]
        dist = float(self.dist_km[node_u, node_v])
        travel_time = float(self.time_mat[node_u, node_v])

        if kind == 'wait':
            # advance a small tick to avoid infinite loops
            tick = 1.0
            self.t += tick
            self.v_idle[veh] = True
            reward -= self.c_time * tick
        else:
            # consume energy to move — guard SoC
            e_cons = dist * self.cons_kwh_per_km
            if self.v_soc[veh] - e_cons < 0.0:
                # Safety layer: not enough energy -> WAIT (no move)
                tick = 1.0
                self.t += tick
                reward -= self.c_time * tick
                obs = self._make_obs()
                info.update({
                    'time': self.t,
                    'total_distance': self.total_distance,
                    'total_travel_time': self.total_travel_time,
                    'total_revenue': self.total_revenue,
                    'total_penalty': self.total_penalty,
                    'safety_override': 'low_soc->wait'
                })
                return obs, reward, terminated, truncated, info

            # Apply travel
            self.v_soc[veh] -= e_cons
            self.v_loc[veh] = node_v
            self.t += travel_time
            self.total_distance += dist
            self.total_travel_time += travel_time
            self.v_idle[veh] = True
            self.v_used_flag[veh] = True

            # Travel cost
            reward -= self.c_km * dist
            reward -= self.c_time * travel_time

            # Arrivals handling
            if kind == 'customer':
                j = cust_idx
                if not self.c_served[j] and self.cust_demand[j] > 0.0:
                    a, b = self.cust_tw[j]
                    # Late penalty if beyond time window (only if window is meaningful)
                    if (b > a) and (self.t > b):
                        reward -= self.c_late
                        self.total_penalty += self.c_late
                    # Service time
                    st = self.cust_service[j]
                    self.t += st
                    # Serve (capacity check)
                    if self.v_cap[veh] >= self.cust_demand[j]:
                        self.v_cap[veh] -= self.cust_demand[j]
                        self.c_served[j] = True
                    else:
                        # not enough capacity — penalize lightly and do not serve
                        reward -= self.c_infeas
                        self.total_penalty += self.c_infeas

            elif kind == 'station':
                k = st_idx
                # Determine if in peak window
                is_peak = self._is_peak_time(k, self.t)
                if is_peak:
                    # discharge amount (kWh) from available above reserve
                    soc_avail = max(0.0, self.v_soc[veh] - self.soc_reserve_kwh)
                    e_kwh = soc_avail * float(np.clip(discharge_frac, 0.0, 1.0))
                    rate = max(self.st_rate[k], 1e-6)
                    t_dis = e_kwh / rate
                    self.t += t_dis
                    self.v_soc[veh] -= e_kwh
                    rev = self.st_price[k] * e_kwh
                    reward += rev
                    self.total_revenue += rev
                # else: no revenue outside peak, just a visit

        # Check termination
        all_served = all(self.c_served)
        timeout = self.t >= self.horizon_T
        if all_served or timeout:
            terminated = True
            # Vehicle usage cost
            n_used = sum(1 for f in self.v_used_flag if f)
            reward -= self.c_vehicle * n_used
            # Missed customers penalty
            n_missed = sum(1 for j in range(self.maxC) if (self.cust_demand[j] > 0.0 and not self.c_served[j]))
            reward -= self.c_miss * n_missed
            self.total_penalty += self.c_miss * n_missed + self.c_vehicle * n_used

        obs = self._make_obs()
        info.update({
            'time': self.t,
            'total_distance': self.total_distance,
            'total_travel_time': self.total_travel_time,
            'total_revenue': self.total_revenue,
            'total_penalty': self.total_penalty,
        })
        return obs, reward, terminated, truncated, info

    # ------------------
    # Observation builder
    # ------------------
    def _make_obs(self) -> np.ndarray:
        # Normalization constants
        T = max(self.horizon_T, 1e-6)
        max_q = max(1.0, max(self.v_soc_max) if len(self.v_soc_max) else 1.0)
        max_cap = max(1.0, max(self.v_cap) if len(self.v_cap) else 1.0)
        max_dist = np.max(self.dist_km) if self.dist_km.size > 0 else 1.0

        # Global
        unserved = sum(1 for j in range(self.maxC) if (self.cust_demand[j] > 0.0 and not self.c_served[j]))
        g = [
            _safe_norm(self.t, T),
            _safe_norm(unserved, max(1, self.maxC)),
        ]

        # Vehicles
        v_feats: List[float] = []
        for i in range(self.maxV):
            x, y = self.node_coords[self.v_loc[i]]
            v_feats += [
                _safe_norm(x, 100.0),
                _safe_norm(y, 100.0),
                _safe_norm(self.v_soc[i], max_q),
                _safe_norm(self.v_cap[i], max_cap),
                1.0 if self.v_idle[i] else 0.0,
                _safe_norm(self.v_tau[i], T),
            ]

        # Customers
        c_feats: List[float] = []
        for j in range(self.maxC):
            x, y = self.cust_xy[j]
            a, b = self.cust_tw[j]
            dist_dep = _euclid(self.depot_xy, (x, y))
            c_feats += [
                _safe_norm(x, 100.0),
                _safe_norm(y, 100.0),
                _safe_norm(self.cust_demand[j], max_cap),
                _safe_norm(a, T),
                _safe_norm(b, T),
                1.0 if self.c_served[j] else 0.0,
                _safe_norm(dist_dep, max_dist),
            ]

        # Stations
        s_feats: List[float] = []
        for k in range(self.maxS):
            x, y = self.st_xy[k]
            is_peak = 1.0 if self._is_peak_time(k, self.t) else 0.0
            dist_dep = _euclid(self.depot_xy, (x, y))
            s_feats += [
                _safe_norm(x, 100.0),
                _safe_norm(y, 100.0),
                _safe_norm(self.st_price[k], 10.0),
                is_peak,
                _safe_norm(dist_dep, max_dist),
            ]

        # Depot
        d = [
            _safe_norm(self.depot_xy[0], 100.0),
            _safe_norm(self.depot_xy[1], 100.0),
        ]

        vec = np.array(g + v_feats + c_feats + s_feats + d, dtype=np.float32)
        return vec

    # ----------------
    # Generic action mask (vehicle-only strict, target permissive)
    # ----------------
    def _compute_action_mask(self) -> Dict[str, np.ndarray]:
        """Return masks for vehicles and targets (generic/permissive target mask).
        Vehicle mask: vehicles that are active (non-zero spec) and idle.
        Target mask: customers not served, all stations, return+wait allowed.
        """
        veh_mask = np.zeros((self.maxV,), dtype=bool)
        for i in range(self.maxV):
            veh_mask[i] = (self.v_cap[i] > 0.0 and self.v_soc_max[i] > 0.0 and self.v_idle[i])

        tgt_mask = np.zeros((self.maxC + self.maxS + 2,), dtype=bool)
        for j in range(self.maxC):
            selectable = (not self.c_served[j]) and (self.cust_demand[j] > 0.0)
            if selectable:
                tgt_mask[j] = True
        for k in range(self.maxS):
            tgt_mask[self.maxC + k] = True
        # 'return' allowed generically; vehicle-specific mask will restrict at depot
        tgt_mask[self.maxC + self.maxS] = True
        tgt_mask[self.maxC + self.maxS + 1] = True  # wait
        return {"vehicle": veh_mask, "target": tgt_mask}

    # ----------------
    # Vehicle-specific action mask (SoC + capacity + return-at-depot handling)
    # ----------------
    def _compute_action_mask_for_vehicle(self, veh_idx: int):
        """Vehicle-specific action mask based on SoC reachability and capacity.
        Returns {'vehicle': <bool[nV]>, 'target': <bool[nTargets]>}.
        Never returns None.
        """
        base = self._compute_action_mask()
        veh_mask = base["vehicle"].copy()
        tgt_mask = np.zeros((self.maxC + self.maxS + 2,), dtype=bool)

        # If veh_idx invalid or vehicle not selectable, only allow WAIT
        if veh_idx < 0 or veh_idx >= self.maxV or not veh_mask[veh_idx]:
            tgt_mask[self.maxC + self.maxS + 1] = True  # WAIT
            return {"vehicle": veh_mask, "target": tgt_mask}

        node_u = self.v_loc[veh_idx]
        soc = self.v_soc[veh_idx]
        cap = self.v_cap[veh_idx]

        # Customers: require not served, demand>0, capacity ok, and SoC to reach
        for j in range(self.maxC):
            if self.cust_demand[j] <= 0.0 or self.c_served[j]:
                continue
            if cap < self.cust_demand[j]:
                continue
            node_v = 1 + j
            dist = float(self.dist_km[node_u, node_v])
            e_cons = dist * self.cons_kwh_per_km
            if soc - e_cons >= 0.0:
                tgt_mask[j] = True

        # Stations: require SoC to reach
        for k in range(self.maxS):
            node_v = 1 + self.maxC + k
            dist = float(self.dist_km[node_u, node_v])
            e_cons = dist * self.cons_kwh_per_km
            if soc - e_cons >= 0.0:
                tgt_mask[self.maxC + k] = True

        # Return requires (i) not already at depot, and (ii) SoC to reach depot
        dist_home = float(self.dist_km[node_u, 0])
        e_home = dist_home * self.cons_kwh_per_km
        if node_u != 0 and (soc - e_home >= 0.0):
            tgt_mask[self.maxC + self.maxS] = True  # 'return'

        # Wait always allowed
        tgt_mask[self.maxC + self.maxS + 1] = True  # 'wait'

        return {"vehicle": veh_mask, "target": tgt_mask}

    # ------------------
    # Helper / utilities
    # ------------------
    def _is_peak_time(self, k: int, t: float) -> bool:
        for (a, b) in self.st_peak[k]:
            if a <= t <= b:
                return True
        return False

    def render(self):
        # Placeholder: extend with matplotlib if desired.
        pass


# -----------------------
# Minimal synthetic sample
# -----------------------

def make_tiny_instance(seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    depot = (50.0, 50.0)
    customers = []
    for _ in range(4):
        xy = (float(rng.uniform(10, 90)), float(rng.uniform(10, 90)))
        demand = float(rng.integers(10, 30))
        a = float(rng.uniform(0, 50))
        b = a + float(rng.uniform(20, 60))
        customers.append({'xy': xy, 'demand': demand, 'tw': (a, b), 'service_time': 2.0})

    stations = [
        {'xy': (20.0, 80.0), 'peak_windows': [(40.0, 80.0)], 'price': 2.0, 'discharge_rate_kw': 10.0},
        {'xy': (80.0, 20.0), 'peak_windows': [(60.0, 120.0)], 'price': 2.5, 'discharge_rate_kw': 10.0},
    ]

    vehicles = [
        {'soc_init_kwh': 18.0, 'soc_max_kwh': 18.0, 'capacity': 80.0},
        {'soc_init_kwh': 18.0, 'soc_max_kwh': 18.0, 'capacity': 80.0},
    ]

    instance = {
        'seed': seed,
        'max_vehicles': 2,
        'max_customers': 6,  # padded beyond actual 4
        'max_stations': 3,   # padded beyond actual 2
        'depot': {'xy': depot},
        'vehicles': vehicles,
        'customers': customers,
        'stations': stations,
        'speed': 1.0,
        'consumption_kwh_per_km': 0.15,
        'horizon_T': 200.0,
        'soc_reserve_kwh': 2.0,
        'costs': {
            'per_km': 1.0,
            'per_time': 0.01,
            'vehicle_fixed': 50.0,
            'late_penalty': 50.0,
            'miss_penalty': 1000.0,
            'infeas_penalty': 1000.0,
        }
    }
    return instance


if __name__ == "__main__":
    # Quick manual smoke test that respects vehicle-specific target masks
    inst = make_tiny_instance(1)
    env = EVRoutingEnv(inst)
    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("mask:", info['action_mask'])

    for _ in range(20):
        veh_mask = env._compute_action_mask()['vehicle']
        veh_choices = np.flatnonzero(veh_mask)
        if len(veh_choices) == 0:
            # no idle vehicles; wait with vehicle 0
            action = {'vehicle': 0, 'target': inst['max_customers'] + inst['max_stations'] + 1, 'discharge_frac': np.array([0.0], dtype=np.float32)}
        else:
            veh = int(np.random.choice(veh_choices))
            mask_v = env._compute_action_mask_for_vehicle(veh)
            if not mask_v:
                mask_v = env._compute_action_mask()  # conservative fallback
            tgt_mask_v = mask_v['target']
            tgt_choices = np.flatnonzero(tgt_mask_v)
            if len(tgt_choices) == 0:
                action = {'vehicle': veh, 'target': inst['max_customers'] + inst['max_stations'] + 1, 'discharge_frac': np.array([0.0], dtype=np.float32)}
            else:
                action = {
                    'vehicle': veh,
                    'target': int(np.random.choice(tgt_choices)),
                    'discharge_frac': np.array([np.random.rand()], dtype=np.float32),
                }
        obs, reward, terminated, truncated, info = env.step(action)
        tag = f" [{info.get('safety_override')} ]" if 'safety_override' in info else ""
        print(f"t={info['time']:.1f}, r={reward:.2f}, term={terminated}, trunc={truncated}{tag}")
        if terminated or truncated:
            break
