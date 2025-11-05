# ev_masked_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple

# This wrapper flattens the (vehicle, target, discharge_frac) action into a single Discrete
# and exposes a boolean action mask for sb3-contrib's ActionMasker.


class EVMaskedDiscreteWrapper(gym.Wrapper):
    """
    Wrap EVRoutingEnv into a single Discrete action space:
        action = (veh_id, target_id, frac_bin)  -> encoded as one integer.

    - veh_id in [0, maxV-1]
    - target_id in [0, maxC + maxS + 2)   # customers | stations | return | wait
    - frac_bin in [0, K-1]                # discharge fraction discretized to K bins

    Masking rules:
      * vehicle must be idle/usable,
      * target must be feasible for that vehicle (strict TW + reach + return + capacity),
      * discharge only valid at stations; if not in peak or no available SoC to sell, only frac_bin=0 allowed.
    """

    def __init__(self, env: gym.Env, discharge_bins: int = 5):
        super().__init__(env)
        assert hasattr(env, "maxV") and hasattr(env, "maxC") and hasattr(env, "maxS"), \
            "Underlying env must be EVRoutingEnv."

        self.maxV = env.maxV
        self.maxC = env.maxC
        self.maxS = env.maxS
        self.K = int(discharge_bins)

        # Flattened Discrete action space size
        self.n_pairs = self.maxV * (self.maxC + self.maxS + 2)
        self.n_actions = self.n_pairs * self.K
        self.action_space = spaces.Discrete(self.n_actions)

        # Observations are unchanged (pass-through)
        self.observation_space = env.observation_space

        # quick indices
        self.IDX_RETURN = self.maxC + self.maxS
        self.IDX_WAIT = self.maxC + self.maxS + 1

    # ---------- encoding / decoding ----------

    def _decode(self, a: int) -> Tuple[int, int, int]:
        """int -> (veh, target, frac_bin)"""
        frac_bin = a % self.K
        pair = a // self.K
        target = pair % (self.maxC + self.maxS + 2)
        veh = pair // (self.maxC + self.maxS + 2)
        return veh, target, frac_bin

    def _encode(self, veh: int, target: int, frac_bin: int) -> int:
        pair = veh * (self.maxC + self.maxS + 2) + target
        return pair * self.K + frac_bin

    def _frac_from_bin(self, frac_bin: int) -> float:
        if self.K <= 1:
            return 0.0
        return float(frac_bin) / float(self.K - 1)

    # ---------- Gym API ----------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # keep last mask around for debugging
        self._last_mask = self.action_mask()
        return obs, info

    def step(self, action: int):
        veh, target, frac_bin = self._decode(action)
        # Build the underlying Dict action
        discharge_frac = self._frac_from_bin(frac_bin)

        action_dict = {
            "vehicle": int(veh),
            "target": int(target),
            "discharge_frac": np.array([discharge_frac], dtype=np.float32),
        }

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        # update latest mask for next step (sb3-contrib asks mask_fn at policy time)
        self._last_mask = self.action_mask()
        return obs, reward, terminated, truncated, info

    # ---------- Masking ----------

    def action_mask(self) -> np.ndarray:
        """
        Return a boolean array of shape (n_actions,) with feasible action flags.
        """
        mask = np.zeros((self.n_actions,), dtype=bool)

        # 1) Which vehicles are selectable?
        veh_mask = self.env._compute_action_mask()["vehicle"]
        # If no idle vehicles: only allow WAIT on any vehicle (choose a consistent vehicle = 0)
        # but we keep the general logic below — WAIT is allowed for any selectable vehicle.

        for v in range(self.maxV):
            if not veh_mask[v]:
                continue  # vehicle not selectable now
            # 2) Target feasibility for this vehicle
            v_mask = self.env._compute_action_mask_for_vehicle(v)["target"]

            for t in range(self.maxC + self.maxS + 2):
                if not v_mask[t]:
                    # whole group disabled
                    continue

                # 3) Discharge bin validity
                if t < self.maxC:
                    # customer: discharge not applicable
                    frac_bins = [0]
                elif t < self.maxC + self.maxS:
                    # station
                    k = t - self.maxC
                    node_v = 1 + self.maxC + k
                    in_peak = self.env._is_peak_time(k, self.env.t)

                    # compute max allowable discharge (must keep soc_reserve + energy to reach depot)
                    node_u = self.env.v_loc[v]
                    dist_to_target = float(self.env.dist_km[node_u, node_v])
                    e_to_target = dist_to_target * self.env.cons_kwh_per_km
                    dist_home = float(self.env.dist_km[node_v, 0])
                    e_home = dist_home * self.env.cons_kwh_per_km
                    soc_after_arrival = self.env.v_soc[v] - e_to_target
                    max_allow = max(0.0, soc_after_arrival - (self.env.soc_reserve_kwh + e_home))

                    if in_peak and (max_allow > 1e-6):
                        # allow all bins, but zero discharge is always allowed too
                        frac_bins = list(range(self.K))
                    else:
                        # not in peak or cannot sell -> only zero
                        frac_bins = [0]
                elif t == self.IDX_RETURN:
                    # return: no discharge
                    frac_bins = [0]
                else:
                    # wait: no discharge
                    frac_bins = [0]

                for fb in frac_bins:
                    idx = self._encode(v, t, fb)
                    mask[idx] = True

        # Edge case: if everything is False (shouldn't happen), allow a safe WAIT with veh 0
        if not np.any(mask):
            idx_safe = self._encode(0, self.IDX_WAIT, 0)
            mask[idx_safe] = True

        return mask
