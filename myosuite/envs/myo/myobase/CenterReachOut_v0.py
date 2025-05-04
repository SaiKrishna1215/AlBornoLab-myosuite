""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import gym
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0


class ReachEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        # "twist_penalty": 2.0  # Optional: you can include this explicitly if needed
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obj_xyz_range=None,
               far_th=.35,
               obs_keys: list = DEFAULT_OBS_KEYS,
               drop_th=0.50,
               qpos_noise_range=None,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
        self.far_th = far_th
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.obj_xyz_range = obj_xyz_range
        self.drop_th = drop_th
        self.qpos_noise_range = qpos_noise_range
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs,
                       )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:].copy() * self.dt
        if self.sim.model.na > 0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        self.obs_dict['obj_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['palm_pos'] = self.sim.data.site_xpos[self.palm_sid]
        self.obs_dict['reach_err'] = np.array(self.obs_dict['palm_pos']) - np.array(self.obs_dict['obj_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()

        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['reach_err'] = np.array(obs_dict['palm_pos']) - np.array(obs_dict['obj_pos'])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0
        far_th = self.far_th
        near_th = 0.05
        drop = reach_dist > self.drop_th

        # === Twist Penalty ===
        # Adjust this index (3) to the correct joint index if needed
        twist_angle = np.abs(obs_dict['hand_qpos'][3])
        twist_penalty = -1.0 * twist_angle  # linear penalty

        rwd_dict = collections.OrderedDict((
            ('reach', -1. * reach_dist),
            ('bonus', 1. * (reach_dist < 2 * near_th) + 1. * (reach_dist < near_th)),
            ('act_reg', -1. * act_mag),
            ('penalty', -1. * (reach_dist > far_th) + twist_penalty),
            # Must keys
            ('sparse', -1. * reach_dist),
            ('solved', reach_dist < near_th),
            ('done', reach_dist < near_th),
        ))

        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def generate_target_pose(self):
        random_index = np.random.randint(0, len(self.obj_xyz_range))
        self.sim.model.body_pos[self.object_bid] = self.obj_xyz_range[random_index]
        self.current_object_pos = self.obj_xyz_range[random_index]
        self.sim.forward()

    def reset(self, reset_qpos=None, reset_qvel=None):
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range * (
                self.sim.model.jnt_range[:, 1] - self.sim.model.jnt_range[:, 0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos

        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs
