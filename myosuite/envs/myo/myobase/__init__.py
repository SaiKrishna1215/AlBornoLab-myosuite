""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from gym.envs.registration import register
from myosuite.envs.env_variants import register_env_variant

import os
import numpy as np

# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )
    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

curr_dir = os.path.dirname(os.path.abspath(__file__))

print("MyoSuite:> Registering Myo Envs")

# MyoArm
register_env_with_variants(id='ArmReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.armreach_v0:ReachEnvV0',
        max_episode_steps=100,
                           kwargs={
                               'model_path': curr_dir+'/../assets/arm/myoarm_reach.xml',
                               'normalize_act': True,
                                'obj_xyz_range': [[-.3, -.5, 0.9], [-.3, -.5, 0.9]], #xyz: neg x, neg y and positive z(around 1)
                               'far_th': 0.05
                           }
    )

register_env_with_variants(id='CenterReachOut-v0',
                           entry_point='myosuite.envs.myo.myobase.CenterReachOut_v0:ReachEnvV0',
                           max_episode_steps=100,
                           kwargs={
                               'model_path': curr_dir + '/../assets/arm/myoarm_centerreachout.xml', 
                               'normalize_act': True,
                               'obj_xyz_range': [[-0.03125069216102037, -0.024064832105254508, 1.09], [0.10199416560603433, -0.3457434044594767, 1.09], [0.15718516789601672, -0.2125, 1.09],[0.1019938068049917, -0.07925623673839724, 1.09], [-0.031249807868617828,-0.40093516789591876, 1.09],[-0.16449389389172694, -0.3457436761747822, 1.09], [-0.21968516789535328, -0.21249949997036174, 1.09],[-0.1680450158763584, -0.07925561667782058, 1.09]],#[[-.2, -.5, 1.09], [-.2, -.25, 1.09],[-.2, -0.1 ,1.09], [0.1, -.5, 1.09], [0.1, -.25, 1.09],[0.1, -0.05 ,1.09], [0.1, -.5, 1.09],[-.05, -0.1 ,1.09]]
                               # xyz: neg x, neg y and positive z(around 1)
                               'far_th': 0.2
                           }
                           )


register_env_with_variants(id='myoArmReachFixed-v1',
                           entry_point='myosuite.envs.myo.myobase.CenterReachOut_v0:ReachEnvV0',
                           max_episode_steps=150,
                           kwargs={
                               'model_path': curr_dir + '/../assets/arm/myoarm_relocate.xml',
                               'normalize_act': True,
                               'frame_skip': 5,
                               'pos_th': 0.1,  # cover entire base of the receptacle
                               'rot_th': np.inf,  # ignore rotation errors
                               'target_xyz_range': {'high': [0.0, 0.0, 0.0], 'low': [0.0, 0.0, 0.0]},
                               'target_rxryrz_range': {'high': [0.0, 0.0, 0.0], 'low': [0.0, 0.0, 0.0]}
                           }
                           )



# Finger-tip reaching ==============================
register_env_with_variants(id='motorFingerReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_reach_range': {'IFtip': ((.1, -.1, .1), (.1, -.1, .1)),},
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='motorFingerReachRandom-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_reach_range': {'IFtip': ((.1, -.1, .1), (0.27, .1, .3)),},
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='myoFingerReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
            'normalize_act': True,
        }
    )
register_env_with_variants(id='myoFingerReachRandom-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_reach_range': {'IFtip': ((.1, -.1, .1), (0.27, .1, .3)),},
            'normalize_act': True,
        }
    )

# Elbow posing ==============================
register_env_with_variants(id='myoElbowPose1D6MFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles.xml',
            'target_jnt_range': {'r_elbow_flex':(2, 2),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random'
        }
    )
register_env_with_variants(id='myoElbowPose1D6MRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles.xml',
            'target_jnt_range': {'r_elbow_flex':(0, 2.27),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random'
        }
    )


# Elbow Exo posing ==============================
register_env_with_variants(id='myoElbowPose1D6MExoFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles_1dofexo.xml',
            'target_jnt_range': {'r_elbow_flex':(2, 2),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random',
            'weighted_reward_keys':{
                                "pose": 1.0,
                                "bonus": 4.0,
                                "act_reg": 5.0,
                                "penalty": 50,
            }
        }
    )
register_env_with_variants(id='myoElbowPose1D6MExoRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles_1dofexo.xml',
            'target_jnt_range': {'r_elbow_flex':(0, 2.27),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random',
            'weight_bodyname':'carry_weight',
            'weight_range':(.1, 2),
            'weighted_reward_keys':{
                                "pose": 1.0,
                                "bonus": 4.0,
                                "act_reg": 5.0,
                                "penalty": 50,
            }
        }
    )


# Finger-Joint posing ==============================
register_env_with_variants(id='motorFingerPoseFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_jnt_range': {'IFadb':(0, 0),
                                'IFmcp':(0, 0),
                                'IFpip':(.75, .75),
                                'IFdip':(.75, .75)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
            'frame_skip': 5,
        }
)
register_env_with_variants(id='motorFingerPoseRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_jnt_range': {'IFadb':(-.2, .2),
                                'IFmcp':(-.4, 1),
                                'IFpip':(.1, 1),
                                'IFdip':(.1, 1)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='myoFingerPoseFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_jnt_range': {'IFadb':(0, 0),
                                'IFmcp':(0, 0),
                                'IFpip':(.75, .75),
                                'IFdip':(.75, .75)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
        }
    )
register_env_with_variants(id='myoFingerPoseRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_jnt_range': {'IFadb':(-.2, .2),
                                'IFmcp':(-.4, 1),
                                'IFpip':(.1, 1),
                                'IFdip':(.1, 1)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
        }
    )

# Hand-Joint posing ==============================

# Remove this when the ASL envs stablizes
register_env_with_variants(id='myoHandPoseFixed-v0', # revisit
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
            'target_jnt_value': np.array([0, 0, 0, -0.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]),
            'normalize_act': True,
            'pose_thd': .7,
            'reset_type': "init",        # none, init, random
            'target_type': 'fixed',      # generate/ fixed
        }
    )

# Create ASL envs ==============================
jnt_namesHand=['pro_sup', 'deviation', 'flexion', 'cmc_abduction', 'cmc_flexion', 'mp_flexion', 'ip_flexion', 'mcp2_flexion', 'mcp2_abduction', 'pm2_flexion', 'md2_flexion', 'mcp3_flexion', 'mcp3_abduction', 'pm3_flexion', 'md3_flexion', 'mcp4_flexion', 'mcp4_abduction', 'pm4_flexion', 'md4_flexion', 'mcp5_flexion', 'mcp5_abduction', 'pm5_flexion', 'md5_flexion']

ASL_qpos={}
ASL_qpos[0]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 1.26466 0 1.40604 0.227795 1.07614 -0.020944 1.46103 0.06284 0.83263 -0.14399 1.571 1.38248'.split(' ')
ASL_qpos[1]='0 0 0 0.0248 0.04536 -0.7854 -1.309 0.366605 0.010473 0.269258 0.111722 1.48459 0 1.45318 1.44532 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[2]='0 0 0 0.0248 0.04536 -0.7854 -1.13447 0.514973 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[3]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.571 -0.036652 1.52387 1.45318 1.40604 -0.068068 1.39033 1.571'.split(' ')
ASL_qpos[4]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[5]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[6]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 1.1861 -0.2618 1.35891 1.48459'.split(' ')
ASL_qpos[7]='0 0 0 0.524 0.01569 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.28036 -0.115192 1.52387 1.45318 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[8]='0 0 0 0.428 0.22338 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.194636 1.39033 0 1.08399 0.573415 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[9]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 0.39275 0 0.18852 0.227795 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')

# ASl Eval envs for each numerals
for k in ASL_qpos.keys():
    register_env_with_variants(id='myoHandPose'+str(k)+'Fixed-v0',
            entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
                'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
                'target_jnt_value': np.array(ASL_qpos[k],'float'),
                'normalize_act': True,
                'pose_thd': .7,
                'reset_type': "init",        # none, init, random
                'target_type': 'fixed',      # generate/ fixed
            }
    )

# ASL Train Env
m = np.array([ASL_qpos[i] for i in range(10)]).astype(float)
Rpos = {}
for i_n, n  in enumerate(jnt_namesHand):
    Rpos[n]=(np.min(m[:,i_n]), np.max(m[:,i_n]))

register_env_with_variants(id='myoHandPoseRandom-v0',  #reconsider
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
            'target_jnt_range': Rpos,
            'normalize_act': True,
            'pose_thd': .7,
            'reset_type': "random",         # none, init, random
            'target_type': 'generate',      # generate/ fixed
        }
    )


# Gait Torso Reaching ==============================
from myosuite.physics.sim_scene import SimBackend
sim_backend = SimBackend.get_sim_backend()
leg_model='/../../../simhive/myo_sim/leg/myolegs.xml'

register_env_with_variants(id='myoLegStandRandom-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:ReachEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+leg_model,
            'joint_random_range': (0.2, -0.2), #range of joint randomization (jnt = init_qpos + random(range)
            'target_reach_range': {
                'pelvis': ((-.05, -.05, 0), (0.05, 0.05, 0)),
                },
            'normalize_act': True,
            'far_th': 0.44
        }
    )

# Gait Torso Walking ==============================
register_env_with_variants(id='myoLegWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:WalkEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None   # if None then the initial root pos will be taken, otherwise provide quat
        }
    )

# Rough Terrain Walking  ==============================
register_env_with_variants(id='myoLegRoughTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'rough',
            'variant': None
        }
    )

# Hilly Walking  ==============================
register_env_with_variants(id='myoLegHillyTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'hilly',
            'variant':'fixed'
        }
    )

# Stair Walking  ==============================
register_env_with_variants(id='myoLegStairTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'stairs',
            'variant':'fixed',
        }
    )



# Hand-Joint Reaching ==============================
register_env_with_variants(id='myoHandReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'target_reach_range': {
                'THtip': ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
                'IFtip': ((-0.151, -0.547, 1.455), (-0.165, -0.537, 1.495)),
                'MFtip': ((-0.146, -0.547, 1.447), (-0.165, -0.537, 1.495)),
                'RFtip': ((-0.148, -0.543, 1.445), (-0.165, -0.537, 1.495)),
                'LFtip': ((-0.148, -0.528, 1.434), (-0.165, -0.537, 1.495)),
                },
            'normalize_act': True,
            'far_th': 0.044
        }
    )
register_env_with_variants(id='myoHandReachRandom-v0',
    entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
    max_episode_steps=100,
    kwargs={
        'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'normalize_act': True,
        'far_th': 0.034
    }
)


# Hand-Joint key turn ==============================
register_env_with_variants(id='myoHandKeyTurnFixed-v0',
        entry_point='myosuite.envs.myo.myobase.key_turn_v0:KeyTurnEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_keyturn.xml',
            'normalize_act': True
        }
    )
register_env_with_variants(id='myoHandKeyTurnRandom-v0',
        entry_point='myosuite.envs.myo.myobase.key_turn_v0:KeyTurnEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_keyturn.xml',
            'normalize_act': True,
            'key_init_range':(-np.pi/2, np.pi/2),
            'goal_th': 2*np.pi
        }
    )


# Hold objects ==============================
register_env_with_variants(id='myoHandObjHoldFixed-v0',
        entry_point='myosuite.envs.myo.myobase.obj_hold_v0:ObjHoldFixedEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_hold.xml',
            'normalize_act': True
        }
    )
register_env_with_variants(id='myoHandObjHoldRandom-v0', # revisit
        entry_point='myosuite.envs.myo.myobase.obj_hold_v0:ObjHoldRandomEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_hold.xml',
            'normalize_act': True
        }
    )

# Pen twirl ==============================
register_env_with_variants(id='myoHandPenTwirlFixed-v0',
            entry_point='myosuite.envs.myo.myobase.pen_v0:PenTwirlFixedEnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_pen.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )
register_env_with_variants(id='myoHandPenTwirlRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pen_v0:PenTwirlRandomEnvV0',
        max_episode_steps=50,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pen.xml',
            'normalize_act': True,
            'frame_skip': 5,
        }
    )

# SAR REORIENT: 8-object ==============================
register_env_with_variants(id='myoHandReorient8-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:Geometries8EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR REORIENT: 100-object
register_env_with_variants(id='myoHandReorient100-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:Geometries100EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR TEST ENVIRONMENT: in-distribution
register_env_with_variants(id='myoHandReorientID-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:InDistribution',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR TEST ENVIRONMENT: out of distribution
register_env_with_variants(id='myoHandReorientOOD-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:OutofDistribution',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )
