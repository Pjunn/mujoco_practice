import numpy as np
import mujoco
from .utils import *

""" CLIPPING ERROR """
POSITION_CLIPPING = 0.02
ROTATION_CLIPPING = 0.2

""" POS ROT RATIO """
POS_ROT_RATIO = 0.57

def get_ik_error_clipped(
        p_current,
        r_current,
        p_target,
        r_target,
        ):
    pos_error = p_target - p_current
    rmat_error = r_target @ r_current.T
    rotvec_error = rmat2rotvec(rmat_error)* POS_ROT_RATIO
    pos_error_clipped = np.clip(pos_error, -POSITION_CLIPPING, POSITION_CLIPPING)
    rotvec_error_clipped = np.clip(rotvec_error, -ROTATION_CLIPPING, ROTATION_CLIPPING)
    return pos_error_clipped, rotvec_error_clipped

def get_jacobian(
        model,
        data,
        name,
        type="body",
        joints_use=None
):
    """
    Get Jacobian of given body/geom/site
    Args
        - model: mujoco model
        - data: mujoco data
        - name: name of the body/geom/site
        - type: "body", "geom", or "site"
        - joints_use: list of joint names to use for Jacobian (if None, use all joints)
    """
    joints_use_idxs = [model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)] for name in joints_use] if joints_use is not None else None
    jacp_full = np.zeros((3, model.nv))
    jacr_full = np.zeros((3, model.nv))
    if type == "body":
        mujoco.mj_jacBody(model, data, jacp_full, jacr_full, data.body(name).id)
    elif type == "geom":
        mujoco.mj_jacGeom(model, data, jacp_full, jacr_full, data.geom(name).id)
    elif type == "site":
        mujoco.mj_jacSite(model, data, jacp_full, jacr_full, data.site(name).id)

    if joints_use_idxs is not None:
        jacp_use = jacp_full[:, joints_use_idxs]
        jacr_use = jacr_full[:, joints_use_idxs]
    else:
        jacp_use = jacp_full
        jacr_use = jacr_full

    return jacp_use, jacr_use

def get_pseudo_inverse(
        jacobian,
        method="svd",
        threshold=1e-3,
        damping=0.1
        ):
    
    row, col = jacobian.shape
    
    if method == "svd":
        U, Sigma, V_T = np.linalg.svd(jacobian, compute_uv=True)

        Sigma_clipped_rev = np.zeros_like(Sigma)

        for i, value in enumerate(Sigma):
            if value < threshold:
                Sigma_clipped_rev[i] = 0
            else:
                Sigma_clipped_rev[i] = 1 / Sigma[i]
        

        
        inverse_Sigma = np.zeros((col, row))
        for i, value in enumerate(Sigma_clipped_rev):
            inverse_Sigma[i,i] = value
        
        inversed = V_T.T @ inverse_Sigma @ U.T
    
    elif method == "dls":

        inversed = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping**2 * np.eye(row))
        
    return inversed

def solve_ik(model, data, joint_names, eef_site_name, q_init, p_target, r_target, max_tick, debug=False):
    eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site_name)

    joint_names_idx = [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in joint_names
    ]
    init_qpos = data.qpos[joint_names_idx].copy()

    q_mins = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in joint_names])
    q_maxs = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in joint_names])

    data.qpos[joint_names_idx] = q_init
    mujoco.mj_forward(model, data)

    for tick in range(max_tick):
        eef_site_pos = data.site_xpos[eef_site_id]
        eef_site_rmat = data.site_xmat[eef_site_id]
        eef_site_rmat = eef_site_rmat.reshape(3, 3)

        pos_error, rotvec_error = get_ik_error_clipped(
        p_current=eef_site_pos,
        r_current=eef_site_rmat,
        p_target=p_target,
        r_target=r_target
        )
        error = np.concatenate([pos_error, rotvec_error])

        # terminate condition
        if np.linalg.norm(pos_error) < 0.01 and np.linalg.norm(rotvec_error) < 0.01:
            # print("Target reached!")
            break


        jacobian_p, jacobian_r = get_jacobian(model, data, eef_site_name, type='site', joints_use=joint_names)

        jacobian = np.concat([jacobian_p, jacobian_r], axis=0)

        inversed_jacobian = get_pseudo_inverse(jacobian, method="dls")

        qpos_error = inversed_jacobian @ error
        before = data.qpos[joint_names_idx].copy()
        new_q = before + qpos_error
        clipped = np.clip(new_q, q_mins, q_maxs)        # ← 한계 강제

        data.qpos[joint_names_idx] = clipped
        mujoco.mj_forward(model, data)

    target_qpos = data.qpos[joint_names_idx].copy()
    data.qpos[joint_names_idx] = init_qpos
    mujoco.mj_forward(model, data)

    return target_qpos, error