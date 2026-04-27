import mujoco
import numpy as np

def get_body_names (model, data):
    body_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY, body_idx) for body_idx in range(model.nbody)]
    return body_names

def rmat2rotvec(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz])
    axis = axis / (2 * np.sin(theta))

    return theta * axis

