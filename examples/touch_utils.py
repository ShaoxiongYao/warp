import time
import glob
import numpy as np
import pathlib

import warp as wp
from warp.sim.integrator_euler import eval_particle_contacts


def compute_contact_forces(model, state, out_state):
        
    wp.launch(
        kernel=eval_particle_contacts,
        dim=model.soft_contact_max,
        inputs=[
            state.particle_q,
            state.particle_qd,
            model.particle_radius,
            model.particle_flags,
            state.body_q,
            state.body_qd,
            model.body_com,
            model.shape_body,
            model.shape_materials,
            model.soft_contact_ke,
            model.soft_contact_kd,
            model.soft_contact_kf,
            model.soft_contact_mu,
            model.particle_adhesion,
            model.soft_contact_count,
            model.soft_contact_particle,
            model.soft_contact_shape,
            model.soft_contact_body_pos,
            model.soft_contact_body_vel,
            model.soft_contact_normal,
            model.soft_contact_max,
        ],
        # outputs
        outputs=[out_state.particle_f, out_state.body_f],
        device=model.device,
    )

    return out_state


class TouchSeq:
    def __init__(self, seq_dir="/media/motion/8AF1-B496/warp_data", seq_id=None) -> None:
        self.seq_dir = seq_dir
        if seq_id is None:
            self.seq_dir = self.seq_dir + f'/seq_{int(time.time())}'
            pathlib.Path(self.seq_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.seq_dir = self.seq_dir + f'/seq_{seq_id}'

            self.q_fn_lst = sorted(glob.glob(self.seq_dir + '/particle_q_*.npy'))
            self.qd_fn_lst = sorted(glob.glob(self.seq_dir + '/particle_qd_*.npy'))
            self.f_fn_lst = sorted(glob.glob(self.seq_dir + '/particle_f_*.npy'))
            self.cid_fn_lst = sorted(glob.glob(self.seq_dir + '/contact_particle_*.npy'))

    def save(self, sim_time, model, state):

        np.save(self.seq_dir + f'/body_q_{sim_time:07.3f}.npy', state.body_q.numpy())
        np.save(self.seq_dir + f'/particle_q_{sim_time:07.3f}.npy', state.particle_q.numpy())
        np.save(self.seq_dir + f'/particle_qd_{sim_time:07.3f}.npy', state.particle_qd.numpy())
        np.save(self.seq_dir + f'/particle_f_{sim_time:07.3f}.npy', state.particle_f.numpy())
        np.save(self.seq_dir + f'/contact_particle_{sim_time:07.3f}.npy', model.soft_contact_particle.numpy())
    
    def load(self, idx):
        q = np.load(self.q_fn_lst[idx])
        qd = np.load(self.qd_fn_lst[idx])
        f = np.load(self.f_fn_lst[idx])
        cid = np.load(self.cid_fn_lst[idx])

        return q, qd, f, cid


