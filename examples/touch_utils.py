import time
import glob
import json
import numpy as np
import pathlib

import warp as wp
from warp.sim.integrator_euler import eval_particle_contacts

def sum_contact_forces(f, cid):
    nz_cid = cid[cid != 0]
    if nz_cid.shape[0] == 0:
        return np.zeros(3)
    else:
        return f[nz_cid, :].sum(axis=0)

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
    def __init__(self, seq_dir="/media/motion/8AF1-B496/warp_data", seq_id=None, 
                 data_keys=['particle_q', 'particle_qd', 'particle_f', 'contact_particle']) -> None:
        self.seq_dir = seq_dir
        self.data_keys = data_keys

        if seq_id is None:
            self.seq_dir = self.seq_dir + f'/seq_{int(time.time())}'
            pathlib.Path(self.seq_dir).mkdir(parents=True, exist_ok=True)
            self.seq_len = 0
            self.sim_time_lst = []
        else:
            self.seq_dir = self.seq_dir + f'/seq_{seq_id}'
            self.sim_time_lst = np.load(self.seq_dir + '/sim_time.npy').tolist()

            self.dict_fn_lst = {}
            for key in data_keys:
                self.dict_fn_lst[key] = sorted(glob.glob(self.seq_dir + f'/{key}_*.npy'))
                assert len(self.sim_time_lst) == len(self.dict_fn_lst[key])

            self.seq_len = len(self.sim_time_lst)
    
    def __len__(self):
        return self.seq_len

    def save(self, sim_time, model, state):
        self.sim_time_lst.append(sim_time)

        if 'body_q' in self.data_keys:
            np.save(self.seq_dir + f'/body_q_{sim_time:07.3f}.npy', state.body_q.numpy())
        if 'particle_q' in self.data_keys:
            np.save(self.seq_dir + f'/particle_q_{sim_time:07.3f}.npy', state.particle_q.numpy())
        if 'particle_qd' in self.data_keys:
            np.save(self.seq_dir + f'/particle_qd_{sim_time:07.3f}.npy', state.particle_qd.numpy())
        if 'particle_f' in self.data_keys:
            np.save(self.seq_dir + f'/particle_f_{sim_time:07.3f}.npy', state.particle_f.numpy())
        if 'contact_particle' in self.data_keys:
            np.save(self.seq_dir + f'/contact_particle_{sim_time:07.3f}.npy', model.soft_contact_particle.numpy())
        if 'contact_normal' in self.data_keys:
            np.save(self.seq_dir + f'/contact_normal_{sim_time:07.3f}.npy', model.soft_contact_normal.numpy())
        if 'contact_body_pos' in self.data_keys:
            np.save(self.seq_dir + f'/contact_body_pos_{sim_time:07.3f}.npy', model.soft_contact_body_pos.numpy())
        if 'shape_transform' in self.data_keys:
            np.save(self.seq_dir + f'/shape_transform_{sim_time:07.3f}.npy', model.shape_transform.numpy())

        self.seq_len += 1
    
    def end_seq(self, config=None):
        if config is not None:
            with open(self.seq_dir + f'/config.json', 'w') as f:
                json.dump(config, f, indent=2)
        np.save(self.seq_dir + f'/sim_time.npy', np.array(self.sim_time_lst))
    
    def load_group(self, idx, with_sim_time=True, group_keys=None):
        sim_time = self.sim_time_lst[idx]

        data_lst = []
        if with_sim_time:
            data_lst.append(sim_time)
        if group_keys is None:
            group_keys = self.data_keys

        for key in group_keys:
            data_lst.append(self.load_by_key(idx, key))

        return tuple(data_lst)
    
    def load_by_key(self, idx, key):
        assert key in self.data_keys
        sim_time = self.sim_time_lst[idx]
        return np.load(self.seq_dir + f'/{key}_{sim_time:07.3f}.npy')



