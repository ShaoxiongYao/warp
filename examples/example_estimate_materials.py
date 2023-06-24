import warp as wp
import warp.sim
import numpy as np
from example_sim_grid_deform import eval_linear_tetrahedra
import matplotlib.pyplot as plt

wp.init()

@wp.kernel
def loss_l2(ary1: wp.array(dtype=wp.vec3f), ary2: wp.array(dtype=wp.vec3f), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    diff = ary1[i] - ary2[i]
    l = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l)

def setup_model(cell_dim=1):
    builder = wp.sim.ModelBuilder()

    cell_size = 2.0 / cell_dim

    center = cell_size * cell_dim * 0.5

    builder.add_soft_grid(
        pos=(-center, 0.0, -center),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=cell_dim,
        dim_y=cell_dim,
        dim_z=cell_dim,
        cell_x=cell_size,
        cell_y=cell_size,
        cell_z=cell_size,
        density=100.0,
        fix_bottom=True,
        fix_top=True,
        k_mu=100.0,
        k_lambda=5000.0,
        k_damp=0.0,
    )

    model = builder.finalize()
    model.ground = False
    model.gravity[1] = 0.0
    return model


if __name__ == '__main__':
    time_id = 1684512104
    
    particle_q_init = np.load(f'outputs/box_deform_{time_id}/particle_q_init.npy')
    particle_f_init = np.load(f'outputs/box_deform_{time_id}/particle_f_init.npy')

    particle_q_lst = []
    particle_v_lst = []
    particle_f_lst = []

    for idx in range(300):
        x = np.load(f'outputs/box_deform_{time_id}/particle_q_{idx}.npy')
        particle_q_lst.append(x)

        v = np.load(f'outputs/box_deform_{time_id}/particle_qd_{idx}.npy')
        particle_v_lst.append(v)

        f = np.load(f'outputs/box_deform_{time_id}/particle_f_{idx}.npy')
        particle_f_lst.append(f)
    
    model = setup_model(cell_dim=1)

    loss_lst = []

    for _ in range(2000):
        state = model.state()
        state.particle_q = wp.from_numpy(particle_q_lst[100], dtype=wp.vec3f, device='cuda', requires_grad=True)
        particle_f = wp.zeros(shape=state.particle_f.shape, dtype=state.particle_f.dtype, 
                            device='cuda', requires_grad=True)
        model.tet_materials.requires_grad = True

        tape = wp.Tape()

        with tape:
            wp.launch(
                kernel=eval_linear_tetrahedra,
                dim=model.tet_count,
                inputs=[
                    state.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                ],
                outputs=[particle_f],
                device=model.device,
            )
            loss = wp.zeros(1, dtype=wp.float32, device='cuda', requires_grad=True)
            target_f = wp.from_numpy(particle_f_lst[100], dtype=wp.vec3f, device='cuda')
            wp.launch(loss_l2, dim=len(particle_f), inputs=[particle_f, target_f], outputs=[loss])

        tape.backward(loss)
        print("loss value:", loss.numpy())
        loss_lst.append(loss.numpy()[0])

        m = model.tet_materials.numpy()
        m_grad = model.tet_materials.grad.numpy()
        m = m - m_grad * 1e-2
        model.tet_materials = wp.from_numpy(m, dtype=wp.float32, device='cuda', requires_grad=True)

        tape.zero()
    
    plt.plot(loss_lst)
    plt.show()

    print("Final estimated material parameters:")
    print(model.tet_materials.numpy())