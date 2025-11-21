import numpy as np

if __name__ == '__main__':
    # every frame = 250 sf YM 100
    spec_list = ["2by2_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k1.npz",
                 "3by3_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1.npz",
                 "4by4_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k1.npz",
                 "6by6_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_k.npz",
                 "8by8_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_k.npz",
                 "10by10_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspline_.npz"]
    
    file_path = "./Simulations/SMASIS_sims/all_post_proc_1sf_YM100/"

    spec_label_list = []

    for spec in spec_list:

        data = np.load(f"{file_path}{spec}", allow_pickle=True)

        spec_label = spec[:4] if spec[:4] != '10by' else spec[:6]
        spec_label_list.append(spec_label)
        print(spec_label)

        input_data = data['input_data']
        output_data = data['output_data']
        time_data = data['time_data']

        print(len(input_data), len(output_data), len(time_data))

        original_dt = time_data[1] - time_data[0]
        new_dt = 100e-3
        downsample_factor = int(new_dt / original_dt)
        print(f"Original dt: {original_dt}, New dt: {new_dt}, Downsample factor: {downsample_factor}")
        input_data_ds = input_data[::downsample_factor]
        output_data_ds = output_data[::downsample_factor]
        time_data_ds = time_data[::downsample_factor]

        print(len(input_data_ds), len(output_data_ds), len(time_data_ds))

        np.savez(f"{file_path}downsampled_{new_dt}_{spec_label}", input_data=input_data_ds, output_data=output_data_ds, time_data=time_data_ds)




