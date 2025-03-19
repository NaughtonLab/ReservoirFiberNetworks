import numpy as np

class unit_scaling:

    def scale(params, scaling_type):
        if scaling_type=="mm_g_s":
            params['thread_length'] *=  1e3
            params['thread_diameter'] *=  1e3
            params['dx'] *= 1e3
            
            params['youngs_modulus'] *= 1
            params['density'] *= 1e-6

            params['tension_force'] *= 1e6
            params['point_force_mag'] *= 1e6

            params['duration'] *= 1

        elif scaling_type=="mm_mg_ms":
            params['thread_length'] *=  1e3
            params['thread_diameter'] *=  1e3
            params['dx'] *= 1e3
            
            params['youngs_modulus'] *= 1e-3
            params['density'] *= 1e-3

            params['tension_force'] *= 1e3
            params['point_force_mag'] *= 1e3
            params['sample_freq_pf'] *= 1e-3
            params['sample_freq_pf'] = np.rint(params['sample_freq_pf']).astype(int)

            params['duration'] *= 1e3
        else:
            raise ValueError("Scaling type not recognized")

        return params


