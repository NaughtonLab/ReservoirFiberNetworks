import numpy as np

class unit_scaling:

    def scale(params, scaling_type):
        match scaling_type:
            case "mm_g_s":
                params['thread_length'] *=  1e3
                params['thread_diameter'] *=  1e3
                params['dx'] *= 1e3
                
                params['youngs_modulus'] *= 1
                params['density'] *= 1e-6

                params['tension_force'] *= 1e6
                params['point_force_mag'] *= 1e6

                params['duration'] *= 1

            case "mm_mg_ms":
                params['thread_length'] *=  1e3
                params['thread_diameter'] *=  1e3
                params['dx'] *= 1e3
                
                params['youngs_modulus'] *= 1e-3
                params['density'] *= 1e-3

                params['tension_force'] *= 1e3
                params['point_force_mag'] *= 1e3

                params['duration'] *= 1e3

        return params


