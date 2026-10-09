from elastica.callback_functions import CallBackBaseClass

class NetworkCallBack(CallBackBaseClass):
    """
    Call back function for Elastica rod
    """

    def __init__(
        self,
        step_skip: int,
        callback_params: dict,
    ):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(
                system.compute_position_center_of_mass()
            )
            self.callback_params["directors"].append(
                system.director_collection.copy()
            )
            self.callback_params["kappa"].append(system.kappa.copy())
            self.callback_params["omega_collection"].append(
                system.omega_collection.copy()
            )
            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["tangents"].append(system.tangents.copy())
            self.callback_params["velocity_collection"].append(
                system.velocity_collection.copy()
            )
            self.callback_params["acceleration_collection"].append(
                system.acceleration_collection.copy()
            )
            self.callback_params["internal_forces"].append(
                system.internal_forces.copy()
            )
            self.callback_params["internal_torques"].append(
                system.internal_torques.copy()
            )
            self.callback_params["angular_acceleration"].append(
                system.alpha_collection.copy()
            )
            self.callback_params["external_forces"].append(
                system.external_forces.copy()
            )
            self.callback_params["external_torques"].append(
                system.external_torques.copy()
            )
            self.callback_params["bending_energy"].append(
                system.compute_bending_energy()
            )
            self.callback_params["translational_energy"].append(
                system.compute_translational_energy()
            )
            self.callback_params["shear_energy"].append(
                system.compute_shear_energy()
            )
            self.callback_params["rotational_energy"].append(
                system.compute_rotational_energy()
            )
            return
        
class NetworkCallBack_less(CallBackBaseClass):
    """
    Call back function for Elastica rod
    """

    def __init__(
        self,
        step_skip: int,
        callback_params: dict,
    ):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            # self.callback_params["radius"].append(system.radius.copy())
            # self.callback_params["com"].append(
            #     system.compute_position_center_of_mass()
            # )
            # self.callback_params["directors"].append(
            #     system.director_collection.copy()
            # )
            # self.callback_params["kappa"].append(system.kappa.copy())
            # self.callback_params["omega_collection"].append(
            #     system.omega_collection.copy()
            # )
            # self.callback_params["sigma"].append(system.sigma.copy())
            # self.callback_params["tangents"].append(system.tangents.copy())
            # self.callback_params["velocity_collection"].append(
                # system.velocity_collection.copy()
            # )
            # self.callback_params["acceleration_collection"].append(
            #     system.acceleration_collection.copy()
            # )
            # self.callback_params["internal_forces"].append(
            #     system.internal_forces.copy()
            # )
            # self.callback_params["internal_torques"].append(
            #     system.internal_torques.copy()
            # )
            # self.callback_params["angular_acceleration"].append(
            #     system.alpha_collection.copy()
            # )
            self.callback_params["external_forces"].append(
                system.external_forces.copy()
            )
            # self.callback_params["external_torques"].append(
            #     system.external_torques.copy()
            # )
            # self.callback_params["bending_energy"].append(
            #     system.compute_bending_energy()
            # )
            # self.callback_params["translational_energy"].append(
            #     system.compute_translational_energy()
            # )
            # self.callback_params["shear_energy"].append(
            #     system.compute_shear_energy()
            # )
            # self.callback_params["rotational_energy"].append(
            #     system.compute_rotational_energy()
            # )
            return