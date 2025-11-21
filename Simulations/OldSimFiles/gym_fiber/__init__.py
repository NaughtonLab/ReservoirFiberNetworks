from gym.envs.registration import register

register(
    id='fiber-v0', 
    entry_point='gym_fiber.envs:FiberEnv'
)

from gym_fiber.config import *

""" Global Configuration Parameters """
RENDERER_CONFIG = RendererType.MATPLOTLIB

# """ Octopus Environment """
# register(
#     id="OctoFlat-v0",
#     entry_point="gym_fiber.envs.octopus:FlatEnv",
# )

# register(
#     id="OctoFlatLite-v0",
#     entry_point="gym_fiber.envs.octopus:FlatEnv",
#     kwargs=dict(n_arm=1, n_action=8),
# )

# register(
#     id='OctoCrawl-v0',
#     entry_point='gym_fiber.envs.octopus:CrawlEnv',
# )

# register(
#    id='OctoReach-v0',
#    entry_point='gym_fiber.envs.octopus:ReachEnv',
# )

# # register(
# #    id='OctoSwim-v0',
# #    entry_point='gym_fiber.envs.octopus:SwimEnv',
# # )

# # register(
# #    id='OctoHunt-v0',
# #    entry_point='gym_fiber.envs.octopus:HuntEnv',
# # )

# register(
#     id="OctoArmSingle-v0",
#     entry_point="gym_fiber.envs.octopus:ArmSingleEnv",
# )

# register(
#     id="OctoArmTwo-v0",
#     entry_point="gym_fiber.envs.octopus:ArmTwoEnv",
# )

# register(
#     id="OctoArmPush-v0",
#     entry_point="gym_fiber.envs.octopus:ArmPushEnv",
# )

# register(
#     id='OctoArmPush-v1',
#     entry_point='gym_fiber.envs.octopus:ArmPushEnv',
#     kwargs=dict(mode="continuous")
# )

# register(
#     id='OctoArmPullWeight-v0',
#     entry_point='gym_fiber.envs.octopus:ArmPullWeightEnv',
#     kwargs=dict(mode="continuous")
# )

# """ Snake Environment """
# register(
#     id="ContinuumSnake-v0",
#     entry_point="gym_fiber.envs.snake:ContinuumSnakeEnv",
# )

# """ Simple Control Environment """
# # register(
# #    id='InertialPull-v0',
# #    entry_point='gym_fiber.envs.simple_control:InertialPullEnv',
# # )

# # """ Soft Arm Environment """
# register(
#     id="SoftArmTracking-v0",
#     entry_point="gym_fiber.envs.soft_arm:SoftArmTrackingEnv",
# )
# # register(
# #    id='SoftArmTracking-v1',
# #    entry_point='gym_fiber.envs.soft_arm:SoftArmTrackingEnv',
# #    kwargs=dict(game_mode=2)
# # )

# """ Soft Pendulum Environment """
# register(
#     id="SoftPendulum-v0", entry_point="gym_fiber.envs.soft_pendulum:SoftPendulumEnv"
# )



