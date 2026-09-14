# class Scheduler:
#     def __init__(self, start_value, end_value, max_iteration, schedule_func):
#         self.start_value = start_value
#         self.end
#         self.max_iteration = max_iteration
#         self.schedule_func = schedule_func
#         self.current_iter = 0
#
#     @property
#     def value(self):
#         return self.schedule_func(self.current_iter, self.max_iteration)
#
#     def step(self):
#         self.current_iter += 1
import numpy as np


def build_linear_scheduler(
    init_value, final_value, max_step
):
    def helper(step):
        if step > max_step:
            return final_value
        slope = (final_value - init_value) / max_step
        value = step * slope + init_value
        return value

    return helper


def build_expon_scheduler(
    init_value, final_value, delay_steps=0, delay_mult=1.0, max_steps=1000000, step_sub=0,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):

        if step < 0 or (init_value == 0.0 and final_value == 0.0):
            # Disable this parameter
            return 0.0
        if delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - step_sub) / (max_steps - step_sub), 0, 1)
        log_lerp = np.exp(np.log(init_value) * (1 - t) + np.log(final_value) * t)
        return delay_rate * log_lerp

    return helper


