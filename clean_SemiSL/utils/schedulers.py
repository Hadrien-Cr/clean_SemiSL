from omegaconf import DictConfig
import numpy as np

def resolve_schedules(schedules: list[DictConfig], t: float):
    n = len(schedules)

    for k, schedule in enumerate(schedules):
        if schedule.schedule_start <= t <= schedule.schedule_end:
            return compute_value(schedule, t)
    
    return compute_value(schedule, t)

def compute_value(schedule_info: DictConfig, t: float) -> float:
    if schedule_info.schedule_type == "constant":
        return schedule_info.v0
    if schedule_info.schedule_type == "decay":
        return exponential_decay(schedule_info.v0, schedule_info.decay_factor, schedule_info.decay_period, t, schedule_info.schedule_start)
    if schedule_info.schedule_type == "cosine":
        return cosine_schedule(schedule_info.v0, schedule_info.vf, t, schedule_info.schedule_start, schedule_info.schedule_end)
    if schedule_info.schedule_type == "linear":
        return linear_schedule(schedule_info.v0, schedule_info.vf, t, schedule_info.schedule_start, schedule_info.schedule_end)
    if schedule_info.schedule_type == "step":
        return step_schedule(schedule_info.v0, schedule_info.decay_factor, schedule_info.decay_period, t, schedule_info.schedule_start)
    else:
        raise ValueError

def step_schedule(initial_value, decay_factor, decay_period, t, t_start):
    progress = np.clip((t-t_start)/decay_period, 0, 10000)
    return initial_value * decay_factor ** progress 

def exponential_decay(initial_value, decay_factor, decay_period, t, t_start):
    progress = np.clip((t-t_start)/decay_period, 0, 10000)
    return initial_value * decay_factor ** progress 

def linear_schedule(initial_value, final_value, t, t_start, t_end):
    progress = np.clip((t-t_start)/(t_end-t_start), 0, 1)
    return initial_value + (final_value - initial_value) * progress

def cosine_schedule(initial_value, final_value, t, t_start, t_end):
    progress = np.clip((t-t_start)/(t_end-t_start), 0, 1)
    return final_value + (initial_value - final_value) * (1 + np.cos(np.pi * progress)) / 2 
