"""Frozen decision logic only; no runtime, files or model dependencies."""
import math

def decide(*, n_minus_m_ci, n_minus_r_ci, physical_difference, host_difference,
           n_cpu_ms, n_wall_p95_ms, rss_sum_bytes,
           paired_replay_cpu_ratio, historical_formal_cpu_ratio):
    values=[*n_minus_m_ci,*n_minus_r_ci,physical_difference,host_difference,n_cpu_ms,n_wall_p95_ms,
            rss_sum_bytes,paired_replay_cpu_ratio,historical_formal_cpu_ratio]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite result')
    if len(n_minus_m_ci)!=2 or len(n_minus_r_ci)!=2 or n_minus_m_ci[0]>n_minus_m_ci[1] or n_minus_r_ci[0]>n_minus_r_ci[1]:
        raise ValueError('Invalid confidence interval')
    if min(n_cpu_ms,n_wall_p95_ms,rss_sum_bytes,paired_replay_cpu_ratio,historical_formal_cpu_ratio)<0:
        raise ValueError('Negative cost')
    retained = n_minus_m_ci[0] > -0.01
    practical = n_minus_r_ci[0] > 0.01
    tasks = physical_difference >= 0 and host_difference >= 0
    absolute_cost = n_cpu_ms <= 10 and n_wall_p95_ms <= 50 and rss_sum_bytes <= 4*1024**3
    savings = paired_replay_cpu_ratio <= 0.8 and historical_formal_cpu_ratio <= 0.8
    if retained and practical and tasks and absolute_cost and savings:
        result = 'frozen_online_world_removal_supported_prepare_simplified_candidate'
    elif n_minus_m_ci[1] < -0.01 or not tasks:
        result = 'direct_removal_not_supported_no_independent_world_claim'
    elif retained and practical and tasks and not (absolute_cost and savings):
        result = 'task_retention_supported_cost_saving_not_accepted'
    else:
        result = 'uncertain_stop_without_more_samples_or_training'
    return {'decision':result,'noninferiority_pass':retained,'increment_vs_rule_pass':practical,
            'task_guard_pass':tasks,'absolute_cost_pass':absolute_cost,'saving_pass':savings,
            'world_independent_contribution_proven':False,'automatic_training':False}
