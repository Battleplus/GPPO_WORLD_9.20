"""Task decisions independent of, and never overriding, the failed cost gate."""
import math

def decide(*,utility_ci,rule_ci,physical_ci,host_ci,loss_family_ci,
           n_cpu_ms,n_wall_p95_ms,rss_sum_bytes,replay_ratio,formal_decision_ratio,formal_episode_ratio):
    cis=[utility_ci,rule_ci,physical_ci,host_ci,*loss_family_ci]
    if len(loss_family_ci)!=3 or any(len(c)!=2 or c[0]>c[1] or not all(math.isfinite(x) for x in c) for c in cis):
        raise ValueError('Invalid CI')
    costs=[n_cpu_ms,n_wall_p95_ms,rss_sum_bytes,replay_ratio,formal_decision_ratio,formal_episode_ratio]
    if not all(math.isfinite(x) and x>=0 for x in costs):raise ValueError('Invalid cost')
    retained=utility_ci[0]>-.01
    practical=rule_ci[0]>.01
    tasks=physical_ci[0]>-.02 and host_ci[0]>-.02
    absolute=n_cpu_ms<=10 and n_wall_p95_ms<=50 and rss_sum_bytes<=4*1024**3
    saving=replay_ratio<=.8 and formal_decision_ratio<=.8 and formal_episode_ratio<=.8
    clear_loss=any(ci[1]<-margin for ci,margin in zip(loss_family_ci,[.01,.02,.02]))
    if clear_loss:result='clear_loss_direct_removal_not_supported'
    elif retained and practical and tasks:result='support_preparing_simplification_task_basis_only_cost_not_qualified'
    else:result='uncertain_pause_no_more_samples'
    return {'decision':result,'utility_retention_pass':retained,'increment_vs_rule_pass':practical,
        'task_guard_pass':tasks,'task_benefit_retention_pass':retained and tasks,
        'absolute_cost_pass_current_C_only':absolute,'cpu_saving_pass_descriptive':saving,
        'prior_A_cost_failure_preserved':True,'practical_acceptance_pass':False,
        'clear_loss_family_pass':clear_loss,'world_training_contribution_proven':False,
        'automatic_training':False,'automatic_more_samples':False}
