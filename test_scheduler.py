from  scheduler import ddpmScheduler

scheduler = ddpmScheduler()

print(scheduler.betas.shape)
print(scheduler.alpha_bar[0])
print(scheduler.alpha_bar[0])
print(scheduler.alpha_bar[-1])
print(scheduler.posterior_vairance[-1])