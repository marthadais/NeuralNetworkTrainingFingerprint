import NNTF


def execute_lr_range(lr_range, k, max_epochs=50, mnt=0.9, wd=5e-3, model_name='lenet', dataset='cifar10'):
    for lr in lr_range:
        NNTF.NNTF_measures(model_name=model_name, dataset=dataset, lr=lr, mt=mnt, wd=wd,
                 max_epochs=max_epochs, batch_size=128, k_steps=k)


lr_range = [0.01, 0.005, 0.001]
mnt_range = [0.5, 0.7, 0.9, 0.95]
wd_range = [0.1, 0.05, 0.01, 0.005, 0.001, 0]
k=10

execute_lr_range(lr_range, k=k, max_epochs=30)

