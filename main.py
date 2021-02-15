import NNTF


def execute_lr_range(param_range, k, max_epochs=50, mnt=0.9, wd=5e-3, model_name='lenet', dataset='cifar10'):
    for prm in param_range:
        res = NNTF.NNTF_measures(model_name=model_name, dataset=dataset, lr=prm, mt=mnt, wd=wd,
                 max_epochs=max_epochs, batch_size=128, k_steps=k)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


def execute_wd_range(param_range, k, max_epochs=50, mnt=0.9, lr=0.001, model_name='lenet', dataset='cifar10'):
    for prm in param_range:
        res = NNTF.NNTF_measures(model_name=model_name, dataset=dataset, lr=lr, mt=mnt, wd=prm,
                 max_epochs=max_epochs, batch_size=128, k_steps=k)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


def execute_mt_range(param_range, k, max_epochs=50, wd=5e-3, lr=0.001, model_name='lenet', dataset='cifar10'):
    for prm in param_range:
        res = NNTF.NNTF_measures(model_name=model_name, dataset=dataset, lr=lr, mt=prm, wd=wd,
                 max_epochs=max_epochs, batch_size=128, k_steps=k)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


lr_range = [0.01, 0.005, 0.001]
mnt_range = [0.5, 0.7, 0.9, 0.95]
wd_range = [0.1, 0.05, 0.01, 0.005, 0.001, 0]
k=10

execute_lr_range(lr_range, k=k, max_epochs=100, model_name='VGG16')
execute_wd_range(wd_range, k=k, max_epochs=100, model_name='VGG16')
execute_mt_range(mnt_range, k=k, max_epochs=100, model_name='VGG16')

