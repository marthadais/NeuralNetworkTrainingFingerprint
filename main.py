from NNTF import NNTF


def execute_lr_range(param_range, k, max_epochs=50, mnt=0.9, wd=5e-3, model_name='lenet', dataset='cifar10', l_min=2, interval=0.05):
    for prm in param_range:
        res = NNTF(model_type=model_name, dataset=dataset, learning_rate=prm, momentum=mnt, weight_decay=wd,
                 max_epochs=max_epochs, batch_size=128, k_steps=k, l_min=l_min, interval=interval)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


def execute_wd_range(param_range, k, max_epochs=50, mnt=0.9, lr=0.001, model_name='lenet', dataset='cifar10', l_min=2, interval=0.05):
    for prm in param_range:
        res = NNTF(model_type=model_name, dataset=dataset, learning_rate=lr, momentum=mnt, weight_decay=prm,
                             max_epochs=max_epochs, batch_size=128, k_steps=k, l_min=l_min, interval=interval)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


def execute_mt_range(param_range, k, max_epochs=50, wd=5e-3, lr=0.001, model_name='lenet', dataset='cifar10', l_min=2, interval=0.05):
    for prm in param_range:
        res = NNTF(model_type=model_name, dataset=dataset, learning_rate=lr, momentum=prm, weight_decay=wd,
                             max_epochs=max_epochs, batch_size=128, k_steps=k, l_min=l_min, interval=interval)
        print(f'Lam: {res.measures.laminarity}')
        print(f'Ent: {res.measures.entropy}')


lr_range = [0.01, 0.005, 0.001]
mnt_range = [0.5, 0.7, 0.9, 0.95]
wd_range = [0.1, 0.05, 0.01, 0.005, 0.001, 0]
k=10

execute_lr_range(lr_range, k=k, max_epochs=100, model_name='VGG16')
execute_wd_range(wd_range, k=k, max_epochs=100, model_name='VGG16')
execute_mt_range(mnt_range, k=k, max_epochs=100, model_name='VGG16')

