from NNTF import NNTF

res = NNTF(model_type='lenet', dataset_name='mnist', learning_rate=0.001, momentum=0.9, weight_decay=0.005,
                 max_epochs=30)
res.print_rqa()
res.plot_figures()