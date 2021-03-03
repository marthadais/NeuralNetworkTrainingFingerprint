import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

IMGDIR = 'figures/'
SNAPSHOTS_DIR = 'data/snapshots_info/'
DMATRIX_DIR = 'data/dist_matrix/'




if (os.path.isdir(IMGDIR) == False):
    os.makedirs(IMGDIR)

snapshots_names = os.listdir(SNAPSHOTS_DIR)
dmatrix_names = os.listdir(DMATRIX_DIR)

for i in range(0,len(dmatrix_names)):
    with open((DMATRIX_DIR+dmatrix_names[i]),"rb") as f:
        ddata = pickle.load(f)
        plt.figure(figsize=(10,10))
        plt.imshow(ddata)
        plt.xlabel("Iteration")
        plt.ylabel("Iteration")
        plt.title("Distance matrix for transition data: \n"+os.path.splitext(dmatrix_names[i])[0])
        plt.colorbar()
        plt.savefig((IMGDIR+'transition_'+os.path.splitext(dmatrix_names[i])[0]+'.png'),bbox_inches='tight')
        plt.close()
    

for i in range(0,len(snapshots_names)):
    with open((SNAPSHOTS_DIR+snapshots_names[i]),"rb") as f:
        sdata = pickle.load(f)
        preds = []
        acc = []
        n = len(sdata)
        for j in range(0,n):
            preds.append(sdata[j]['predictions'])
            acc.append(sdata[j]['accuracy'])
        
        preds = np.array(preds)
        acc = np.array(acc)
        
        diff = np.ndarray((n,n),dtype=int)
        for j in range(0,n):
            for k in range(0,n):
                diff[j,k] = (preds[j] != preds[k]).sum()
    
        #plt.subplots(2, 1, figsize=(10,15),gridspec_kw={'height_ratios': [2, 1]})
        plt.figure(figsize=(10,15))
        plt.subplot(211)
        plt.title("Distance matrix for raw classification: \n"+os.path.splitext(snapshots_names[i])[0])
        plt.imshow(diff)
        plt.colorbar()
        plt.subplot(212)
        plt.title("Accuracy per iteration")
        plt.plot(acc)
        plt.tight_layout()
        plt.savefig((IMGDIR+'predictions_'+os.path.splitext(snapshots_names[i])[0]+'.png'),bbox_inches='tight')
        plt.close()



