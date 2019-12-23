import numpy as np
import os

dir1 = '1shot'
dir2 = '5shot'

def read_files(d, nshots):
    dirs = sorted(os.listdir(d))
    exp_set_metrics = {}
    for exp_set in dirs:
        tokens = exp_set.split(',')
        exp_type = tokens[0].split('=')[1]
        if exp_type not in exp_set_metrics:
            exp_set_metrics[exp_type] = {}

        fold = int(tokens[1].split('=')[1])
        if fold not in exp_set_metrics[exp_type]:
            exp_set_metrics[exp_type][fold] = []

        pthf = d + '/' + exp_set + '/testing/' + 'fo=%d'%fold + '/final_test_miou_%d.txt'%nshots
        if not os.path.exists(pthf):
            print('Couldnt find exp ', exp_set)
            continue
        testf = open(pthf, 'r')

        for line in testf:
            if line == '':
                break
            metrics = line.split(',')
            miou = float(metrics[0].split(':')[1].strip())
            biou = float(metrics[1].split(':')[1].strip())
            exp_set_metrics[exp_type][fold].append((miou, biou))
    return exp_set_metrics

def compute_mean(exp_set):
    means = {}
    bmeans = {}
    stds = {}
    bstds = {}
    for exp_type, exp in exp_set.items():
        means_fold = []
        bmeans_fold = []
        for j in range(5):
            bious = []
            mious = []
            for i in range(4):
                if j > len(exp[i])-1:
                    continue
                mious.append(exp[i][j][0])
                bious.append(exp[i][j][1])
            means_fold.append(np.mean(mious))
            bmeans_fold.append(np.mean(bious))
        means[exp_type] = np.mean(means_fold)
        bmeans[exp_type] = np.mean(bmeans_fold)

        stds[exp_type] = np.std(means_fold)
        bstds[exp_type] = np.std(bmeans_fold)
    return means, bmeans, stds, bstds

print('1-shot Exp Set:')
metrics = read_files(dir1, 1)
means, bmeans, stds, bstds = compute_mean(metrics)

for exp_type, mean in means.items():
    print('Exp ', exp_type, ' miou = ', str(mean), ' +/- ', str(1.96*stds[exp_type]/np.sqrt(5)),
          ' ,  biou = ', str(bmeans[exp_type]), ' +/- ', str(1.96*bstds[exp_type]/np.sqrt(5)) )

print('5-shot Exp Set:')
metrics = read_files(dir2, 5)
means, bmeans, stds, bstds = compute_mean(metrics)
for exp_type, mean in means.items():
    print('Exp ', exp_type, ' miou = ', str(mean), ' +/- ', str(1.96*stds[exp_type]/np.sqrt(5)),
          ' ,  biou = ', str(bmeans[exp_type]), ' +/- ', str(1.96*bstds[exp_type]/np.sqrt(5)))



