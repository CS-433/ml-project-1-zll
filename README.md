### ML project 1
This repository contains the code for the project1 of the Machine Learning Course. The team is composed of

- Xinwei Li (xinwei.li@epfl.ch)
- Xingyue Zhang (xingyue.zhang@epfl.ch)
- Zhan Li (zhan.li@epfl.ch)

------------------
tx_train = tx_train[0:5000]
y_train = y_train[0:5000].reshape(5000,1)

y_train[y_train == -1] = 0

y_train = y_train.reshape(len(y_train),1)

degree = 3
tx_train = build_poly(tx_train, degree)
tx_train = np.c_[np.ones((tx_train.shape[0], 1)), tx_train]

num = y_train.shape[0]
seed = 111
k_fold = 5
k_indices = build_k_indices(num, k_fold, seed)

def ridge_plot(mse_rr, acc_rr, prec_rr, lambdas):
    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    ax[0].semilogx(lambdas, mse_rr, label="MSE", color='r', marker='x', markersize = 4, linestyle='solid', linewidth=2)
    ax[0].set_xlabel("lambda")
    ax[0].set_ylabel("MSE Loss")
    ax[0].legend(loc=0)
    ax[0].set_title("varied lambda reflected on MSE Loss")

    ax[1].semilogx(lambdas, acc_rr, label="ACC", color='g', marker='x', markersize = 4, linestyle='--', linewidth=2)
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("accuracy")
    ax[1].legend(loc=0)
    ax[1].set_title("varied lambda reflected on accuracy")
    
    ax[2].semilogx(lambdas, prec_rr, label="prec", color='b', marker='x', markersize = 4, linestyle='-.', linewidth=2)
    ax[2].set_xlabel("lambda")
    ax[2].set_ylabel("precision")
    ax[2].legend(loc=0)
    ax[2].set_title("varied lambda reflected on precision")

    
    plt.show()
    
def ridge_regression_loop(y_train, tx_train, k_indices, k_fold):
    """loop to find best lambda"""
    lambdas = np.logspace(-20, 5, 20)
    mse_rr = []
    acc_rr = []
    prec_rr = []
    # ridge regression with different lambda
    for idx, lam in enumerate(lambdas):
        losses_temp = []
        accs_temp = []
        prec_temp = []
        for k in range(k_fold):
            x_tr, x_val, y_tr, y_val = cross_validation(y_train, tx_train, k_indices, k)
            w, loss = ridge_regression(y_tr, x_tr, lam)
            y_pred = predict_labels(w, x_val)
            acc = _accuracy(y_pred, y_val)
            prec = _precision(y_pred, y_val)
            print(sum(y_pred == 1), sum(y_pred == 0))
            print(sum(y_val == 1), sum(y_val == 0))
            accs_temp.append(acc)
            losses_temp.append(loss)
            prec_temp.append(prec)
        mse_rr.append(np.mean(losses_temp))
        acc_rr.append(np.mean(accs_temp))
        prec_rr.append(np.mean(prec_temp))
        # print("Average test prediction accuracy over " + str(k_fold) + " folds is " + str(np.mean(accs_temp)))


    ridge_plot(mse_rr, acc_rr, prec_rr, lambdas)
    
ridge_regression_loop(y_train, tx_train, k_indices, k_fold)
