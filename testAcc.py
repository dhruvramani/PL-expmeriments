import os
import matplotlib
import matplotlib.pyplot as plt

def write_table():
    _MAX_NUM = 4 
    _SPLITS  = [500]
    _ACTIVAT = ['relu', 'paramlog']

    for model_num in range(_MAX_NUM):
        for act in _ACTIVAT:
                for i in _SPLITS:
                    path = "./logs-2/{}/a{}_{}_test_acc.log".format(i, model_num + 1, act)
                    try :
                        with open(path, "r") as f:
                            maxi = -1.0
                            for j in f:
                                try :
                                    k = float(j.split("\n")[0].lstrip().rstrip())
                                    if(maxi < k):
                                        maxi = k
                                except :
                                    pass
                        print("& {} ".format(round(maxi*100, 2)), end='')
                    except :
                        print("& FnF ", end='')
                print("\n")


def plot_models(files):
    matplotlib.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (15,10)
    for filepath in files:
        values = list()
        with open(filepath, "r") as f:
            for log in f:
                try :
                    values.append(float(log))
                except :
                    pass
        plt.plot(values)
        print(filepath)
    plt.legend(loc=2, prop={'size': 30})
    plt.legend(['Parametric Log', 'ReLU'])
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.savefig("/Users/mac/Desktop/New Logs/{}".format("_".join(filepath.split("/")[-2:]).split(".")[0]))
    #plt.show()


if __name__ == '__main__':
    write_table()
    #models = ['./logs-2/5000/a3_paramlog_trainloss.log', './logs-2/5000/a3_relu_trainloss.log']
    #plot_models(models)