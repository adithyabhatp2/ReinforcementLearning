
# import seaborn as sns
import matplotlib.pyplot as plt

def loadNumericSeriesFromFile(filePath):
    input_file = open(filePath, 'r')

    # fp = open(filename, "r")
    # lines = fp.read().split("\n")
    lines = input_file.readlines()
    lines = filter(None, lines)
    x = []
    y = []
    numPoints = 0

    lines = lines[500:]

    for line in lines:
        if line.startswith('episode'):
            tokens = line.split('\t')
            x.append(float(tokens[1]))
            y.append(float(tokens[3]))
            numPoints += 1

    return y, x


if __name__ == "__main__":

    xs = []
    ys = []
    legends = []

    for l in xrange(1,3):
        for n in xrange(10,41,10):
            filename = 'cartpole_v1_l'+str(l)+"_n"+str(n)+"_e1000.out"
            y, x = loadNumericSeriesFromFile('../Reinf2/runs_v1/'+filename)
            # y.sort()

            ys.append(y)
            xs.append(x)
            legends.append('h=' + str(l) + ', u=' + str(n))
            print('h={}, u={}'.format(l,n))


    # sns.set(style="darkgrid")
    for i in range(0,len(ys)):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)


    plt.ylim(ymin=0)
    plt.legend(legends, fontsize=15, loc='best')
    plt.title('Cartpole - Different Network Structures', fontsize=20)
    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Score', fontsize=15)


    plt.autoscale(enable=True)
    plt.show()