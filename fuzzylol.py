import numpy as np
import matplotlib.pyplot as plt


class fuzzy:

    def __init__(self):
        self.lav = [100, 100, 160, 170]
        self.medium = [165, 175, 185]
        self.hoy = [180, 190, 220, 220]

        self.tynn = [10, 10, 75, 85]
        self.feit = [75, 85, 150, 150]

    def fuzzify(self, input, var):
        if(var == "h"):
            return [self.membershipFunction(input, self.lav), self.membershipFunction(input, self.medium),
                self.membershipFunction(input, self.hoy)]
        elif(var == "f"):
            return [self.membershipFunction(input, self.tynn), self.membershipFunction(input, self.feit)]
        else:
            print("Fuck deg")

    def membershipFunction(self, x, limits):
        if (len(limits) == 4):
            a = limits[0]
            b = limits[1]
            c = limits[2]
            d = limits[3]
            if (a <= x <= d):
                if (a <= x <= b):
                    try:
                        return (x - a) / (b - a)
                    except ZeroDivisionError:
                        return 1
                elif (c <= x <= d):
                    return (d - x) / (d - c)
                else:
                    return 1
            else:
                return 0
        elif (len(limits) == 3):
            a = limits[0]
            b = limits[1]
            c = limits[2]
            if (a <= x <= b):
                return (x - a) / (b - a)
            elif (b <= x <= c):
                return (c - x) / (c - b)
            else:
                return 0

    def plotThatShitv2(self):
        x2 = list(range(100, 220))
        y = []
        y2 = []
        y3 = []

        for i in x2:
            y.append(self.membershipFunction(i, self.lav))
            y2.append(self.membershipFunction(i, self.medium))
            y3.append(self.membershipFunction(i, self.hoy))

        plt.plot(x2, y)
        plt.plot(x2, y2)
        plt.plot(x2, y3)
        plt.xlim([150, 210])
        plt.show()

        z = []
        z2 = []
        xf = list(range(10,150))

        for j in xf:
            z.append(self.membershipFunction(j, self.tynn))
            z2.append(self.membershipFunction(j, self.feit))

        plt.plot(xf, z)
        plt.plot(xf, z2)
        plt.show()



if __name__ == "__main__":
    h = 167 # cm
    v = 78 # kg

    mf = fuzzy()
    mf.plotThatShitv2()
    fuzzheight = mf.fuzzify(h, "h")

    plt.plot(h, fuzzheight[0], color='r', marker='o')
    plt.plot(h, fuzzheight[1], color='r', marker='o')
    plt.plot(h, fuzzheight[2], color='r', marker='o')


    fuzzfeit = mf.fuzzify(v, "f")

    plt.plot(v, fuzzfeit[0], color='r', marker='o')
    plt.plot(v, fuzzfeit[1], color='r', marker='o')

    maxheight = np.amax(np.array(fuzzheight))
    maxfeit = np.amax(np.array(fuzzfeit))




