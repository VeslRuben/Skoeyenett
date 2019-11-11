import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D




class fuzzy:

    def __init__(self, inputVar):
        # Market value fuzzy sets
        self.mvLow = [0, 0, 75000, 100000]
        self.mvMedium = [50000, 100000, 200000, 250000]
        self.mvHigh = [200000, 300000, 650000, 850000]
        self.mvVHigh = [650000, 850000, 1000000, 1000000]
        self.marketValue = ['Market Value', self.mvLow, self.mvMedium, self.mvHigh, self.mvVHigh]

        # Location fuzzy sets
        self.locBad = [0, 0, 1.5, 4]
        self.locFair = [2.5, 5, 6, 8.5]
        self.locExc = [6, 8.5, 10, 10]
        self.location = ['Location', self.locBad, self.locFair, self.locExc]

        # House fuzzy sets
        self.houseVLow = [0, 0, 3]
        self.houseLow = [0, 3, 6]
        self.houseMed = [2, 5, 8]
        self.houseHigh = [4, 7, 10]
        self.houseVHigh = [7, 10, 10]
        self.house = ['House Evaluation', self.houseVLow, self.houseLow, self.houseMed, self.houseHigh, self.houseVHigh]

        # Asset fuzzy sets
        self.assetLow = [0, 0, 0, 150000]
        self.assetMed = [50000, 250000, 450000, 650000]
        self.assetHigh = [500000, 700000, 1000000, 1000000]
        self.asset = ['Assets', self.assetLow, self.assetMed, self.assetHigh]

        # Income fuzzy sets
        self.incomeLow = [0, 0, 10000, 25000]
        self.incomeMed = [15000, 35000, 55000]
        self.incomeHigh = [40000, 60000, 80000]
        self.incomeVHigh = [60000, 80000, 100000, 100000]
        self.income = ['Income', self.incomeLow, self.incomeMed, self.incomeHigh, self.incomeVHigh]

        # Applicant fuzzy sets
        self.applLow = [0, 0, 2, 4]
        self.applMed = [2, 5, 8]
        self.applHigh = [6, 8, 10, 10]
        self.applicant = ['Applicant Evaluation', self.applLow, self.applMed, self.applHigh]

        # Interest fuzzy sets
        self.interestLow = [0, 0, 2, 5]
        self.interestMed = [2, 4, 6, 8]
        self.interestHigh = [6, 8.5, 10, 10]
        self.interest = ['Interest', self.interestLow, self.interestMed, self.interestHigh]

        # Credit fuzzy sets
        self.creditVLow = [0, 0, 125000]
        self.creditLow = [0, 125000, 250000]
        self.creditMed = [125000, 250000, 375000]
        self.creditHigh = [250000, 375000, 500000]
        self.creditVHigh = [375000, 500000, 500000]
        self.credit = ['Credit', self.creditVLow, self.creditLow, self.creditMed, self.creditHigh, self.creditVHigh]

        self.fuzzySets = [self.marketValue, self.location, self.house, self.asset, self.income, self.applicant,
                          self.interest, self.credit]

        # Outputs
        self.houseEvaluation = [0, 0, 0, 0, 0]
        self.applicantEvaluation = [0, 0, 0]
        self.creditEvaluation = [0, 0, 0, 0, 0]

        # Input

        # [MarketValue, Location, Asset, Income, Interest]
        self.input = inputVar

        # [[Low, Med, High, VHigh], [Bad, Fair, Excellent], [Low, Med, High], [Low, Med, High, VHigh], [Low, Med, High]]
        self.fuzzifiedInput = []

        # Rule base

        for sets in self.fuzzySets:
            self.plotFunctions(sets)

    def fuzzify(self):
        fuzzyList = [self.marketValue, self.location, self.asset, self.income, self.interest]
        self.fuzzifiedInput = []
        i = 0
        for bigSets in fuzzyList:
            temp = []
            for sets in bigSets:
                temp.append(self.membershipFunction(self.input[i], sets))
            self.fuzzifiedInput.append(temp)
            i += 1

    def membershipFunction(self, x, limits):
        if (len(limits) == 4):
            a = limits[0]
            b = limits[1]
            c = limits[2]
            d = limits[3]
            if (a <= x <= d):
                if (a <= x <= b):
                    temp = (x - a) / (b - a)
                    if math.isnan(temp):
                        return 1
                    else:
                        return temp
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
                temp = (x - a) / (b - a)
                if math.isnan(temp):
                    return 1
                else:
                    return temp
            elif (b <= x <= c):
                temp = (c - x) / (c - b)
                if math.isnan(temp):
                    return 1
                else:
                    return temp
            else:
                return 0

    def plotFunctions(self, fuzzySet):
        title = fuzzySet.pop(0)
        for set in fuzzySet:
            if max(set) > 10:
                x = np.arange(min(set), max(set))
            else:
                x = np.arange(min(set)-0.1, max(set)+0.1, 0.1)

            y = []
            for i in x:
                y.append(self.membershipFunction(i, set))

            plt.plot(x, y)
        plt.grid(True)
        plt.title(title)
        plt.show()

    def homeEvaluation(self):
        # [MarketValue, Location, Asset, Income, Interest]
        # [[Low, Med, High, VHigh], [Bad, Fair, Excellent], [Low, Med, High], [Low, Med, High, VHigh], [Low, Med, High]]

        marketValue = self.fuzzifiedInput[0]
        location = self.fuzzifiedInput[1]

        if marketValue[0] > 0:
            self.houseEvaluation[1] = marketValue[0]
        if location[0] > 0:
            self.houseEvaluation[1] = max(self.houseEvaluation[1], location[0])
        if location[0] > 0 and marketValue[0] > 0:
            self.houseEvaluation[0] = min(marketValue[0], location[0])
        if location[0] > 0 and marketValue[1] > 0:
            self.houseEvaluation[1] = max(self.houseEvaluation[1], min(marketValue[1], location[0]))
        if location[0] > 0 and marketValue[2] > 0:
            self.houseEvaluation[2] = min(marketValue[2], location[0])
        if location[0] > 0 and marketValue[3] > 0:
            self.houseEvaluation[3] = min(marketValue[3], location[0])
        if location[1] > 0 and marketValue[0] > 0:
            self.houseEvaluation[1] = max(self.houseEvaluation[1], min(location[1], marketValue[0]))
        if location[1] > 0 and marketValue[1] > 0:
            self.houseEvaluation[2] = max(self.houseEvaluation[2], min(location[1], marketValue[1]))
        if location[1] > 0 and marketValue[2] > 0:
            self.houseEvaluation[3] = max(self.houseEvaluation[3], min(location[1], marketValue[2]))
        if location[1] > 0 and marketValue[3] > 0:
            self.houseEvaluation[4] = max(self.houseEvaluation[4], min(location[1], marketValue[3]))
        if location[2] > 0 and marketValue[0] > 0:
            self.houseEvaluation[2] = max(self.houseEvaluation[2], min(location[2], marketValue[0]))
        if location[2] > 0 and marketValue[1] > 0:
            self.houseEvaluation[3] = max(self.houseEvaluation[3], min(location[2], marketValue[1]))
        if location[2] > 0 and marketValue[2] > 0:
            self.houseEvaluation[4] = max(self.houseEvaluation[4], min(location[2], marketValue[2]))
        if location[2] > 0 and marketValue[3] > 0:
            self.houseEvaluation[4] = max(self.houseEvaluation[4], min(location[2], marketValue[3]))


        crisp = self.findCentreOfGravity(self.house, self.houseEvaluation)

        temp = []
        for sets in self.house:
            temp.append(self.membershipFunction(crisp, sets))

        self.houseEvaluation = temp

        return crisp
        # Output
        # [VLow, Low, Med, High, VHigh]

    def applicEvaluation(self):
        # [MarketValue, Location, Asset, Income, Interest]
        # [[Low, Med, High, VHigh], [Bad, Fair, Excellent], [Low, Med, High], [Low, Med, High, VHigh], [Low, Med, High]]
        asset = self.fuzzifiedInput[2]
        income = self.fuzzifiedInput[3]

        if asset[0] > 0 and income[0] > 0:
            self.applicantEvaluation[0] = self.minMax(self.applicantEvaluation[0], asset[0], income[0])
        if asset[0] > 0 and income[1] > 0:
            self.applicantEvaluation[0] = self.minMax(self.applicantEvaluation[0], asset[0], income[1])
        if asset[0] > 0 and income[2] > 0:
            self.applicantEvaluation[1] = self.minMax(self.applicantEvaluation[1], asset[0], income[2])
        if asset[0] > 0 and income[3] > 0:
            self.applicantEvaluation[2] = self.minMax(self.applicantEvaluation[2], asset[0], income[3])
        if asset[1] > 0 and income[0] > 0:
            self.applicantEvaluation[0] = self.minMax(self.applicantEvaluation[0], asset[1], income[0])
        if asset[1] > 0 and income[1] > 0:
            self.applicantEvaluation[1] = self.minMax(self.applicantEvaluation[1], asset[1], income[1])
        if asset[1] > 0 and income[2] > 0:
            self.applicantEvaluation[2] = self.minMax(self.applicantEvaluation[2], asset[1], income[2])
        if asset[1] > 0 and income[3] > 0:
            self.applicantEvaluation[2] = self.minMax(self.applicantEvaluation[2], asset[1], income[3])
        if asset[2] > 0 and income[0] > 0:
            self.applicantEvaluation[1] = self.minMax(self.applicantEvaluation[1], asset[2], income[0])
        if asset[2] > 0 and income[1] > 0:
            self.applicantEvaluation[1] = self.minMax(self.applicantEvaluation[1], asset[2], income[1])
        if asset[2] > 0 and income[2] > 0:
            self.applicantEvaluation[2] = self.minMax(self.applicantEvaluation[2], asset[2], income[2])
        if asset[2] > 0 and income[3] > 0:
            self.applicantEvaluation[2] = self.minMax(self.applicantEvaluation[2], asset[2], income[3])

        crisp = self.findCentreOfGravity(self.applicant, self.applicantEvaluation)

        temp = []
        for sets in self.applicant:
            temp.append(self.membershipFunction(crisp, sets))

        self.applicantEvaluation = temp

        return crisp
        # Output
        # [Low, Medium, High]

    def creditEval(self):
        # [MarketValue, Location, Asset, Income, Interest]
        # [[Low, Med, High, VHigh], [Bad, Fair, Excellent], [Low, Med, High], [Low, Med, High, VHigh], [Low, Med, High]]

        income = self.fuzzifiedInput[3]
        interest = self.fuzzifiedInput[4]
        applicant = self.applicantEvaluation
        house = self.houseEvaluation

        if income[0] > 0 and interest[1] > 0:
            self.creditEvaluation[0] = self.minMax(self.creditEvaluation[0], income[0], interest[1])
        if income[0] > 0 and interest[2] > 0:
            self.creditEvaluation[0] = self.minMax(self.creditEvaluation[0], income[0], interest[2])
        if income[1] > 0 and interest[2] > 0:
            self.creditEvaluation[1] = self.minMax(self.creditEvaluation[1], income[1], interest[2])
        if applicant[0] > 0:
            self.creditEvaluation[0] = max(self.creditEvaluation[0], applicant[0])
        if house[0] > 0:
            self.creditEvaluation[0] = max(self.creditEvaluation[0], house[0])
        if applicant[1] > 0 and house[0] > 0:
            self.creditEvaluation[1] = self.minMax(self.creditEvaluation[1], applicant[1], house[0])
        if applicant[1] > 0 and house[1] > 0:
            self.creditEvaluation[1] = self.minMax(self.creditEvaluation[1], applicant[1], house[1])
        if applicant[1] > 0 and house[2] > 0:
            self.creditEvaluation[2] = self.minMax(self.creditEvaluation[2], applicant[1], house[2])
        if applicant[1] > 0 and house[3] > 0:
            self.creditEvaluation[3] = self.minMax(self.creditEvaluation[3], applicant[1], house[3])
        if applicant[1] > 0 and house[4] > 0:
            self.creditEvaluation[3] = self.minMax(self.creditEvaluation[3], applicant[1], house[4])
        if applicant[2] > 0 and house[0] > 0:
            self.creditEvaluation[1] = self.minMax(self.creditEvaluation[1], applicant[2], house[0])
        if applicant[2] > 0 and house[1] > 0:
            self.creditEvaluation[2] = self.minMax(self.creditEvaluation[2], applicant[2], house[1])
        if applicant[2] > 0 and house[2] > 0:
            self.creditEvaluation[3] = self.minMax(self.creditEvaluation[3], applicant[2], house[2])
        if applicant[2] > 0 and house[3] > 0:
            self.creditEvaluation[3] = self.minMax(self.creditEvaluation[3], applicant[2], house[3])
        if applicant[2] > 0 and house[4] > 0:
            self.creditEvaluation[4] = self.minMax(self.creditEvaluation[4], applicant[2], house[4])

        crisp = self.findCentreOfGravity(self.credit, self.creditEvaluation)

        return crisp
        # Output
        # [VLow, Low, Med, High, VHigh]

    def minMax(self, assign, input1, input2):
        return max(assign, min(input1, input2))

    def findCentreOfGravity(self, fuzzySetList:list, vektor:list):
        sumListOver = []
        sumListUnder = []

        indexList = []
        limitList = []
        setList = []

        for x in vektor:
            if x == 0:
                continue
            else:
                limitList.append(x)
                indexList.append(vektor.index(x))
                setList.append(fuzzySetList[vektor.index(x)])

        lowRange = fuzzySetList[indexList[0]][0]
        maxRange = fuzzySetList[indexList[len(indexList) - 1]][len(fuzzySetList[indexList[len(indexList) - 1]]) - 1]

        j = 0

        for set in setList:
            tempOver = []
            tempUnder = []
            for i in np.arange(lowRange, maxRange, (maxRange-lowRange)/100):
                mfValue = self.membershipFunction(i, set)
                if mfValue > limitList[j]:
                    mfValue = limitList[j]
                xValue = i
                result = mfValue * xValue
                tempOver.append(result)
                tempUnder.append(mfValue)
            sumListOver.append(tempOver)
            sumListUnder.append(tempUnder)
            j += 1

        finalListOver = []
        finalListUnder = []
        for i in range(len(sumListUnder[0])):
            highestUnder = 0
            highestOver = 0
            for l in sumListUnder:
                if l[i] > highestUnder:
                    highestUnder = l[i]
            for l in sumListOver:
                if l[i] > highestOver:
                    highestOver = l[i]


            finalListUnder.append(highestUnder)
            finalListOver.append(highestOver)

        cog = (sum(finalListOver) / sum(finalListUnder))

        return cog

    def plot3d(self):
        # Market value
        market = np.arange(0, 1000000, 25000)

        # Location
        loc = np.arange(0, 10, 0.25)

        z = np.empty((40,40))
        #Home evaluation

        #for (x,y) in zip(market, loc):
        #    self.input[0] = x
        #    self.input[1] = y
        #    self.fuzzify()
        #    z.append(self.homeEvaluation())

        j = 0
        for x in market:
            i = 0
            for y in loc:
                self.input[0] = x
                self.input[1] = y
                self.fuzzify()
                z[j][i] = self.homeEvaluation()
                i += 1
            j += 1

        print()
        z = np.array(z)
        x = np.array(market)
        y = np.array(loc)

        X,Y = np.meshgrid(x,y)

        Z = np.sqrt(X ** 2 + Y ** 2)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(Y, X, z)

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

if __name__ == "__main__":
    # [MarketValue, Location, Asset, Income, Interest]
    innputt = [560000, 5, 473250, 30000, 6.5]
    f = fuzzy(innputt)
    f.fuzzify()
    z1 = f.homeEvaluation()
    z2 = f.applicEvaluation()
    z3 = f.creditEval()
    f.plot3d()

