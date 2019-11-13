import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import axes3d, Axes3D


class fuzzyv2:

    def __init__(self):
        # Market value fuzzy sets
        self.mvLow = [0, 0, 75000, 100000]
        self.mvMedium = [50000, 100000, 200000, 250000]
        self.mvHigh = [200000, 300000, 650000, 850000]
        self.mvVHigh = [650000, 850000, 1000000, 1000000]
        self.marketValue = [self.mvLow, self.mvMedium, self.mvHigh, self.mvVHigh]

        # Location fuzzy sets
        self.locBad = [0, 0, 1.5, 4]
        self.locFair = [2.5, 5, 6, 8.5]
        self.locExc = [6, 8.5, 10, 10]
        self.location = [self.locBad, self.locFair, self.locExc]

        # House fuzzy sets
        self.houseVLow = [0, 0, 3]
        self.houseLow = [0, 3, 6]
        self.houseMed = [2, 5, 8]
        self.houseHigh = [4, 7, 10]
        self.houseVHigh = [7, 10, 10]
        self.house = [self.houseVLow, self.houseLow, self.houseMed, self.houseHigh, self.houseVHigh]

        # Asset fuzzy sets
        self.assetLow = [0, 0, 0, 150000]
        self.assetMed = [50000, 250000, 450000, 650000]
        self.assetHigh = [500000, 700000, 1000000, 1000000]
        self.asset = [self.assetLow, self.assetMed, self.assetHigh]

        # Income fuzzy sets
        self.incomeLow = [0, 0, 10000, 25000]
        self.incomeMed = [15000, 35000, 55000]
        self.incomeHigh = [40000, 60000, 80000]
        self.incomeVHigh = [60000, 80000, 100000, 100000]
        self.income = [self.incomeLow, self.incomeMed, self.incomeHigh, self.incomeVHigh]

        # Applicant fuzzy sets
        self.applLow = [0, 0, 2, 4]
        self.applMed = [2, 5, 8]
        self.applHigh = [6, 8, 10, 10]
        self.applicant = [self.applLow, self.applMed, self.applHigh]

        # Interest fuzzy sets
        self.interestLow = [0, 0, 2, 5]
        self.interestMed = [2, 4, 6, 8]
        self.interestHigh = [6, 8.5, 10, 10]
        self.interest = [self.interestLow, self.interestMed, self.interestHigh]

        # Credit fuzzy sets
        self.creditVLow = [0, 0, 125000]
        self.creditLow = [0, 125000, 250000]
        self.creditMed = [125000, 250000, 375000]
        self.creditHigh = [250000, 375000, 500000]
        self.creditVHigh = [375000, 500000, 500000]
        self.credit = [self.creditVLow, self.creditLow, self.creditMed, self.creditHigh, self.creditVHigh]

        self.fuzzySets = [self.marketValue, self.location, self.house, self.asset, self.income, self.applicant,
                          self.interest, self.credit]

        # Outputs
        self.houseEvaluation = [0, 0, 0, 0, 0]
        self.applicantEvaluation = [0, 0, 0]
        self.creditEvaluation = [0, 0, 0, 0, 0]

    def fuzzify(self, input, keyword):
        if keyword == 'mv':
            return self.membershipFunction(input, self.marketValue)
        elif keyword == 'loc':
            return self.membershipFunction(input, self.location)
        elif keyword == 'hv':
            return self.membershipFunction(input, self.house)
        elif keyword == 'ass':
            return self.membershipFunction(input, self.asset)
        elif keyword == 'inc':
            return self.membershipFunction(input, self.income)
        elif keyword == 'ae':
            return self.membershipFunction(input, self.applicant)
        elif keyword == 'intr':
            return self.membershipFunction(input, self.interest)
        elif keyword == 'cred':
            return self.membershipFunction(input, self.credit)
        else:
            raise Exception('Keyword not recognized')

    def membershipFunction(self, x, limitsList):
        out = []
        for limits in limitsList:
            if (len(limits) == 4):
                a = limits[0]
                b = limits[1]
                c = limits[2]
                d = limits[3]
                if (a <= x <= d):
                    if (a <= x <= b):
                        temp = (x - a) / (b - a)
                        if math.isnan(temp):
                            out.append(1)
                        else:
                            out.append(temp)
                    elif (c <= x <= d):
                        out.append((d - x) / (d - c))
                    else:
                        out.append(1)
                else:
                    out.append(0)
            elif (len(limits) == 3):
                a = limits[0]
                b = limits[1]
                c = limits[2]
                if (a <= x <= b):
                    temp = (x - a) / (b - a)
                    if math.isnan(temp):
                        out.append(1)
                    else:
                        out.append(temp)
                elif (b <= x <= c):
                    temp = (c - x) / (c - b)
                    if math.isnan(temp):
                        out.append(1)
                    else:
                        out.append(temp)
                else:
                    out.append(0)
        return out

    def plotLimit(self, fuzzySet, title):
        minimum = fuzzySet[0][0]
        maximum = fuzzySet[len(fuzzySet) - 1][len(fuzzySet[len(fuzzySet) - 1]) - 1]
        if maximum > 10:
            x = np.arange(minimum, maximum)
        else:
            x = np.arange(minimum, maximum + 0.1, 0.1)

        y = []
        for i in x:
            y.append(self.membershipFunction(i, fuzzySet))

        maxplot = len(y[0])
        y2 = []
        for j in range(maxplot):
            temp = []
            for k in range(len(y)):
                temp.append(y[k][j])
            y2.append(temp)

        for vects in y2:
            plt.plot(x, vects)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plotLimits(self):
        titles = ['Market Value', 'Location', 'House Evaluation', 'Asset', 'Income', 'Applicant Evaluation',
                  'Interest', 'Credit Evaluation']

        for sets, title in zip(self.fuzzySets, titles):
            self.plotLimit(sets, title)

    def homeEvaluation(self, marketValue, location):
        houseEvaluation = {'VLow': 0, 'Low': 0, 'Med': 0, 'High': 0, 'VHigh': 0}
        mValue = {'Low': marketValue[0], 'Med': marketValue[1], 'High': marketValue[2], 'VHigh': marketValue[3]}
        loc = {'Bad': location[0], 'Fair': location[1], 'Exc': location[2]}

        # Rule 1:
        houseEvaluation['Low'] = max(houseEvaluation['Low'], mValue['Low'])

        # Rule 2:
        houseEvaluation['Low'] = max(houseEvaluation['Low'], loc['Bad'])

        # Rule 3:
        houseEvaluation['VLow'] = self.minMax(houseEvaluation['VLow'], mValue['Low'], loc['Bad'])

        # Rule 4:
        houseEvaluation['Low'] = self.minMax(houseEvaluation['Low'], mValue['Med'], loc['Bad'])

        # Rule 5:
        houseEvaluation['Med'] = self.minMax(houseEvaluation['Med'], mValue['High'], loc['Bad'])

        # Rule 6:
        houseEvaluation['High'] = self.minMax(houseEvaluation['High'], mValue['VHigh'], loc['Bad'])

        # Rule 7:
        houseEvaluation['Low'] = self.minMax(houseEvaluation['Low'], mValue['Low'], loc['Fair'])

        # Rule 8:
        houseEvaluation['Med'] = self.minMax(houseEvaluation['Med'], mValue['Med'], loc['Fair'])

        # Rule 9:
        houseEvaluation['High'] = self.minMax(houseEvaluation['High'], mValue['High'], loc['Fair'])

        # Rule 10:
        houseEvaluation['VHigh'] = self.minMax(houseEvaluation['VHigh'], mValue['VHigh'], loc['Fair'])

        # Rule 11:
        houseEvaluation['Med'] = self.minMax(houseEvaluation['Med'], mValue['Low'], loc['Exc'])

        # Rule 12:
        houseEvaluation['High'] = self.minMax(houseEvaluation['High'], mValue['Med'], loc['Exc'])

        # Rule 13:
        houseEvaluation['VHigh'] = self.minMax(houseEvaluation['VHigh'], mValue['High'], loc['Exc'])

        # Rule 14:
        houseEvaluation['VHigh'] = self.minMax(houseEvaluation['VHigh'], mValue['VHigh'], loc['Exc'])

        returnList = [houseEvaluation['VLow'], houseEvaluation['Low'], houseEvaluation['Med'], houseEvaluation['High'],
                      houseEvaluation['VHigh']]

        return returnList

    def applicEvaluation(self, asset, income):
        applicantEvaluation = {'Low': 0, 'Med': 0, 'High': 0}

        ass = {'Low': asset[0], 'Med': asset[1], 'High': asset[2]}
        inc = {'Low': income[0], 'Med': income[1], 'High': income[2], 'VHigh': income[3]}

        # Rule 1:
        applicantEvaluation['Low'] = self.minMax(applicantEvaluation['Low'], ass['Low'], inc['Low'])

        # Rule 2:
        applicantEvaluation['Low'] = self.minMax(applicantEvaluation['Low'], ass['Low'], inc['Med'])

        # Rule 3:
        applicantEvaluation['Med'] = self.minMax(applicantEvaluation['Med'], ass['Low'], inc['High'])

        # Rule 4:
        applicantEvaluation['High'] = self.minMax(applicantEvaluation['High'], ass['Low'], inc['VHigh'])

        # Rule 5:
        applicantEvaluation['Low'] = self.minMax(applicantEvaluation['Low'], ass['Med'], inc['Low'])

        # Rule 6:
        applicantEvaluation['Med'] = self.minMax(applicantEvaluation['Med'], ass['Med'], inc['Med'])

        # Rule 7:
        applicantEvaluation['High'] = self.minMax(applicantEvaluation['High'], ass['Med'], inc['High'])

        # Rule 8:
        applicantEvaluation['High'] = self.minMax(applicantEvaluation['High'], ass['Med'], inc['VHigh'])

        # Rule 9:
        applicantEvaluation['Med'] = self.minMax(applicantEvaluation['Med'], ass['High'], inc['Low'])

        # Rule 10:
        applicantEvaluation['Med'] = self.minMax(applicantEvaluation['Med'], ass['High'], inc['Med'])

        # Rule 11:
        applicantEvaluation['High'] = self.minMax(applicantEvaluation['High'], ass['High'], inc['High'])

        # Rule 12:
        applicantEvaluation['High'] = self.minMax(applicantEvaluation['High'], ass['High'], inc['VHigh'])

        returnList = [applicantEvaluation['Low'], applicantEvaluation['Med'], applicantEvaluation['High']]

        return returnList

    def creditEval(self, income, interest, applicant, house):
        creditEvaluation = {'VLow': 0, 'Low': 0, 'Med': 0, 'High': 0, 'VHigh': 0}

        inc = {'Low': income[0], 'Med': income[1], 'High': income[2], 'VHigh': income[3]}
        intr = {'Low': interest[0], 'Med': interest[1], 'High': interest[2]}
        appl = {'Low': applicant[0], 'Med': applicant[1], 'High': applicant[2]}
        hv = {'VLow': house[0], 'Low': house[1], 'Med': house[2], 'High': house[3], 'VHigh': house[4]}

        # Rule 1:
        creditEvaluation['VLow'] = self.minMax(creditEvaluation['VLow'], inc['Low'], intr['Med'])

        # Rule 2:
        creditEvaluation['VLow'] = self.minMax(creditEvaluation['VLow'], inc['Low'], intr['High'])

        # Rule 3:
        creditEvaluation['Low'] = self.minMax(creditEvaluation['Low'], inc['Med'], intr['High'])

        # Rule 4:
        creditEvaluation['VLow'] = max(creditEvaluation['VLow'], appl['Low'])

        # Rule 5:
        creditEvaluation['VLow'] = max(creditEvaluation['VLow'], hv['VLow'])

        # Rule 6:
        creditEvaluation['Low'] = self.minMax(creditEvaluation['Low'], appl['Med'], hv['VLow'])

        # Rule 7:
        creditEvaluation['Low'] = self.minMax(creditEvaluation['Low'], appl['Med'], hv['Low'])

        # Rule 8:
        creditEvaluation['Med'] = self.minMax(creditEvaluation['Med'], appl['Med'], hv['Med'])

        # Rule 9:
        creditEvaluation['High'] = self.minMax(creditEvaluation['High'], appl['Med'], hv['High'])

        # Rule 10:
        creditEvaluation['High'] = self.minMax(creditEvaluation['High'], appl['Med'], hv['VHigh'])

        # Rule 11:
        creditEvaluation['Low'] = self.minMax(creditEvaluation['Low'], appl['High'], hv['VLow'])

        # Rule 12:
        creditEvaluation['Med'] = self.minMax(creditEvaluation['Med'], appl['High'], hv['Low'])

        # Rule 13:
        creditEvaluation['High'] = self.minMax(creditEvaluation['High'], appl['High'], hv['Med'])

        # Rule 14:
        creditEvaluation['High'] = self.minMax(creditEvaluation['High'], appl['High'], hv['High'])

        # Rule 15:
        creditEvaluation['VHigh'] = self.minMax(creditEvaluation['VHigh'], appl['High'], hv['VHigh'])

        returnList = [creditEvaluation['VLow'], creditEvaluation['Low'], creditEvaluation['Med'],
                      creditEvaluation['High'], creditEvaluation['VHigh']]

        return returnList

    def minMax(self, assign, input1, input2):
        return max(assign, min(input1, input2))

    def findCentreOfGravity(self, fuzzySetList: list, vektor: list):
        sumListOver = []
        sumListUnder = []

        maxRange = max(fuzzySetList[len(fuzzySetList) - 1])

        for i in np.arange(0, maxRange, maxRange / 100):
            temp = self.membershipFunction(i, fuzzySetList)
            for j in range(len(temp)):
                if temp[j] > vektor[j]:
                    temp[j] = vektor[j]
            maxMf = max(temp)
            sumListUnder.append(maxMf)
            sumListOver.append(maxMf * i)

        cog = ((sum(sumListOver)) / (sum(sumListUnder)))

        return cog

    def findCentreOfGravityScale(self, fuzzySetList: list, vektor: list):
        sumListOver = []
        sumListUnder = []

        maxRange = max(fuzzySetList[len(fuzzySetList) - 1])

        for i in np.arange(0, maxRange, maxRange / 100):
            temp = self.membershipFunction(i, fuzzySetList)
            for j in range(len(temp)):
                temp[j] = temp[j] * vektor[j]
            maxMf = max(temp)
            sumListUnder.append(maxMf)
            sumListOver.append(maxMf * i)

        cog = ((sum(sumListOver)) / (sum(sumListUnder)))

        return cog

    def plot3d(self, *args, flipFlag=False):
        if len(args) == 4:
            input1 = args[0]
            input2 = args[1]
            fx = args[2]
            fuzzyZ = args[3]

            xvektor = np.arange(0, max(input1[len(input1) - 1]), max(input1[len(input1) - 1]) / 20)
            yvektor = np.arange(0, max(input2[len(input2) - 1]), max(input2[len(input2) - 1]) / 20)

            zvektor = np.empty((20, 20))

            j = 0
            for x in xvektor:
                i = 0
                for y in yvektor:
                    xtemp = self.membershipFunction(x, input1)
                    ytemp = self.membershipFunction(y, input2)
                    zlist = fx(xtemp, ytemp)
                    zvektor[j][i] = self.findCentreOfGravity(fuzzyZ, zlist)
                    i += 1
                j += 1

            X, Y = np.meshgrid(xvektor, yvektor)

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.invert_xaxis()
            surf = ax.plot_surface(Y, X, zvektor)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            fig.show()
        elif len(args) == 6:
            input1 = args[0]
            input2 = args[1]
            input3 = args[2]
            input4 = args[3]
            fx = args[4]
            fuzzyZ = args[5]

            xvektor = np.arange(0, max(input1[len(input1) - 1]), max(input1[len(input1) - 1]) / 20)
            yvektor = np.arange(0, max(input2[len(input2) - 1]), max(input2[len(input2) - 1]) / 20)

            zvektor = np.empty((20, 20))

            j = 0
            for x in xvektor:
                i = 0
                for y in yvektor:
                    xtemp = self.membershipFunction(x, input1)
                    ytemp = self.membershipFunction(y, input2)
                    if flipFlag:
                        zlist = fx(input3, ytemp, xtemp, input4)
                    else:
                        zlist = fx(input3, input4, xtemp, ytemp)
                    zvektor[j][i] = self.findCentreOfGravity(fuzzyZ, zlist)
                    i += 1
                j += 1

            X, Y = np.meshgrid(xvektor, yvektor)

            fig = plt.figure()
            ax = Axes3D(fig)
            if not flipFlag:
                ax.invert_xaxis()
            surf = ax.plot_surface(X, Y, zvektor)
            fig.colorbar(surf, shrink=0.5, aspect=5)

            fig.show()

    def getEvaluationClipped(self, inputVar):
        # Input:
        # [MarketValue, Location, Assets, Income, Interest]
        mvfuzz = self.fuzzify(inputVar[0], 'mv')
        locfuzz = self.fuzzify(inputVar[1], 'loc')
        hvfuzz = self.homeEvaluation(mvfuzz, locfuzz)
        hvCrisp = self.findCentreOfGravity(f.house, hvfuzz)
        print(f'House Value Crisp Clipped: {hvCrisp}')

        assfuzz = self.fuzzify(inputVar[2], 'ass')
        incfuzz = self.fuzzify(inputVar[3], 'inc')
        appfuzz = self.applicEvaluation(assfuzz, incfuzz)
        appCrisp = self.findCentreOfGravity(f.applicant, appfuzz)
        # print(appCrisp)

        intfuzz = self.fuzzify(inputVar[4], 'intr')
        hvfuzz2 = self.fuzzify(hvCrisp, 'hv')
        appfuzz2 = self.fuzzify(appCrisp, 'ae')
        creditFuzz = self.creditEval(incfuzz, intfuzz, appfuzz2, hvfuzz2)
        creditCrisp = self.findCentreOfGravity(f.credit, creditFuzz)

        return creditCrisp

    def getEvaluationScaled(self, inputVar):
        # Input:
        # [MarketValue, Location, Assets, Income, Interest]
        mvfuzz = self.fuzzify(inputVar[0], 'mv')
        locfuzz = self.fuzzify(inputVar[1], 'loc')
        hvfuzz = self.homeEvaluation(mvfuzz, locfuzz)
        hvCrisp = self.findCentreOfGravityScale(f.house, hvfuzz)
        print(f'House Value Crisp Scaled: {hvCrisp}')

        assfuzz = self.fuzzify(inputVar[2], 'ass')
        incfuzz = self.fuzzify(inputVar[3], 'inc')
        appfuzz = self.applicEvaluation(assfuzz, incfuzz)
        appCrisp = self.findCentreOfGravityScale(f.applicant, appfuzz)
        # print(appCrisp)

        intfuzz = self.fuzzify(inputVar[4], 'intr')
        hvfuzz2 = self.fuzzify(hvCrisp, 'hv')
        appfuzz2 = self.fuzzify(appCrisp, 'ae')
        creditFuzz = self.creditEval(incfuzz, intfuzz, appfuzz2, hvfuzz2)
        creditCrisp = self.findCentreOfGravityScale(f.credit, creditFuzz)

        return creditCrisp


if __name__ == "__main__":
    # input = [MarketValue, Location, Assets, Income, Interest]
    innputt = [700000, 6.5, 70200, 33000, 7]
    print(innputt)
    f = fuzzyv2()
    f.plotLimits()
    credit = f.getEvaluationClipped(innputt)
    print(f'Your clipped credit is evaluated to: {credit}')

    f.plot3d(f.marketValue, f.location, f.homeEvaluation, f.house)
    f.plot3d(f.asset, f.income, f.applicEvaluation, f.applicant)
    f.plot3d(f.applicant, f.house, [1, 0, 0, 0], [1, 0, 0], f.creditEval, f.credit)
    f.plot3d(f.applicant, f.interest, [0, 0, 0, 0], [0, 0, 1, 0, 0], f.creditEval, f.credit, flipFlag=True)

    credit = f.getEvaluationScaled(innputt)
    print(f'Your scaled credit is evaluated to: {credit}')
