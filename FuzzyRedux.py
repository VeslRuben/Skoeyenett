import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class fuzzy:

    def __init__(self):
        # Market value fuzzy sets
        self.marketLow = [0, 0, 75000, 100000]
        self.marketMedium = [50000, 100000, 200000, 250000]
        self.marketHigh = [200000, 300000, 650000, 850000]
        self.marketVHigh = [650000, 850000, 1000000, 1000000]
        self.marketValueSet = [self.marketLow, self.marketMedium, self.marketHigh, self.marketVHigh]

        # Location fuzzy sets
        self.locationBad = [0, 0, 1.5, 4]
        self.locationFair = [2.5, 5, 6, 8.5]
        self.locationExc = [6, 8.5, 10, 10]
        self.location = [self.locationBad, self.locationFair, self.locationExc]

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
        self.applicantLow = [0, 0, 2, 4]
        self.applicantMed = [2, 5, 8]
        self.applicantHigh = [6, 8, 10, 10]
        self.applicant = [self.applicantLow, self.applicantMed, self.applicantHigh]

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

        self.fuzzySets = [self.marketValueSet, self.location, self.house, self.asset, self.income, self.applicant,
                          self.interest, self.credit]

        # Outputs
        self.houseEvaluation = [0, 0, 0, 0, 0]
        self.applicantEvaluation = [0, 0, 0]
        self.creditEvaluation = [0, 0, 0, 0, 0]

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

    def fuzzify(self, input, keyword):
        if keyword == 'mv':
            return self.membershipFunction(input, self.marketValueSet)
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

    def minOfMax(self, assign, input1, input2):
        return max(assign, min(input1, input2))

    def creditEval(self, income, interest, applicant, house):
        creditEvaluation = {'VLow': 0, 'Low': 0, 'Med': 0, 'High': 0, 'VHigh': 0}

        incom = {'Low': income[0], 'Med': income[1], 'High': income[2], 'VHigh': income[3]}
        intrest = {'Low': interest[0], 'Med': interest[1], 'High': interest[2]}
        applicant = {'Low': applicant[0], 'Med': applicant[1], 'High': applicant[2]}
        houseValue = {'VLow': house[0], 'Low': house[1], 'Med': house[2], 'High': house[3], 'VHigh': house[4]}

        creditEvaluation['VLow'] = self.minOfMax(creditEvaluation['VLow'], incom['Low'], intrest['Med'])

        creditEvaluation['VLow'] = self.minOfMax(creditEvaluation['VLow'], incom['Low'], intrest['High'])

        creditEvaluation['Low'] = self.minOfMax(creditEvaluation['Low'], incom['Med'], intrest['High'])

        creditEvaluation['VLow'] = max(creditEvaluation['VLow'], applicant['Low'])

        creditEvaluation['VLow'] = max(creditEvaluation['VLow'], houseValue['VLow'])

        creditEvaluation['Low'] = self.minOfMax(creditEvaluation['Low'], applicant['Med'], houseValue['VLow'])

        creditEvaluation['Low'] = self.minOfMax(creditEvaluation['Low'], applicant['Med'], houseValue['Low'])

        creditEvaluation['Med'] = self.minOfMax(creditEvaluation['Med'], applicant['Med'], houseValue['Med'])

        creditEvaluation['High'] = self.minOfMax(creditEvaluation['High'], applicant['Med'], houseValue['High'])

        creditEvaluation['High'] = self.minOfMax(creditEvaluation['High'], applicant['Med'], houseValue['VHigh'])

        creditEvaluation['Low'] = self.minOfMax(creditEvaluation['Low'], applicant['High'], houseValue['VLow'])

        creditEvaluation['Med'] = self.minOfMax(creditEvaluation['Med'], applicant['High'], houseValue['Low'])

        creditEvaluation['High'] = self.minOfMax(creditEvaluation['High'], applicant['High'], houseValue['Med'])

        creditEvaluation['High'] = self.minOfMax(creditEvaluation['High'], applicant['High'], houseValue['High'])

        creditEvaluation['VHigh'] = self.minOfMax(creditEvaluation['VHigh'], applicant['High'], houseValue['VHigh'])

        return_list = [creditEvaluation['VLow'], creditEvaluation['Low'], creditEvaluation['Med'],
                      creditEvaluation['High'], creditEvaluation['VHigh']]

        return return_list

    def applicEvaluation(self, asset, income):
        applicantEvaluation = {'Low': 0, 'Med': 0, 'High': 0}

        assets = {'Low': asset[0], 'Med': asset[1], 'High': asset[2]}
        income = {'Low': income[0], 'Med': income[1], 'High': income[2], 'VHigh': income[3]}

        applicantEvaluation['Low'] = self.minOfMax(applicantEvaluation['Low'], assets['Low'], income['Low'])

        applicantEvaluation['Low'] = self.minOfMax(applicantEvaluation['Low'], assets['Low'], income['Med'])

        applicantEvaluation['Med'] = self.minOfMax(applicantEvaluation['Med'], assets['Low'], income['High'])

        applicantEvaluation['High'] = self.minOfMax(applicantEvaluation['High'], assets['Low'], income['VHigh'])

        applicantEvaluation['Low'] = self.minOfMax(applicantEvaluation['Low'], assets['Med'], income['Low'])

        applicantEvaluation['Med'] = self.minOfMax(applicantEvaluation['Med'], assets['Med'], income['Med'])

        applicantEvaluation['High'] = self.minOfMax(applicantEvaluation['High'], assets['Med'], income['High'])

        applicantEvaluation['High'] = self.minOfMax(applicantEvaluation['High'], assets['Med'], income['VHigh'])

        applicantEvaluation['Med'] = self.minOfMax(applicantEvaluation['Med'], assets['High'], income['Low'])

        applicantEvaluation['Med'] = self.minOfMax(applicantEvaluation['Med'], assets['High'], income['Med'])

        applicantEvaluation['High'] = self.minOfMax(applicantEvaluation['High'], assets['High'], income['High'])

        applicantEvaluation['High'] = self.minOfMax(applicantEvaluation['High'], assets['High'], income['VHigh'])

        return_list = [applicantEvaluation['Low'], applicantEvaluation['Med'], applicantEvaluation['High']]

        return return_list

    def homeEvaluation(self, marketValue, location):
        houseEvaluation = {'VLow': 0, 'Low': 0, 'Med': 0, 'High': 0, 'VHigh': 0}
        marketValue = {'Low': marketValue[0], 'Med': marketValue[1], 'High': marketValue[2], 'VHigh': marketValue[3]}
        location = {'Bad': location[0], 'Fair': location[1], 'Exc': location[2]}

        houseEvaluation['Low'] = max(houseEvaluation['Low'], marketValue['Low'])

        houseEvaluation['Low'] = max(houseEvaluation['Low'], location['Bad'])

        houseEvaluation['VLow'] = self.minOfMax(houseEvaluation['VLow'], marketValue['Low'], location['Bad'])

        houseEvaluation['Low'] = self.minOfMax(houseEvaluation['Low'], marketValue['Med'], location['Bad'])

        houseEvaluation['Med'] = self.minOfMax(houseEvaluation['Med'], marketValue['High'], location['Bad'])

        houseEvaluation['High'] = self.minOfMax(houseEvaluation['High'], marketValue['VHigh'], location['Bad'])

        houseEvaluation['Low'] = self.minOfMax(houseEvaluation['Low'], marketValue['Low'], location['Fair'])

        houseEvaluation['Med'] = self.minOfMax(houseEvaluation['Med'], marketValue['Med'], location['Fair'])

        houseEvaluation['High'] = self.minOfMax(houseEvaluation['High'], marketValue['High'], location['Fair'])

        houseEvaluation['VHigh'] = self.minOfMax(houseEvaluation['VHigh'], marketValue['VHigh'], location['Fair'])

        houseEvaluation['Med'] = self.minOfMax(houseEvaluation['Med'], marketValue['Low'], location['Exc'])

        houseEvaluation['High'] = self.minOfMax(houseEvaluation['High'], marketValue['Med'], location['Exc'])

        houseEvaluation['VHigh'] = self.minOfMax(houseEvaluation['VHigh'], marketValue['High'], location['Exc'])

        houseEvaluation['VHigh'] = self.minOfMax(houseEvaluation['VHigh'], marketValue['VHigh'], location['Exc'])

        return_list = [houseEvaluation['VLow'], houseEvaluation['Low'], houseEvaluation['Med'], houseEvaluation['High'],
                      houseEvaluation['VHigh']]

        return return_list

    def findCOG(self, fuzzySetList: list, vektor: list):
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

    def findCOGScale(self, fuzzySetList: list, vektor: list):
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

    def plot(self, *args, flipFlag=False):
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
                    zvektor[j][i] = self.findCOG(fuzzyZ, zlist)
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
                    zvektor[j][i] = self.findCOG(fuzzyZ, zlist)
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

    def getClipped(self, inputVar):
        # Input:
        # [MarketValue, Location, Assets, Income, Interest]
        mvfuzz = self.fuzzify(inputVar[0], 'mv')
        locfuzz = self.fuzzify(inputVar[1], 'loc')
        hvfuzz = self.homeEvaluation(mvfuzz, locfuzz)
        hvCrisp = self.findCOG(fuzz.house, hvfuzz)
        print(f'House Value Crisp Clipped: {hvCrisp}')

        assfuzz = self.fuzzify(inputVar[2], 'ass')
        incfuzz = self.fuzzify(inputVar[3], 'inc')
        appfuzz = self.applicEvaluation(assfuzz, incfuzz)
        appCrisp = self.findCOG(fuzz.applicant, appfuzz)
        # print(appCrisp)

        intfuzz = self.fuzzify(inputVar[4], 'intr')
        hvfuzz2 = self.fuzzify(hvCrisp, 'hv')
        appfuzz2 = self.fuzzify(appCrisp, 'ae')
        creditFuzz = self.creditEval(incfuzz, intfuzz, appfuzz2, hvfuzz2)
        creditCrisp = self.findCOG(fuzz.credit, creditFuzz)

        return creditCrisp

    def getScaled(self, inputVar):
        # Input:
        # [MarketValue, Location, Assets, Income, Interest]
        mvfuzz = self.fuzzify(inputVar[0], 'mv')
        locfuzz = self.fuzzify(inputVar[1], 'loc')
        hvfuzz = self.homeEvaluation(mvfuzz, locfuzz)
        hvCrisp = self.findCOGScale(fuzz.house, hvfuzz)
        print(f'House Value Crisp Scaled: {hvCrisp}')

        assfuzz = self.fuzzify(inputVar[2], 'ass')
        incfuzz = self.fuzzify(inputVar[3], 'inc')
        appfuzz = self.applicEvaluation(assfuzz, incfuzz)
        appCrisp = self.findCOGScale(fuzz.applicant, appfuzz)
        # print(appCrisp)

        intfuzz = self.fuzzify(inputVar[4], 'intr')
        hvfuzz2 = self.fuzzify(hvCrisp, 'hv')
        appfuzz2 = self.fuzzify(appCrisp, 'ae')
        creditFuzz = self.creditEval(incfuzz, intfuzz, appfuzz2, hvfuzz2)
        creditCrisp = self.findCOGScale(fuzz.credit, creditFuzz)

        return creditCrisp


if __name__ == "__main__":
    # input = [MarketValue, Location, Assets, Income, Interest]
    innputt = [246123, 7, 56280, 91220, 3.2]

    fuzz = fuzzy()
    fuzz.plotLimits()
    credit = fuzz.getClipped(innputt)
    print(f"clipped credit is: {credit}")

    fuzz.plot(fuzz.marketValueSet, fuzz.location, fuzz.homeEvaluation, fuzz.house)
    fuzz.plot(fuzz.asset, fuzz.income, fuzz.applicEvaluation, fuzz.applicant)
    fuzz.plot(fuzz.applicant, fuzz.house, [1, 0, 0, 0], [1, 0, 0], fuzz.creditEval, fuzz.credit)
    fuzz.plot(fuzz.applicant, fuzz.interest, [0, 0, 0, 0], [0, 0, 1, 0, 0], fuzz.creditEval, fuzz.credit, flipFlag=True)

    credit = fuzz.getScaled(innputt)
    print(f"scaled credit is: {credit}")
