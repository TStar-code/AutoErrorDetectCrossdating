import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time


def prepareSamples(lines_0):
    liness = []

    for line in lines_0:
        liness.append(line.split(','))

    num_of_colums = len(liness[0])

    columns = []

    for y in range(num_of_colums):
        temp_column = []
        for i in range(len(liness)):
            temp_column.append(liness[i][y])
        columns.append(temp_column)

    corColumns = []
    corBuffer = []

    for col in columns:
        buffer = ""
        corEles = []
        counter = 0
        dataFound = False
        for ele in col:
            if dataFound is False:
                if ele == col[0]:
                    counter = counter + 1
                elif ele == 'NA\n':
                    counter = counter + 1
                elif ele == 'NA':
                    counter = counter + 1
                else:
                    corEles.append(float(ele))
                    dataFound = True

            if ele == col[0]:
                buffer = "Ignore"
            elif ele == 'NA\n':
                buffer = "Ignore"
            elif ele == 'NA':
                buffer = "Ignore"
            else:
                corEles.append(float(ele))
                dataFound = True
        corColumns.append(corEles)
        corBuffer.append(counter)

    for col in corColumns:
        col.pop(0)

    return corColumns, corBuffer


def generateTupleOverlap(sampleX, sampleY):
    tempOverlap = []
    for i in range(len(sampleX)):
        tempOverlap.append((sampleX[i], sampleY[i]))
    return tempOverlap


def calculatePearsonCoefficient(overlap):
    '''
                            A - B
        pearson = -------------------------
                  sqrt(C - D) * sqrt(E - F)
    '''

    N = len(overlap)
    Xvalues = []
    Yvalues = []
    for i in range(len(overlap)):
        Xvalues.append(overlap[i][0])
        Yvalues.append(overlap[i][1])
    Xmean = sum(Xvalues) / len(Xvalues)
    Ymean = sum(Yvalues) / len(Yvalues)

    #PART A
    AValues = []
    for i in range(len(overlap)):
        temp = overlap[i][0] * overlap[i][1]
        AValues.append(temp)
    A = sum(AValues)

    #PART B
    B = N * Xmean * Ymean

    #PART C
    CValues = []
    for i in range(len(Xvalues)):
        temp = Xvalues[i] * Xvalues[i]
        CValues.append(temp)
    C = sum(CValues)

    #PART D
    D = N * (Xmean * Xmean)

    #PART E
    EValues = []
    for i in range(len(Yvalues)):
        temp = Yvalues[i] * Yvalues[i]
        EValues.append(temp)
    E = sum(EValues)

    #PART F
    F = N * (Ymean * Ymean)

    #Final Calc
    top = A - B
    BottomPartOne = math.sqrt(C - D)
    BottomPartTwo = math.sqrt(E - F)

    PearsonCoefficient = top / (BottomPartOne * BottomPartTwo)

    return PearsonCoefficient


def calculateTValue(overlap, pearson_coefficient):
    N = len(overlap)
    R = pearson_coefficient
    top = R * (math.sqrt(N-2))
    bottom = math.sqrt(1 - (R * R))
    tValue = top / bottom
    return tValue


'''
    SAMPLE Y IS RAN ACROSS SAMPLE X, I.E. SAMPLE Y STARTS AT ITS LEFT MOST OVERLAP WITH X AND RUNS UNTIL ITS RIGHT MOST
    OVERLAP. (KEY DISTINCTION TO MAKE FOR WHEN USING THIS FUNCTION TO CROSSDATE)
'''
def calculateTValues(sampleX, sampleY):
    prepX = []
    prepY = []

    for i in range((len(sampleX) + len(sampleY))):
        x = 0
        y = 0
        if i >= len(sampleY):
            x = sampleX[i - len(sampleY)]
        if i <= len(sampleY) - 1:
            y = sampleY[i]
        prepX.append(x)
        prepY.append(y)

    allOverlaps = []
    npX = np.array(prepX)
    npY = np.array(prepY)

    allOverlaps.append(generateTupleOverlap(npX, npY))

    tempSampleY = npY
    while tempSampleY[-1] == 0:
        tempSampleY = np.roll(tempSampleY, 1)
        allOverlaps.append(generateTupleOverlap(npX, tempSampleY))

    tempSampleX = npX
    while tempSampleX[0] == 0:
        tempSampleX = np.roll(tempSampleX, -1)
        allOverlaps.append(generateTupleOverlap(tempSampleX, tempSampleY))

    #allOverlaps is an array of arrays of tuples containing every possible overlap of the two given samples
    #where the first array is the starting position (no overlap) and the last array is the ending position (no overlap)

    allOverlaps.pop(0)
    allOverlaps.pop(-1)

    newAllOverlaps = []

    for i in range(len(allOverlaps)):
        tempArray = []
        stringBuffer = ""
        for j in range(len(allOverlaps[i])):
            if allOverlaps[i][j][0] == 0:
                stringBuffer = "ignore"
            elif allOverlaps[i][j][1] == 0:
                stringBuffer = "ignore"
            else:
                tempArray.append(allOverlaps[i][j])
        newAllOverlaps.append(tempArray)


    #newAllOverlaps now is an array of arrays of tuples containing every possible useful overlap of two samples


    '''this is the code for calculation the pearson and tvalue for a single overlap section'''
    '''temp_p = calculatePearsonCoefficient(newAllOverlaps[2])
    temp_a = calculateTValue(newAllOverlaps[2], temp_p)'''


    '''for over in newAllOverlaps:
        print(over)'''



    '''THIS SECTION IS THERE TO REMOVE SCENARIOS THAT CANNOT BE CALCULATED'''
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    '''END SECTION'''

    '''THIS SECTION IS THERE TO REMOVE STATISTICAL ANOMALIES'''
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    '''END SECTION'''

    pearsons = []
    tValues = []

    for over in newAllOverlaps:
        temp_p = calculatePearsonCoefficient(over)
        temp_t = calculateTValue(over, temp_p)
        pearsons.append(temp_p)
        tValues.append(temp_t)

    # pearsons is a list of all of the pearson coefficients for all of the sample overlap posistions
    # tValues is their corresponding t values

    # sample y moves over x starting from the left most position and wokring to the right most

    # lag indicated by index representing position from starting position, i.e 0th index is sample y placed before sample
    # x with no overlap. Thus 1st index (index=1) is sample y placed before sample x with 1 overlap

    lagAmount = []

    for i in range(len(pearsons)):
        lagAmount.append(i+3)

    return tValues, lagAmount, pearsons



main_file = open("test_data/mt108_detrend.csv", "r")
lines = main_file.readlines()
main_file.close()

#samples is an array of all of the samples given, including a dating column
#samplebuffers is an array of all of the buffers to correctly place the sample
samples, sampleBuffers = prepareSamples(lines)

basesample = samples[1]
overlapsample = samples[2]

#tV is an array of all of the tvalues for the given overlap of the two samples
#lA is an array of lag amounts corresponding to each tvalue
tV, lA, pC = calculateTValues(basesample, overlapsample)


samples.pop(0)

allComps = []
Comps = itertools.combinations(samples, 2)
for comp in Comps:
    allComps.append(comp)

index = []
for i in range(len(samples)):
    index.append(i)

indexs = itertools.combinations(index, 2)

compIndex = []
for comp in indexs:
    compIndex.append(comp)

#allComps is a list of every combination of samples to be crossdated
#compIndex is a list of all of the indexs for each combination


allTValues = []
allLagAmounts = []
allPearsons = []


counter = 0
for comp in allComps:
    temp_tV, temp_lA, temp_pC = calculateTValues(comp[0], comp[1])
    allTValues.append(temp_tV)
    allLagAmounts.append(temp_lA)
    allPearsons.append(temp_pC)
    counter = counter + 1
    if counter % 400 == 0:
        print("iteration", counter)

#samples is a list of all of the samples
#compIndex is a list of tuples containing the index for each combination of samples
#allComps is a list of all possible combinations of all of the samples
#allTValues is a list of all of the tvalues for every combination
#allLagAmounts is a list of all of the lag amounts for every combination
#allPearsons is a list of all of the pearson coefficients for every combination


highest = []
samplesPlaced = []
samplesyettoplace = list(range(0, len(samples)))
all_pairings = []
all_lags = []

temporary_list = []

for i in range(len(allTValues)):
    temp_h = max(allTValues[i])
    highest.append(temp_h)
    temp_o = allTValues[i].index(temp_h)
    temporary_list.append(temp_o)


temp_l = max(highest)

first_pair_index = highest.index(temp_l)

first_sample = compIndex[first_pair_index][0]
second_sample = compIndex[first_pair_index][1]

temporary_lag_amount = temporary_list[first_pair_index] + 3
all_lags.append(temporary_lag_amount)

all_pairings.append((first_sample, second_sample))

samplesPlaced.append(first_sample)
samplesPlaced.append(second_sample)

samplesyettoplace.remove(first_sample)
samplesyettoplace.remove(second_sample)

while len(samplesyettoplace) > 0:
    comps_to_check = []
    comps_to_check_values = []
    for samp in samplesPlaced:
        #print("index", samp, "has been placed")
        for ele in compIndex:
            if ele[0] == samp and ele[1] not in samplesPlaced:
                comps_to_check.append(ele)
                comps_to_check_values.append(allTValues[compIndex.index(ele)])
            elif ele[1] == samp and ele[0] not in samplesPlaced:
                comps_to_check.append(ele)
                comps_to_check_values.append(allTValues[compIndex.index(ele)])

    t_highest = []
    t_highest_index = []

    for i in range(len(comps_to_check_values)):
        temp_k = max(comps_to_check_values[i])
        temp_o = comps_to_check_values[i].index(temp_k)
        t_highest.append(temp_k)
        t_highest_index.append(temp_o)

    next_to_add = max(t_highest)
    next_to_add_index = t_highest.index(next_to_add)

    lag_amount = t_highest_index[next_to_add_index] + 3

    next_to_add_index_0 = comps_to_check[next_to_add_index][0]
    next_to_add_index_1 = comps_to_check[next_to_add_index][1]

    pairing = (next_to_add_index_0, next_to_add_index_1)

    temp_string = ""
    if next_to_add_index_0 in samplesPlaced:
        #print("already there", next_to_add_index_0)
        temp_string = "yyy"
    elif next_to_add_index_0 not in samplesPlaced:
        #print("being added now", next_to_add_index_0)
        samplesPlaced.append(next_to_add_index_0)
        samplesyettoplace.remove(next_to_add_index_0)

    if next_to_add_index_1 in samplesPlaced:
        #print("already there", next_to_add_index_1)
        temp_string = "yyy"
    elif next_to_add_index_1 not in samplesPlaced:
        #print("being added now", next_to_add_index_1)
        samplesPlaced.append(next_to_add_index_1)
        samplesyettoplace.remove(next_to_add_index_1)

    all_pairings.append(pairing)
    all_lags.append(lag_amount)

    #print(samplesPlaced)
    #print(samplesyettoplace)

print(all_pairings)
print(all_lags)


#all pairings should be a list of all of the best possible pairs in order.
#all lags should be a list of the corersponding lags to put each pair in the right spot.