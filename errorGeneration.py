import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import random
import copy
import statistics
from statistics import mean

def prepareSets(lines_0):
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

def prepareSamples(lines_1):
    total_set = []
    temp_set = []

    for ele in lines_1:
        temp_set.append(ele)
        if ele == "---seperatorline---\n":
            total_set.append(temp_set)
            temp_set = []
        if ele == "---seperatorline---":
            total_set.append(temp_set)
            temp_set = []

    for ele in total_set:
        ele.pop(-1)

    all_sets = []
    all_buffers = []
    all_samples = []

    for ele in total_set:
        temp_samples, temp_buffers = prepareSets(ele)
        all_sets.append(temp_samples)
        all_buffers.append(temp_buffers)

    for set in all_sets:
        set.pop(0)

    for set in all_sets:
        for sample in set:
            all_samples.append(sample)
    return all_samples

def prepareChrono(lines_2):
    liness = []

    for line in lines_2:
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

def buildChrono(lines_3):
    samples, buff = prepareChrono(lines_3)

    samples.pop(0)
    buff.pop(0)

    for i in range(len(samples)):
        num_of_zeros = buff[i]
        for j in range(num_of_zeros - 1):
            samples[i].insert(0, 0)

    length_of_chron = len(samples[0])
    num_of_samples = len(samples)

    all_averages = []
    for i in range(length_of_chron):
        temp_total = 0
        temp_total_amount = 0
        temp_average = 0
        for j in range(num_of_samples):
            if samples[j][i] != 0:
                temp_total = temp_total + samples[j][i]
                temp_total_amount = temp_total_amount + 1
        temp_average = temp_total / temp_total_amount
        all_averages.append(temp_average)


    return all_averages, samples, buff

def generateErrors(sample): #takes in 1 sample and returns a variation of that sample with errors randomly placed.
    temporary_sample = copy.deepcopy(sample)
    localised_error_chance = (1/100)*(len(temporary_sample))*incor_per_hundred
    error_pos = []
    for i in range(len(temporary_sample)):
        error_chance = random.uniform(0, 1)
        if error_chance < localised_error_chance:
            #error spawned
            error_pos.append(i)
            miss_or_add = random.uniform(0, 1)
            if miss_or_add < 0.5:
                #add rings
                addi_error.append(0)
                err_clump_size = random.randint(1, 3)
                #print(err_clump_size, "additional rings at position", i)
                temp_i = i
                for j in range(err_clump_size):
                    if temp_i+j < len(temporary_sample):
                        value = temporary_sample[temp_i + j]
                        split = random.uniform(1.5, 2.5)
                        temp = value / split
                        other_temp = value - temp
                        temporary_sample[temp_i + j] = temp
                        temporary_sample.insert(temp_i + j + 1, other_temp)
                        temp_i = temp_i + 1
            else:
                #miss rings
                miss_error.append(0)
                err_clump_size = random.randint(1, 4)
                #print(err_clump_size, "missing rings at position", i)
                total = 0
                for j in range(err_clump_size):
                    if len(temporary_sample) > i+j:
                        total = total + temporary_sample[i+j]
                    else:
                        total = total + 0
                for p in range(err_clump_size):
                    if len(temporary_sample) > i+0:
                        temporary_sample.pop(i+0)
                temporary_sample.insert(i+0, total)
    return temporary_sample, error_pos

def addErrors(samples):
    test_samples = copy.deepcopy(samples)
    num_of_oth_ver = 4
    new_all_samples = []
    new_all_samples_errors = []

    for samp in test_samples:
        new_all_samples.append(samp)
        new_all_samples_errors.append([])
        for i in range(num_of_oth_ver):
            test_samp, test_error = generateErrors(samp)
            new_all_samples.append(test_samp)
            new_all_samples_errors.append(test_error)

    return new_all_samples, new_all_samples_errors

def databuilding(lines_4):
    total_set = []
    temp_set = []

    for ele in lines_4:
        temp_set.append(ele)
        if ele == "---seperatorline---\n":
            total_set.append(temp_set)
            temp_set = []
        if ele == "---seperatorline---":
            total_set.append(temp_set)
            temp_set = []

    for ele in total_set:
        ele.pop(-1)

    counter = 0

    for set in total_set:
        chron, samps, starting_pos = buildChrono(set)
        all_samples = copy.deepcopy(samps)

        for i in range(len(all_samples)):
            for j in range(starting_pos[i] - 1):
                all_samples[i].pop(0)

        all_samples_incl_errors, all_error_positions = addErrors(all_samples)
        minimum_sample_length = (sample_length + number_of_versions_of_each_sample) / sample_length

        for i in range(len(all_samples_incl_errors)):
            for j in range(number_of_versions_of_each_sample):
                if len(all_samples_incl_errors[i]) > (sample_length * minimum_sample_length):
                    random_sample_pos = random.randint(0, len(all_samples_incl_errors[i]) - sample_length)

                    sample_values = all_samples_incl_errors[i][random_sample_pos:random_sample_pos + sample_length]
                    chron_values = chron[starting_pos[math.floor(i/5)] - 1 + random_sample_pos:starting_pos[math.floor(i/5)] - 1 + random_sample_pos + sample_length]

                    error_values = []
                    if len(all_error_positions[i]) != 0:
                        for err_val in all_error_positions[i]:
                            if err_val > random_sample_pos and err_val < random_sample_pos + sample_length:
                                error_values.append(err_val - random_sample_pos)

                    f = open("neural_data/input_data.txt", "a")
                    for idx, val in enumerate(sample_values):
                        f.write(str(val) + ",")
                    for idx, val in enumerate(chron_values):
                        if chron_values[idx] == chron_values[-1]:
                            f.write(str(val) + "\n")
                        else:
                            f.write(str(val) + ",")
                    f.close()

                    f = open("neural_data/output_data_binary.txt", "a")
                    if len(error_values) == 0:
                        f.write(str(0) + "\n")
                        no_error.append(0)
                    else:
                        f.write(str(1) + "\n")
                        with_error.append(0)
                    f.close()

                    f = open("neural_data/output_data_position.txt", "a")
                    tempppoo = error_values
                    if len(tempppoo) == 0:
                        f.write("0" + "\n")
                        with_error_amount_total.append(0)
                    else:
                        with_error_amount.append(len(tempppoo))
                        with_error_amount_total.append(len(tempppoo))
                        for idx, val in enumerate(tempppoo):
                            if tempppoo[idx] == tempppoo[-1]:
                                f.write(str(val) + "\n")
                            else:
                                f.write(str(val) + ",")


                    f.close()

                    #print(sample_values)
                    #print(chron_values)

                    #print("---end of version---")
            print("---end of sample---")
        print("---end of chronology---")
    return 0

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

    #NEXT: calculate pearson coefficients and t values
    #AFTER: return the array of t values



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


incor_per_hundred = 0.0047 # 0.0047 best to average 1 err clump per 100 rings
sample_length = 100
number_of_versions_of_each_sample = 30

main_file = open("test_data/final_total_one_chron.csv", "r")
lines = main_file.readlines()
main_file.close()

f = open("neural_data/input_data.txt", "w")
f.close()
f = open("neural_data/output_data_binary.txt", "w")
f.close()
f = open("neural_data/output_data_position.txt", "w")
f.close()

samples = prepareSamples(lines)

#sample is new sample with errors generated
#error pos is the starting position of the error clump
#new_sample, error_positions = generateErrors(samples[0])

with_error = []
no_error = []

addi_error = []
miss_error = []

with_error_amount = []
with_error_amount_total = []

johnson = databuilding(lines)





input_file = open("neural_data/input_data.txt", "r")
input_lines = input_file.readlines()
input_file.close()

target_file = open("neural_data/output_data_binary.txt", "r")
target_lines = target_file.readlines()
target_file.close()

target_file_2 = open("neural_data/output_data_position.txt", "r")
target_lines_2 = target_file_2.readlines()
target_file_2.close()

def cleanData(lines):
    split_lines = []

    for lin in lines:
        split_lines.append(lin.split(','))

    #print(split_lines)

    for i in range(len(split_lines)):
        for j in range(len(split_lines[i])):
            split_lines[i][j] = split_lines[i][j].replace("\n", "")

    for i in range(len(split_lines)):
        for j in range(len(split_lines[i])):
            split_lines[i][j] = float(split_lines[i][j])

    return split_lines

jim = cleanData(input_lines)
bob = cleanData(target_lines)
barbara = cleanData(target_lines_2)

indices_to_remove = []

for i in range(len(jim)):
    if len(jim[i]) != 200:
        indices_to_remove.append(i)

for i in range(len(indices_to_remove)):
    indices_to_remove[i] = indices_to_remove[i] - i

for val in indices_to_remove:
    jim.pop(val)
    bob.pop(val)
    barbara.pop(val)

features = []
for i, jimmy in enumerate(jim):
    sample = jimmy[:len(jimmy) // 2]
    chron = jimmy[len(jimmy) // 2:]
    tvalues, bang, bong = calculateTValues(chron, sample)
    samp_max = [max(sample)]
    samp_min = [min(sample)]
    chron_max = [max(chron)]
    chron_min = [min(chron)]
    tval_std = [np.std(tvalues)]
    temp_value = sample + samp_max + samp_min + chron + tvalues + chron_max + chron_min + tval_std
    features.append(temp_value)

for i in range(len(features)):
    if len(features[i]) != 396:
        features.pop(i)
        bob.pop(i)
        barbara.pop(i)
        break

f = open("neural_data/input_data.txt", "w")
f.close()
f = open("neural_data/output_data_binary.txt", "w")
f.close()
f = open("neural_data/output_data_position.txt", "w")
f.close()

f = open("neural_data/input_data.txt", "a")
for i in range(len(features)):
    for idx, val in enumerate(features[i]):
        if features[i][idx] == features[i][-1]:
            f.write(str(val) + "\n")
        else:
            f.write(str(val) + ",")
f.close()

f = open("neural_data/output_data_binary.txt", "a")
for i in range(len(bob)):
    f.write(str(bob[i][0]) + "\n")
f.close()

f = open("neural_data/output_data_position.txt", "a")
for i in range(len(barbara)):
    for idx, val in enumerate(barbara[i]):
        if barbara[i][idx] == barbara[i][-1]:
            f.write(str(val) + "\n")
        else:
            f.write(str(val) + ",")
f.close()




