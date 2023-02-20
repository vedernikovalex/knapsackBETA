import random
import itertools
import time
import numpy
import multiprocessing


def get_items_count_user_input():
    itemCountInput = ""
    print("How much items do you want to generate for solving?")
    while itemCountInput == "":
        itemCountInput = input(">> ")
        try:
            itemCount = int(itemCountInput)
            return itemCount
        except ValueError:
            print("Only numbers allowed")
            itemCountInput = ""


def generate_items(itemCount, wMin, wMax, vMin, vMax):
    # item creation
    weights = []
    values = []
    for i in range(itemCount):
        weights.append(random.randint(wMin, wMax))
        values.append(random.randint(vMin, vMax))
        print(i, weights[i], values[i])
    return weights, values


def get_min_max_user_input():
    inputString = ""
    min = 0
    max = 0
    while inputString == "":
        inputString = input(">> ")
        try:
            numbers = inputString.split(",")
            if len(numbers) == 2:
                num1 = int(numbers[0])
                num2 = int(numbers[1])
                if num1 > num2:
                    min = num2
                    max = num1
                else:
                    min = num1
                    max = num2
            else:
                print("Exactly two numbers required!")
                inputString = ""
        except ValueError:
            print("Only numbers allowed!")
            inputString = ""
    return min, max


# brute force algorithm
def knapsack_bruteforce(weights, values, itemCount, knapsackCapacity):
    maxValue = 0
    bestCombination = []
    for i in range(1, itemCount + 1):
        for combination in itertools.combinations(range(itemCount), i):
            totalWeight = sum(weights[j] for j in combination)
            if totalWeight <= knapsackCapacity:
                totalValue = sum(values[j] for j in combination)
                if totalValue > maxValue:
                    maxValue = totalValue
                    bestCombination = combination
    if not bestCombination:
        return None

    return maxValue, list(bestCombination)


# combinations algorithm
def calc_combination(knapsackCapacity, weights, values, starIndex, endIndex, combinations, queue):
    print("invoke calc")
    maxValue = 0
    bestCombination = []
    print(starIndex, endIndex)
    for i in range(starIndex, endIndex):
        combination = combinations[i]
        totalWeight = sum(weights[j] for j in combination)
        if totalWeight <= knapsackCapacity:
            totalValue = sum(values[j] for j in combination)
            if totalValue > maxValue:
                maxValue = totalValue
                bestCombination = combination

    queue.put((maxValue, bestCombination))


def knapsack_bruteforce_processing(weights, values, itemCount, knapsackCapacity):
    cpuCount = multiprocessing.cpu_count()
    queue = multiprocessing.Queue()
    maxValue = 0
    bestCombination = []
    combinations = []
    for i in range(1, itemCount + 1):
        itemCombinations = itertools.combinations(range(itemCount), i)
        combinations.extend(itemCombinations)

    processes = []
    partSize = len(combinations) // cpuCount
    for i in range(cpuCount):
        print("invoke process")
        start = i * partSize
        end = start + partSize

        process = multiprocessing.Process(target=calc_combination, args=(knapsackCapacity, weights, values, start, end, combinations, queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    for i in range(len(processes)):
        result = queue.get()
        if result[0] > maxValue:
            maxValue = result[0]
            bestCombination = result[1]

    return maxValue, list(bestCombination)


if __name__ == "__main__":
    itemCount = get_items_count_user_input()

    print("What range of values you want to generate an items with?")
    print("Use ',' between two numbers!")
    vMin, vMax = get_min_max_user_input()

    print("What range of weights you want to generate an items with?")
    print("Use ',' between two numbers!")
    wMin, wMax = get_min_max_user_input()

    weights, values = generate_items(itemCount, wMin, wMax, vMin, vMax)

    knapsackCapacity = 15

    # Non parallelized
    timeStart = time.time()
    theBest = knapsack_bruteforce(weights, values, itemCount, knapsackCapacity)
    timeStop = time.time()

    timeResult = timeStop - timeStart


    print("Non parallelized")
    print(theBest)
    print("TIME " + str(timeResult))

    # parallelized
    timeStart2 = time.time()
    theBestParallel = knapsack_bruteforce_processing(weights, values, itemCount, knapsackCapacity)
    timeStop2 = time.time()

    timeResultParallelized = timeStop2 - timeStart2


    print("parallelized")
    print(theBestParallel)
    print("TIME " + str(timeResultParallelized))