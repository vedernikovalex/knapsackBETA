import random
import itertools
import time
import multiprocessing
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def generate_items(itemCount, wMin, wMax, vMin, vMax):
    # item creation
    weights = []
    values = []
    for i in range(itemCount):
        weights.append(random.randint(wMin, wMax))
        values.append(random.randint(vMin, vMax))
        print(i, weights[i], values[i])
    return weights, values


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


def greedy_algorithm(weights, values, capacity):
    valueWeightCoeficient = []
    for v, w in zip(values, weights):
        valueWeightCoeficient.append((v / w, v, w))
    valueWeightCoeficient.sort(reverse=True)
    bestCombination = []
    weight = 0
    value1 = 0
    for i, v, w in valueWeightCoeficient:
        if weight + w <= capacity:
            bestCombination.append((v, w))
            weight += w
            value1 += v
        else:
            secondCombination = [(v, w)]
            value2 = v
            if value1 > value2:
                return value1, bestCombination
            else:
                return value2, secondCombination


if __name__ == "__main__":
    itemCount = int(config['Test1']['itemCount'])
    vMin = int(config['Test1']['vMin'])
    vMax = int(config['Test1']['vMax'])
    wMin = int(config['Test1']['wMin'])
    wMax = int(config['Test1']['wMax'])
    knapsackCapacity = int(config['Test1']['knapsackCapacity'])

    weights, values = generate_items(itemCount, wMin, wMax, vMin, vMax)

    # Non parallelized
    timeStart = time.time()
    theBest = knapsack_bruteforce(weights, values, itemCount, knapsackCapacity)
    timeStop = time.time()

    timeResult = timeStop - timeStart

    print("Non parallelized")
    print(theBest)
    print("TIME " + str(timeResult))

    # parallelized
    #timeStart2 = time.time()
    #theBestParallel = knapsack_bruteforce_processing(weights, values, itemCount, knapsackCapacity)
    #timeStop2 = time.time()

    #timeResultParallelized = timeStop2 - timeStart2

    #print("parallelized")
    #print(theBestParallel)
    #print("TIME " + str(timeResultParallelized))

    timeStartGreedy = time.time()
    theBestGreedy = greedy_algorithm(weights, values, knapsackCapacity)
    timeStopGreedy = time.time()

    timeResultGreedy = timeStopGreedy - timeStartGreedy

    print("Greedy heuristic")
    print(theBestGreedy)
    print("TIME " + str(timeResultGreedy))
