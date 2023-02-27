import random
import itertools
import time
import multiprocessing
import configparser

""" Loading configuration file as 'config' """
config = configparser.ConfigParser()
config.read('config.ini')


def generate_items(itemCount, wMin, wMax, vMin, vMax):
    """
    Generating knapsack items as two lists of 'weights', 'values' based on inputs
    :param itemCount: Count of items that will be generated
    :param wMin: Minimal value of range of weights
    :param wMax: Maximal value of range of weights
    :param vMin: Minimal value of range of values
    :param vMax:Maximal value of range of values
    :return: Tuple of two lists as weights and values
    """
    # item creation
    weights = []
    values = []
    for i in range(itemCount):
        weights.append(random.randint(wMin, wMax))
        values.append(random.randint(vMin, vMax))
        print(i, weights[i], values[i])
    return weights, values

# brute force algorithm
def calc_combination(knapsackCapacity, weights, values, starIndex, endIndex, combinations, queue):
    """
    algorithm for parallelized bruteforce that will compare combinations one by one and put it's chunnk into queue
    :param knapsackCapacity: Capacity of knapsack
    :param weights: List of weights
    :param values: List of values
    :param starIndex: index to start from
    :param endIndex: index to end
    :param combinations: list of combinations
    :param queue: queue to put the process result
    """
    maxValue = 0
    bestCombination = []
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
    """
    Parallelized bruteforce algorithm that will generate all combinations of all available items and compare them one by
    one in different processes using 'calc_combination' function
    :param weights: List of weights
    :param values: List of values
    :param itemCount: Count of items
    :param knapsackCapacity: Available capacity of knapsack
    :return: Best Value of combination and list of items numbers that are inside a combination
    """
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
        start = i * partSize
        end = start + partSize

        process = multiprocessing.Process(target=calc_combination, args=(knapsackCapacity, weights, values, start, end, combinations, queue))
        process.start()
        processes.append(process)
        print(len(processes))

    for process in processes:
        process.join()

    for i in range(len(processes)):
        result = queue.get()
        if result[0] > maxValue:
            maxValue = result[0]
            bestCombination = result[1]

    return maxValue, list(bestCombination)


""" Main of application """
if __name__ == "__main__":
    """ Getting values from config files """
    itemCount = int(config['Test1']['itemCount'])
    vMin = int(config['Test1']['vMin'])
    vMax = int(config['Test1']['vMax'])
    wMin = int(config['Test1']['wMin'])
    wMax = int(config['Test1']['wMax'])
    knapsackCapacity = int(config['Test1']['knapsackCapacity'])

    """ Returning generated values """
    weights, values = generate_items(itemCount, wMin, wMax, vMin, vMax)


    def run_bruteforce_parallelized():
        """Executing 'knapsack_bruteforce_processing' with time measuring and printing a result with time"""
        timeStart2 = time.time()
        theBestParallel = knapsack_bruteforce_processing(weights, values, itemCount, knapsackCapacity)
        timeStop2 = time.time()

        timeResultParallelized = timeStop2 - timeStart2

        print("parallelized")
        print(theBestParallel)
        print("TIME " + str(timeResultParallelized))


    run_bruteforce_parallelized()