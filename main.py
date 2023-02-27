import random
import itertools
import time
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
def knapsack_bruteforce(weights, values, itemCount, knapsackCapacity):
    """
    Simple bruteforce algorithm that will generate all combinations of all available items and compare them one by one
    :param weights: List of weights
    :param values: List of values
    :param itemCount: Count of items
    :param knapsackCapacity: Available capacity of knapsack
    :return: Best Value of combination and list of items that are inside a combination
    """
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


    def run_bruteforce():
        """Executing 'knapsack_bruteforce' with time measuring and printing a result with time"""
        timeStart = time.time()
        theBest = knapsack_bruteforce(weights, values, itemCount, knapsackCapacity)
        timeStop = time.time()

        timeResult = timeStop - timeStart

        print("Non parallelized")
        print(theBest)
        print("TIME " + str(timeResult))

    run_bruteforce()