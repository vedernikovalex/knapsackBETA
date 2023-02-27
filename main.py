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
    return weights,


# brute force algorithm
def greedy_algorithm(weights, values, capacity):
    """
    Greedy heuristic algorithm that will sort items by its coefficient of value:weight and choosing the best possible
    combination that can fit in knapsack and continues to the next item untill its full
    :param weights: List of weights
    :param values: List of values
    :param capacity: Capacity of knapsack
    :return: Best Value of combination and list of items numbers that are inside a combination
    """
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


def dynamic_knapsack(weights, values, itemCount, capacity):
    """
    Optimized Algorithm for solving knapsack problem by checking only items that can be fitted inside knapsack
    it calculates total value and weight and compares each combination with current best combination
    :param weights: List of weights
    :param values: List of values
    :param itemCount: Count of all items
    :param capacity: Knapsack capacity
    :return: Best Value of combination and list of items numbers that are inside a combination
    """
    if values[0] <= capacity:
        bestCombination = [(0, 0, set()), (values[0], weights[0], {1})]
    else:
        bestCombination = [(0, 0, set())]
    for i in range(0, itemCount):
        temp = bestCombination
        secondCombination = []
        for k, w, T in temp:
            if w + weights[i] <= capacity:
                secondCombination.append((k + values[i], w + weights[i], T.union({i + 1})))
        temp.extend(secondCombination)
        temp = sorted(temp, key=lambda x: (x[0], -x[1]))
        bestCombination = [temp[0]]
        for j in range(1, len(temp)):
            if temp[j][0] != temp[j-1][0]:
                bestCombination.append(temp[j])
    m, w, T = max(bestCombination, key=lambda x: x[0])
    return m, list(T)


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

    def run_greedy():
        """Executing 'greedy_algorithm' with time measuring and printing a result with time"""
        timeStartGreedy = time.time()
        theBestGreedy = greedy_algorithm(weights, values, knapsackCapacity)
        timeStopGreedy = time.time()

        timeResultGreedy = timeStopGreedy - timeStartGreedy

        print("Greedy heuristic")
        print(theBestGreedy)
        print("TIME " + str(timeResultGreedy))

    # dynamic
    def run_dynamic():
        """Executing 'dynamic_knapsack' with time measuring and printing a result with time"""
        timeStartDynamic = time.time()
        theBestDynamic = dynamic_knapsack(weights, values, itemCount, knapsackCapacity)
        timeStopDynamic = time.time()

        timeResultDynamic = timeStopDynamic - timeStartDynamic

        print("Dynamic")
        print(theBestDynamic)
        print("TIME " + str(timeResultDynamic))


    run_greedy()
    run_dynamic()
