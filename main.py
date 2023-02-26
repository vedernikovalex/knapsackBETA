import random
import itertools
import time
import multiprocessing
import configparser
from mpi4py import MPI

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


def dynamic_knapsack(weights, values, itemCount, capacity):
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


if __name__ == "__main__":
    itemCount = int(config['Test1']['itemCount'])
    vMin = int(config['Test1']['vMin'])
    vMax = int(config['Test1']['vMax'])
    wMin = int(config['Test1']['wMin'])
    wMax = int(config['Test1']['wMax'])
    knapsackCapacity = int(config['Test1']['knapsackCapacity'])

    weights, values = generate_items(itemCount, wMin, wMax, vMin, vMax)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        port = MPI.COMM_WORLD.Bind(port=65525)
        print("Bound to port", port)
    else:
        port = None

    port = comm.bcast(port, root=0)

    if port is not None:
        comm = MPI.COMM_WORLD.Connect(port)
        print("Connected to rank 0 on port", port)

    size = comm.Get_size()
    print(size)

    chunk_size = itemCount // size
    chunks = [None] * size
    chunks[rank] = (weights[rank * chunk_size:(rank + 1) * chunk_size], values[rank * chunk_size:(rank + 1) * chunk_size])
    chunk = comm.scatter(chunks, root=0)
    w_chunk, v_chunk = chunk

    def mpi_implementation_documentation():
        result = dynamic_knapsack(w_chunk, v_chunk, len(w_chunk), knapsackCapacity)

        results = comm.gather(result, root=0)
        return results

    def mpi_implementation_my():
        total = 0
        if rank == 0:
            for i in range(1, size):
                received_value = comm.recv(source=i)
                total += received_value
            return total
        elif rank == 0 and size == 1:
            total = dynamic_knapsack(w_chunk, v_chunk, len(w_chunk), knapsackCapacity)
            return total
        elif rank > 0:
            total = dynamic_knapsack(w_chunk, v_chunk, len(w_chunk), knapsackCapacity)
            comm.send(total, dest=0)

    results = mpi_implementation_documentation()

    if rank == 0:
        result = max(results, key=lambda x: x[0])[0]
        print('Max value: ', result)

    # Non parallelized
    def run_bruteforce():
        timeStart = time.time()
        theBest = knapsack_bruteforce(weights, values, itemCount, knapsackCapacity)
        timeStop = time.time()

        timeResult = timeStop - timeStart

        print("Non parallelized")
        print(theBest)
        print("TIME " + str(timeResult))

    # parallelized
    def run_bruteforce_parallelized():
        timeStart2 = time.time()
        theBestParallel = knapsack_bruteforce_processing(weights, values, itemCount, knapsackCapacity)
        timeStop2 = time.time()

        timeResultParallelized = timeStop2 - timeStart2

        print("parallelized")
        print(theBestParallel)
        print("TIME " + str(timeResultParallelized))

    # greedy
    def run_greedy():
        timeStartGreedy = time.time()
        theBestGreedy = greedy_algorithm(weights, values, knapsackCapacity)
        timeStopGreedy = time.time()

        timeResultGreedy = timeStopGreedy - timeStartGreedy

        print("Greedy heuristic")
        print(theBestGreedy)
        print("TIME " + str(timeResultGreedy))

    # dynamic
    def run_dynamic():
        timeStartDynamic = time.time()
        theBestDynamic = dynamic_knapsack(weights, values, itemCount, knapsackCapacity)
        timeStopDynamic = time.time()

        timeResultDynamic = timeStopDynamic - timeStartDynamic

        print("Dynamic")
        print(theBestDynamic)
        print("TIME " + str(timeResultDynamic))


    # run_bruteforce()
    # run_dynamic()
