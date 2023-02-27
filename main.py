import random
import time
import configparser
from mpi4py import MPI

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


def distributed_algorithm(weights, values, itemCount, knapsackCapacity):
    comm = MPI.COMM_WORLD
    """ Get rank of current computer """
    rank = comm.Get_rank()
    print("rank: " + str(rank))
    #if rank == 0:
        #port = MPI.COMM_WORLD.Bind(port=65525)
        #print("Bound to port", port)
    #else:
        #port = None

    #port = comm.bcast(port, root=0)

    #if port is not None:
    #comm = MPI.COMM_WORLD.Connect(port)
    #print("Connected to rank 0 on port", port)

    size = comm.Get_size()
    print(size)

    """ Determining a size of chunk from all of combinations to spread even chunks across all processes """
    chunk_size = itemCount // size
    chunks = [None] * size
    chunks[rank] = (weights[rank * chunk_size:(rank + 1) * chunk_size], values[rank * chunk_size:(rank + 1) * chunk_size])
    chunk = comm.scatter(chunks, root=0)
    w_chunk, v_chunk = chunk

    """ getting all available computers to calculate """
    result = dynamic_knapsack(w_chunk, v_chunk, len(w_chunk), knapsackCapacity)

    """ Gather all results from available coomputers """
    results = comm.gather(result, root=0)
    return results

    results = mpi_implementation_documentation()

    """ Outputs result """
    if rank == 0:
        result = max(results, key=lambda x: x[0])[0]
        print('Max value: ', result)


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

    # dynamic
    def run_dynamic():
        """Executing 'dynamic_knapsack' with time measuring and printing a result with time"""
        timeStartDynamic = time.time()
        theBestDynamic = distributed_algorithm(weights, values, itemCount, knapsackCapacity)
        timeStopDynamic = time.time()

        timeResultDynamic = timeStopDynamic - timeStartDynamic

        print("Dynamic")
        print(theBestDynamic)
        print("TIME " + str(timeResultDynamic))


    """ Executing """
    run_dynamic()
