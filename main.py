import random
import itertools
import time

weights = []
values = []
itemCount = 20

#item creation
for i in range(itemCount):
    weights.append(random.randint(1, 4))
    values.append(random.randint(50, 200))
    print(i,weights[i],values[i])

knapsackCapacity = 20

# brute force algorithm
def knapsack_bruteforce(values, weights, capacity):
    max_value = 0

    for i in range(1, itemCount + 1):
        for combination in itertools.combinations(range(itemCount), i):
            total_weight = sum(weights[j] for j in combination)
            if total_weight <= capacity:
                total_value = sum(values[j] for j in combination)
                if total_value > max_value:
                    max_value = total_value
                    best_combination = combination

    return max_value, list(best_combination)


timeStart = time.time()
theBest = knapsack_bruteforce(values,weights,knapsackCapacity)
timeStop = time.time()

timeResult = timeStop - timeStart

print(theBest)
print("TIME "+str(timeResult))