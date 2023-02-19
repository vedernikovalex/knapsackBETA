import random
import itertools
import time

weights = []
values = []

itemCountInput = ""
itemCount = 0
print("How much items do you want to generate for solving?")
while itemCountInput == "":
    itemCountInput = input(">> ")
    try:
        itemCount = int(itemCountInput)
    except ValueError:
        print("Only numbers allowed")
        itemCountInput = ""


def get_min_max_user_input():
    inputString = ""
    min = 0
    max = 0
    while inputString == "":
        inputString = input(">> ")
        try:
            numbers = inputString.split(",")
            if len(numbers) <= 2:
                num1 = int(numbers[0])
                num2 = int(numbers[1])
                if num1 > num2:
                    min = num2
                    max = num1
                else:
                    min = num1
                    max = num2
            else:
                print("More than two numbers inserted")
                inputString = ""
        except ValueError:
            print("Only numbers allowed")
            inputString = ""
    return min, max


print("What range of values you want to generate an items with?")
print("Use ',' between two numbers!")
vMin, vMax = get_min_max_user_input()

print("What range of weights you want to generate an items with?")
print("Use ',' between two numbers!")
wMin, wMax = get_min_max_user_input()

# item creation
for i in range(itemCount):
    weights.append(random.randint(wMin, wMax))
    values.append(random.randint(vMin, vMax))
    print(i, weights[i], values[i])

knapsackCapacity = 15


# brute force algorithm
def knapsack_bruteforce(values, weights, capacity):
    maxValue = 0
    best_combination = []
    for i in range(1, itemCount + 1):
        for combination in itertools.combinations(range(itemCount), i):
            totalWeight = sum(weights[j] for j in combination)
            if totalWeight <= capacity:
                totalValue = sum(values[j] for j in combination)
                if totalValue > maxValue:
                    maxValue = totalValue
                    best_combination = combination
    if not best_combination:
        return None

    return maxValue, list(best_combination)


timeStart = time.time()
theBest = knapsack_bruteforce(values, weights, knapsackCapacity)
timeStop = time.time()

timeResult = timeStop - timeStart

print(theBest)
print("TIME " + str(timeResult))
