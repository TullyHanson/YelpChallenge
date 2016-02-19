import matplotlib.pyplot as plt
import os
import re
import time
import mltools as ml
import numpy as np

#Take a subset of businesses that meet category == restaurant
def findRestaurants(businessFile):
    restaurantArray = []
    #Read in the first line
    currentLine = businessFile.readline()
    while currentLine:
        currentLine = currentLine.strip("\n")

        if "Restaurants" in currentLine:
            restaurantArray.append(currentLine)

        currentLine = businessFile.readline()
    businessFile.close()
    return restaurantArray


def findDirtyAttributeList(restaurantArray):
    dirtyAttributeArray = []
    for line in restaurantArray:
        # Finds the match for the text "attributes" in the given line
        matchText = re.search(r"[^a-zA-Z](attributes)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        #Sanity check to make sure we're getting the text containing all the attributes
        #print(line[stringStartPos+14:-21].replace("\"", ""))
        # Add the list of attributes to the array
        dirtyAttributeArray.append(line[stringStartPos+14:-21].replace("\"", ""))

    return dirtyAttributeArray


def parseNestedAttributes(attributeList, currentLine):
    attributeFound = False
    finishedNested = False

    attributeName = ""
    attributeValue = ""
    currentAttribute = []

    i = 0
    if currentLine[i] == '}':
        return 2

    while not finishedNested:
        if not attributeFound:
            if currentLine[i] != ':':
                attributeName += currentLine[i]
            else:
                attributeFound = True
                currentAttribute.append(attributeName)
                attributeName = ""
                i += 1
        else:
            if currentLine[i] == '}':
                finishedNested = True
                currentAttribute.append(attributeValue)
                attributeList.append(currentAttribute)
                i += 1
            elif currentLine[i] != ',':
                attributeValue += currentLine[i]
            else:
                currentAttribute.append(attributeValue)
                attributeValue = ""
                attributeList.append(currentAttribute)
                currentAttribute = []
                attributeFound = False
                i += 1
        i += 1

    return i


# This is where we want to parse the attribute list for the separate attributes.
# Slightly more complicated because attributes can either be:
# "<attribute>": <bool>,      or  "<attribute>": {"<subattribute>": <bool>, ...}
# It may be easier to run through all the attributes once, only grabbing those enclosed in " ",
# and add it to a hashset so we can find all unique attributes.
# Then, run through again and for each line find which of those unique attributes it contains...
def parseForAttributes(dirtyAttributeArray):
    attributeMatrix = []

    for line in dirtyAttributeArray:
        attributeName = ""
        attributeValue = ""

        attributeFound = False

        attributeList = []
        currentAttribute = []

        i = 0
        while i in range(0, len(line)):
            if attributeName == "}, Outdoor Seating":
                print(i)
            if not attributeFound:
                if line[i] != ':':
                    attributeName += line[i]
                else:
                    attributeFound = True
                    currentAttribute.append(attributeName)
                    attributeName = ""
                    i += 1
            else:
                if line[i] == '{':
                    currentAttribute = []
                    numberAhead = parseNestedAttributes(attributeList, line[i+1:])
                    i += numberAhead + 1
                    attributeFound = False
                    attributeName = ""

                #Special case: At end of line (grab Attribute 'name' and 'value')
                elif line[i] == '}':
                    currentAttribute.append(attributeValue)
                    attributeList.append(currentAttribute)
                    i += 1
                elif line[i] != ',':
                    attributeValue += line[i]
                else:
                    currentAttribute.append(attributeValue)
                    attributeList.append(currentAttribute)
                    currentAttribute = []
                    attributeValue = ""
                    attributeFound = False
                    i += 1
            i += 1
        attributeMatrix.append(attributeList)

    return attributeMatrix


def determineUnique(attributeMatrix, restaurantArray):
    unique = set()
    for i, restaurant in enumerate(attributeMatrix):
        for attributes in restaurant:
            if not unique.__contains__(attributes[0]):
                unique.add(attributes[0])

    return unique


def createTargetList(restaurantArray):
    targetArray = []
    for line in restaurantArray:
        # Finds the match for the text "stars" in the given line
        matchText = re.search(r"[^a-zA-Z](\"stars\":)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        starRating = line[stringStartPos+9 : stringStartPos + 12]
        floatRating = float(starRating)

        # Using the above start position, increments slightly forward past the text and only grabs the star rating
        targetArray.append(floatRating)

        # Sanity check to make sure that we are grabbing the correct part of the string
        #print(line[matchText.start(1)+8 : matchText.start(1) + 11])
    return targetArray

def determineNumAttributes(attributeMatrix):
    numberOfAttributes = []

    for restaurant in attributeMatrix:
        numberOfAttributes.append(float(len(restaurant)))

    return numberOfAttributes

def determineDecencyAttributes(numberOfAttributes, starTargetArray):

    attributeNumGood = []
    attributeNumBad = []

    for i in range(0, 63):
        good = 0
        bad = 0
        for index, currentNum in enumerate(numberOfAttributes):
            if currentNum == i:
                if starTargetArray[index] >= 4.0:
                    good += 1
                elif starTargetArray[index] < 3.0:
                    bad += 1
        attributeNumGood.append(good)
        attributeNumBad.append(bad)

    return attributeNumGood, attributeNumBad



def plotTesting(numberOfAttributes, starTargetArray):
    attributeNumGood, attributeNumBad = determineDecencyAttributes(numberOfAttributes, starTargetArray)

    attributeX = []
    for i in range(0, 63):
        attributeX.append(i)

    percentGood = []
    for i in range(0, 63):
        denominator = attributeNumGood[i] + attributeNumBad[i]
        if denominator == 0:
            percentGood.append(0)
        else:
            percentGood.append(attributeNumGood[i] / denominator)

    plt.scatter(attributeX, percentGood)
    plt.show()

def createBinaryAttributeList(attributeMatrix, uniqueAttributes):

    # Create a dictionary for index lookup while passing over attributes
    dict = {}
    for i, unique in enumerate(uniqueAttributes):
        dict[unique] = i

    binaryAttributeList = []
    for restaurant in attributeMatrix:
        currentAttributeList = [0] * 63

        for attribute in restaurant:
            if(attribute[1] == "true"):
                currentAttributeList[dict[attribute[0]]] = 1
            elif(attribute[1] == "false"):
                pass
            else:
                currentAttributeList[dict[attribute[0]]] = 1
                #print(attribute[1])

        binaryAttributeList.append(currentAttributeList)
    return binaryAttributeList

def createBinaryTargetList(starTargetList):

    binaryTargetList = []
    for thisStar in starTargetList:
        if(thisStar >= 4.0):
            binaryTargetList.append(1)
        else:
            binaryTargetList.append(0)

    return binaryTargetList


if __name__ == '__main__':

    #Load in business file
    businessFile = open(os.getcwd() + "/yelp_academic_dataset_business.json", "r")
    restaurantArray = findRestaurants(businessFile)

    dirtyAttributeArray = findDirtyAttributeList(restaurantArray)
    attributeMatrix = parseForAttributes(dirtyAttributeArray)

    uniqueAttributes = determineUnique(attributeMatrix, restaurantArray)

    starTargetList = createTargetList(restaurantArray)

    binaryAttributeList = createBinaryAttributeList(attributeMatrix, uniqueAttributes)
    binaryTargetList = createBinaryTargetList(starTargetList)

    binaryAttributeList = np.array(binaryAttributeList)
    binaryTargetList = np.array(binaryTargetList)

    #Where we begin to learn on the data
    Xtrain,Xtest,Ytrain,Ytest = ml.splitData(binaryAttributeList,binaryTargetList, 0.75)

    errTrain = [0, 0, 0, 0, 0, 0, 0]
    errTest = [0, 0, 0, 0, 0, 0, 0]
    learner = ml.knn.knnClassify()

    K = [1, 2, 5, 10, 50, 100, 200]
    for i,k in enumerate(K):
        learner.train(Xtrain, Ytrain, k)

        Yhat = learner.predict(Xtrain)
        for index in range(0, len(Yhat)):
            if Yhat[index] != Ytrain[index]:
                errTrain[i] += 1

        Yhattest = learner.predict(Xtest)
        for index in range(0, len(Yhattest)):
            if Yhattest[index] != Ytest[index]:
                errTest[i] += 1

    for i in range(0, 7):
        errTrain[i] = errTrain[i] / len(Yhat)
        errTest[i] = errTest[i] / len(Yhattest)

    print(errTrain)
    print(errTest)

    plt.semilogx(K, errTrain, color = 'red')
    plt.semilogx(K, errTest, color = 'green')
    plt.show()
    #numberOfAttributes = determineNumAttributes(attributeMatrix)
    #plotTesting(numberOfAttributes, starTargetArray)

