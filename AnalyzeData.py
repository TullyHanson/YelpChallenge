import matplotlib.pyplot as plt
import os
import re
import time
import mltools as ml
import numpy as np

#Take a subset of businesses that meet category == restaurant
def findRestaurants(businessFile):
    restaurantList = []

    currentLine = businessFile.readline()
    while currentLine:
        currentLine = currentLine.strip("\n")
        # If this current line corresponds to a restaurant, add it to our list
        if "Restaurants" in currentLine:
            restaurantList.append(currentLine)
        currentLine = businessFile.readline()

    businessFile.close()
    return restaurantList


# Find the subset of each restaurant's data that contains the attributes
def findDirtyAttributeList(restaurantList):
    dirtyAttributeList = []
    for line in restaurantList:
        # Finds the match for the text "attributes" in the given line
        matchText = re.search(r"[^a-zA-Z](attributes)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        #Sanity check to make sure we're getting the text containing all the attributes
        #print(line[stringStartPos+14:-21].replace("\"", ""))

        # Add the list of attributes to the array
        dirtyAttributeList.append(line[stringStartPos+14:-21].replace("\"", ""))

    return dirtyAttributeList


# Function that parses the nested portion of an attribute list. Functions similarly
# to the standard parseAttributes function, but has different stop conditions to account
# for its nested nature.
#
# At the end, it returns the total number of characters that we iterated through
# in this function. This is to allow the other parseAttributes function
# to take off where we ended.
#
# *** NOTE
# This function ignores the general attributes. For example, given the nested attribute:
#       Ambience: {Romantic: true, Casual: false, Fine Dining: true}
# This function would ignore "Ambience" and instead parse it into the 3 attributes:
#       ['Romantic', 'true'], ['Casual', 'False'], ['Fine Dining', 'true']
def parseNestedAttributes(attributeList, currentLine):
    attributeFound = False
    finishedNested = False

    attributeName = ""
    attributeValue = ""
    currentAttribute = []

    if currentLine[0] == '}':
        return 2

    relativeIndex = 0
    while not finishedNested:
        if not attributeFound:
            if currentLine[relativeIndex] != ':':
                attributeName += currentLine[relativeIndex]
            else:
                attributeFound = True
                currentAttribute.append(attributeName)
                attributeName = ""
                relativeIndex += 1
        else:
            if currentLine[relativeIndex] == '}':
                finishedNested = True
                currentAttribute.append(attributeValue)
                attributeList.append(currentAttribute)
                relativeIndex += 1
            elif currentLine[relativeIndex] != ',':
                attributeValue += currentLine[relativeIndex]
            else:
                currentAttribute.append(attributeValue)
                attributeValue = ""
                attributeList.append(currentAttribute)
                currentAttribute = []
                attributeFound = False
                relativeIndex += 1
        relativeIndex += 1

    return relativeIndex


# This is where we want to parse the attribute list for the separate attributes.
# Slightly more complicated because attributes can either be:
# "<attribute>": <value>,      or    "<attribute>": {"<subattribute>": <value>, ...}
# For the latter, we instead call parseNestedAttributes above.
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


# Determines the unique Attribute Names that exist within the Yelp Dataset
# Examples include [Good For, Valet, Take Out, etc]
def determineUnique(attributeMatrix, restaurantArray):
    unique = set()
    for i, restaurant in enumerate(attributeMatrix):
        for attributes in restaurant:
            if not unique.__contains__(attributes[0]):
                unique.add(attributes[0])
    return unique


# Searches a restaurant's data to determine its current star count.
# Adds this to a list that we use as our metric to determine the success of the restaurant
def createTargetList(restaurantArray):
    targetArray = []
    for line in restaurantArray:
        # Finds the match for the text "stars": in the given line
        matchText = re.search(r"[^a-zA-Z](\"stars\":)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        # Sanity check to make sure that we are grabbing the correct part of the string
        #print(line[stringStartPos+9 : stringStartPos + 12])

        # Grabs only the rating, and stores it as a float (2.5, 4.0, 1.5, etc)
        starRating = line[stringStartPos+9 : stringStartPos + 12]
        floatRating = float(starRating)
        targetArray.append(floatRating)

    return targetArray


# Determines how many attributes each restaurant contains.
# This data is used to determine if there is a correlation
# between number of attributes and restaurant success
def determineNumAttributes(attributeMatrix):
    numberOfAttributes = []
    for restaurant in attributeMatrix:
        numberOfAttributes.append(float(len(restaurant)))
    return numberOfAttributes


# Determines how many "Good" and "Bad" restaurants there
# are for each given number of attributes.
def determineNumGoodBad(numberOfAttributes, starTargetArray):
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



# Plots the % of restaurants that are succesful with respect to number of attributes.
# Used to determine if there is a correlation between the two
def plotTesting(numberOfAttributes, starTargetArray, uniqueAttributes):

    numGoodRestaurants, numBadRestaurants = determineNumGoodBad(numberOfAttributes, starTargetArray)

    # Creates a list with values ranging from 0 to #UniqueAttributes
    # Used as the x values when plotting
    attributeX = []
    for i in range(0, len(uniqueAttributes)):
        attributeX.append(float(i))

    # Calculates the percentage of "Good" restaurants at each attribute count
    percentGood = []
    for i in range(0, len(uniqueAttributes)):
        denominator = (numGoodRestaurants[i] + numBadRestaurants[i])
        if denominator == 0:
            percentGood.append(float(0))
        else:
            percentGood.append(float(float(numGoodRestaurants[i]) / float(denominator)))

    # Take a subset of our attribute data to avoid the inaccurate 0's at high attribute count
    # Turn them into np arrays for use with our learner
    xStandard = np.array(attributeX[:52])
    xStandard = xStandard[:,np.newaxis]    # By default xStandard is a (52,) list, this forces it to be (52, 1)
    yPercentGood = np.array(percentGood[:52])

    # Creates polynomial features of degree 2, allows our learner to find a more accurate best-fit line
    xPolynomial = ml.transforms.fpoly(xStandard, 2, bias=False)
    # Rescales our data so they have similar ranges and variance
    xPolynomial,params = ml.transforms.rescale(xPolynomial)
    # Train our learner on the polynomial features
    lr = ml.linear.linearRegress(xPolynomial, yPercentGood)

    # Creates a sample of possible x values to use with our prediction
    possibleXValues = np.linspace(0, 52, 200)
    possibleXValues = possibleXValues[:,np.newaxis]     # Forces xs to be (200,1)
    # Creates polynomial x values of degree 2
    possiblePolyXValues = ml.transforms.fpoly(possibleXValues, 2, False)
    # Rescale the x values so they have similar ranges and variance
    possiblePolyXValues,_ = ml.transforms.rescale(possiblePolyXValues)

    # Using our learner, predict for all possiblePolyXValues. This calculates our best-fit line
    ysP = lr.predict(possiblePolyXValues)

    plt.suptitle("Percent \"Good\" given Number of Attributes")
    plt.xlabel("Number of Attributes")
    plt.ylabel("% \"Good\" Restaurants")
    # Scatter plot of our original data
    plt.scatter(xStandard, yPercentGood)
    # Plots our best-fit line
    plt.plot(possibleXValues, ysP, 'g')
    plt.legend(["Prediction", "Actual Data"], loc=4)
    plt.xlim([0, 55])
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
            if attribute[1] == "true":
                currentAttributeList[dict[attribute[0]]] = 1
            elif attribute[1] == "false":
                pass
            else:
                currentAttributeList[dict[attribute[0]]] = 1

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
    #Find the subset of businesses that are restaurants
    restaurantList = findRestaurants(businessFile)

    # Find the subset of each restaurant's data that contains the attributes
    dirtyAttributeList = findDirtyAttributeList(restaurantList)
    # Further parse the dirtyAttributeArray for specific attributes to create our matrix
    attributeMatrix = parseForAttributes(dirtyAttributeList)


    uniqueAttributes = determineUnique(attributeMatrix, restaurantList)
    binaryAttributeList = np.array(createBinaryAttributeList(attributeMatrix, uniqueAttributes))

    starTargetList = createTargetList(restaurantList)
    binaryTargetList = np.array(createBinaryTargetList(starTargetList))

    numberOfAttributes = determineNumAttributes(attributeMatrix)
    plotTesting(numberOfAttributes, starTargetList, uniqueAttributes)

