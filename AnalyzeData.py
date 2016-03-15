import matplotlib.pyplot as plt
import os
import re
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
            percentGood.append(float(float(numGoodRestaurants[i]) / float(denominator)) * 100.0)

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

    plt.suptitle("Percent \"Good\" Restaurants given Number of Attributes")
    plt.xlabel("Number of Attributes")
    plt.ylabel("Percent \"Good\" Restaurants")
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
            #if attribute[0] == "Price Range":
            #    currentAttributeList[dict[attribute[0]]] = int(attribute[1])
            #elif attribute[1] == "true":
            #    currentAttributeList[dict[attribute[0]]] = 1
            #else:
            #    currentAttributeList[dict[attribute[0]]] = 1
            if attribute[1] == "true":
                currentAttributeList[dict[attribute[0]]] = 1
            elif attribute[1] == "false":
                currentAttributeList[dict[attribute[0]]] = 0
            else:
                currentAttributeList[dict[attribute[0]]] = computeSpecialValue(attribute)

        binaryAttributeList.append(currentAttributeList)

    return binaryAttributeList


def computeSpecialValue(attribute):

    name = attribute[0]
    val = attribute[1]

    if name == "Noise Level":
        if val == "average":
            return 1
        if val == "loud":
            return -1
        if val == "quiet":
            return 2
        if val == "very_loud":
            return -2

    elif name == "Attire":
        if val == "casual":
            return 1
        if val == "dressy":
            return 2
        if val == "formal":
            return 3

    elif name == "Alcohol":
        if val == "none":
            return -1
        if val == "full_bar":
            return 2
        if val == "beer_and_wine":
            return 1

    elif name == "Price Range":
        return int(val)

    elif name == "Wi-Fi":
        if val == "free":
            return 1
        if val == "paid":
            return -2
        if val == "no":
            return -1

    else:
        return 0


def createSpecificBinaryAttributeList(attributeMatrix):
    # Create a dictionary for index lookup while passing over attributes
    dict = {}
    specificAttributes = {"Take-out", "Noise Level", "Takes Reservations", "street", "lot", "valet", "Has TV", "Outdoor Seating",
                          "Attire", "Alcohol", "Good for Kids", "Good For Groups", "Price Range", "Happy Hour", "Wi-Fi"}

    for i, unique in enumerate(specificAttributes):
        dict[unique] = i

    binaryAttributeList = []
    for restaurant in attributeMatrix:
        currentAttributeList = [0] * 15

        for attribute in restaurant:
            if dict.__contains__(attribute[0]):
                #currentAttributeList[dict[attribute[0]]] = int(attribute[1])
                if attribute[1] == "true":
                    currentAttributeList[dict[attribute[0]]] = 1
                elif attribute[1] == "false":
                    currentAttributeList[dict[attribute[0]]] = 0
                else:
                    currentAttributeList[dict[attribute[0]]] = computeSpecialValue(attribute)

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


def gradientDescent(x, y, theta, alpha, m, numIterations):
    y = y[:,np.newaxis]

    for i in range(0, numIterations):
        hypothesis = np.dot(theta, x.T)
        loss = y.T - hypothesis

        for j, value in enumerate(loss.T):
            if value > 5.0:
                print("Greater: " + str(value))
                print(str(hypothesis.T[j]) + "    " + str(y[j]))

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss) / (m)

        # avg gradient per example
        gradient = np.dot(loss, x)
        #gradient = gradient * -2
        gradient = gradient / m

        # update
        theta = theta + alpha * gradient

    return theta


def predictLinearRegress(binaryAttributeList, starTargetList):

    starTargetList = np.array(starTargetList)
    Xtrain, Xtest, Ytrain, Ytest = ml.splitData(binaryAttributeList, starTargetList, 0.75)

    lr = ml.linear.linearRegress(Xtrain, Ytrain)

    yHatInitial = lr.predict(Xtest)

    correct = 0
    total = 0
    for i, value in enumerate(yHatInitial):
        if value >= 4.0 and Ytest[i] >= 4.0:
            correct += 1
        elif value < 4.0 and Ytest[i] < 4.0:
            correct += 1

        total += 1

    ratioCorrect = float(float(correct) / float(total))
    print("Ratio correct: " + str(ratioCorrect))



    onesCol = np.ones((len(Xtrain),1))
    Xtrain = np.concatenate((onesCol, Xtrain), 1)

    m, n = np.shape(Xtrain)
    print(str(m) + " " + str(n))

    #Use the calculated theta values to repeatedly do gradient descent
    newThetas = gradientDescent(Xtrain, Ytrain, lr.theta, 0.1, m, 2000)


    lr.theta = newThetas
    yHat = lr.predict(Xtest)


    correct = 0
    total = 0
    for i, value in enumerate(yHat):
        if value >= 4.0 and Ytest[i] >= 4.0:
            correct += 1
        elif value < 4.0 and Ytest[i] < 4.0:
            correct += 1
        total += 1

    ratioCorrect = float(float(correct) / float(total))
    print("Ratio correct: " + str(ratioCorrect))



def plotStarRatingBarGraph(starTargetList):
    numStarCount = [0] * 9
    starRating = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for rating in starTargetList:
        numStarCount[int(2*rating) - 2] += 1

    width = 1 / 3.0
    plt.suptitle("Distribution of Star Ratings")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Restaurants")
    plt.bar(starRating, numStarCount, width, color="blue")
    plt.show()



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
    #binaryAttributeList = np.array(createSpecificBinaryAttributeList(attributeMatrix))


    starTargetList = createTargetList(restaurantList)
    binaryTargetList = np.array(createBinaryTargetList(starTargetList))

    plotStarRatingBarGraph(starTargetList)

    #numberOfAttributes = determineNumAttributes(attributeMatrix)
    #plotTesting(numberOfAttributes, starTargetList, uniqueAttributes)

    #predictLinearRegress(binaryAttributeList, starTargetList)

    """
    Bayes
    -----

    Xtrain, Xtest, Ytrain, Ytest = ml.splitData(binaryAttributeList, starTargetList)

    bayes = ml.bayes.gaussClassify()
    bayes.train(Xtrain, Ytrain, True)
    YtestHat = bayes.predictSoft(Xtest)

    print(YtestHat.shape)
    """



    """
    KNN
    ----


    K = [1]#, 20, 50, 100, 500, 1000, 1500, 2000]

    for i in range(0, 1):
        knn = ml.knn.knnClassify()
        knn.train(Xtrain, Ytrain, K[i])
        YtestHat = knn.predict(Xtrain)

        total = 0
        numCorrect = 0
        for i, value in enumerate(Ytest):
            #print(str(Ytest[i]) + " == " + str(YtestHat[i]))
            if abs(Ytest[i] - YtestHat[i]) <= 0.5:
                numCorrect += 1
            total += 1

        print("   -->   Total correct: " + str(float(numCorrect / total)))
"""