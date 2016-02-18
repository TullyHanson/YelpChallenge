import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
from itertools import islice

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
                print(currentAttribute)
                i += 1
            elif currentLine[i] != ',':
                attributeValue += currentLine[i]
            else:
                currentAttribute.append(attributeValue)
                attributeValue = ""
                attributeList.append(currentAttribute)
                print(currentAttribute)
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

        print(line)
        print("")

        i = 0
        while i in range(0, len(line)):
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
                    print("------ NESTED -------")
                    numberAhead = parseNestedAttributes(attributeList, line[i+1:])
                    i += numberAhead + 1
                    print("------ END -------")
                    attributeFound = False
                elif line[i] == '}': #Special case: At end of line (grab Attribute 'name' and 'value')
                    currentAttribute.append(attributeValue)
                    attributeList.append(currentAttribute)
                    print(currentAttribute)
                elif line[i] != ',':
                    attributeValue += line[i]
                else:
                    currentAttribute.append(attributeValue)
                    attributeValue = ""
                    attributeList.append(currentAttribute)
                    print(currentAttribute)
                    currentAttribute = []
                    attributeFound = False
                    i += 1
            i += 1
        print("")
        attributeMatrix.append(attributeList)

    return attributeMatrix


def determineUnique(attributeArray):

    unique = set()
    for line in attributeArray:
        if not unique.__contains__(line):
            unique.add(line)

    return unique


def createTargetArray(restaurantArray):

    targetArray = []
    for line in restaurantArray:
        # Finds the match for the text "stars" in the given line
        matchText = re.search(r"[^a-zA-Z](stars)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        # Using the above start position, increments slightly forward past the text and only grabs the star rating
        targetArray.append(line[stringStartPos+8 : stringStartPos + 11])

        # Sanity check to make sure that we are grabbing the correct part of the string
        #print(line[matchText.start(1)+8 : matchText.start(1) + 11])
    return targetArray



if __name__ == '__main__':

    #Load in business file
    businessFile = open(os.getcwd() + "/yelp_academic_dataset_business.json", "r")
    restaurantArray = findRestaurants(businessFile)

    dirtyAttributeArray = findDirtyAttributeList(restaurantArray)
    attributeArray = parseForAttributes(dirtyAttributeArray)

    #uniqueAttributes = determineUnique(attributeArray)

    starTargetArray = createTargetArray(restaurantArray)