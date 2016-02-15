import numpy as np
import matplotlib.pyplot as plt
import os
import re

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
        #print(line[stringStartPos+13:-21])
        # Add the list of attributes to the array
        dirtyAttributeArray.append(line[stringStartPos+14:-22].replace("\"", ""))

    return dirtyAttributeArray


def parseForAttributes(dirtyAttributeArray):

    for line in dirtyAttributeArray:

        print(line)
        # This is where we want to parse the attribute list for the separate attributes.
        # Slightly more complicated because attributes can either be:
        # "<attribute>": <bool>,      or  "<attribute>": {"<subattribute>": <bool>, ...}
        # It may be easier to run through all the attributes once, only grabbing those enclosed in " ",
        # and add it to a hashset so we can find all unique attributes.
        # Then, run through again and for each line find which of those unique attributes it contains...



def createTargetArray(restaurantArray):

    targetArray = []
    for line in restaurantArray:
        # Finds the match for the text "stars" in the given line
        matchText = re.search(r"[^a-zA-Z](stars)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        # Using the above start position, increments slightly forward past the text and only grabs the star rating
        targetArray.append(line[stringStartPos+8 : stringStartPos + 11])

        # Sanity check to make sure that we are grabbing the correct part of the string
        #print(line[matchText.start(1)-1 : matchText.start(1) + 11])
    return targetArray



if __name__ == '__main__':

    #Load in business file
    businessFile = open(os.getcwd() + "\\yelp_academic_dataset_business.json", "r")
    restaurantArray = findRestaurants(businessFile)

    dirtyAttributeArray = findDirtyAttributeList(restaurantArray)
    attributeArray = parseForAttributes(dirtyAttributeArray)
    starTargetArray = createTargetArray(restaurantArray)