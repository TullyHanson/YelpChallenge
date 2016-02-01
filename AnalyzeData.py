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


def createAttributeFeatureArray(restaurantArray):

    attributeArray = []
    for line in restaurantArray:
        # Finds the match for the text "attributes" in the given line
        matchText = re.search(r"[^a-zA-Z](attributes)[^a-zA-Z]", line)
        stringStartPos = matchText.start(1)

        #Sanity check to make sure we're getting the text containing all the attributes
        #print(line[stringStartPos-1:-21])

    return attributeArray


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

    attributeArray = createAttributeFeatureArray(restaurantArray)
    starTargetArray = createTargetArray(restaurantArray)