# YelpChallenge
Work in progress entry into the Yelp Dataset Challenge 2016.

The goal for this project is to determine whether the success of a restaurant can be predicted solely on its non-human attributes. For example, does the fact that a restaurant has late hours, serves expensive food, and has a valet imply that it will be more popular? And inversely, does the lacking of certain attributes start a restaurant off on a bad foot? 

Using logistic regression, I hope to create a learner that can predict whether a restaurant will fall into one of two categories: Succesful (4 or 5 stars) or Unsuccesful (1 or 2 stars), using only the attributes listed on its Yelp portfolio. 


The following is a graph showing the correlation between the number of attributes a restaurant contains, and it's likelihood of success:
![alt tag](https://github.com/TullyHanson/YelpChallenge/blob/master/CorrelationFigure.png)

# To Do:
- ~~Parse for individual attributes in each line~~
- ~~Using above attribute list, create a list of all unique attributes~~
- ~~Create a Restaurant by Attribute feature matrix that describes, for each restaurant, which attributes it contains~~
- ~~Create a binary attribute list for each restaurant to be used with a learner~~
- ~~Create a binary target list for each restaurant to be used with a learner~~
- Partition feature vector/target vector into training and test data
- Evaluate accuracy of learner and adjust as necessary
