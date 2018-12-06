# This Python 3 environment comes with many helpful analytics libraries installed  dsI..14t is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from math import sqrt, log
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# The quick brown fox jumped over the lazy dog
# `1234567890-=
# []\
# ;'
#,./
#78461266+.+782*6............................................


"""1                                         Problem statement  
This year, Rudolph believes he can motivate the overworked Reindeer team by wisely choosing the order 
in which they visit the houses on Santa's list. The houses in prime cities always leave carrots for the Reindeers 
alongside the usual cookies and milk. These carrots are just the sustenance the Reindeers need to keep pace. 
In fact, Rudolph has found that if the Reindeer team doesn't visit a prime city exactly every 10th stop, 
it takes the 10% longer than it normally would to make their next destination!                  """


class SavingPrivateRudolph:

    def __init__(self):

        ### File I/O ###
        self.path = "/kaggle/input/cities.csv"
        self.df = pd.read_csv(self.path)

        ### Basic min/max finding on sample data set, trying to get understanding of the problem.
        self.current_vector = None  # Keeps track of current distance travelled
        self.count_cities = self.df['X'].count()  # Count of total cities
        self.max_x_val = self.df['X'].max()  # Max X Coordinate (Gives an idea of the boundaries of region.)
        self.max_y_val = self.df['Y'].max()  # Max Y Coordinate ... same
        self.min_x_val = self.df['X'].min()  # Min X Coordinate ... same
        self.min_y_val = self.df['Y'].min()  # Min Y Coordinate ... same
        self.cities_traveled = 0  # Counter to keep track of cities that have been travelled to.

    def is_prime(self):
        
        """ What is a prime by any other name?"""
        
        prime = [True for i in range(self.count_cities + 1)]             ## taken from https://www.geeksforgeeks.org/sieve-of-eratosthenes                                                           ## modified to fit this program.
        p = 2
        while (p * p <= self.count_cities):
            if (prime[p] == True):
                for i in range(p * p, self.count_cities + 1, p):
                    prime[i] = False
            p += 1

        self.primes = [] # Store all primes in a list format.
        for p in range(2, self.count_cities):
            if prime[p]:
                self.primes.append(p)
        # print(self.primes)
        self.number_of_primes =  len(self.primes)

    def apply_penalty(self, distance):

        """
        Here we apply the penalty if the reindeer haven't hit a city with a prime number after 10 stops. 
        """

        self.cities_hit += 1  # Adds to the counter of cities without visiting a prime.

        if self.cities_hit > 10:  # If Santa has visted more than 10 cities ...
            penalty_distance = (
                               distance * 0.1) + distance  # ...Applies the penalty for not showing up to a prime city...
            return penalty_distance  # ...and returns the value based on the penalty
        else:
            return distance  # Else return the distance.

    def printer(self):
        print(self.df)  # Returns the data set

    def find_number_of_cities_without_penalty(self, penalty_number):

        total_paths_without_slowing_down = self.number_of_primes * penalty_number
        # print(self.count_cities)
        # print(total_paths_without_slowing_down)
        paths_that_need_optimizing = self.count_cities - total_paths_without_slowing_down
        # print(paths_that_need_optimizing)
        return paths_that_need_optimizing
        
    def kmeans_cluster(self):
        
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        X_data = self.df['X'].values    
        Y_data = self.df['Y'].values
        
        #plt.scatter(X_data,Y_data, c='black', s=1)
        
        #plt.scatter(X_data[0], Y_data[0], s=10, c='r', marker="o")
        #show()
        
        self.prime_df = self.df.loc[self.df.index.isin(self.primes)]
        self.notprime_df = self.df.loc[~self.df.index.isin(self.primes)]
        # print(self.notprime_df)
        
        x_prime = self.prime_df['X'].values
        y_prime = self.prime_df['Y'].values
        
        #plt.scatter(x_prime, y_prime, c='black', s=1)
        
    def nearest_neighbors(self):
        
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(self.prime_df)
        self.distances, self.indices = nbrs.kneighbors(self.df)

        self.index = pd.Series(x for x in self.indices)
        self.closest = pd.Series(y for y in self.distances)
        
        data = pd.DataFrame({'closest_prime': self.index, 'distance': self.closest, 'idx_col': self.index.index})
        # print(data)
        
        #The best part about these new switches is how loud they are
        # The best part about this is the asmr feeling you get when hearing the switches. 
        # closest_one = pd.Dataframe(self.primes,self.indices,self.indicies)
        
        """
        
        Ok Algorithm time... So let me explain it to my self so I can code it up. I now need to compare the list
        of cities with closest primes with each other. how many say the 1st prime is their closest city, 2nd prime and so on. 
        
        After I gather this list I will need to bring with each shortest prime its distance and compare this to each other distance,
        the distance between this city and the next normal city will be calculated. If the distance between the cities is shorter than
        the distance to the next shortest prime AND either the reindeer are tired (cities >= 10) or the distance traveled is shorter than
        the average distance.
    
        """
        
        first_city = data['closest_prime'][0]
        second_city = data['closest_prime'][1]
        common = []
         
        for item in data['closest_prime']:
            for stuff in data['closest_prime']:
                common = set(item) & set(stuff)
        
        print(common)
        
        
        
        

if __name__ == '__main__':
    SPR = SavingPrivateRudolph()
    #SPR.printer()
    SPR.is_prime()
    SPR.find_number_of_cities_without_penalty(10)
    SPR.kmeans_cluster()
    SPR.nearest_neighbors()

    
