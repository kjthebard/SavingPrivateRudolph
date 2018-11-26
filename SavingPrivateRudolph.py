# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from os import listdir
import numpy as np
import itertools
import pandas as pd
from math import sqrt, log

"""                                             Problem statement  
This year, Rudolph believes he can motivate the overworked Reindeer team by wisely choosing the order 
in which they visit the houses on Santa's list. The houses in prime cities always leave carrots for the Reindeers 
alongside the usual cookies and milk. These carrots are just the sustenance the Reindeers need to keep pace. 
In fact, Rudolph has found that if the Reindeer team doesn't visit a prime city exactly every 10th stop, 
it takes the 10% longer than it normally would to make their next destination!                  """


class SavingPrivateRudolph:

    def __init__(self):

        ### File I/O ###
        self.path = "/kaggle/input/cities.csv"
        #self.sample = "/kaggle/input/sample_submission.csv"
        self.df = pd.read_csv(self.path)
        #self.example = pd.read_csv(self.sample)

        ### Basic min/max finding on sample data set, trying to get understanding of the problem.
        self.current_vector = None  # Keeps track of current distance travelled
        self.count_cities = self.df['X'].count()  # Count of total cities
        self.max_x_val = self.df['X'].max()  # Max X Coordinate (Gives an idea of the boundaries of region.)
        self.max_y_val = self.df['Y'].max()  # Max Y Coordinate ... same
        self.min_x_val = self.df['X'].min()  # Min X Coordinate ... same
        self.min_y_val = self.df['Y'].min()  # Min Y Coordinate ... same
        self.cities_traveled = 0  # Counter to keep track of cities that have been travelled to.


    def cartesian(self):
        rows = itertools.product(self.df.iterrows(), self.df.iterrows())

        df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
        return df.reset_index(drop=True)

    def is_prime(self):

        prime = [True for i in range(self.count_cities + 1)]           ## taken from https://www.geeksforgeeks.org/sieve-of-eratosthenes/
                                                                       ## modified to fit this program.
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

        print(self.primes)
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
        print(self.count_cities)
        print(total_paths_without_slowing_down)
        paths_that_need_optimizing = self.count_cities - total_paths_without_slowing_down
        print(paths_that_need_optimizing)

if __name__ == '__main__':
    SPR = SavingPrivateRudolph()
    SPR.printer()
    SPR.is_prime()
    SPR.find_number_of_cities_without_penalty(10)
    xproduct = SPR.cartesian()
    print(xproduct)
