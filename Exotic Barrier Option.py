# -*- coding: UTF-8 -*

################################################
# Exotic Barrier Option (up-and-out) Pricer
# By: Haiyang Zhang
################################################
# -----------------------------------------------
################################################
# Monte Carlo Simulation pricing an up-and-out #
# call option. This call option is a barrier   #
# option in which pyoffs are zero if the       #
# asset crosses some predifned barrier at some #
# time in [0,T]. If the barrier is not crossed,#
# the payoff becomes that of a European call.  #
# Note: Monte Carlo tends to overestimate the  #
# price of an option. Same for Barrier Options.#
################################################
# -----------------------------------------------

# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from random import gauss
from math import exp, sqrt


# -----------------------------------------------
# Class for all parameters
# -----------------------------------------------

class parameters():
    """parameters to be used for program"""

    # --------------------------
    def __init__(self, vol):
        """initializa parameters"""

        self.S = 100  # underlying asset price
        self.v = vol  # volatility
        self.r = 0.032  # risk free rate
        self.T = 0.5  # years until maturity
        self.K = 100.0  # strike price
        self.B = 130.0  # high barrier price
        self.B2 = 70  # low barrier price

        self.delta_t = 0.004  # timestep
        self.N = self.T / self.delta_t  # Number of discrete time points
        self.simulations = 1000  # num of simulations
        self.numberofexecutions = 0  # number of execution

        self.european_length = 126
        self.american_point = self.european_length
        self.downward_point = 1000
        self.downward_probability = 0.6

    # ---------------------------
    def print_parameters(self):
        """print parameters"""

        print "---------------------------------------------"
        print "---------------------------------------------"
        print "Pricing a Barrier option"
        print "---------------------------------------------"
        print "Parameters of Barrier Option Pricer:"
        print "---------------------------------------------"
        print "Underlying Asset Price = ", self.S
        print "Volatility = ", self.v
        print "Risk-Free Rate =", self.r
        print "Years Until Expiration = ", self.T
        print "Time-Step = ", self.delta_t
        print "Discrete time points =", self.N
        print "Number of Simulations = ", self.simulations
        print "---------------------------------------------"
        print "---------------------------------------------"


# -----------------------------------------------
# Class for Monte Carlo
# -----------------------------------------------

class up_and_out_mc(parameters):
    """This is the class to price the barrier option
       defined as an up and in option. Paramaters are
       fed in as a subclass"""

    # ---------------------------
    def __init__(self, vol):
        """initialize the class including the subclass"""
        parameters.__init__(self, vol)
        self.payoffs = []
        self.price_trajectories = []
        self.discount_factor = exp(-self.r * self.T)

    # ---------------------------
    def call_payoff(self, s):
        """use to price a call"""
        self.cp = 1.0 * max(s - self.K, 0.0)
        return self.cp

    # ---------------------------
    def calculate_payoff_vector(self):  # choose knock in or knock out option
        """Main iterative loop to run the
        Monte Carlo simulation. Returns a vector
        of different potential payoffs"""

        for i in xrange(0, self.simulations):
            self.stock_path = []
            self.S_j = self.S

            up_count = 0
            low_count = 0

            up_recorder = -1
            low_recorder = -1
            for j in xrange(0, int(self.N - 1)):
                self.xi = gauss(0, 1.0)

                self.S_j *= (
                    exp((self.r - .5 * self.v * self.v) * self.delta_t + self.v * sqrt(self.delta_t) * self.xi))

                self.stock_path.append(self.S_j)

                # 欧式到期后超过转股价即行权
                if j == self.european_length and self.S_j > self.K:
                    break

                # if j > self.european_length:
                #     if self.S_j > self.B:
                #         up_count += 1
                #         if up_recorder == -1:
                #             up_recorder = j
                #         elif j - up_recorder >= 30 and up_count >= 15:
                #             break

                    # elif self.S_j < self.B2:
                    #     low_count += 1
                    #     if low_recorder == -1:
                    #         low_recorder = j
                    #     elif j - low_recorder >= 30 and low_count >= 15:
                    #         break
            self.payoffs.append(self.call_payoff(self.S_j))
            self.stock_path.extend([self.S_j] * int(self.N - len(self.stock_path) - 1))
            self.price_trajectories.append(self.stock_path)
            # if max(self.stock_path) < self.B and min(self.stock_path) > self.B2:
            #     self.payoffs.append(self.call_payoff(self.stock_path[-1]))
            #     self.numberofexecutions = self.numberofexecutions + 1
            # elif max(self.stock_path) > self.B:
            #     self.payoffs.append(0)
            # elif min(self.stock_path) < self.B2:  # knock out option
            #     self.payoffs.append(0)

        return self.payoffs

    # ---------------------------
    def compute_price(self):
        """Uses payoff vector and discount
        factor to compute the price of the
        option. Numpy used for efficiency"""
        self.np_payoffs = np.array(self.payoffs, dtype=float)
        self.np_Vi = self.discount_factor * self.np_payoffs
        self.price = np.average(self.np_Vi)

    # ---------------------------
    def print_price(self):
        """prints the option price to terminal"""
        print str("Call Price: %.4f") % self.price
        print "---------------------------------------------"

    # ---------------------------
    def calc_statistics(self):
        """uses payoffs and price to calc
        variance, standard deviation, and
        a 95% confidence interval for price"""
        self.variance = np.var(self.np_Vi)
        self.sd = np.std(self.np_Vi)
        # 95% C.I. uses 1.96 z-value
        self.CI = [self.price - (1.96 * self.sd / sqrt(float(self.simulations))),
                   self.price + (1.96 * self.sd / sqrt(float(self.simulations)))]

    # ---------------------------
    def print_statistics(self):
        """prints option statistics that were
        calculated to the terminal"""
        print "Percent of Execution:", 100 * self.numberofexecutions / float(self.simulations)
        print "Variance: %.4f" % self.variance
        print "Standard Deviation: %.4f" % self.sd
        print "95% Confidence Interval:", self.CI
        print "---------------------------------------------"
        print "---------------------------------------------\n"

    # ---------------------------
    def plot_trajectories(self):
        print "Creating Plot..."
        # use numpy to plot
        self.np_price_trajectories = np.array(self.price_trajectories, dtype=float)
        self.times = np.linspace(0, self.T, self.N - 1)

        # style/plot/barrier line
        style.use('dark_background')

        self.fig = plt.figure()
        self.ax1 = plt.subplot2grid((1, 1), (0, 0))
        for sublist in self.np_price_trajectories:
            if max(sublist) > self.B or min(sublist) < self.B2:
                self.ax1.plot(self.times, sublist, color='cyan')
            else:
                self.ax1.plot(self.times, sublist, color='#e2fb86')
        plt.axhline(y=self.B, xmin=0, xmax=1, linewidth=2, color='red', label='Barrier')
        # rotate and add grid
        for label in self.ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        self.ax1.grid(True)

        # plotting stuff
        plt.xticks(np.arange(0, self.T + self.delta_t, .1))
        plt.suptitle('Stock Price Trajectory', fontsize=20)
        plt.legend()
        self.leg = plt.legend(loc=2)
        self.leg.get_frame().set_alpha(0.4)
        plt.xlabel('Time (in years)', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.show()


# -----------------------------------------------
# Main Program
# -----------------------------------------------

# Initialize and print parameters
vol = 0.3
prm = parameters(vol)
prm.print_parameters()

# Price/print the option
uo_mc = up_and_out_mc(vol)
uo_mc.calculate_payoff_vector()
uo_mc.compute_price()
uo_mc.print_price()

# caclulate/print stats
uo_mc.calc_statistics()
uo_mc.print_statistics()

# plot trajectories
uo_mc.plot_trajectories()
