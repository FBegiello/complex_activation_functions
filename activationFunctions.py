#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:17:59 2019

@author: fbegiello
"""

from keras.backend import sigmoid, sin, cos, tanh
from keras.backend import maximum, minimum, square, exp
from keras.backend import switch, equal, greater, zeros_like, ones_like
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from sys import maxsize


def getCustomActivations():
    get_custom_objects().update({'xSigmoid': Activation(xSigmoid)})
    get_custom_objects().update({'maxSigmoid': Activation(maxSigmoid)})
    get_custom_objects().update({'cosDiff': Activation(cosDiff)})
    get_custom_objects().update({'minSin': Activation(minSin)})
    get_custom_objects().update({'quadTg': Activation(quadTg)})
    get_custom_objects().update({'quadTgCapped': Activation(quadTg)})
    get_custom_objects().update({'maxTanh': Activation(maxTanh)})
    get_custom_objects().update({'sincPlus': Activation(sincPlus)})
    get_custom_objects().update({'quadSinh': Activation(quadSinh)})
    get_custom_objects().update({'quadSinhCapped': Activation(quadSinh)})

def xSigmoid(x, b=1):
    """
    Activation function
    x * sigmoid(b*x)
    """
    return x * sigmoid(b*x)

def maxSigmoid(x):
    return maximum(x, sigmoid(x))

def cosDiff(x):
    """
    Activation function
    cos(x)-x
    """
    return cos(x)-x

def minSin(x):
    """
    Activation function
    min(x, sin(x))
    """
    return minimum(x, sin(x))

def quadTg(x):
    """
    Activation function
    (tan^-1 (x))^2 -x
    """
    return switch(
                    equal(sin(x)/cos(x), 0), 
                    ones_like(x)*maxsize,
                    square(1/(sin(x)/cos(x)))
                  )
    
def quadTgCapped(x, valCap=2):
    """
    Activation function
    (tan^-1 (x))^2 -x
    capped at valCap
    """
    
    y = switch( 
                equal(sin(x)/cos(x), 0), 
                ones_like(x)*maxsize,
                square(1/(sin(x)/cos(x)))
               )
    return switch(
                    greater(y, valCap), 
                    ones_like(x)*valCap,
                    y
                  )

def maxTanh(x):
    """
    Activation function
    max(x, tanh(x))
    """
    return maximum(x, tanh(x))

def sinc(x):
    return switch(
                    equal(x, 0), 
                    ones_like(x),
                    sin(x)/x
                  )

def sincPlus(x):
    """
    Activation function
    sinc(x)+x
    """
    return sinc(x)+x
    
def sinh(x):
    return (exp(x)-exp(-x))/2

def quadSinh(x):
    """
    Activation function
    x * (sinh^-1 (x))^-2
    """
    return switch(
                    equal(x, 0) , 
                    zeros_like(x),
                    x*square(1/sinh(x))
                  )

def quadSinhCapped(x, valCap=2):
    """
    Activation function
    (tan^-1 (x))^2 -x
    capped at valCap
    """
    
    y = switch( 
                equal(x, 0) , 
                zeros_like(x),
                x*square(1/sinh(x))
               )
    return switch(
                    greater(y, valCap), 
                    ones_like(x)*valCap,
                    y
                  )