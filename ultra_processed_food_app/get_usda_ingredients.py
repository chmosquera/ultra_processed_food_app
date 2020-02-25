import csv
from os import environ
from sys import stderr
from requests import get as GET
from usda import UsdaClient
import pickle
import requests

from fdc import FdcClient

DISCARD_CACHE = ".~discard_cache.pkl"
CACHE_SAVE_INTERVAL = 500

ALMOND_MILK_INGREDIENTS = "ALMONDMILK (FILTERED WATER, ALMONDS), CANE SUGAR, CONTAINS 2% OR LESS OF: VITAMIN AND MINERAL BLEND (CALCIUM CARBONATE, VITAMIN E ACETATE, ZINC GLUCONATE, VITAMIN A PALMITATE, RIBOFLAVIN [B2], VITAMIN B12, VITAMIN D2), SEA SALT, NATURAL FLAVOR, SUNFLOWER LECITHIN, LOCUST BEAN GUM, GELLAN GUM."

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += (ele + "\n")   
    
    # return string   
    return str1 


# Input: a barcode of a food product (Int)
# Output: a list of ingredients OR 'None' if no item found with given barcode
def get_usda_ingredients(barcode):
    client = FdcClient("BqVzh5bSi91ZKipGWp8e3axaH1jA8ujsYWpZp9WY")
    
    response = client.search(barcode)

    
    try:
        details = next(response)
        ingredients = ingredients_to_list(str(details['ingredients']))
        #print("==============\n " + listToString(ingredients) + "\n("==============")
        return ingredients

    except StopIteration:
        print("get_usda_ingredients -- ERROR: nothing found in database")
        return None
    

# Input: string of ingredients seperated by commas
# Output: A list of ingredient strings
def ingredients_to_list(ingredients):
    
    outlist = []

    remaining = ingredients

    while(len(remaining) > 0):

        pair = strip_next_ingredient(remaining)

        addme = pair[0]
        remaining = pair[1]

        outlist.append(addme)

    return outlist
    

# Input: string of ingredients
# Output: a tuple -> (first ingredient in input as string, string of the remaining ingredients)
def strip_next_ingredient(ingredients):

    ingredient = ""

    parens = []
    for index in range(0, len(ingredients)):
        char = ingredients[index]

        if(len(parens) == 0 and char == "," and char != "."):
            break

        if(char == "("):
            parens.append(char)
        elif(char == ")"):
            parens.pop()

        ingredient += char

    last_index = len(ingredients) - 1
    remaining = ingredients[min(last_index, len(ingredient) + 1):]
    if(len(remaining) == 1): remaining = ""

    return (ingredient, remaining)

get_usda_ingredients("025293000995")


#olive oil 00300902

#almond milk 025293000995
# ALMONDMILK (FILTERED WATER, ALMONDS), CANE SUGAR, CONTAINS 2% OR LESS OF: VITAMIN AND MINERAL BLEND (CALCIUM CARBONATE, VITAMIN E ACETATE, ZINC GLUCONATE, VITAMIN A PALMITATE, RIBOFLAVIN [B2], VITAMIN B12, VITAMIN D2), SEA SALT, NATURAL FLAVOR, SUNFLOWER LECITHIN, LOCUST BEAN GUM, GELLAN GUM.