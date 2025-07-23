# Matching code for the Malevolent Big Bang
# Last updated: July 23, 2025 by JRV

# import packages

import numpy as np
from operator import itemgetter
import pandas as pd
import random
import sys
import time
import os
from random import shuffle
import math
import argparse

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd) # printEnd
    # Print New Line on Complete
    if iteration == total: 
        print()
        
if __name__ == '__main__':
    
    # pull our arguments
    parser = argparse.ArgumentParser(description = "Matching script for the Malevolent Big Bang")
    parser.add_argument("-x", "--Excel", help = "The name of the Excel sheet containing your claims", required = True, default = "")
    parser.add_argument("-f", "--Fics", help = "The total number of fics", required = True, default = "")
    parser.add_argument("-i", "--Iterations", help = "Number of iterations/runs", required = True, default = "")
    parser.add_argument("-n", "--NullFics", help = "Fics to exclude from the matching process, with commas separating them (eg. 1,2,3)", required = False, default = "0")
    parser.add_argument("-p", "--Preference", help = "The sort preference toggle; 1 sorts by preference, 0 does not", required = False, default = "1")
    parser.add_argument("-s", "--Shuffle", help = "The shuffle condition; 0 does not shuffle, 1 shuffles all fics for each iteration, and 2 shuffles fics within their order categories for each iteration", required = False, default = "1")
    parser.add_argument("-d", "--DistributionWeight", help = "How much you would like to weight the distribution parameter of the performance index", required = False, default = "1")
    parser.add_argument("-r", "--RankWeight", help = "How much you would like to weight the rank parameter of the performance index", required = False, default = "1")
    parser.add_argument("-u", "--UndermatchWeight", help = "How much you would like to weight the undermatch parameter of the performance index", required = False, default = "0")
    parser.add_argument("-o", "--OvermatchWeight", help = "How much you would like to weight the overmatch parameter of the performance index", required = False, default = "0")

    argument = parser.parse_args()

    # set values
    totalFics = int(argument.Fics) # total number of fics
    filename = argument.Excel
    numIterations = int(argument.Iterations) # number of runs (5000 recommended)
    
    nullFics = [int(x) for x in np.array(argument.NullFics.split(','))]

    sortPreference = int(argument.Preference)
    ### sortPreference = 1 gives better overall performance index (ie higher rankings)
    ### sortPreference = 0 gives lower performance index but is more likely to converge for tricky cases 

    shuffleFics = int(argument.Shuffle)
    ### 0 will not shuffle the fics at all
    ### 1 will shuffle all the fics for each iteration
    ### 2 will shuffle fics within their order categories (ie shuffle all fics 
    ### picked once, all fics picked twice, etc.)

    ### performance index weights
    ### if you're not getting the performance you like, feel free to tweak these

    distributionWeight = int(argument.DistributionWeight)
    rankWeight = int(argument.RankWeight)
    undermatchWeight = int(argument.UndermatchWeight)
    overmatchWeight = int(argument.OvermatchWeight)

    ### the last two do similar things, but the overmatchWeight is more effective at getting rid of
    ### two artists matched on one fic when there are unmatched fics still around

    # load excel data

    print('Reading excel file...')
    claimsData = pd.read_excel(filename)

    FIELDS_OF_INTEREST = [
                          "Number",            # artist number
                          "Preferred Name",    # artist name
                          "1st Choice Fic",  # fic ranked 1
                          "2nd Choice Fic", # fic ranked 2
                          "3rd Choice Fic",  # fic ranked 3
                          "4th Choice Fic", # fic ranked 4
                          "5th Choice Fic"]  # fic ranked 5

    # pull relevant data from excel file

    artistID = claimsData.loc[:,FIELDS_OF_INTEREST[0]].tolist()
    artistName = claimsData.loc[:,FIELDS_OF_INTEREST[1]].tolist()

    firstChoices = claimsData.loc[:,FIELDS_OF_INTEREST[2]].tolist()
    firstChoices = [int(i.split('-')[0]) for i in firstChoices] 
    secondChoices = claimsData.loc[:,FIELDS_OF_INTEREST[3]].tolist()
    secondChoices = [int(i.split('-')[0]) for i in secondChoices] 
    thirdChoices = claimsData.loc[:,FIELDS_OF_INTEREST[4]].tolist()
    thirdChoices = [int(i.split('-')[0]) for i in thirdChoices] 
    fourthChoices = claimsData.loc[:,FIELDS_OF_INTEREST[5]].tolist()
    fourthChoices = [int(i.split('-')[0]) for i in fourthChoices] 
    fifthChoices = claimsData.loc[:,FIELDS_OF_INTEREST[6]].tolist()
    fifthChoices = [int(i.split('-')[0]) for i in fifthChoices] 

    # define list of viable fics
    originalFics = np.arange(1,totalFics+1)
    originalFics = np.delete(originalFics,np.searchsorted(originalFics,nullFics)) # remove null fics
    finalMatchesFicIDs = np.arange(1,totalFics+1)

    # sort fics by the number of people who preferenced them
    # also remove fics that have not been preferenced at all

    totalRankedFics = firstChoices + secondChoices + thirdChoices + fourthChoices + fifthChoices
    totalRankedFics.sort()

    unrankedMask = np.zeros(totalFics)

    numRanks = [];

    originalFicsCopy = originalFics
    for i in range(len(originalFicsCopy)):
        ficI = originalFicsCopy[i]
        temp = sum(totalRankedFics == ficI)
        if temp == 0: # if nobody ranked a certain fic
            unrankedMask[ficI-1] = 1
            originalFics = np.delete(originalFics,np.searchsorted(originalFics,ficI)) # remove unranked fic
        else:
            numRanks.append(temp)

    sorti = sorted(range(len(numRanks)),key=lambda k: numRanks[k])
    sorta = [numRanks[i] for i in sorti]
    originalFicsSorted = originalFics[sorti]

    # create dictionary with fic numbers as keys
    # and amount of people ranking those fics as values
    unique,counts = np.unique(totalRankedFics,return_counts = True)
    countFics = dict(zip(unique,counts))

    originalFicsSortedCopy = originalFicsSorted.copy()
    ranksA = np.unique(sorta) # find all the number of times someone ranked a fic
    for rankingI in ranksA:
        tempMask = [i for i, x in enumerate(sorta) if x == rankingI]
        tempFics = [originalFicsSortedCopy[i] for i in tempMask]

        x = list(enumerate(tempFics))
        shuffle(x)
        randI,tempFicsShuffled = zip(*x) # shuffle fics and extract indices

        originalFicsSortedCopy[tempMask] = tempFicsShuffled

    # assign artists to fics - main matching code
    # there is an option to sort/match by rank and to not sort/match by rank
    # code will also calculate a performance index along the way to evaluate performance at the end

    printProgressBar(0, numIterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for step in range(numIterations):
        
        # create blank matrixes and copies of our data
        currentArtistIDs = np.empty((totalFics,4,))
        currentArtistIDs[:] = np.nan
        currentRanks = np.empty((totalFics,4,))
        currentRanks[:] = np.nan

        artistIDCopy = artistID.copy()
        firstChoicesCopy = firstChoices.copy()
        secondChoicesCopy = secondChoices.copy()
        thirdChoicesCopy = thirdChoices.copy()
        fourthChoicesCopy = fourthChoices.copy()
        fifthChoicesCopy = fifthChoices.copy()

        rank1Copy = [1]*(len(firstChoicesCopy))
        rank2Copy = [2]*(len(secondChoicesCopy))
        rank3Copy = [3]*(len(thirdChoicesCopy))
        rank4Copy = [4]*(len(fourthChoicesCopy))
        rank5Copy = [5]*(len(fifthChoicesCopy))

        # establish some beginning parameters
        j = 0
        rank = 0
        rankNumbers = []
        distribution = 0
        popFics = []

        if shuffleFics == 0:
            originalFicsSortedCopy = originalFicsSorted.copy()

        if shuffleFics == 1: # do a random sorting of the fics each time, but keep fics with only one artist first
            originalFicsSortedCopy = originalFicsSorted.copy()
            ranksA = np.unique(sorta) # find all the number of times someone ranked a fic
            if ranksA[0] == 1: # if we have fic(s) where only one artist picked them
                # find the fics ranked more than once
                tempMask = [i for i, x in enumerate(sorta) if x != 1]
                tempFics = [originalFicsSortedCopy[i] for i in tempMask]

                x = list(enumerate(tempFics))
                shuffle(x)
                randI,tempFicsShuffled = zip(*x) # shuffle fics ranked more than once and extract indices
                originalFicsSortedCopy[tempMask] = tempFicsShuffled # replace fics picked more than once with randomized fics
            else: # if all fics picked more than once
                shuffle(originalFicsSortedCopy)

        if shuffleFics == 2: # if shuffling fics within their order categories
            originalFicsSortedCopy = originalFicsSorted.copy()
            ranksA = np.unique(sorta) # find all the number of times someone ranked a fic
            for rankingI in ranksA:
                tempMask = [i for i, x in enumerate(sorta) if x == rankingI]
                tempFics = [originalFicsSortedCopy[i] for i in tempMask]

                x = list(enumerate(tempFics))
                shuffle(x)
                randI,tempFicsShuffled = zip(*x) # shuffle fics and extract indices

                originalFicsSortedCopy[tempMask] = tempFicsShuffled # replace fics for this ranking with randomized fics

        # walk through the artists and create matches
        while len(artistIDCopy) > 0: # while we have artists left to match
            artistIDComparison = artistIDCopy.copy()
            numFicsRemaining = len(originalFicsSortedCopy)
            for i in range(numFicsRemaining):
                automatchIndex = 0 # assume no automatching (0)
                ficI = originalFicsSortedCopy[i] # this is the fic NUMBER
                ficIdx = originalFicsSortedCopy[i]-1 # this is the fic INDEX

                # if any fic has one total request, 
                # automatically match artist with that fic
                if countFics[ficI] <= 1:
                    tempFics = firstChoicesCopy+secondChoicesCopy+thirdChoicesCopy+fourthChoicesCopy+fifthChoicesCopy
                    tempArtists = artistIDCopy+artistIDCopy+artistIDCopy+artistIDCopy+artistIDCopy
                    tempRanks = rank1Copy+rank2Copy+rank3Copy+rank4Copy+rank5Copy
                    findFics = [i for i, x in enumerate(tempFics) if x == ficI]
                    tempArtists = [tempArtists[i] for i in findFics]
                    tempRanks = [tempRanks[i] for i in findFics]
                    for k in range(len(tempArtists)):
                        currentArtistIDs[ficIdx][k] = tempArtists[k]
                        rank = rank + tempRanks[k]
                        rankNumbers.append(tempRanks[k])
                        currentRanks[ficIdx][k] = tempRanks[k];

                        # remove artist(s) and choices from pool
                        popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[k]]
                        artistIDCopy = [artistIDCopy[i] for i in popMask]
                        firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                        secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                        thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                        fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                        fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                        rank1Copy = [rank1Copy[i] for i in popMask]
                        rank2Copy = [rank2Copy[i] for i in popMask]
                        rank3Copy = [rank3Copy[i] for i in popMask]
                        rank4Copy = [rank4Copy[i] for i in popMask]
                        rank5Copy = [rank5Copy[i] for i in popMask]

                    popFics.append(ficI) # we'll remove this fic from the pool later
                    automatchIndex = 1 # we have automatched, so set this index to 1

                if automatchIndex != 1: # if we did not already automatically match the fic
                    if sortPreference == 1: # preferentially match artists by ranking
                        if ficI in firstChoicesCopy: # if there are artists who ranked this fic first
                            findFics = [i for i, x in enumerate(firstChoicesCopy) if x == ficI]
                            tempArtists = [artistIDCopy[i] for i in findFics] # find artists who ranked this fic first
                            shuffle(tempArtists) # shuffle the artists
                            if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                                currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                                rank = rank + 1
                                rankNumbers.append(1)
                                currentRanks[ficIdx][j] = 1

                                # remove artist and choices from pools
                                popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                                artistIDCopy = [artistIDCopy[i] for i in popMask]
                                firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                                secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                                thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                                fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                                fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                        elif ficI in secondChoicesCopy:
                            findFics = [i for i, x in enumerate(secondChoicesCopy) if x == ficI]
                            tempArtists = [artistIDCopy[i] for i in findFics] # find artists who ranked this fic second
                            shuffle(tempArtists) # shuffle the artists
                            if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                                currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                                rank = rank + 2
                                rankNumbers.append(2)
                                currentRanks[ficIdx][j] = 2

                                # remove artist and choices from pools
                                popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                                artistIDCopy = [artistIDCopy[i] for i in popMask]
                                firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                                secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                                thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                                fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                                fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                        elif ficI in thirdChoicesCopy:
                            findFics = [i for i, x in enumerate(thirdChoicesCopy) if x == ficI]
                            tempArtists = [artistIDCopy[i] for i in findFics] # find artists who ranked this fic third
                            shuffle(tempArtists) # shuffle the artists
                            if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                                currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                                rank = rank + 3
                                rankNumbers.append(3)
                                currentRanks[ficIdx][j] = 3

                                # remove artist and choices from pools
                                popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                                artistIDCopy = [artistIDCopy[i] for i in popMask]
                                firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                                secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                                thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                                fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                                fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                        elif ficI in fourthChoicesCopy:
                            findFics = [i for i, x in enumerate(fourthChoicesCopy) if x == ficI]
                            tempArtists = [artistIDCopy[i] for i in findFics] # find artists who ranked this fic fourth
                            shuffle(tempArtists) # shuffle the artists
                            if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                                currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                                rank = rank + 4
                                rankNumbers.append(4)
                                currentRanks[ficIdx][j] = 4

                                # remove artist and choices from pools
                                popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                                artistIDCopy = [artistIDCopy[i] for i in popMask]
                                firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                                secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                                thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                                fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                                fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                        elif ficI in fifthChoicesCopy:
                            findFics = [i for i, x in enumerate(fifthChoicesCopy) if x == ficI]
                            tempArtists = [artistIDCopy[i] for i in findFics] # find artists who ranked this fic fifth
                            shuffle(tempArtists) # shuffle the artists
                            if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                                currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                                rank = rank + 5
                                rankNumbers.append(5)
                                currentRanks[ficIdx][j] = 5

                                # remove artist and choices from pools
                                popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                                artistIDCopy = [artistIDCopy[i] for i in popMask]
                                firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                                secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                                thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                                fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                                fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                    else: # if we are not sorting by ranking
                        # make sure each fic has at least one artist before moving on
                        tempFics = firstChoicesCopy+secondChoicesCopy+thirdChoicesCopy+fourthChoicesCopy+fifthChoicesCopy
                        tempArtistsA = artistIDCopy+artistIDCopy+artistIDCopy+artistIDCopy+artistIDCopy
                        tempRanks = rank1Copy+rank2Copy+rank3Copy+rank4Copy+rank5Copy

                        findFics = [i for i, x in enumerate(tempFics) if x == ficI]
                        tempArtists = [tempArtistsA[i] for i in findFics] # find artists who ranked this fic at all
                        tempRanks = [tempRanks[i] for i in findFics] # get ranks

                        if sum(tempArtists) > 0: # if we still have artists that we can match with this fic
                            x = list(enumerate(tempArtists))
                            shuffle(x)
                            artIdx,tempArtists = zip(*x) # shuffle artists and extract indices

                            tempRanks = [tempRanks[i] for i in artIdx] # shuffle ranks using those indices

                            currentArtistIDs[ficIdx][j] = tempArtists[0] # assign first artist to fic
                            rank = rank + tempRanks[0]
                            rankNumbers.append(tempRanks[0])
                            currentRanks[ficIdx][j] = tempRanks[0]

                            # remove artist and choices from pools
                            popMask = [i for i, x in enumerate(artistIDCopy) if x != tempArtists[0]]
                            artistIDCopy = [artistIDCopy[i] for i in popMask]
                            firstChoicesCopy = [firstChoicesCopy[i] for i in popMask]
                            secondChoicesCopy = [secondChoicesCopy[i] for i in popMask]
                            thirdChoicesCopy = [thirdChoicesCopy[i] for i in popMask]
                            fourthChoicesCopy = [fourthChoicesCopy[i] for i in popMask]
                            fifthChoicesCopy = [fifthChoicesCopy[i] for i in popMask]

                            rank1Copy = [rank1Copy[i] for i in popMask]
                            rank2Copy = [rank2Copy[i] for i in popMask]
                            rank3Copy = [rank3Copy[i] for i in popMask]
                            rank4Copy = [rank4Copy[i] for i in popMask]
                            rank5Copy = [rank5Copy[i] for i in popMask]

                    distribution = distribution + j + 1 # calculates running total of current column/# of artists per fic

            # if we automatched any artists, remove those fics from the pool
            if len(popFics) > 0:
                for idx in range(len(popFics)):
                    popMask = [i for i, x in enumerate(originalFicsSortedCopy) if x != popFics[idx]]
                    originalFicsSortedCopy = [originalFicsSortedCopy[i] for i in popMask]

            if len(artistIDComparison) > len(artistIDCopy): # if we have actually matched artists
                j = j + 1

        # determine effectiveness of run
        # based on two factors:
        # 1. percentages of preferred fics (rank, lower is better)
        # 2. distribution of artists amongst fics (distribution, lower is better)

        performanceIndex = rankWeight*rank + distributionWeight*distribution

        # find fics that only have one artist and weight the performance index
        # to penalize these matching configurations
        temp = currentArtistIDs.T[1]
        if len(temp[~np.isnan(temp)]) < len(originalFics):
            undermatchPenalty = undermatchWeight * (len(originalFics) - len(temp[~np.isnan(temp)]))
            performanceIndex = performanceIndex + undermatchPenalty

        # weight performance index against having two artists
        temp = currentArtistIDs.T[1]
        if len(temp[~np.isnan(temp)]) > 0:
            overmatchPenalty = overmatchWeight * len(temp[~np.isnan(temp)])
            performanceIndex = performanceIndex + overmatchPenalty

        if step == 0: # grab performance from first iteration
            bestPerformance = performanceIndex
            finalMatchesArtistIDs = currentArtistIDs.copy() # use artist matches from this run
            finalMatchesArtistRanks = currentRanks.copy() # use ranks from this run

            currentArtistNames = [[0]*totalFics,[0]*totalFics,[0]*totalFics,[0]*totalFics]
            for m in range(currentArtistIDs.shape[0]):
                for k in range(currentArtistIDs.shape[1]):
                    temp = currentArtistIDs[m][k]
                    if ~np.isnan(temp):
                        artIdx = [i for i, x in enumerate(artistID) if x == currentArtistIDs[m][k]]
                        currentArtistNames[k][m] = [artistName[i] for i in artIdx]
                    else:
                        if unrankedMask[i] == 1:
                            currentArtistNames[k][m] = 'NO ARTIST RANKS';
                        else:
                            currentArtistNames[k][m] = '-'

            finalMatchesArtistNames = currentArtistNames.copy()
            
            stats = "\nCurrent best run: \n{} 1st-Choice Artists\n{} 2nd-Choice Artists\n{} 3rd-Choice Artists\n{} 4th-Choice Artists\n{} 5th-Choice Artists\n\r".format(rankNumbers.count(1), rankNumbers.count(2), rankNumbers.count(3), rankNumbers.count(4), rankNumbers.count(5))

        elif step > 0: # compare all other runs to the first run
            if performanceIndex < bestPerformance: # if our performance index is smaller than our best performance
                bestPerformance = performanceIndex # set new best performance
                finalMatchesArtistIDs = currentArtistIDs.copy() # use artist matches from this run
                finalMatchesArtistRanks = currentRanks.copy() # use ranks from this run

                currentArtistNames = [[0]*totalFics,[0]*totalFics,[0]*totalFics,[0]*totalFics]
                for m in range(currentArtistIDs.shape[0]):
                    for k in range(currentArtistIDs.shape[1]):
                        temp = currentArtistIDs[m][k]
                        if ~np.isnan(temp):
                            artIdx = [i for i, x in enumerate(artistID) if x == currentArtistIDs[m][k]]
                            currentArtistNames[k][m] = [artistName[i] for i in artIdx]
                        else:
                            if unrankedMask[m] == 1:
                                currentArtistNames[k][m] = 'NO ARTIST RANKS';
                            else:
                                currentArtistNames[k][m] = '-'

                finalMatchesArtistNames = currentArtistNames.copy()

                stats = "\nCurrent best run: \n{} 1st-Choice Artists\n{} 2nd-Choice Artists\n{} 3rd-Choice Artists\n{} 4th-Choice Artists\n{} 5th-Choice Artists\n\r".format(rankNumbers.count(1), rankNumbers.count(2), rankNumbers.count(3), rankNumbers.count(4), rankNumbers.count(5))

        # Update Progress Bar
        printProgressBar(step + 1, numIterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        sys.stdout.flush()
        if(step == numIterations - 1): 
            sys.stdout.write("\x1b[1A"*1)
        print(stats, end='\r')
        sys.stdout.write("\x1b[1A"*7)
                
    # export resulting data to an excel file
    print("\n\n\n\r\r\r\n\r\n\r\n\r\n")

    dfoutput = {'Fic IDs': finalMatchesFicIDs, 
                'Artist 1 ID': finalMatchesArtistIDs.T[0], 
                'Artist 1 Name': finalMatchesArtistNames[0], 
                'Artist 1 Rank': finalMatchesArtistRanks.T[0],
                'Artist 2 ID': finalMatchesArtistIDs.T[1], 
                'Artist 2 Name': finalMatchesArtistNames[1], 
                'Artist 2 Rank': finalMatchesArtistRanks.T[1],
                'Artist 3 ID': finalMatchesArtistIDs.T[2], 
                'Artist 3 Name': finalMatchesArtistNames[2], 
                'Artist 3 Rank': finalMatchesArtistRanks.T[2],
                'Artist 4 ID': finalMatchesArtistIDs.T[3], 
                'Artist 4 Name': finalMatchesArtistNames[3], 
                'Artist 4 Rank': finalMatchesArtistRanks.T[3]}

    df2 = pd.DataFrame(data=dfoutput)
    df2.to_excel("finalMatches_Python.xlsx") 
