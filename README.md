# BigBangMatching
Command line-based Python code that takes in an Excel file of artist claims and outputs matches with writers

# Usage
```
matching.py [-h] -x EXCEL -f FICS -i ITERATIONS [-n NULLFICS] [-p PREFERENCE] [-s SHUFFLE] [-d DISTRIBUTIONWEIGHT] [-r RANKWEIGHT] [-u UNDERMATCHWEIGHT] [-o OVERMATCHWEIGHT]
```

# Details about usage and matching process
This code works by running through multiple different matching iterations/runs and finding the one that has the smallest performance index. By default, any fics with only one artist ranking that fic automatically matches that artist with that fic, and any fics with no artists ranking it are removed from the list of possible fics to match. Those fics will output with NO ARTIST RANKINGS as the artist names so that they can be easily identified and manually matched.

Details about the parameters you can define when running this code via the command line are below:

## Excel
The name of the Excel file with your claims information in it.

## Fics
Total number of fics. This should be your highest writer number that an artist can choose during the matching process.

## Iterations
Iterations are the number of runs you wish to use. More iterations may result in better performance, but not always. I usually use somewhere between 5000 and 10000.

## Null Fics
Null fics are fics that are not part of the matching process (ie, writers who have dropped out). Because this code runs through a list of numbers from 1 to your highest writer ID number, you need to specify which fics it should skip (or the code will assume these fics have not been selected yet and will return NO ARTIST MATCH for those fics).

## Preference
Default = 1

Sort Preference determines whether or not you wish to assign matches based on artist rankings/preferences. 0 ignores all preferences (1st to 5th choice) and just determines if an artist chose a fic at all. 1 takes preference into account. Both parameters do still generate a rank weight that contributes to the performance index, but Preference = 1 specifically takes it into account when matching. Preference = 1 is generally better, but Preference = 0 can be used if your code is having difficulty creating viable matches.

## Shuffle
Default = 1

Shuffle determines how the fics are shuffled each iteration. Shuffle = 0 does not shuffle the fics at all. This often leads to the same configurations of artists and writers being generated over and over again and is generally not the best method of matching. Shuffle = 1 shuffles all fics randomly before each iteration, and Shuffle = 2 shuffles fics while still maintaining order of ranking. In other words, this code--by default--sorts fics based on how frequently they were ranked by artists (eg. if fic 7 was ranked 2 times and fic 11 was ranked 5 times, fic 7 would be matched before fic 11). Shuffle = 2 maintains this sorting while shuffling all the fics that were ranked 2 times, all the fics that were ranked 3 times, etc. (Eg. if 7 and 8 were each ranked twice and 11 and 12 were each ranked three times, we might have a fic order of 7 8 11 12, or we might have an order of 8 7 12 11.) Shuffle = 1 is adequate for most situations, but Shuffle = 2 can be used if you are having difficulty getting the code to match a particular fic that was not ranked by many artists, especially if many artists ranked it lower on their list.

## Distribution Weight
Default = 1

This is how evenly the artists are distributed amongst your fics. For example, if you have 20 writers and 30 artists, weighing this parameter more heavily/using a larger number will prompt the code to trend towards 10 fics having 2 artists and 10 having 1 artist, rather than eg. 10 fics having 2 artists, 5 having 1 artist, and 5 having 3 artists.

## Rank Weight
Default = 1

This is how many artists have higher-ranked fics rather than lower-ranked fics. Weighing this parameter more heavily will prompt the code to trend towards more artists having their 1st choice fics (or 2nd, or 3rd, etc.), sometimes at the cost of having a more uneven distribution of artists amongst fics

## Undermatch Weight
Default = 0

This is a penalty imposed against fics that are only assigned one writer that scales with the number of fics that only have one writer. This parameter can be scaled more heavily, in conjunction with the distribution index, to specifically discourage situations where a fic is only assigned one writer while other fics are assigned two or three writers. May not be necessary in situations where the ratio of writers to artists is closer, or where you have fewer artists than writers. Works well in situations where you have a large amount of artists compared to writers. In this case, set this value to 0.

## Overmatch Weight
Default = 0

This is a penalty imposed against fics that have two artists. This is very similar to the undermatch weight, but this parameter works well for situations where most fics have one artist but only a few have two, or where the code is leaving fics unmatched in favor of giving a fic two artists. Unlike with undermatch weight, this parameter can be very useful in situations where you have a close ratio of writers to artists, or when you have fewer artists than writers, and is less useful in situations where you have a lot of artists compared to writers. If you do not wish to use this parameter, set the value to 0.

# Structure of Excel file
This code works for an Excel file that contains at least the following columns, named as such:
- Number: Artist number
- Preferred Name: Artist name
- 1st Choice Fic: Fic ranked 1st by artist
- 2nd Choice Fic: Fic ranked 2nd by artist
- 3rd Choice Fic: Fic ranked 3rd by artist
- 4th Choice Fic: Fic ranked 4th by artist
- 5th Choice Fic: Fic ranked 5th by artist

Code can be edited to have more or fewer ranks taken into account
