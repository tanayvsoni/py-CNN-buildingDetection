# Code for Building Dectection using CNN 
This repo contains an building dectection and data organization used to train this CNN.
The repo is part of the larger research project in collabroation with professors at the University of Alberta. 

This project is to build a Image-Based Post-Disaster Recovery Estimation Using Bayesian & Convolutional Neural Network.

## Purpose:
Communities that go through disasters (earthquakes, floods, etc), often recover with certain recovery curves

![infrastructures-04-00030-g003](https://user-images.githubusercontent.com/104151592/199405860-2a40b242-a027-486d-81b4-ac7f8abdf10b.png)

These recovery curves usually are steeper for richer communities than poorer ones. Using a Bayesian Nerual Network, we predicte these curves with a high degree of accuracy with only first 'x' amount of days worth of information.

The only bottleneck to scale the BNN to whole countries, and thousands of communities is that gathering data on how the recovery is going is difficult and slow, however this Building Dectection CNN code is suppsoe to solve this issue.

## Method:
If can use infrastructure damage and rebuilt buildings as a good indicator of how well a communitity is recovering, then to gather data we just need to compare how many existed before and after the disaster and how well the rebuilding is going. 

We can observe this using free data from google maps, however doing this manual is still very slow. To remendy this issue this CNN was created.

- Images and labels (exisiting usable buildings) were gather manually to train and test the network
- A data organization script was written in python to make data collection more streamlined
- Currently data collection is still ongoing, accuracy numbers will be posted after development is complete.


## Here's a general description of each of the files:
`data_org.py` - Script that was used to streamline data collection/

`CNN.py` - Main CNN code, and data interpatation implementation.
