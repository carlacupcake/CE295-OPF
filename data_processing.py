# CE 295 - Energy Systems and Control
# Final Project: Model Predictive Control and the Optimal Power Flow Problem in the IEEE 39-bus Test Feededr
# Author: Carla Becker
# File description: TODO

import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

# Define global variables
global day_length
day_length = 96 # data point every 15 minutes --> 4 points per hour, 24 hours per day
global num_points_per_entity
num_points_per_entity = 41085 # the fewest points available for a single generator/consumer
global num_days
num_days = int(np.floor(num_points_per_entity/day_length))

#--------------------------------------------------------------------------------------------#
# FUNCTIONS FOR PV GENERATION DATA 
#--------------------------------------------------------------------------------------------#

def generate_json_from_pv_data(directory):

    '''
    Inputs: 
    directory = string, where the csv files are stored e.g. "PV Generation Data" 

    Outputs: 
    json file for real power keyed by generator name

    Notes: 
    The csv files should contain 2 columns each: DateTime, RealPower [kW]
    A single csv may produce several generators (entities) if they contain more data then the shortest file
    '''
    
    # Import raw csv data (takes 3- 5 minutes to read-in)
    pv_data = pd.DataFrame()

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        
        if file.endswith(".csv"):                                      # only iterate over csv files
            filepath = os.path.join(directory, file)                   # Get the full file path
            filename = os.path.splitext(os.path.basename(filepath))[0] # use filename as column name
            
            print("Reading:", filepath)
            data_in = pd.read_csv(filepath, parse_dates=['DateTime'])

            # Normalize the data
            data_in['RealPower'] /= np.abs(data_in['RealPower']).max()

            # Replace negative values with zeros          
            power_data = data_in['RealPower']  # only extract real power, not datetime

            # Find how many entities we can extract from this single entity
            num_data_points = power_data.shape[0]
            num_entities = int(np.floor(num_data_points/num_points_per_entity))

            # Loop over the entities and store daily data for each
            for entity in range(num_entities):
                start = entity * num_points_per_entity
                end = start + num_points_per_entity
                key_name = filename + str(entity)
                pv_data[key_name] = power_data[start:end]

    pv_dict = {}
    for column in pv_data.columns:
        daily_data = {}

        for day in range(num_days):
            start = day_length * day
            end = start + day_length       
            daily_data[day] = pv_data[column][start:end].tolist()

        pv_dict[column] = daily_data

    with open(os.path.join(directory, 'pv_data.json'), 'w') as json_file:
        json.dump(pv_dict, json_file)

def plot_pv_data(time_data, power_data, save_figure=False, filename='plot_data.png'):

    '''
    Positional Inputs: 
    time_data = values from DateTime column of raw csv (dataframe, ndarray, or list)
    power_data = either real power generated (dataframe, ndarray, or list)

    Keyword Inputs:
    save_figure = boolean for whether or not you want to save the plot (bool)
    filename = the name you want tp save the plot under, end with '.png' (string)

    Outputs: 
    A plot of time_data on the x-axis and power_data on the y-axis
    '''

    plt.figure(figsize=(8, 4))
    plt.plot(time_data, power_data)
    plt.xlabel('Date')
    plt.ylabel('PV Generation (kW)')
    plt.title(filename)
    plt.show()

    if save_figure:
        plt.savefig(filename)

#--------------------------------------------------------------------------------------------#
# FUNCTIONS FOR BUILDING LOAD DATA 
#--------------------------------------------------------------------------------------------#

def generate_json_from_bldg_data(directory):

    '''
    Inputs: 
    directory = string, where the csv files are stored e.g. "Building Load Data" 

    Outputs: 
    json file for real power keyed by consumer name

    Notes: 
    The csv files should contain 3 columns each: DateTime, RealPower [kW], ReactivePower [kW]
    A single csv may produce several consumers (entities) if they contain more data then the shortest file
    '''
    
    # Import raw csv data (takes 3- 5 minutes to read-in)
    real_data = pd.DataFrame()
    reactive_data = pd.DataFrame()

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        
        if file.endswith(".csv"):                                      # only iterate over csv files
            filepath = os.path.join(directory, file)                   # Get the full file path
            filename = os.path.splitext(os.path.basename(filepath))[0] # use filename as column name
            
            print("Reading:", filepath)
            data_in = pd.read_csv(filepath, parse_dates=['DateTime'])

            # Normalize the data
            data_in['RealPower'] /= np.abs(data_in['RealPower']).max()
            data_in['ReactivePower'] /= np.abs(data_in['ReactivePower']).max()

            # Replace negative values with zeros          
            real_power     = data_in['RealPower']   
            reactive_power = data_in['ReactivePower'] 

            # Find how many entities we can extract from this single entity
            num_data_points = real_power.shape[0]
            num_entities = int(np.floor(num_data_points/num_points_per_entity))

            # Loop over the entities and store daily data for each
            for entity in range(num_entities):
                start = entity * num_points_per_entity
                end = start + num_points_per_entity
                key_name = filename + str(entity)
                real_data[key_name] = real_power[start:end]
                reactive_data[key_name] = reactive_power[start:end]

    real_dict = {}
    reactive_dict = {}
    for column in real_data.columns:
        daily_real_data = {}
        daily_reactive_data = {}

        for day in range(num_days):
            start = day_length * day
            end = start + day_length       
            daily_real_data[day] = real_data[column][start:end].tolist() 
            daily_reactive_data[day] = reactive_data[column][start:end].tolist()

        real_dict[column] = daily_real_data
        reactive_dict[column] = daily_reactive_data

    with open(os.path.join(directory, 'real_data.json'), 'w') as json_file:
        json.dump(real_dict, json_file)

    with open(os.path.join(directory, 'reactive_data.json'), 'w') as json_file:
        json.dump(reactive_dict, json_file)

def plot_load_data(time_data, load_data, save_figure=False, filename='plot_data.png'):

    '''
    Positional Inputs: 
    time_data = values from DateTime column of raw csv (dataframe, ndarray, or list)
    load_data = either real or reactive power consumed (dataframe, ndarray, or list)

    Keyword Inputs:
    save_figure = boolean for whether or not you want to save the plot (bool)
    filename = the name you want tp save the plot under, end with '.png' (string)

    Outputs: 
    A plot of time_data on the x-axis and load_data on the y-axis
    '''

    plt.figure(figsize=(8, 4))
    plt.plot(time_data, load_data)
    plt.xlabel('Date')
    plt.ylabel('Building Consumption (kW)')
    plt.title(filename)
    plt.show()

    if save_figure:
        plt.savefig(filename)

#--------------------------------------------------------------------------------------------#
# FUNCTIONS FOR WIND GENERATION DATA 
#--------------------------------------------------------------------------------------------#

def downsample_wind_data(original_csv, n):

    '''
    Inputs: 
    original_csv = original data that you want to downsample, 
                   e.g. 5 minute day you want to downsample to 15 minutes
    n = downsample every n samples e.g. n=3 for 5 min --> 15 min

    Outputs: 
    An equivalent csv file, just with every n rows

    Notes:
    Assumes there is a header row
    Header row gets copied to the new file
    '''

    directory, filename = os.path.split(original_csv)
    filename, file_extension = os.path.splitext(filename)
    new_csv = filename + '_downsampled_by_' + str(n) + '.' + file_extension

    # Open the original csv file for reading
    with open(original_csv, mode='r') as original_file:
        csv_reader = csv.reader(original_file)
        
        # Open a new csv file for writing
        with open(os.path.join(directory, new_csv), mode='w', newline='') as new_file:
            csv_writer = csv.writer(new_file)
            
            # Read and write the header row
            header_row = next(csv_reader)
            csv_writer.writerow(header_row)
            
            # Use a counter to keep track of the rows
            row_counter = 0
            
            # Iterate over the remaining rows in the original csv file
            for row in csv_reader:
                
                # Write every noth row to the new csv file, starting after the header row
                if row_counter % n == 0:
                    csv_writer.writerow(row)
                
                # Increment the row counter
                row_counter += 1

    print("New CSV file created successfully.")

def generate_individual_wind_csvs(cumulative_csv):

    '''
    Inputs: 
    cumulative_csv = original data, organized with columns corresponding to individual generators

    Outputs: 
    A csv file for each generator with date, time, and power as the columns
    '''
    
    cumulative_wind_data = pd.read_csv(cumulative_csv)
    directory, filename = os.path.split(cumulative_csv)

    for col in cumulative_wind_data.columns:
        if (col == 'Date') or (col == 'Time'):
            continue
        turbine_df = cumulative_wind_data[['Date', 'Time', col]]
        filename = col + '.csv'
        filepath = os.path.join(directory, filename)
        turbine_df.to_csv(filepath, index=False)

def generate_json_from_wind_data(directory):

    '''
    Inputs: 
    directory = string, where the csv files are stored e.g. "Wind Generation Data" 

    Outputs: 
    json file for real power keyed by generator name

    Notes: 
    The csv files should contain 3 columns each: Date, Time, RealPower [kW]
    A single csv may produce several consumers (entities) if they contain more data then the shortest file
    '''
    
    # Import raw csv data (takes 3- 5 minutes to read-in)
    wind_data = pd.DataFrame()

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        
        if file.endswith(".csv"):                                      # only iterate over csv files
            filepath = os.path.join(directory, file)                   # Get the full file path
            filename = os.path.splitext(os.path.basename(filepath))[0] # use filename as column name
            
            print("Reading:", filepath)
            data_in = pd.read_csv(filepath)

            # Normalize the data
            data_in.iloc[:, 2] /= np.abs(data_in.iloc[:, 2]).max()

            # Replace negative values with zeros          
            wind_power     = data_in.iloc[:, 2]   

            # Find how many entities we can extract from this single entity
            num_data_points = wind_power.shape[0]
            num_entities = int(np.floor(num_data_points/num_points_per_entity))

            # Loop over the entities and store daily data for each
            for entity in range(num_entities):
                start = entity * num_points_per_entity
                end = start + num_points_per_entity
                key_name = filename + str(entity)
                wind_data[key_name] = wind_power[start:end]

    wind_dict = {}
    for column in wind_data.columns:
        daily_data = {}

        for day in range(num_days):
            start = day_length * day
            end = start + day_length       
            daily_data[day] = wind_data[column][start:end].tolist() 

        wind_dict[column] = daily_data

    with open(os.path.join(directory, 'wind_data.json'), 'w') as json_file:
        json.dump(wind_dict, json_file)

def plot_wind_data(time_data, wind_data, save_figure=False, filename='plot_data.png'):

    '''
    Positional Inputs: 
    time_data = values from DateTime column of raw csv (dataframe, ndarray, or list)
    wind_data = real power generated (dataframe, ndarray, or list)

    Keyword Inputs:
    save_figure = boolean for whether or not you want to save the plot (bool)
    filename = the name you want tp save the plot under, end with '.png' (string)

    Outputs: 
    A plot of time_data on the x-axis and wind_data on the y-axis
    '''

    plt.figure(figsize=(8, 4))
    plt.plot(time_data, wind_data)
    plt.xlabel('Date')
    plt.ylabel('Winder Power Generated (kW)')
    plt.title(filename)
    plt.show()

    if save_figure:
        plt.savefig(filename)