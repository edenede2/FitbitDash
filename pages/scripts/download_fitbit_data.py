import requests
import pickle
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import datetime
import time as ti
from datetime import datetime, timedelta, time
from datetime import date as dt
import json
import zipfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import os
import sys
import polars as pl

import UTILS.utils as ut




global FILES_DICT

FILES_DICT = {'csv': [], 'json': []}


# Function to combine time with the date and reformat
def merge_date_and_time(given_date, time_str):
    # Combine given date and time string
    combined_str = f"{given_date} {time_str}"
    # Convert to datetime object
    dt = datetime.strptime(combined_str, "%Y-%m-%d %H:%M:%S")
    # Reformat to desired format
    return dt.strftime("%m/%d/%y %H:%M:%S")

def getTempFiles(path, start_date, end_date, token):
    print(f'Starting to download temperature data')

    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')
    SleepPath = Path(path).joinpath('Sleep')
    if not SleepPath.exists():
        SleepPath.mkdir()

    responses = []

    date_ranges_dict = {}

    if len(date_range) >= 30:
        if len(date_range) >= 60:
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[60]
            date_range_2 = [start_date_1, end_date_1]

            start_date_2 = date_range[61]
            end_date_2 = date_range[-1]
            date_range_3 = [start_date_2, end_date_2]

            date_ranges_dict = {1: date_range_1, 2: date_range_2, 3: date_range_3}

        else:
            # Generate start and end dates that have a maximum of 30 days between them
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[-1]
            date_range_2 = [start_date_1, end_date_1]

            date_ranges_dict = {1: date_range_1, 2: date_range_2}
    else:
        start_date = date_range[0]
        end_date = date_range[-1]
        date_ranges_dict = {1: [start_date, end_date]}

    for key, date_range in date_ranges_dict.items():
        
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
        print(f'Start date: {start_date}, End date: {end_date}')
        url = f'https://api.fitbit.com/1/user/-/temp/skin/date/{start_date}/{end_date}.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        responses.append(response)

        if response.status_code == 200:
            response = response.json()
            # create a csv file with the data
            for item in response['tempSkin']:
                timestamp = f"{item['dateTime']} 00:00"
                dailyTempDF = pd.DataFrame({'recorded_time': [timestamp] , 'temperature': [item['value']['nightlyRelative']], 'sensor_type': 'UNKNOWN'}, index=[0])
                dailyTempDF.to_csv(SleepPath.joinpath('Device Temperature - ' + item['dateTime'] + ' API.csv'), index=False)
                with open(SleepPath.joinpath('Device Temperature - ' + item['dateTime'] + ' API.csv'), 'w') as newFTemp:
                    FILES_DICT['csv'].append(newFTemp)
        else:
            print(f'Error {response.status_code}: {response.text}')
            print(f'While downloading temperature data for {start_date} to {end_date}')
            print(f'Reason: {response.reason}')
        print(f'Finished downloading temperature data')


    return responses

def getHRVfiles(path, start_date, end_date, token):
    print(f'Starting to download Heart Rate Variability data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')

    completed_dates = []

    SleepPath = Path(path).joinpath('Sleep')
    if not SleepPath.exists():
        SleepPath.mkdir()

    responses = []

    for date in date_range:
        print(f'Downloading HRV data for {date}')
        newDfHRV = pd.DataFrame({'timestamp': [], 'rmssd': [], 'coverage': [], 'high_frequency': [], 'low_frequency': []})

        date = date.strftime('%Y-%m-%d')
        url = f'https://api.fitbit.com/1/user/-/hrv/date/{date}/all.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        responses.append(response)

        if response.status_code == 200:
            response = response.json()
            # create a csv file with the data
            for item in response['hrv']:
                date = item['dateTime']
                for minute in item['minutes']:
                    timestamp = minute['minute'].replace('T', ' ')
                    timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    rmssd = minute['value']['rmssd']
                    coverage = minute['value']['coverage']
                    highFrequency = minute['value']['hf']
                    lowFrequency = minute['value']['lf']
                    new_row = pd.DataFrame({'timestamp': [timestamp], 'rmssd': [rmssd], 'coverage': [coverage], 'high_frequency': [highFrequency], 'low_frequency': [lowFrequency]})
                    newDfHRV = pd.concat([newDfHRV, new_row], ignore_index=True)
            print(f'Finished downloading HRV data for {date}')
            completed_dates.append(date)

        else:
            print(f'Error {response.status_code}: {response.text}')
            print(f'While downloading HRV data for {date}')
            print(f'Reason: {response.reason}')

        newDfHRV.to_csv(SleepPath.joinpath('Heart Rate Variability Details - ' + date + '.csv'), index=False)
        with open(SleepPath.joinpath('Heart Rate Variability Details - ' + date + '.csv'), 'w') as newFHRV:
            FILES_DICT['csv'].append(newFHRV)
            

    print(f'Finished downloading HRV data')
    print(f'Completed dates: {completed_dates}')
    return responses




def getfilesHR(path, start_date, end_date, token):
    print(f'Starting to download heart rate data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')
    responses = []

    completed_dates = []

    PhysicalActivityPath = Path(path).joinpath('Physical Activity')
    if not PhysicalActivityPath.exists():
        PhysicalActivityPath.mkdir()

    for date in date_range:
        print(f'Downloading heart rate data for {date}')
        date = date.strftime('%Y-%m-%d')
        url = f'https://api.fitbit.com/1.2/user/-/activities/heart/date/{date}/1d/1sec/time/00:00/23:59.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            response = response.json()
            responses.append(response)

            # open the json file
            if 'Request failed with status code 502' in response:
                print(f'Error 502: Bad Gateway - {date} from {path}')
                continue
            # take the date from the json file
            date = response['activities-heart'][0]['dateTime']
            # create a new file
            newFile = 'api-heart_rate-' + date + '.json'
            # write the data to the new file [{"dateTime": "${date} ${data.activities-heart-intraday.dataset.time}", "value": {"bpm": ${data.activities-heart-intraday.dataset.value}, "confidence": 2}}]
            with open(PhysicalActivityPath.joinpath(newFile), 'w') as newFHR:
                # newFHR.write('[{\n')
                for item in response['activities-heart-intraday']['dataset']:
                    item['dateTime'] = merge_date_and_time(date, item['time'])
                    # newFHR.write('"dateTime": "' + date + ' ' + item['time'] + '", "value": {"bpm": ' + str(
                    #     item['value']) + ', "confidence": 2}\n')
                    # newFHR.write('},{\n')
                    item['value'] = {'bpm': item['value'], 'confidence': 2}                
                
                newFHR.write(json.dumps(response['activities-heart-intraday']['dataset'], indent=4))
                FILES_DICT['json'].append(newFHR)
        else:
            print(f'Error {response.status_code}: {response.text}')
            print(f'While downloading heart rate data for {date}')
            print(f'Reason: {response.reason}')

        print(f'Finished downloading heart rate data for {date}')
        completed_dates.append(date)

    print(f'Finished downloading heart rate data')
    print(f'Completed dates: {completed_dates}')
    return responses

def getfilesSteps(path, start_date, end_date, token):
    print(f'Starting to download steps data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')
    completed_dates = []
    steps_df = pd.DataFrame(columns=['timestamp', 'steps'])

    responses = []

    PhysicalActivityPath = Path(path).joinpath('Physical Activity')
    if not PhysicalActivityPath.exists():
        PhysicalActivityPath.mkdir()

    for date in date_range:
        date = date.strftime('%Y-%m-%d')
        url = f'https://api.fitbit.com/1/user/-/activities/steps/date/{date}/1d/1min/time/00:00/23:59.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            response = response.json()
            responses.append(response)

            # open the json file
            if 'Request failed with status code 502' in response:
                continue
            # take the date from the json file
            date = response['activities-steps'][0]['dateTime']
            # create a new file
            newFile = 'api-steps-' + date + '.json'
            # write the data to the new file [{"dateTime": "${date} ${data.activities-heart-intraday.dataset.time}", "value": {"bpm": ${data.activities-heart-intraday.dataset.value}, "confidence": 2}}]
            
            
            for item in response['activities-steps-intraday']['dataset']:
                item['dateTime'] = merge_date_and_time(date, item['time'])

                # newF.write('"dateTime": "' + date + ' ' + item['time'] + '", "value": ' + str(item['value']))
                # newF.write('\n},{\n')

                new_row = pd.DataFrame({'timestamp': [date + ' ' + item['time']], 'steps': [item['value']]})
                steps_df = pd.concat([steps_df, new_row], ignore_index=True)
            # newF.write(json.dumps(response['activities-steps-intraday']['dataset'], indent=4))
            FILES_DICT['json'].append(response['activities-steps-intraday']['dataset'])
            completed_dates.append(date)
        else:
            print(f'Error {response.status_code}: {response.text}')
            print(f'While downloading steps data for {date}')
            print(f'Reason: {response.reason}')

    steps_df.to_csv(PhysicalActivityPath.joinpath('steps.csv'), index=False)
    print(f'Finished downloading steps data')
    print(f'Completed dates: {completed_dates}')
    return responses


def generateRespiratoryRateCSV(path, start_date, end_date, token):
    print(f'Starting to download respiratory rate data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')

    responses = []

    SleepPath = Path(path).joinpath('Sleep')
    if not SleepPath.exists():
        SleepPath.mkdir()
    date_ranges_dict = {}

    if len(date_range) >= 30:
        if len(date_range) >= 60:
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[60]
            date_range_2 = [start_date_1, end_date_1]

            start_date_2 = date_range[61]
            end_date_2 = date_range[-1]
            date_range_3 = [start_date_2, end_date_2]

            date_ranges_dict = {1: date_range_1, 2: date_range_2, 3: date_range_3}

        else:
            # Generate start and end dates that have a maximum of 30 days between them
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[-1]
            date_range_2 = [start_date_1, end_date_1]

            date_ranges_dict = {1: date_range_1, 2: date_range_2}
    else:
        start_date = date_range[0]
        end_date = date_range[-1]
        date_ranges_dict = {1: [start_date, end_date]}

    for key, date_range in date_ranges_dict.items():
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
        url = f'https://api.fitbit.com/1/user/-/br/date/{start_date}/{end_date}/all.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            response = response.json()
            responses.append(response)

            # open the json file
            if 'Request failed with status code 502' in response:
                continue
            for item in response['br']:
                timestamp = f"{item['dateTime']} 0:00:00"
                dailyRespiratoryRateDF = pd.DataFrame({'timestamp': [timestamp] , 'daily_respiratory_rate': [item['value']['fullSleepSummary']['breathingRate']]}, index=[0])
                dailyRespiratoryRateDF.to_csv(SleepPath.joinpath('Daily Respiratory Rate Summary - ' + item['dateTime'] + '.csv'), index=False)
                with open(SleepPath.joinpath('Daily Respiratory Rate Summary - ' + item['dateTime'] + '.csv'), 'w') as newF:
                    FILES_DICT['csv'].append(newF)
        else:
            print(f'Error {response.status_code}: {response.text}')
            print(f'While downloading respiratory rate data for {start_date} to {end_date}')
            print(f'Reason: {response.reason}')
        
    print(f'Finished downloading respiratory rate data')
    return responses

def getCalories(path, start_date, end_date, token):
    print(f'Starting to download calories data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')
    responses = []

    calories_df = pd.DataFrame(columns=['timestamp', 'calories'])
    completed_dates = []

    PhysicalActivityPath = Path(path).joinpath('Physical Activity')
    if not PhysicalActivityPath.exists():
        PhysicalActivityPath.mkdir()

    for date in date_range:
        print(f'Downloading calories data for {date}')
        date = date.strftime('%Y-%m-%d')
        url = f'https://api.fitbit.com/1/user/-/activities/calories/date/{date}/1d/1min/time/00:00/23:59.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            response = response.json()
            responses.append(response)

            # open the json file
            if 'Request failed with status code 502' in response:
                continue
            # take the date from the json file
            date = response['activities-calories'][0]['dateTime']
            # create a new file
            newFile = 'calories-' + date + '.json'
            # write the data to the new file [{"dateTime": "${date} ${data.activities-heart-intraday.dataset.time}", "value": {"bpm": ${data.activities-heart-intraday.dataset.value}, "confidence": 2}}]
            with open(PhysicalActivityPath.joinpath(newFile), 'w') as newF:
                
                for item in response['activities-calories-intraday']['dataset']:
                    item['dateTime'] = merge_date_and_time(date, item['time'])

                    new_row = pd.DataFrame({'timestamp': [date + ' ' + item['time']], 'calories': [item['value']]})
                    calories_df = pd.concat([calories_df, new_row], ignore_index=True)
                newF.write(json.dumps(response['activities-calories-intraday']['dataset'], indent=4))
                FILES_DICT['json'].append(newF)
        print(f'Finished downloading calories data for {date}')
        completed_dates.append(date)

    print(f'Finished downloading calories data')
    print(f'Completed dates: {completed_dates}')
    calories_df.to_csv(PhysicalActivityPath.joinpath('calories.csv'), index=False)


def getSleepfiles(path, start_date, end_date, token):
    print(f'Starting to download sleep data')
    date_range = pd.date_range(start_date, end_date)
    print(f'Date range: {date_range}')

    responses = []

    SleepPath = Path(path).joinpath('Sleep')
    if not SleepPath.exists():
        SleepPath.mkdir()
    
    date_ranges_dict = {}

    if len(date_range) >= 30:
        if len(date_range) >= 60:
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[60]
            date_range_2 = [start_date_1, end_date_1]

            start_date_2 = date_range[61]
            end_date_2 = date_range[-1]
            date_range_3 = [start_date_2, end_date_2]

            date_ranges_dict = {1: date_range_1, 2: date_range_2, 3: date_range_3}

        else:
            # Generate start and end dates that have a maximum of 30 days between them
            start_date = date_range[0]
            end_date = date_range[30]
            date_range_1 = [start_date, end_date]

            start_date_1 = date_range[31]
            end_date_1 = date_range[-1]
            date_range_2 = [start_date_1, end_date_1]

            date_ranges_dict = {1: date_range_1, 2: date_range_2}
    else:
        start_date = date_range[0]
        end_date = date_range[-1]
        date_ranges_dict = {1: [start_date, end_date]}

    for key, date_range in date_ranges_dict.items():
        
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
        print(f'Start date: {start_date}, End date: {end_date}')
        url = f'https://api.fitbit.com/1.2/user/-/sleep/date/{start_date}/{end_date}.json'
        headers = {
            'Authorization': 'Bearer ' + token,
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            response = response.json()['sleep']
            responses.append(response)

            # open the json file

            if 'Request failed with status code 502' in response:
                continue
            if '492' in response:
                print(response)
                continue
            # save the data to a json file
            response = json.dumps(response, indent=4)
            with open(SleepPath.joinpath('sleep-' + start_date + '.json'), 'w') as newF:
                newF.write(response)
                FILES_DICT['json'].append(newF)
    print(f'Finished downloading sleep data')
    return responses


def saveAsZIP(path, token, start_date, end_date,sub_name):
    print(f'Starting to download data for {sub_name}')
    with TemporaryDirectory() as temp_dir:
        print(f'Temporary directory created: {temp_dir}')


        temp_dir = Path(temp_dir)
        sub_temp_path = temp_dir.joinpath(sub_name)
        if not sub_temp_path.exists():
            sub_temp_path.mkdir()
            sub_temp_path.joinpath('FITBIT').mkdir()
        sub_temp_path = sub_temp_path.joinpath('FITBIT')
        sub_temp_path.joinpath('Sleep').mkdir()
        sub_temp_path.joinpath('Physical Activity').mkdir()
        sub_temp_path.joinpath('Stress').mkdir()
        print(f'Subject temporary directory created: {sub_temp_path}')
        sleep_responses_nova = getSleepfiles(sub_temp_path, start_date, end_date, token)
        steps_responses_nova = getfilesSteps(sub_temp_path, start_date, end_date, token)
        HR_responses_nova = getfilesHR(sub_temp_path, start_date, end_date, token)
        Respiratory_responses_nova = generateRespiratoryRateCSV(sub_temp_path, start_date, end_date, token)
        HRV_responses_nova = getHRVfiles(sub_temp_path, start_date, end_date, token)
        Temp_responses_nova = getTempFiles(sub_temp_path, start_date, end_date, token)
        Calories_responses_nova = getCalories(sub_temp_path, start_date, end_date, token)
        print(f'Finished downloading data for {sub_name}')

        print(f'Cleaning steps data for {sub_name}')
        clean_steps(sub_temp_path)
        # Create a ZipFile Object
        with ZipFile(path.joinpath(f'{sub_name}.zip'), 'w', compression=ZIP_DEFLATED) as zipObj:
            for folderName, subfolders, filenames in os.walk(temp_dir):
            
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, os.path.relpath(filePath, temp_dir))

        print(f'Zip file created for {sub_name}')
        print(f'Cleaning temporary directory for {sub_name}')
        shutil.rmtree(temp_dir)
        print(f'Temporary directory cleaned for {sub_name}')

        print(f'Extracting zip file for {sub_name}')
        with ZipFile(path.joinpath(f'{sub_name}.zip'), 'r') as zip_ref:
            zip_ref.extractall(path)
        print(f'Zip file extracted for {sub_name}')


def clean_steps(physical_activity_path):
    print('Cleaning steps data')
    # read the steps and calories dataframes that generated from the API
    steps_df = pd.read_csv(physical_activity_path.joinpath('Physical Activity').joinpath('steps.csv'))
    print(f'steps file found: {steps_df.head()}')
    steps_df['timestamp'] = pd.to_datetime(steps_df['timestamp'])
    # read the calories dataframe that generated from the API
    calories_df = pd.read_csv(physical_activity_path.joinpath('Physical Activity').joinpath('calories.csv'))
    print(f'calories file found: {calories_df.head()}')
    calories_df['timestamp'] = pd.to_datetime(calories_df['timestamp'])
    

    # merge the two dataframes on the timestamp column
    merged_df = pd.merge(steps_df, calories_df, on='timestamp', how='outer')
    print(f'merged file: {merged_df.head()}')
    # save the merged dataframe to a csv file
    merged_df.to_csv(physical_activity_path.joinpath('Physical Activity').joinpath('merged.csv'), index=False)
    print(f'merged file saved to {physical_activity_path.joinpath("Physical Activity").joinpath("merged.csv")}')
    min_calories_per_date = {}


    merged_df['date'] = merged_df['timestamp'].dt.date

    for date in merged_df['date'].unique():
        print(f'Getting min calories for {date}')
        min_calories_per_date[date] = merged_df[merged_df['date'] == date]['calories'].min()

    for index, row in merged_df.iterrows():
        date = row['date']
        if row['calories'] == min_calories_per_date[date] and row['steps'] == 0:
            merged_df.at[index, 'steps'] = np.nan
            merged_df.at[index, 'timestamp'] = np.nan



    # get new steps json file
    new_steps_df = pd.DataFrame(columns=['timestamp', 'steps'])
    # for index, row in merged_df.iterrows():
    #     new_row = pd.DataFrame({'timestamp': [row['timestamp']], 'steps': [row['steps']]})
    #     new_steps_df = pd.concat([new_steps_df, new_row], ignore_index=True)
    new_steps_df = merged_df[['timestamp', 'steps']]
    print(f'new steps file: {new_steps_df.head()}')


    # save the new steps json file
    new_steps = []
    for index, row in new_steps_df.iterrows():
        if pd.notna(row['timestamp']) and pd.notna(row['steps']):
            new_steps.append({
                "dateTime": row['timestamp'].strftime('%m/%d/%y %H:%M:%S'),
                "value": str(int(row['steps']))
            })
    print(f'new steps: {new_steps}')
    physical_activity_path = Path(physical_activity_path.joinpath('Physical Activity'))
    with open(physical_activity_path.joinpath(f'api-steps-{date}.json'), 'w') as newF:
        new_steps = json.dumps(new_steps, indent=4)
        newF.write(new_steps)
        FILES_DICT['json'].append(newF)
    print(f'new steps file saved to {physical_activity_path.joinpath("api_steps-2024-04-27.json")}')


def main(project, username, now):

    try:
        print('Downloading data...')
        exeHistory_path = Path(r'.\pages\ExecutionHis\exeHistory.parquet')
        exeHistory = pl.read_parquet(exeHistory_path)

        paths_json = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))

    
        project_path = Path(paths_json[project])



        # DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

        # AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

        # if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
        #     os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)
        DATA_PATH = Path(project_path)
        if not DATA_PATH.exists():
            DATA_PATH.mkdir()


        api_data_path = Path(rf'.\pages\sub_selection\{project}_sub_selection_download_api.parquet')
    
        data_df = pl.read_parquet(api_data_path)

        print('Folder created successfully')

        for sub in data_df.iter_rows():
            print(f'Downloading data for {sub[0]}')
            sub_name = sub[0]
            token = sub[5]
            start_date = pd.to_datetime(sub[1], format='%d/%m/%Y')
            end_date = pd.to_datetime(sub[3], format='%d/%m/%Y')
            path = Path(DATA_PATH)
            
            if not path.exists():
                path.mkdir()
            saveAsZIP(path, token, start_date, end_date, sub_name)
    except Exception as e:
        print(e)
        ti.sleep(15)
        



    # for sub in data_df.iter_rows():
    #     if not sub[0] in ['sub_011_T4','sub_012_T2', 'sub_012_T3','sub_012_T4', 'sub_015_T4','sub_016_T3', 'sub_016_T4', 'sub_017_T3']:
    #         continue
    #     sub_name = sub[0]
    #     token = sub[21]
    #     start_date = sub[22]
    #     end_date = sub[23]
    #     path = Path('/Users/edeneldar/JS learn/fitbitApiCall/data').joinpath(sub_name)
    #     if not path.exists():
    #         path.mkdir()
    #     saveAsZIP(path, token, start_date, end_date, sub_name)
        





if __name__ == '__main__':
    try:
        project = sys.argv[1]
        username = sys.argv[2]
        now = sys.argv[3]
        main(project, username, now)
    except Exception as e:
            project = 'MDMA'
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            username = 'PsyLab-6028'

            main(project, username, now)

        
        

    