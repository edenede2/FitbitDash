import requests
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import datetime
from datetime import time
from datetime import datetime, timedelta
import json
import zipfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import os
import time as ti
import polars as pl




global FILES_DICT

FILES_DICT = {'csv': [], 'json': []}


def getTempFiles(path, start_date, end_date, token):

    date_range = pd.date_range(start_date, end_date)

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


    return responses

def getHRVfiles(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)

    SleepPath = Path(path).joinpath('Sleep')
    if not SleepPath.exists():
        SleepPath.mkdir()

    responses = []

    for date in date_range:
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
        else:
            if response.status_code == 403:
                print(f'Error 403: Forbidden - {date} from {path}')
            # print(response.json())
        newDfHRV.to_csv(SleepPath.joinpath('Heart Rate Variability Details - ' + date + '.csv'), index=False)
        with open(SleepPath.joinpath('Heart Rate Variability Details - ' + date + '.csv'), 'w') as newFHRV:
            FILES_DICT['csv'].append(newFHRV)

    return responses




def getfilesHR(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)

    responses = []

    PhysicalActivityPath = Path(path).joinpath('Physical Activity')
    if not PhysicalActivityPath.exists():
        PhysicalActivityPath.mkdir()

    for date in date_range:
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
                continue
            # take the date from the json file
            date = response['activities-heart'][0]['dateTime']
            # create a new file
            newFile = 'api-heart_rate-' + date + '.json'
            # write the data to the new file [{"dateTime": "${date} ${data.activities-heart-intraday.dataset.time}", "value": {"bpm": ${data.activities-heart-intraday.dataset.value}, "confidence": 2}}]
            with open(PhysicalActivityPath.joinpath(newFile), 'w') as newFHR:
                newFHR.write('[{\n')
                for item in response['activities-heart-intraday']['dataset']:
                    newFHR.write('"dateTime": "' + date + ' ' + item['time'] + '", "value": {"bpm": ' + str(
                        item['value']) + ', "confidence": 2}\n')
                    newFHR.write('},{\n')
                    
                newFHR.write('}]')

                FILES_DICT['json'].append(newFHR)

                
    return responses

def getfilesSteps(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)

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
            with open(PhysicalActivityPath.joinpath(newFile), 'w') as newF:
                newF.write('[{\n')
                for item in response['activities-steps-intraday']['dataset']:
                    newF.write('"dateTime": "' + date + ' ' + item['time'] + '", "value": ' + str(item['value']))
                    newF.write('\n},{\n')
                    new_row = pd.DataFrame({'timestamp': [date + ' ' + item['time']], 'steps': [item['value']]})
                    steps_df = pd.concat([steps_df, new_row], ignore_index=True)
                newF.write('\n}]')
                FILES_DICT['json'].append(newF)

    steps_df.to_csv(PhysicalActivityPath.joinpath('steps.csv'), index=False)

    return responses


def generateRespiratoryRateCSV(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)


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

    return responses

def getCalories(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)

    responses = []

    calories_df = pd.DataFrame(columns=['timestamp', 'calories'])

    PhysicalActivityPath = Path(path).joinpath('Physical Activity')
    if not PhysicalActivityPath.exists():
        PhysicalActivityPath.mkdir()

    for date in date_range:
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
                newF.write('[{\n')
                for item in response['activities-calories-intraday']['dataset']:
                    newF.write('"dateTime": "' + date + ' ' + item['time'] + '", "value": ' + str(item['value']))
                    newF.write('\n},{\n')
                    new_row = pd.DataFrame({'timestamp': [date + ' ' + item['time']], 'calories': [item['value']]})
                    calories_df = pd.concat([calories_df, new_row], ignore_index=True)
                newF.write('\n}]')
                FILES_DICT['json'].append(newF)


    calories_df.to_csv(PhysicalActivityPath.joinpath('calories.csv'), index=False)


def getSleepfiles(path, start_date, end_date, token):
    date_range = pd.date_range(start_date, end_date)

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
        url = f'https://api.fitbit.com/1.2/user/-/sleep/date/{start_date}/{end_date}.json'
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
            if '492' in response:
                print(response)
                continue
            # save the data to a json file
            with open(SleepPath.joinpath('sleep-' + start_date + '.json'), 'w') as newF:
                json.dump(response, newF)
                FILES_DICT['json'].append(newF)

    return responses


def saveAsZIP(path, token, start_date, end_date,sub_name):
    with TemporaryDirectory() as temp_dir:
        # Nova131_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JXVEQiLCJzdWIiOiJCVDRHUEIiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBycHJvIHJudXQgcnNsZSByYWN0IHJyZXMgcmxvYyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ1MjQyMjYyLCJpYXQiOjE3MTM3MDYyNjZ9.wmP3VhhaoxoGxUEqALN284VW--DQpR7Tum37CdHLX7I'
        # Nova131_start_date = '2024-03-07'
        # Nova131_end_date = '2024-04-21'

        temp_dir = Path(temp_dir)
        sub_temp_path = temp_dir.joinpath(sub_name)
        if not sub_temp_path.exists():
            sub_temp_path.mkdir()
        sub_temp_path.joinpath('Sleep').mkdir()
        sub_temp_path.joinpath('Physical Activity').mkdir()
        sub_temp_path.joinpath('Stress').mkdir()

        sleep_responses_nova = getSleepfiles(sub_temp_path, start_date, end_date, token)
        steps_responses_nova = getfilesSteps(sub_temp_path, start_date, end_date, token)
        HR_responses_nova = getfilesHR(sub_temp_path, start_date, end_date, token)
        Respiratory_responses_nova = generateRespiratoryRateCSV(sub_temp_path, start_date, end_date, token)
        HRV_responses_nova = getHRVfiles(sub_temp_path, start_date, end_date, token)
        Temp_responses_nova = getTempFiles(sub_temp_path, start_date, end_date, token)

        # Create a ZipFile Object
        with ZipFile(path.joinpath(f'{sub_name}.zip'), 'w', compression=ZIP_DEFLATED) as zipObj:
            for folderName, subfolders, filenames in os.walk(temp_dir):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, os.path.relpath(filePath, temp_dir))


def clean_steps(physical_activity_path):
    # read the steps and calories dataframes that generated from the API
    steps_df = pd.read_csv(physical_activity_path.joinpath('Physical Activity').joinpath('steps.csv'))
    steps_df['timestamp'] = pd.to_datetime(steps_df['timestamp'])

    # read the calories dataframe that generated from the API
    calories_df = pd.read_csv(physical_activity_path.joinpath('Physical Activity').joinpath('calories.csv'))
    calories_df['timestamp'] = pd.to_datetime(calories_df['timestamp'])

    # merge the two dataframes on the timestamp column
    merged_df = pd.merge(steps_df, calories_df, on='timestamp', how='outer')
    
    # save the merged dataframe to a csv file
    merged_df.to_csv(physical_activity_path.joinpath('Physical Activity').joinpath('merged.csv'), index=False)

    min_calories_per_date = {}


    merged_df['date'] = merged_df['timestamp'].dt.date

    for date in merged_df['date'].unique():
        min_calories_per_date[date] = merged_df[merged_df['date'] == date]['calories'].min()

    for index, row in merged_df.iterrows():
        date = row['date']
        if row['calories'] == min_calories_per_date[date] and row['steps'] == 0:
            merged_df.at[index, 'steps'] = np.nan
            merged_df.at[index, 'timestamp'] = np.nan



    # get new steps json file
    new_steps_df = pd.DataFrame(columns=['timestamp', 'steps'])
    for index, row in merged_df.iterrows():
        new_row = pd.DataFrame({'timestamp': [row['timestamp']], 'steps': [row['steps']]})
        new_steps_df = pd.concat([new_steps_df, new_row], ignore_index=True)

    # save the new steps json file
    new_steps = []
    for index, row in new_steps_df.iterrows():
        if pd.notna(row['timestamp']) and pd.notna(row['steps']):
            new_steps.append({
                "dateTime": row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "value": row['steps']
            })

    with open(physical_activity_path.joinpath('api_steps-2024-04-27.json'), 'w') as newF:
        newF.write('[\n')
        newF.write(',\n'.join([str(step).replace("'", '"') for step in new_steps]))
        newF.write('\n]')


        
sub_011_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1BKWEIiLCJzdWIiOiJCWTJOTVciLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJsb2MgcnJlcyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ5NTY0Mzc1LCJpYXQiOjE3MTgwMjgzNzV9.T28rPF3oIpqtlSOOGJvDadhZEq-7zTep9RoSSmlzUCw'
sub_012_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1MyTEMiLCJzdWIiOiJCWFJMUUYiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJyZXMgcmxvYyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ0MTM0NjkwLCJpYXQiOjE3MTI1OTg2OTB9.iSItUVSWtlOS3C-Hf9MeVdW9TDn7_RHS0-TyfhBcnII'
sub_015_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1MySE0iLCJzdWIiOiJCWFJMOFkiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJsb2MgcnJlcyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ0MTM2MzE0LCJpYXQiOjE3MTI2MDAzMTR9.b01r4LuR8ffMofs3WUg18mGV5gGITy_eVAwILuUgezs'
sub_016_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JaWkoiLCJzdWIiOiJCWFJMODMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJyZXMgcmxvYyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ0MTM1NDYzLCJpYXQiOjE3MTI1OTk0NjN9.p3VUeVq3_E-kEy2xYvuMLmnlp2wGKHjzq8LhLpRQCRc'
sub_017_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1MyTDUiLCJzdWIiOiJCUUxaQkQiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJsb2MgcnJlcyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ0MTM0MDQ2LCJpYXQiOjE3MTI1OTgwNDZ9.oQ3Zjav-smW8p4ZQ_V4Bzsa7izn3uimmBJ4KINmHWsE'


data_df = (
    pl.read_csv('/Users/edeneldar/Downloads/Subjects List MDMA.csv')
    .with_columns(
        sub_name = pl.col('Id').str.split('_').list.slice(0,2)
    )
    .with_columns(
        sub_name = pl.col('sub_name').list.join('_')
    )
    .with_columns(
        token = pl.when(pl.col('sub_name') == 'sub_011').then(pl.lit(sub_011_token))
        .when(pl.col('sub_name') == 'sub_012').then(pl.lit(sub_012_token))
        .when(pl.col('sub_name') == 'sub_015').then(pl.lit(sub_015_token))
        .when(pl.col('sub_name') == 'sub_016').then(pl.lit(sub_016_token))
        .when(pl.col('sub_name') == 'sub_017').then(pl.lit(sub_017_token)),
        ExperimentStart = pl.col('ExperimentStartDate').str.strptime(pl.Date, "%d/%m/%Y"),
        ExperimentEnd = pl.col('ExperimentEndDate').str.strptime(pl.Date, "%d/%m/%Y")
    )
    .with_columns(
        ExperimentStart = pl.col('ExperimentStart').dt.strftime("%Y-%m-%d"),
        ExperimentEnd = pl.col('ExperimentEnd').dt.strftime("%Y-%m-%d")
    )
)


for sub in data_df.iter_rows():
    if not sub[0] in ['sub_011_T4','sub_012_T2', 'sub_012_T3','sub_012_T4', 'sub_015_T4','sub_016_T3', 'sub_016_T4', 'sub_017_T3']:
        continue
    sub_name = sub[0]
    token = sub[21]
    start_date = sub[22]
    end_date = sub[23]
    path = Path('/Users/edeneldar/JS learn/fitbitApiCall/data').joinpath(sub_name)
    if not path.exists():
        path.mkdir()
    saveAsZIP(path, token, start_date, end_date, sub_name)
    

ti.sleep(15)

# Nova131_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JXVEQiLCJzdWIiOiJCVDRHUEIiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBycHJvIHJudXQgcnNsZSByYWN0IHJyZXMgcmxvYyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ1MjQyMjYyLCJpYXQiOjE3MTM3MDYyNjZ9.wmP3VhhaoxoGxUEqALN284VW--DQpR7Tum37CdHLX7I'
# HotMobile_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JaWk4iLCJzdWIiOiJCVEJOTE4iLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJsb2MgcnJlcyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQzNjg4NTczLCJpYXQiOjE3MTIxNTI1NzN9.yoQxXR2wN1syDQVIjYXIJIPVKG8nrELMfsV3vVWt7P4'
# # Einat = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JYM1AiLCJzdWIiOiJCVEJXUlYiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcm94eSBybnV0IHJwcm8gcnNsZSByYWN0IHJsb2MgcnJlcyByd2VpIHJociBydGVtIiwiZXhwIjoxNzQ2NjIyNTYxLCJpYXQiOjE3MTUwODY1NjF9.dAP30ih8LTiXK1fepCRXCBRxjPYe-f0l2uyeVLEJz8w'


# Nova131_start_date = '2024-03-07'
# Nova131_end_date = '2024-04-21'

# HotMobile_start_date = '2024-02-22'
# HotMobile_end_date = '2024-03-22'

# # Einat_start_date = '2024-05-01'
# # Einat_end_date = '2024-05-26'

# Nova_path = Path('/Users/edeneldar/JS learn/fitbitApiCall/data/E326')
# HotMobile_path = Path('/Users/edeneldar/JS learn/fitbitApiCall/data/E352')
# # Einat_path = Path('/Users/edeneldar/JS learn/fitbitApiCall/data/nova_042')
# # if not Einat_path.exists():
# #     Einat_path.mkdir()
# if not Nova_path.exists():
#     Nova_path.mkdir()
# if not HotMobile_path.exists():
#     HotMobile_path.mkdir()


# outputPath = Path('/Users/edeneldar/JS learn/fitbitApiCall/data')
# subjects = ['E326', 'E352', 'nova_042']


# calories_responses_nova = getCalories(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# # calories_responses_einat = getCalories(Einat_path, Einat_start_date, Einat_end_date, Einat)
# calories_responses_hotmobile = getCalories(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)

# sleep_responses_nova = getSleepfiles(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# sleep_responses_hotmobile = getSleepfiles(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)

# steps_responses_nova = getfilesSteps(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# steps_responses_hotmobile = getfilesSteps(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)
# steps_responses_einat = getfilesSteps(Einat_path, Einat_start_date, Einat_end_date, Einat)

# clean_steps(Einat_path)
# clean_steps(Nova_path)
# clean_steps(HotMobile_path)

# HR_responses_nova = getfilesHR(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# HR_responses_hotmobile = getfilesHR(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)

# Respiratory_responses_nova = generateRespiratoryRateCSV(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# # Respiratory_responses_hotmobile = generateRespiratoryRateCSV(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)

# HRV_responses_nova = getHRVfiles(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# HRV_responses_hotmobile = getHRVfiles(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)

# Temp_responses_nova = getTempFiles(Nova_path, Nova131_start_date, Nova131_end_date, Nova131_token)
# Temp_responses_hotmobile = getTempFiles(HotMobile_path, HotMobile_start_date, HotMobile_end_date, HotMobile_token)


# ZipPath = Path('/Users/edeneldar/JS learn/fitbitApiCall/data')
        


# saveAsZIP(ZipPath)





# getfilesSteps(correctedPathSteps)


# getfilesHR(correctedPathHR)

# generateRespiratoryRateCSV(correctedPathBR)
                    
# getTempFiles(inputPathHotMobile)
# getTempFiles(inputPathNova)
                    
# getHRVfiles(inputPathHotMobile)
# getHRVfiles(inputPathNova, '2024-03-07', '2024-04-06')
# getHRVfiles(inputPathHotMobile, '2024-03-07', '2024-03-22')
                    