from typing import Optional, Union
from pathlib import Path
import pandas as pd
import configparser
import datetime
# from md2pdf.core import md2pdf
import shutil
import os
import numpy as np
import glob
import re
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
sys.path.append(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages')

from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term


def add_feature_2_column(subject_heart_rate_steps_df, sub_id):
    
    return pd.DataFrame()

def concate_to_old(term, path, new_df):
    old_path = new_get_latest_file_by_term(term, root=path)
    if new_df.empty and old_path != Path("NOT_EXISTS_PATH"):
        return pd.read_csv(old_path)
    if old_path.exists():
        old_df = pd.read_csv(old_path)
        
        for subject in new_df['Id'].unique():
            if subject in old_df['Id'].unique():
                old_df = old_df[old_df['Id'] != subject]
        new_df = pd.concat([old_df, new_df])
        new_df = new_df.sort_values(by=['Id'])
        return new_df
    else:
        return new_df

def calculate_sleep_halfs(df):
    """
    Take the df that include feature column and return the df with the halfs of the sleep columns.
    the index column is the DateAndMinute column.
    The halfs columns are: 'first_half_of_sleep', 'last_half_of_sleep'
    :param df: DataFrame
    :return: DataFrame
    
    """
    sleep_duration = 0
    for index, row in df.iterrows():
            
        if row['Feature'] == 'sleep' and 'ValidSleep':
            sleep_duration += 1


    half = sleep_duration / 2
    first_half = half
    last_half = sleep_duration

    i = 0 


    for idx, row in df.iterrows():
        if i < first_half:
            df.loc[idx, 'first_half_of_sleep'] = True
            df.loc[idx, 'last_half_of_sleep'] = False
        elif first_half <= i < last_half:
            df.loc[idx, 'last_half_of_sleep'] = True
            df.loc[idx, 'first_half_of_sleep'] = False
        i += 1

    return df

def calculate_sleep_thirds(df):
    """
    Take the df that include feature column and return the df with the thirds of the sleep columns.
    the index column is the DateAndMinute column.
    The thirds columns are: 'first_third_of_sleep', 'second_third_of_sleep', 'last_third_of_sleep'
    :param df: DataFrame
    :return: DataFrame
    """
    # Iterate day by day and calculate the thirds of the sleep time
    # Get the rows of the current date
    sleep_duration = 0
    for index, row in df.iterrows():
        
        # If the feature is 'sleep' 
        if row['Feature'] == 'sleep' and 'ValidSleep':
            sleep_duration += 1
    # Calculate the thirds of the sleep time
    third = sleep_duration / 3
    first_third = third
    second_third = third * 2
    last_third = sleep_duration

    i = 0

    # Set the thirds of the sleep time to the DataFrame
    for idx, row in df.iterrows():

        if i < first_third:
            df.loc[idx, 'first_third_of_sleep'] = True
            df.loc[idx, 'second_third_of_sleep'] = False
            df.loc[idx, 'last_third_of_sleep'] = False
        elif first_third <= i < second_third:
            df.loc[idx, 'second_third_of_sleep'] = True
            df.loc[idx, 'first_third_of_sleep'] = False
            df.loc[idx, 'last_third_of_sleep'] = False
        elif second_third <= i < last_third:
            df.loc[idx, 'last_third_of_sleep'] = True
            df.loc[idx, 'first_third_of_sleep'] = False
            df.loc[idx, 'second_third_of_sleep'] = False
        i += 1


    return df


def get_subject_dates_of_experiment(subject: str, METADATA_PATH) -> pd.DataFrame:
    """
    Description:
        read the subjects dates of experiment file and return the dates of the specified subject.
        ignore rows that have nan value in any column.
    :param subject: sub_xxx String
    :return: A pandas DataFrame (one row) containing the experiment start and end dates for the specified subject.
    """
    # Load the subjects dates of experiment file.
    subjects_dates_path = METADATA_PATH.joinpath('Subjects Dates.csv')

    not_in_israel_path = r'H:\Shared drives\Hevra - Roy Salomon Lab - HIPAA\SafeHeart_research_(NovaHelp)\Experiment\not_in_israel_df.xlsx'
    # subjects_dates_df = pd.read_csv(subjects_dates_path,
    #                                 usecols=['Id', 'ExperimentStartDate', 'ExperimentEndDate',
    #                                            'ExperimentStartTime', 'ExperimentEndTime'],
    #                                 parse_dates=['ExperimentStartDate', 'ExperimentEndDate'],
    #                                 date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    try:
        subjects_dates_df = pd.read_csv(subjects_dates_path,
                                        usecols=['Id', 'ExperimentStartDate', 'ExperimentEndDate',
                                                'ExperimentStartTime', 'ExperimentEndTime'],
                                        parse_dates=['ExperimentStartDate', 'ExperimentEndDate'],
                                        date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'),
                                        encoding='ISO-8859-1')
    except:
        subjects_dates_df = pd.read_csv(subjects_dates_path,
                                        usecols=['Id', 'ExperimentStartDate', 'ExperimentEndDate',
                                                'ExperimentStartTime', 'ExperimentEndTime'],
                                        parse_dates=['ExperimentStartDate', 'ExperimentEndDate'],
                                        date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    
    try:
        if os.path.exists(not_in_israel_path):
            not_in_israel_df = pd.read_excel(not_in_israel_path,
                                             usecols=['Id', 'start_date', 'end_date', 'state','start_date_1','end_date_1','state_1'],
                                                parse_dates=['start_date', 'end_date'],
                                                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        else:
            not_in_israel_df = pd.DataFrame(columns=['Id', 'start_date', 'end_date', 'state','start_date_1','end_date_1','state_1'])
    except:
        not_in_israel_df = pd.DataFrame(columns=['Id', 'start_date', 'end_date', 'state','start_date_1','end_date_1','state_1'])

    # Merge the two dataframes on the 'Id' column
    subjects_dates_df = pd.merge(subjects_dates_df, not_in_israel_df, on='Id', how='left')
    for idx, row in subjects_dates_df.iterrows():
        if row['start_date'] is not pd.NaT and row['state'] is np.nan:
            subjects_dates_df.loc[idx, 'state'] = 'NotIsrael'
        elif row['start_date'] is pd.NaT:
            subjects_dates_df.loc[idx, 'state'] = 'InIsrael'

    


    # ignore rows that have nan value in any column
    # subjects_dates_df = subjects_dates_df.dropna()
    # convert time to datetime.time
    subjects_dates_df['ExperimentStartTime'] = pd.to_datetime(subjects_dates_df['ExperimentStartTime']).dt.time
    subjects_dates_df['ExperimentEndTime'] = pd.to_datetime(subjects_dates_df['ExperimentEndTime']).dt.time
    # merge date and time one cell
    subjects_dates_df['ExperimentStartDateTime'] = pd.to_datetime(
        subjects_dates_df['ExperimentStartDate'].dt.strftime('%Y-%m-%d') + ' ' + subjects_dates_df['ExperimentStartTime'].astype(
            str))
    subjects_dates_df['ExperimentEndDateTime'] = pd.to_datetime(
        subjects_dates_df['ExperimentEndDate'].dt.strftime('%Y-%m-%d') + ' ' + subjects_dates_df['ExperimentEndTime'].astype(
            str))
    
    not_in_israel_mask = [x for _, x in subjects_dates_df.iterrows() if x['start_date'] != 'NaT' and not re.search(r'(קפריסין|InIsrael)', x['state'])]

    subjects_dates_df['state_1'] = subjects_dates_df['state_1'].fillna('InIsrael')

    not_in_israel_mask_second = [x for _, x in subjects_dates_df.iterrows() if x['start_date_1'] != 'NaT' and not re.search(r'(קפריסין|InIsrael)', x['state_1'])]

    subjects_dates_df['NotInIsrael'] = subjects_dates_df['Id'].apply(lambda x: any(x == row['Id'] for row in not_in_israel_mask))
    subjects_dates_df['NotInIsrael_1'] = subjects_dates_df['Id'].apply(lambda x: any(x == row['Id'] for row in not_in_israel_mask_second))

    # rename the columns
    subjects_dates_df = subjects_dates_df.rename(columns={'start_date': 'NotInIsraelStartDate',
                                                            'end_date': 'NotInIsraelEndDate',
                                                            'start_date_1': 'NotInIsraelStartDate_1',
                                                            'end_date_1': 'NotInIsraelEndDate_1'})

    return subjects_dates_df.loc[subjects_dates_df['Id'] == subject][['Id', 'ExperimentStartDateTime', 'ExperimentEndDateTime', 'ExperimentStartDate', 'ExperimentEndDate'
                                                                        , 'NotInIsrael', 'NotInIsraelStartDate', 'NotInIsraelEndDate','NotInIsrael_1', 'NotInIsraelStartDate_1', 'NotInIsraelEndDate_1']]


def valid_sleep(subject_sleep_df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function gets a DataFrame with sleep data of a subject and returns a DataFrame column with the valid sleep data.
    Valid sleep data is defined as:
                1. Between 20:00 to 08:00
                2. More than 3 hours of sleep
                3. If there are multiple sleep at night, the function will set True for the longest sleep and False for the rest.
    # the column names that the dataframe must have are:
    SleepStartTime
    DateOfSleepEvening
    If there are multiple sleep at night, the function will set True for the longest sleep and False for the rest.
    It does it by group by date and select the longest sleep at night.
    :param subject_sleep_df: DataFrame with sleep data of a subject
    :return: DataFrame column with the valid sleep column (boolean column)

    '''
    # Create a copy of the original DataFrame within the function and modify the copy.
    subject_sleep_df_copy = subject_sleep_df.copy()
    # Create a boolean mask to identify rows with valid sleep times based on the hour between 20:00 in the evening to 08:00 at the morning.
    mask_valid_rows_by_hour = (subject_sleep_df_copy['SleepStartTime'].dt.hour < 8) | (subject_sleep_df_copy['SleepStartTime'].dt.hour >= 20)
    # Add a new column to the DataFrame indicating whether each row has valid sleep times
    subject_sleep_df_copy['ValidSleep'] = mask_valid_rows_by_hour
    valid_subject_sleep_df_copy = subject_sleep_df_copy.loc[subject_sleep_df_copy['ValidSleep'] == True]
    # group by DateOfSleepEvening and iterate over each group
    for date, group in valid_subject_sleep_df_copy.groupby('DateOfSleepEvening'):
        max_index = group['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].idxmax()
        # Set False for all rows in the group
        for idx in group.index:
            subject_sleep_df_copy.at[idx, 'ValidSleep'] = False
        # Set True for the row with the longest sleep duration
        subject_sleep_df_copy.at[max_index, 'ValidSleep'] = True
    # Validate that each valid sleep has more than 3 hours of sleep, if not set False in ValidSleep column
    subject_sleep_df_copy.loc[
        (subject_sleep_df_copy['ValidSleep'] == True) & (subject_sleep_df_copy['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'] <= 180),'ValidSleep'] = False

    return subject_sleep_df_copy['ValidSleep']


def config_project(project_name, now):
    config_path = Path(r'G:\Shared drives\AdmonPsy - Lab Resources\Python Scripts\current_project_details.ini')
    config = load_project_details(config_path)

    project_path = Path(config['project_details']['project_path'])

    log_dir = project_path /'Metadata' / 'Logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(f'{log_dir.joinpath(project_path.stem)} {project_name} {now}.txt', 'w')

    return project_path

def load_project_details(config_path: Path = None):
    """
    Description:
        Load the project details from the config file:
        project_path: The path to the project folder.
    """
    if not config_path:
        raise Exception('config_path parameter is required')
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config:
        raise Exception(f'Could not load config file. Please check config name. it should be {details_config}')
    if len(config.sections()) == 0:
        raise Exception(f'The config file is empty. Please check config name. it should be {details_config}')
    return config

def load_config(METADATA_PATH: Path):
    """
    Load the config file
    Returns:

    """
    # The config file contains the names of the events
    config = configparser.ConfigParser()
    config_path = METADATA_PATH.joinpath('config.ini')
    config.read(config_path)
    if not config:
        raise Exception(f'Could not load config file. Please check config name. it should be {config_path}')
    if len(config.sections()) == 0:
        raise Exception(f'The config file is empty or you need to rename it. Please check config name. it should be {config_path}')

    return config

def write_config(METADATA_PATH: Path, section, option, value):
    """
    Only wrrite to the section that given in the function
    """
    config = configparser.ConfigParser()
    config_path = METADATA_PATH.joinpath('config.ini')
    config.read(config_path)
    if section not in config.sections():
        config[section] = {}

    config.set(section, option, value)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def get_subject_dates_of_experiment_old(subject: str, METADATA_PATH: Path) -> pd.DataFrame:
    """
    Description:
        read the subjects dates of experiment file and return the dates of the specified subject.
        ignore rows that have nan value in any column.
    :param subject: sub_xxx String
    :return: A pandas DataFrame (one row) containing the experiment start and end dates for the specified subject.
    """
    # Load the subjects dates of experiment file.
    subjects_dates_path = METADATA_PATH.joinpath('Subjects Dates.csv')

    subjects_dates_df = pd.read_csv(subjects_dates_path,
                                    usecols=['Id', 'ExperimentStartDate', 'ExperimentEndDate',
                                               'ExperimentStartTime', 'ExperimentEndTime'],
                                    parse_dates=['ExperimentStartDate', 'ExperimentEndDate'],
                                    date_format='%d/%m/%Y')
                                    # date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    # ignore rows that have nan value in any column
    subjects_dates_df = subjects_dates_df.dropna()
    # convert time to datetime.time
    subjects_dates_df['ExperimentStartTime'] = pd.to_datetime(subjects_dates_df['ExperimentStartTime'], format='%H:%M').dt.time
    subjects_dates_df['ExperimentEndTime'] = pd.to_datetime(subjects_dates_df['ExperimentEndTime'], format='%H:%M').dt.time
    # merge date and time one cell
    subjects_dates_df['ExperimentStartDateTime'] = pd.to_datetime(
        subjects_dates_df['ExperimentStartDate'].dt.strftime('%Y-%m-%d') + ' ' + subjects_dates_df['ExperimentStartTime'].astype(
            str))
    subjects_dates_df['ExperimentEndDateTime'] = pd.to_datetime(
        subjects_dates_df['ExperimentEndDate'].dt.strftime('%Y-%m-%d') + ' ' + subjects_dates_df['ExperimentEndTime'].astype(
            str))


    return subjects_dates_df.loc[subjects_dates_df['Id'] == subject][['Id', 'ExperimentStartDateTime', 'ExperimentEndDateTime', 'ExperimentStartDate', 'ExperimentEndDate']]


def split_event(file: Path, event_to_split: str, num_of_splits: int):
    """
    Description:
        split event and create a new file with added rows for the event.
    Args:
        file:
        event_to_split:
        num_of_splits:

    Returns:
    """
    # Iterate through each line of the file
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # find the event_to_split which is in the regex F\d
            regex_to_find = event_to_split

            print(line)


def loop_event_files_and_split(project_file: Path,  event_to_split, num_of_splits):
    """
    Description:
        loop each subject folder and split_event on the event file.
    Args:
        project_file:
        file:
        event_to_split:
        num_of_splits:

    Returns:
    """
    data_path = project_file.joinpath('Data')
    for subject in os.listdir(data_path):
        subject_path = data_path.joinpath(subject)
        for file in os.listdir(subject_path):
            if re.fullmatch('sub_[0-9]{3}_events.txt', file):
                split_event(subject_path.joinpath(file), event_to_split, num_of_splits)


def copy_files_from_src_to_dst(src: Path, dst: Path, new_folder: Path = ""):
    """
    Description:
        Copy files from src to dst.
        files that will be copied:
            - sub_[0-9]{3}_data.txt
            - sub_[0-9]{3}_events.txt

    Args:
        src: Path to the source folder.
        dst: Path to the destination folder.
        new_folder: parent folder for the copied files.

    Returns: None
    """
    for subject_folder in os.listdir(src):
        if re.fullmatch('sub_[0-9]{3}', subject_folder):
            subject_folder_path = src.joinpath(subject_folder).joinpath('HRV')
            if subject_folder_path.exists():
                for f in os.listdir(subject_folder_path):
                    if re.fullmatch('sub_[0-9]{3}_data.txt', f) and f.endswith('.txt'):
                        if not dst.joinpath(subject_folder).joinpath(new_folder).exists():
                            os.makedirs(dst.joinpath(subject_folder).joinpath(new_folder))
                        shutil.copy(subject_folder_path.joinpath(f), dst.joinpath(subject_folder).joinpath(new_folder).joinpath(f))
                    if re.fullmatch('sub_[0-9]{3}_events.txt', f) and f.endswith('.txt'):
                        if not dst.joinpath(subject_folder).joinpath(new_folder).exists():
                            os.makedirs(dst.joinpath(subject_folder).joinpath(new_folder))
                        shutil.copy(subject_folder_path.joinpath(f), dst.joinpath(subject_folder).joinpath(new_folder).joinpath(f))
        if re.fullmatch('s_[0-9]{3}', subject_folder):
            subject_folder_path = src.joinpath(subject_folder)
            for folder in os.listdir(subject_folder_path):
                if folder == 'MyFitbitData':
                    subject_folder_path = subject_folder_path.joinpath(folder)
                    for folder in os.listdir(subject_folder_path):
                        if folder == 'idfidf':
                            subject_fitbit_folder_path = subject_folder_path.joinpath(folder)
                            dst_folder = dst.joinpath('RAW_DATA').joinpath('sub_'+subject_folder[-3:]).joinpath(new_folder)
                            # Copy subject_fitbit_folder_path folder
                            shutil.copytree(subject_fitbit_folder_path, dst_folder)



def change_MINDWARE_name_to_standard(file_path: Path = None, file_type: str = ''):
    """
    ! Called by fix_file_names() !
    Description:
        This function fix the file names of the MINDWARE device.

    Args:
        file_path:
        file_type:

    Returns:

    """
    if file_type == '':
        raise Exception('file_type is required')

    file_path = Path(file_path)
    file_name = file_path.stem
    file_ext = file_path.suffix
    if file_type == 'data':
        # sub_001 --> sub_001_data.txt
        if re.fullmatch('sub_[0-9]{3}', file_name):
            if not file_name.endswith('data') and file_ext == '.txt':
                shutil.copy(file_path, file_path.parent.joinpath(file_name + '_data.txt'))

        # s_001 --> sub_001_data.txt
        if re.fullmatch('s_[0-9]{3}', file_name):
            file_path.rename(file_path.parent.joinpath(file_name.replace('s_', 'sub_')))
            if not file_name.endswith('data') and file_ext == '.txt':
                shutil.copy(file_path, file_path.parent.joinpath(file_name + '_data.txt'))
    # sub_001_event --> sub_001_events.txt
    elif file_type == 'events':
        if re.fullmatch('sub_[0-9]{3}_event', file_name):
            if not file_name.endswith('events') and file_ext == '.txt':
                shutil.copy(file_path, file_path.parent.joinpath(file_name + 's.txt'))
    else:
        raise Exception('file_type should be "data" or "events"')


def fix_file_names(data_path):
    """
    This function fix the file names of the MINDWARE device.
    Its loop over all the subjects and search for files inside MINDWARE folder and if its find files that relevant to
    MINDWARE device and relevant to code (data or events) it will change the name of the file to the standard name and copy it to
    :param data_path:
    :return:
    """
    SUBJECT_FOLDER_FORMAT = 'sub_[0-9]{3}'
    for subject in os.listdir(data_path):
        if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
            continue

        subject_folder = data_path.joinpath(subject)
        for device in os.listdir(subject_folder):
            if device != 'HRV':
                continue
            HRV_subject_folder = subject_folder.joinpath(device)
            for file in os.listdir(HRV_subject_folder):
                file_path = HRV_subject_folder.joinpath(file)
                change_MINDWARE_name_to_standard(file_path, 'data')
                change_MINDWARE_name_to_standard(file_path, 'events')

# NEW cersion of declare_project_global_variables
def declare_project_global_variables(project_path):
    # Folders
    DATA_PATH = Path(project_path.joinpath('Data'))
    OUTPUT_PATH = Path(project_path.joinpath('Outputs'))
    AGGREGATED_OUTPUT_PATH = Path(OUTPUT_PATH.joinpath('Aggregated Output'))
    ARCHIVE_PATH = Path(OUTPUT_PATH.joinpath('Archive'))
    METADATA_PATH = Path(project_path.joinpath('Metadata'))
    
        
    # Formats
    SUBJECT_FOLDER_FORMAT = 'sub_[0-9]{3}' # Set the format for the folder names (regular expression)
    return DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT


# NEW cersion of declare_project_global_variables
def declare_project_global_variables_HR_TEST(project_path):
    # Folders
    DATA_PATH = project_path.joinpath('Data_HR_TEST')
    OUTPUT_PATH = project_path.joinpath('Outputs_HR_TEST')
    AGGREGATED_OUTPUT_PATH = OUTPUT_PATH.joinpath('Aggregated Output')
    ARCHIVE_PATH = OUTPUT_PATH.joinpath('Archive')
    METADATA_PATH = project_path.joinpath('Metadata')
    
        
    # Formats
    SUBJECT_FOLDER_FORMAT = 'sub_[0-9]{3}' # Set the format for the folder names (regular expression)
    return DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT

# NEW cersion of declare_project_global_variables
def declare_project_global_variables_NOT_IL_TEST(project_path):
    # Folders
    DATA_PATH = project_path.joinpath('Data_HR_TEST')
    OUTPUT_PATH = project_path.joinpath('Outputs_NOT_IL')
    AGGREGATED_OUTPUT_PATH = OUTPUT_PATH.joinpath('Aggregated Output')
    ARCHIVE_PATH = OUTPUT_PATH.joinpath('Archive')
    METADATA_PATH = project_path.joinpath('Metadata')
    
        
    # Formats
    SUBJECT_FOLDER_FORMAT = 'sub_[0-9]{3}' # Set the format for the folder names (regular expression)
    return DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT


# OLD cersion of declare_project_global_variables

# def declare_project_global_variables(project_path):
#     # Folders
#     global DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH
#     DATA_PATH = project_path.joinpath('Data')
#     OUTPUT_PATH = project_path.joinpath('Outputs')
#     AGGREGATED_OUTPUT_PATH = OUTPUT_PATH.joinpath('Aggregated Output')
#     ARCHIVE_PATH = OUTPUT_PATH.joinpath('Archive')
#     METADATA_PATH = project_path.joinpath('Metadata')
# 
#     # Formats
#     global SUBJECT_FOLDER_FORMAT
#     SUBJECT_FOLDER_FORMAT = 'sub_[0-9]{3}' # Set the format for the folder names (regular expression)


    

def get_latest_file_by_term(term: str, subject: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """
    Description:
        Searching the last created file with <term> that indicates on specific file format to search.
    Parameters:
        term: String (string the indicates the file format.)
        subject: if the file is in subject's folder, the caller should specify the subject Id.
    Assumptions:
        files in Aggregated folder is in the format: "<term> Aggregated yyyy-mm-dd_HH-MM-SS"
    Return:
        Path (if not found, return path that not exists.)
    """
    # Define the regular expression pattern to match the date
    path = ''
    if term == 'Sleep All Subjects':
        pattern = r'^Sleep All Subjects (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where sleep file is.
        path = root
    elif term == 'Steps':
        pattern = r'^Steps Aggregated (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where sleep file is.
        path = root
    elif term == 'Heart Rate':
        if subject is None:
            raise Exception('Subject Id is required for Heart Rate file.')
        pattern = r'^.*\d{3} Heart Rate (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv$'
        # Set up path to the folder where Heart Rate file is.
        path = root.joinpath(subject)
    elif term == 'EDA':
        if subject is None:
            raise Exception('Subject Id is required for EDA file.')
        pattern = r'^.*\d{3} EDA (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv$'
        # Set up path to the folder where EDA file is.
        path = root.joinpath(subject)
    elif term == 'HRV Temperature Respiratory':
        if subject is None:
            raise Exception('Subject Id is required for HRV Temperature Respiratory file.')
        pattern = r'^.*\d{3} HRV Temperature Respiratory At Sleep (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv$'
        # Set up path to the folder where HRV Temperature Respiratory file is.
        path = root.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Heart Rate and Steps and Sleep Aggregated':
        pattern = r'^.*\d{3} Heart Rate and Steps and Sleep Aggregated (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv$'
        if subject is None:
            raise Exception('Subject Id is required for Heart Rate and Steps and Sleep Aggregated file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root.joinpath(subject)
    elif term == 'Metrics of Heart Rate By Activity':
        pattern = r'^.*\d{3} Metrics of Heart Rate By Activity \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.csv$'
        if subject is None:
            raise Exception('Subject Id is required for Metrics of Heart Rate By Activity file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Full Week Summary of Heart Rate Metrics By Activity':
        pattern = r'^Full Week Summary of Heart Rate Metrics By Activity (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Summary Of HRV Temperature Respiratory At Sleep':
        pattern = r'^Summary Of HRV Temperature Respiratory At Sleep (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'EDA Summary':
        pattern = r'^EDA Summary (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Final All Subjects Aggregation':
        pattern = r'^Final All Subjects Aggregation (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'EMA_raw':
        pattern = r'^.*\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.xlsx$'  # TODO: validate that it reads
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA with extra metrics':
        pattern = r'^.*\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_with_extra_metrics\.xlsx$'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA Daily Mean + Fitbit Night':
        # pattern match 'sub_032 EMA Daily Mean + Fitbit Night 2023-11-12_12-57-49.csv'
        pattern = r'^.*\d{3} EMA Daily Mean \+ Fitbit Night \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv$'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA metrics and diff':
        # pattern match 'sub_032 EMA Daily Mean + Fitbit Night 2023-11-12_12-57-49.csv'
        pattern =  r'^.*\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_with_extra_metrics_with_diff\.xlsx$'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA metrics and diff Updated':
        # transformation of EMA metrics and diff for fibro's subjects data after adding values with Random Forest algo
        pattern =  r'_EMA_Imputation.xlsx$'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)


    if path == '':
        raise Exception(f'No path was found for term: {term}')
    # Get only the relevant files
    files = [f for f in os.listdir(path) if re.search(pattern, f)]
    # Sort the files by their last created time
    files_sorted_by_date = sorted(files, key=lambda f: os.path.getctime(path.joinpath(f)), reverse=True)
    # Check if there are any files in the sorted list, and return the file with the latest date
    if files_sorted_by_date:
        return path.joinpath(files_sorted_by_date[0])

    # If no files were found, return a path that does not exists.
    if not subject:
        print(f'{pattern} not found')
    else:
        print(f'{pattern} not found for subject {subject}')
    return Path("NOT_EXISTS_PATH")

    

# define archive_project_data_subject that run before to_csv or to_excel in order to archive the data before overwrite it.
def archive_project_data_subject(file_path: Path = None, new_archive_folder: Path = None, subject: str = None, file_name: str = None):
    """
    Description:
        Archive the data of the subject before overwrite it.
    Args:
        file_path: Path to the project folder.
        new_archive_folder: The path to the new archive folder.
        subject: The subject Id.
        file_name: The file name to archive.
    return:
        None
    """
   
    
    # subject archive folder
    new_subject_archive_folder = new_archive_folder.joinpath(subject)
    
    # check if necessry to create a new folders for new subject or aggregate
    if not new_subject_archive_folder.exists():
        os.makedirs(new_subject_archive_folder)
        new_subject_archive_folder.mkdir(parents=True, exist_ok=True)
    
    #check if the file exist in the data folder
    if file_path.joinpath(file_name).exists():
        # move the only file to the archive folder.
        shutil.move(file_path.joinpath(file_name), new_subject_archive_folder)
        

# define archive_project_data_aggregated that run before to_csv or to_excel in order to archive the data before overwrite it.
def archive_project_data_aggregated(file_path: Path = None, new_archive_folder: Path = None, file_name: str = None):
    """
    Description:
        Archive the data of the subject before overwrite it.
    Args:
        file_path: Path to the project folder.
        new_archive_folder: The path to the new archive folder.
        file_name: The file name to archive.
    return:
        None
    """
    
    # aggregate archive folder
    new_aggregate_archive_folder = new_archive_folder.joinpath("Aggregated Output")
    
    # check if necessry to create a new folders for new aggregate
    if not new_aggregate_archive_folder.exists():
        os.makedirs(new_aggregate_archive_folder)
        new_aggregate_archive_folder.mkdir(parents=True, exist_ok=True)
        
    # Use glob to find all files that match the pattern
    for file in glob.glob(os.path.join(file_path, file_name)):
        # Move each file to the new archive folder
        shutil.move(file, new_aggregate_archive_folder)   
    # move the file to the archive folder.
    
    
    # move the last file with the same name to the archive folder.


def output_record(output_path: Path = None, subject: Optional[str] = None, user_name: str = None, date: str = None):
    """ 
    Input - output folder of the specific project.
    description - define a function that open a new folder in the output folder,
    ask the user for his name and call the folder by the user name 
    with addition of the exact date and time of the creation of the folder.
    output - a new output path that direct to the new folder.
    example - "{user_name} {date} {time}"

    """
    
    #define the new folder name
    folder_name = f'{user_name} {date}'
    if subject != None:
        output_path = output_path.joinpath(subject)
        
    history_path = output_path.joinpath('History')
    first_history_path = history_path.joinpath("first_history")

    if not first_history_path.exists():
        os.makedirs(first_history_path)
        # Create a list of the files in the output folder except the History folder
        output_files = [f for f in os.listdir(output_path) if f != 'History']
        for item in output_files:
            if item == 'desktop.ini':
                continue
            s = os.path.join(output_path, item)
            d = os.path.join(first_history_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, ignore=ignore_history)
            else:
                shutil.copy2(s, d)

    # return the new path for the output folder
    return output_path.joinpath('History').joinpath(folder_name)
        


# check for duplications in the output folder
def check_for_duplications(output_path: Path = None, history_path: Path = None, file_name: str = None):
    """
    Description:
        Check for duplications in the output folder.
    Args:
        output_path: Path to the output folder.
        subject: The subject Id.
    return:
        None
    """
    
    intersection = set(os.listdir(output_path)).intersection(os.listdir(history_path))
    
    # compare the files in the output folder to the files in the history folder of the subject
    if file_name:
        # search the file in the output folder and the history folder of the subject
        if file_name in os.listdir(output_path) and file_name in os.listdir(history_path):
            # overwrite the file in the output folder that have the same name
            shutil.move(output_path.joinpath(file_name), history_path.joinpath(file_name))
    
    else:    
        # overwrite the files in the output folder that have the same name as the files in the history folder of the subjec
        for file in os.listdir(history_path):
            if file == 'first_history' or file == 'History' or file == 'desktop.ini' or r'^.*.pkl$' in file:
                continue
            shutil.copy(history_path.joinpath(file), output_path.joinpath(file))
        
def ignore_history(dir, filenames):
    return ['History'] if 'History' in filenames else []

def project_path_selection_window(project_path):
    """
    Description:
        Create a window for the user to select the project path.
        The keys are numbers and the values are the project names.
        After the user selects the project,
        there will be a message box that will show the selected project name and
        ask the user if the selection is correct.
    Args:
        config_section: The section in the config file that contains the project paths.
    Returns:
        The selected project number (dictionary key).
    """
    
    project_paths_dict = {}
    # Split the values under the project_path ( config section) by the "\" separator
    for i in project_path:
        # add to the dictionary the key and the value
        project_paths_dict[i] = project_path[i]
        # split by the separator and take the value of the string after the shared drive
        if "\Shared drives\\" in project_paths_dict[i]:
            project_paths_dict[i] = project_paths_dict[i].split("\Shared drives\\")[1]
            # Take the first string after the shared drive
            project_paths_dict[i] = project_paths_dict[i].split("\\")[0]
        
    # Create a window
    window = tk.Tk()
    window.title("Select Project")
    window.geometry("300x200")
    # if close the window without selecting a project exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    
    # Create a label
    label = tk.Label(window, text="Select Project")
    label.pack()
    
    # Create a listbox and adjust the size
    listbox = tk.Listbox(window, width=50)
    listbox.pack()
    
    # Insert the projects to the listbox
    for key, value in project_paths_dict.items():
        listbox.insert(tk.END, f"{key}. {value}")
    selected_project_number = None
    
    # Create a function that will be called when the user selects a project
    def onselect(event):
        nonlocal selected_project_number
        selected_project_number = int(listbox.get(listbox.curselection())[0])
        
    # Bind the function to the listbox
    listbox.bind('<<ListboxSelect>>', onselect)
    
    # Create a function that will be called when the user clicks the "Select" button
    def select_project():
        nonlocal selected_project_number
        if selected_project_number:
            # Validate the selection
            selected_project_name = project_paths_dict[f'{selected_project_number}']
            # Ask the user to confirm the selection. red bold text
            answer = messagebox.askyesno("Confirm Selection", f"Selected Project: {selected_project_name}. Is this correct?")
            
            # If the user confirms the selection, close the window else, let him select again
            if answer:
                window.destroy()
            else:
                selected_project_number = None
        else:
            messagebox.showinfo("Error", "Please select a project")
    
    # Create a button
    button = tk.Button(window, text="Select", command=select_project)
    button.pack()
    
    # Create a function that will pop up a message box with the selected project name to confirm the selection
    def confirm_selection():
        nonlocal selected_project_number
        if selected_project_number:
            selected_project_name = project_paths_dict[selected_project_number]
            answer = messagebox.askyesno("Confirm Selection", f"Selected Project: {selected_project_name}. Is this correct?")
            # If the user confirms the selection, close the window else, let him select again
            if answer:
                window.destroy()
            else:
                selected_project_number = None
                
                
                
    # Create a button
    confirm_button = tk.Button(window, text="Confirm Selection", command=confirm_selection)
    confirm_button.pack()
    
    
    # Run the window
    window.mainloop()
    if selected_project_number:
        return selected_project_number
    else:
        return project_path_selection_window(project_path)
    
    

# define a function that will create a window for the user name input
def user_name_input_window(runner):
    """
    Description:
        Create a window for the user to input his name.
        The user will be asked to input his name and then click the "Submit" button.
        After the user clicks the "Submit" button, the window will close and the user name will be returned.
    Returns:
        The user name.
    """
    # Create a window
    window = tk.Tk()
    window.title("User Name")
    window.geometry("300x200")
    # if close the window without selecting a project exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Enter Your Name")
    label.pack()
    user_name = runner
    
    # Create an entry with a default value
    entry = tk.Entry(window)
    entry.insert(0, user_name)
    entry.pack()
    
    
    # Create a function that will be called when the user clicks the "Submit" button
    def submit():
        nonlocal user_name
        # show the default user name
        user_name = entry.get()
        if user_name:
            window.destroy()
        else:
            messagebox.showinfo("Error", "Please enter your name")

    
    # Create a button
    button = tk.Button(window, text="Submit", command=submit)
    button.pack()
    
    # Run the window
    window.mainloop()
    return user_name

# define a function that will create a window for the user to select the subjects he wants to analyze take argument of list or pd.series
def select_subjects_window(subjects: Union[list, pd.Series], run_on, to_exclude):
    """
    Description:
        Create a window for the user to select the subjects he wants to analyze.
        The user will be asked to select the subjects from a listbox and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selected subjects will be returned.
    Args:
        subjects: A list of the subjects.
    Returns:
        The selected subjects.
    """
    # Create a window
    window = tk.Tk()
    window.title("Select Subjects")
    window.geometry("350x300")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Select Subjects")
    label.pack()
    
    # Create a listbox and adjust the size (height and width)
    listbox = tk.Listbox(window, width=50, selectmode=tk.MULTIPLE)
    listbox.pack()
    
    # Insert the subjects to the listbox
    for subject in subjects:
        listbox.insert(tk.END, subject)
    selected_subjects = []
    
    # Enter the pre selected subjects by the name of the subject
    try:
        try:
            # split the string by the separator and remove the white spaces
            run_on = run_on.split(',')
            run_on = [subject.strip() for subject in run_on]
            for subject in run_on:
                listbox.select_set(subjects.index(subject))     
        except:
            for subject in run_on:
                listbox.select_set(subjects.tolist().index(subject))
    except:
        pass
    
    
    try:
        if to_exclude != 'empty':
            to_exclude = to_exclude.split(',')
            to_exclude = [subject.strip() for subject in to_exclude]
        else:
            to_exclude = []
    except:
        pass
    
    
    # try:
    #     # if run_on = 'all' select all the subjects in the listbox and check if the user want to exclude some subjects
    #     if run_on == 'all':
    #         listbox.select_set(0, tk.END)
    #         if to_exclude != 'empty':
    #             to_exclude = to_exclude.split(',')
    #             to_exclude = [subject.strip() for subject in to_exclude]
    #             for subject in to_exclude:
    #                 listbox.selection_clear(subjects.tolist().index(subject))
    # except:
    #     pass
    # Create a function that will be called when the user clicks the "Select" button
    def select_subjects():
        nonlocal selected_subjects
        selected_subjects = [listbox.get(i) for i in listbox.curselection()]
        if selected_subjects:
            window.destroy()
        else:
            messagebox.showinfo("Error", "Please select at least one subject")
    
    # Create a function that will be called and deselect all the subjects in to_exclude
    def deselect_subjects():
        nonlocal selected_subjects
        if to_exclude != 'empty':
            selected_subjects = [listbox.get(i) for i in listbox.curselection()]
            if selected_subjects:
                for subject in to_exclude:
                    if subject in selected_subjects:
                        listbox.selection_clear(subjects.index(subject))
                    else:
                        continue
            else:
                messagebox.showinfo("Error", "Please select at least one subject")
        else:
            messagebox.showinfo("Error", "There are no subjects to exclude")
            
            
            
    
    
    def select_all():
        listbox.select_set(0, tk.END)
        
    def deselect_all():
        listbox.selection_clear(0, tk.END)
    
    
    
    # Create a button
    select_all_button = tk.Button(window, text="Select All", command=select_all)
    select_all_button.pack()
    
    deselect_all_button = tk.Button(window, text="Deselect All", command=deselect_all)
    deselect_all_button.pack()
    
    deselect_button_to_exclude = tk.Button(window, text="Deselect subjects to exclude", command=deselect_subjects)
    deselect_button_to_exclude.pack()
    
    button = tk.Button(window, text="Select", command=select_subjects)
    button.pack()
    
    
    
    # Run the window
    window.mainloop()
    return selected_subjects


# define a function that will create a window for the user to select if he wants to copy the data from the source data folder to the project data folder
def copy_data_window():
    """
    Description:
        Create a window for the user to select if he wants to copy the data from the source data folder to the project data folder.
        The user will be asked to select "Yes" or "No" and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selection will be returned.
    Returns:
        The selection.
    """
    # Create a window
    window = tk.Tk()
    window.title("Copy Data")
    window.geometry("450x100")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Any new data to copy?")
    label.pack()
    
    # Create 2 buttons Yes and No
    selection = None
    def yes():
        nonlocal selection
        selection = True
        window.destroy()
        
    def no():
        nonlocal selection
        selection = False
        window.destroy()
        
    # Create a the two buttons in distance from each other in the same line but close to the center
    yes_button = tk.Button(window, text="Yes", command=yes)
    yes_button.pack(side=tk.LEFT, padx=50)
    no_button = tk.Button(window, text="No", command=no)
    no_button.pack(side=tk.RIGHT, padx=50)
    
    
    # Run the window
    window.mainloop()
    return selection


# define a function that will create a window for the user to select the subjects to copy the data from the source data folder to the project data folder
def select_subjects_to_copy_data_window(subjects: Union[list, pd.Series]):
    """
    Description:
        Create a window for the user to select the subjects he wants to copy the data from the source data folder to the project data folder.
        The user will be asked to select the subjects from a listbox and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selected subjects will be returned.
    Args:
        subjects: A list of the subjects.
    Returns:
        The selected subjects.
    """
    # Create a window
    window = tk.Tk()
    window.title("Select Subjects to Copy Data")
    window.geometry("350x300")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Select Subjects to Copy Data")
    label.pack()
    
    # Create a listbox and adjust the size (height and width)
    listbox = tk.Listbox(window, width=50, selectmode=tk.MULTIPLE)
    listbox.pack()
    
    # Insert the subjects to the listbox
    for subject in subjects:
        listbox.insert(tk.END, subject)
    selected_subjects = []
    
    # Create a function that will be called when the user clicks the "Select" button
    def select_subjects():
        nonlocal selected_subjects
        selected_subjects = [listbox.get(i) for i in listbox.curselection()]
        if selected_subjects:
            window.destroy()
        else:
            messagebox.showinfo("Error", "Please select at least one subject")
    
    def select_all():
        listbox.select_set(0, tk.END)
        
    def deselect_all():
        listbox.selection_clear(0, tk.END)
    
    # Create a button
    select_all_button = tk.Button(window, text="Select All", command=select_all)
    select_all_button.pack()
    
    deselect_all_button = tk.Button(window, text="Deselect All", command=deselect_all)
    deselect_all_button.pack()
    
    button = tk.Button(window, text="Select", command=select_subjects)
    button.pack()
    
    
    # Run the window
    window.mainloop()
    return selected_subjects

# define a function that will check if there is an empty folder in the path and add the string "empty" to the folder name
def check_empty_folder(path: Path, remove_empty_folders):
    """
    Description:
        Check if there is an empty folder in the path and add the string "empty" to the folder name.
    Args:
        path: The path to check.
    """

    try:
        # Get the folders in the path
        folders = [f for f in os.listdir(path) if os.path.isdir(path.joinpath(f))]
        # Check if there are any empty folders
        for folder in folders:
            if not os.listdir(path.joinpath(folder)):
                if remove_empty_folders:
                    shutil.rmtree(path.joinpath(folder))
                else:
                    if 'empty' not in folder:
                        os.rename(path.joinpath(folder), path.joinpath(f"{folder} empty"))
            else:
                if 'empty' in folder:
                    os.rename(path.joinpath(folder), path.joinpath(folder.replace(' empty', '')))
    except:
        return             
                
                  
# define a function that will create a window for the user to select if he wants to remove the empty folders at the end of the process
def remove_empty_folders_window():
    """
    Description:
        Create a window for the user to select if he wants to remove the empty folders at the end of the process.
        The user will be asked to select "Yes" or "No" and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selection will be returned.
    Returns:
        The selection.
    """
    # Create a window
    window = tk.Tk()
    window.title("Remove Empty Folders")
    window.geometry("450x100")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Remove Empty Folders at the end of the process?")
    label.pack()
    
    # Create 2 buttons Yes and No
    selection = None
    def yes():
        nonlocal selection
        selection = True
        window.destroy()
        
    def no():
        nonlocal selection
        selection = False
        window.destroy()
        
    # Create a the two buttons in distance from each other in the same line but close to the center
    yes_button = tk.Button(window, text="Yes", command=yes)
    yes_button.pack(side=tk.LEFT, padx=50)
    
    no_button = tk.Button(window, text="No", command=no)
    no_button.pack(side=tk.RIGHT, padx=50)
    
    # Run the window
    window.mainloop()
    return selection

# define a function that will create a window for the user to select if he wants to create a new Sleep All Subjects csv
def create_new_sleep_all_subjects_csv_window():
    """
    Description:
        Create a window for the user to select if he wants to create a new Sleep All Subjects csv.
        The user will be asked to select "Yes" or "No" and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selection will be returned.
    Returns:
        The selection.
    """
    # Create a window
    window = tk.Tk()
    window.title("Create New Sleep All Subjects CSV")
    window.geometry("450x100")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Create a new Sleep All Subjects csv?")
    label.pack()
    
    # Create 2 buttons Yes and No
    selection = None
    def yes():
        nonlocal selection
        selection = True
        window.destroy()
        
    def no():
        nonlocal selection
        selection = False
        window.destroy()
        
    # Create a the two buttons in distance from each other in the same line but close to the center
    yes_button = tk.Button(window, text="Yes", command=yes)
    yes_button.pack(side=tk.LEFT, padx=50)
    no_button = tk.Button(window, text="No", command=no)
    no_button.pack(side=tk.RIGHT, padx=50)
    
    
    # Run the window
    window.mainloop()
    return selection

# define a function that will create a window for the user to select if he wants to run get all the files if no ask him to select the files he wants to get
def get_all_files_window():
    """
    Description:
        Create a window for the user to select if he wants to get all the files.
        The user will be asked to select "Yes" or "No" and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selection will be returned.
    Returns:
        The selection.
    """
    # Create a window
    window = tk.Tk()
    window.title("Get All Files")
    window.geometry("450x100")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Get all the files?")
    label.pack()
    
    # Create 2 buttons Yes and No
    selection = None
    def yes():
        nonlocal selection
        selection = True
        window.destroy()
        
    def no():
        nonlocal selection
        selection = False
        window.destroy()
        
    # Create a the two buttons in distance from each other in the same line but close to the center
    yes_button = tk.Button(window, text="Yes", command=yes)
    yes_button.pack(side=tk.LEFT, padx=50)
    no_button = tk.Button(window, text="No", command=no)
    no_button.pack(side=tk.RIGHT, padx=50)
    
    
    # Run the window
    window.mainloop()
    return selection

# define a function that will create a window for the user to select the files he wants to get
def select_files_to_get_window(files: Union[list, pd.Series]):
    """
    Description:
        Create a window for the user to select the files he wants to get.
        The user will be asked to select the files from a listbox and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the selected files will be returned.
    Args:
        files: A list of the files.
    Returns:
        The selected files.
    """
    # Create a window
    window = tk.Tk()
    window.title("Select Files to Get")
    window.geometry("350x300")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Select Files to Get")
    label.pack()
    
    # Create a listbox and adjust the size (height and width)
    listbox = tk.Listbox(window, width=50, selectmode=tk.MULTIPLE)
    listbox.pack()
    
    # Insert the files to the listbox
    for file in files:
        listbox.insert(tk.END, file)
    selected_files = []
    
    # Create a function that will be called when the user clicks the "Select" button
    def select_files():
        nonlocal selected_files
        selected_files = [listbox.get(i) for i in listbox.curselection()]
        if selected_files:
            window.destroy()
        else:
            messagebox.showinfo("Error", "Please select at least one file")
    
    # Create a button
    button = tk.Button(window, text="Select", command=select_files)
    button.pack()
    
    # Run the window
    window.mainloop()
    return selected_files


# define a function that will create a window for the user to select if he wants to run the function
def run_function_window(functions_dict, previous_runned_functions):
    """
    Description:
        Create a window for the user to select which file he wants to get.
        Present the user with a list of the files from the dictionary and then click the "Select" button.
        After the user clicks the "Select" button, the window will close and the list of the selected files will be returned.
    Args:
        functions_dict: A dictionary of the functions ( keys: numbers, values: file names).
    Returns:
        List of the selected files numbers (dict's keys)
    """
    # Create a window
    window = tk.Tk()
    window.title("Run Function")
    window.geometry("700x600")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen and a bit up
    # lift the window a bit up
    sceen_width = window.winfo_screenwidth()
    sceen_height = window.winfo_screenheight()

    x_coordinate = (sceen_width/2) - (500/2)
    y_coordinate = (sceen_height/2) - (600/2) - 50
    window.geometry(f"600x600+{int(x_coordinate)}+{int(y_coordinate)}")    
    # Create a label
    label = tk.Label(window, text="Select Functions")
    label.pack()
    
    # Create a checkbutton for each function
    selected_functions = []
    for key, value in functions_dict.items():
        var = tk.IntVar()
        try:
            if value in previous_runned_functions:
                var.set(1)
        except:
            pass
        checkbutton = tk.Checkbutton(window, text=f"{key}. {value}", variable=var)
        checkbutton.pack()
        selected_functions.append(var)
        
    # Create a function that will be called when the user clicks the "Select" button
    def select_functions():
        nonlocal selected_functions
        selected_functions = [i.get() for i in selected_functions]
        selected_functions = [i for i, x in enumerate(selected_functions) if x]
        if selected_functions:
            window.destroy()
        else:
            messagebox.showinfo("Error", "Please select at least one function")
            
    
    def select_all():
        for var in selected_functions:
            var.set(1)
            
    def deselect_all():
        for var in selected_functions:
            var.set(0)
            
            
    
            
    # Create a button
    select_all_button = tk.Button(window, text="Select All", command=select_all)
    select_all_button.pack()
    
    deselect_all_button = tk.Button(window, text="Deselect All", command=deselect_all)
    deselect_all_button.pack()
    
    button = tk.Button(window, text="Select", command=select_functions)
    button.pack()
    
    # Run the window
    window.mainloop()
    selected_functions = [i+1 for i in selected_functions]
    
    return selected_functions

# define a function that will create a ini file that will contain the current run variables
def create_ini_log_file(path_to_save, runner_name, selected_subjects, runned_functions,functions_dict, date):
    """
    Description:
        Create a ini file that will contain the current run variables.
    Args:
        path_to_save: The path to save the ini file.
        runner_name: The name of the runner.
        selected_subjects: The selected subjects.
        new_data_subjects: The new data subjects.
        runned_functions: The runned functions.
        date: The date of the run.
    Returns:
        None
    """
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # create a list of the selected functions by the name of the function
    runned_functions = [functions_dict.get(i) for i in runned_functions if i in functions_dict.keys()]
    # Add the sections and the keys
    config['Run'] = {'Runner': runner_name, 'Selected Subjects': ', '.join(selected_subjects), 'Runned Functions': ', '.join(runned_functions), 'Date': date}
    # Write the config to the file
    with open(path_to_save.joinpath(f"last_run.ini"), 'w') as configfile:
        config.write(configfile)
    
    
def last_run_load_ini_file(metadata_path):
    """
    Description:
        Load the last run ini file.
    Args:
        metadata_path: The path to the metadata folder.
    Returns:
        The last run ini file.
    """
    # Get the last run ini file
    last_run_ini_file = metadata_path.joinpath('last_run.ini')
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    if not last_run_ini_file.exists():
        return None
    # Read the config from the file
    config.read(last_run_ini_file)
    return config

# define a function that will create a window for the user to enter the mindware parameters
def mindware_parameters_window(project_config):
    """
    Description:
        Create a window for the user to enter the mindware parameters.
        The user will be asked to enter the parameters and then click the "Submit" button.
        After the user clicks the "Submit" button, the window will close and the parameters will 
        be written to the project config.
    Args:
        project_config: The project config.
    Returns:
        The mindware parameters.
    """
    # Create a window
    window = tk.Tk()
    window.title("Mindware Parameters")
    window.geometry("300x200")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Enter Mindware Parameters")
    label.pack()
    
    # Create an entry for each parameter
    parameters = {}
    for key, value in project_config['MINDWARE-parameters'].items():
        label = tk.Label(window, text=f"{key}:")
        label.pack()
        entry = tk.Entry(window)
        entry.insert(0, value)
        entry.pack()
        parameters[key] = entry
        
    # Create a function that will be called when the user clicks the "Submit" button
    def submit():
        nonlocal parameters
        parameters = {key: entry.get() for key, entry in parameters.items()}
        window.destroy()
        
    # Create a button
    button = tk.Button(window, text="Submit", command=submit)
    button.pack()
    
    # Run the window
    window.mainloop()
    
    # Write the parameters to the project config
    for key, value in parameters.items():
        project_config['MINDWARE-parameters'][key] = value
    
    return project_config

# define a function that will create a window for the user to select the number of events
def number_of_events_window(project_config):
    """
    Description:
        Create a window for the user to select the number of events.
        The user will be asked to select the number of events and then click the "Submit" button.
        After the user clicks the "Submit" button, the window will close and the number of events will 
        be returned.
    Args:
        project_config: The project config.
    Returns:
        The number of events.
    """
    # Create a window
    window = tk.Tk()
    window.title("Number of Events")
    window.geometry("300x200")
    # if close the window exit the program
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    # put the window in the center of the screen
    window.eval('tk::PlaceWindow . center')
    
    # Create a label
    label = tk.Label(window, text="Enter Number of Events")
    label.pack()
    
    # count the number of events
    events = project_config['events']
    number_of_events = len(events)
    # Create an entry with the default value
    entry = tk.Entry(window)
    entry.insert(0, number_of_events)
    entry.pack()
    
    # Create a function that will be called when the user clicks the "Submit" button
    def submit():
        nonlocal number_of_events
        number_of_events = entry.get()
        window.destroy()
        
    # Create a button
    button = tk.Button(window, text="Submit", command=submit)
    button.pack()
    
    # Run the window
    window.mainloop()
    return number_of_events


    
    

# define a function that ask the subject if he want to copy the data from the source data folder to the project data folder, if e want to remove the empty folders at the end of the process, if its a rerun or a new run and if he want to use the previous steps and heartrate data
def ask_subject_for_input():
    """
    Description:
        Ask the subject if he wants to copy the data from the source data folder to the project data folder,
        if he wants to remove the empty folders at the end of the process,
        if it's a rerun or a new run and if he wants to use the previous steps and heartrate data.
    Returns:
        The selections.
    """
    window = tk.Tk()
    window.title("Subject Input")
    window.geometry("600x300")
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    window.eval('tk::PlaceWindow . center')
    window.eval('tk::PlaceWindow . -20 -20')

    # Create a label
    label = tk.Label(window, text="Subject Input")
    label.pack()


    # Create a variable for each question
    remove_empty_folders_var = tk.IntVar()
    rerun_var = tk.IntVar()
    remove_heart_rate_pickle_var = tk.IntVar()
    remove_steps_pickle_var = tk.IntVar()


    # Create a function that will be called when the user clicks the "Submit" button
    def submit():
        nonlocal remove_empty_folders_var, rerun_var, remove_heart_rate_pickle_var, remove_steps_pickle_var
        remove_empty_folders = remove_empty_folders_var.get()
        rerun = rerun_var.get()
        remove_heart_rate_pickle = remove_heart_rate_pickle_var.get()
        remove_steps_pickle = remove_steps_pickle_var.get()

        window.destroy()



    remove_empty_folders_checkbutton = tk.Checkbutton(window, text="Remove Empty Folders at the end of the process from the history folder?",
                                                      variable=remove_empty_folders_var)
    remove_empty_folders_checkbutton.pack()

    rerun_checkbutton = tk.Checkbutton(window,text="Is this a rerun?",
                                        variable=rerun_var)
    rerun_checkbutton.pack()

    remove_heart_rate_pickle_checkbutton = tk.Checkbutton(window, text="Remove the previous heartrate pickle file? (if exists)\nIf yes then the heart rate data for individual will be deleted \nand calculted again from the raw data.",
                                                        variable=remove_heart_rate_pickle_var)
    remove_heart_rate_pickle_checkbutton.pack()

    remove_steps_pickle_checkbutton = tk.Checkbutton(window, text="Remove the previous steps pickle file? (if exists)\nIf yes then the ste[s] data for individual will be deleted \nand calculted again from the raw data.",
                                                    variable=remove_steps_pickle_var)
    remove_steps_pickle_checkbutton.pack()

    # Create a button
    button = tk.Button(window, text="Submit", command=submit)
    button.pack()

    # Run the window
    window.mainloop()
    return remove_empty_folders_var.get(), rerun_var.get(), remove_heart_rate_pickle_var.get(), remove_steps_pickle_var.get()

# define a function to vaerify with the user if he accept the label on the warning message
def verify_label(string):
    """
    Description:
        Verify with the user if he accepts the label on the warning message.
    Args:
        string: The warning message.
    Returns:
        The selection.
    """
    window = tk.Tk()
    window.title("Verify Label")
    window.geometry("350x100")
    window.protocol("WM_DELETE_WINDOW", sys.exit)
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (250/2))
    y_cordinate = int((screen_height/2) - (100/2))
    window.geometry(f"{250}x{100}+{x_cordinate}+{y_cordinate}")

    # Create a label
    label = tk.Label(window, text=string)
    label.pack()

    # Create a variable
    var = tk.IntVar()

    # Create a function that will be called when the user clicks the "Submit" button
    def submit():
        nonlocal var
        var = var.get()
        window.destroy()

    # Create a checkbutton
    checkbutton = tk.Checkbutton(window, text="I accept", variable=var)
    checkbutton.pack()

    # Create a button
    button = tk.Button(window, text="Submit", command=submit)
    button.pack()

    # Run the window
    window.mainloop()
    return bool(var)


def new_get_latest_file_by_term(term: str, subject: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """
    Description:
        Searching the last created file with <term> that indicates on specific file format to search.
    Parameters:
        term: String (string the indicates the file format.)
        subject: if the file is in subject's folder, the caller should specify the subject Id.
    Assumptions:
        files in Aggregated folder is in the format: "<term> Aggregated yyyy-mm-dd_HH-MM-SS"
    Return:
        Path (if not found, return path that not exists.)
    """
    # Define the regular expression pattern to match the date
    path = ''
    if term == 'Sleep All Subjects':
        pattern = r'^Sleep All Subjects.csv'
        # Set up path to the folder where sleep file is.
        path = root
    elif term == 'subject steps':
        pattern = r'^.*\d{3} steps.csv$'
        # Set up path to the folder where sleep file is.
        path = root
        
    elif term == 'Steps':
        pattern = r'^Steps Aggregated.csv'
        # Set up path to the folder where sleep file is.
        path = root
    elif term == 'Heart Rate':
        if subject is None:
            raise Exception('Subject Id is required for Heart Rate file.')
        pattern = r'^.*\d{3} Heart Rate.csv$'
        # Set up path to the folder where Heart Rate file is.
        path = root.joinpath(subject)
    elif term == 'EDA':
        if subject is None:
            raise Exception('Subject Id is required for EDA file.')
        pattern = r'^.*\d{3} EDA.csv$'
        # Set up path to the folder where EDA file is.
        path = root.joinpath(subject)
    elif term == 'HRV Temperature Respiratory At Sleep All Subjects':
        pattern = r'^HRV Temperature Respiratory At Sleep All Subjects.csv'
        # Set up path to the folder where HRV Temperature Respiratory At Sleep All Subjects file is.
        path = root
    elif term == 'HRV Temperature Respiratory':
        if subject is None:
            raise Exception('Subject Id is required for HRV Temperature Respiratory file.')
        pattern = r'^.*\d{3} HRV Temperature Respiratory At Sleep.csv$'
        # Set up path to the folder where HRV Temperature Respiratory file is.
        path = root.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Heart Rate and Steps and Sleep Aggregated':
        pattern = r'^.*\d{3} Heart Rate and Steps and Sleep Aggregated.csv$'
        if subject is None:
            raise Exception('Subject Id is required for Heart Rate and Steps and Sleep Aggregated file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root.joinpath(subject)
    elif term == 'Metrics of Heart Rate By Activity':
        pattern = r'^.*\d{3} Metrics of Heart Rate By Activity.csv$'
        if subject is None:
            raise Exception('Subject Id is required for Metrics of Heart Rate By Activity file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Full Week Summary of Heart Rate Metrics By Activity':
        pattern = r'^Full Week Summary of Heart Rate Metrics By Activity.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Summary Of HRV Temperature Respiratory At Sleep':
        pattern = r'^Summary Of HRV Temperature Respiratory At Sleep.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'EDA Summary':
        pattern = r'^EDA Summary.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Final All Subjects Aggregation':
        pattern = r'^Final All Subjects Aggregation.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'EMA_raw':
        pattern = r'^EMA_raw_sub_\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.xlsx'  # TODO: validate that it reads
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA with extra metrics':
        pattern = r'^EMA_raw_sub_\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_with_extra_metrics\.xlsx'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA Daily Mean + Fitbit Night':
        # pattern match 'sub_032 EMA Daily Mean + Fitbit Night 2023-11-12_12-57-49.csv'
        pattern = '^sub_\d{3} EMA Daily Mean \+ Fitbit Night \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EMA metrics and diff':
        # pattern match 'sub_032 EMA Daily Mean + Fitbit Night 2023-11-12_12-57-49.csv'
        pattern =  r'^EMA_raw_sub_\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_with_extra_metrics_with_diff\.xlsx'
        # Set up path to the folder where EMA file is.
        path = root.joinpath(subject)
    elif term == 'EDA All Subjects':
        pattern = r'^EDA All Subjects.csv'
        # Set up path to the folder where EDA file is.
        path = root
    elif term == 'No Weekends Summary of Heart Rate Metrics By Activity':
        pattern = r'^No Weekends Summary of Heart Rate Metrics By Activity.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'No Weekends All Subjects of Heart Rate Metrics By Activity':
        pattern = r'^No Weekends All Subjects of Heart Rate Metrics By Activity.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Full Week Summary of Heart Rate Metrics By Activity':
        pattern = r'^Full Week Summary of Heart Rate Metrics By Activity.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Full Week All Subjects of Heart Rate Metrics By Activity':
        pattern = r'^Full Week All Subjects of Heart Rate Metrics By Activity.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Details Exclude Thursday and Friday':
        pattern = r'^Sleep Daily Details Exclude Thursday and Friday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Summary Exclude Thursday and Friday':
        pattern = r'^Sleep Daily Summary Exclude Thursday and Friday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Summary Exclude Thursday':
        pattern = r'^Sleep Daily Summary Exclude Thursday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Details Exclude Thursday':
        pattern = r'^Sleep Daily Details Exclude Thursday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Summary Exclude Friday':
        pattern = r'^Sleep Daily Summary Exclude Friday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Details Exclude Friday':
        pattern = r'^Sleep Daily Details Exclude Friday.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'Sleep Daily Details Full Week':
        pattern = r'^Sleep Daily Details Full Week.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root
    elif term == 'cosinor':
        pattern = r'^.*\d{3} cosinor.csv$'
        if subject is None:
            raise Exception('Subject Id is required for cosinor file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = root.joinpath(subject)
        
    if path == '':
        raise Exception(f'No path was found for term: {term}')
    # Get only the relevant files
    files = [f for f in os.listdir(path) if re.search(pattern, f)]
    # Sort the files by their last created time
    files_sorted_by_date = sorted(files, key=lambda f: os.path.getctime(path.joinpath(f)), reverse=True)
    # Check if there are any files in the sorted list, and return the file with the latest date
    if files_sorted_by_date:
        return path.joinpath(files_sorted_by_date[0])

    # If no files were found, return a path that does not exists.
    if not subject:
        print(f'{pattern} not found')
    else:
        print(f'{pattern} not found for subject {subject}')
    return Path("NOT_EXISTS_PATH")

    