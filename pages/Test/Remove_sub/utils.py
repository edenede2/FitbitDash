from typing import Optional
from pathlib import Path
import pandas as pd
import configparser
import datetime
# from md2pdf.core import md2pdf
import shutil
import os
import glob
import re
import sys

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


def get_subject_dates_of_experiment(subject: str, METADATA_PATH: Path) -> pd.DataFrame:
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
                                    date_format='%d/%m/%Y',
                                    date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
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
    DATA_PATH = project_path.joinpath('Data')
    OUTPUT_PATH = project_path.joinpath('Outputs')
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
            shutil.copy(history_path.joinpath(file), output_path.joinpath(file))
        
def ignore_history(dir, filenames):
    return ['History'] if 'History' in filenames else []


# 