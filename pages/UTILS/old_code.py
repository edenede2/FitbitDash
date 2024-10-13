def get_latest_file_by_term(term: str, subject: Optional[str] = None) -> Path:
    """
    Description:
        Searching the last created file with <term> that indicates on specific file format to search
        in Outputs/Aggregated Output folder.
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
    if term == 'Sleep Full Details':
        pattern = r'^Sleep Full Details (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where sleep file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Steps':
        pattern = r'^Steps Aggregated (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where sleep file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Heart Rate':
        pattern = r'^Heart Rate (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Heart Rate file is.
        path = OUTPUT_PATH.joinpath(subject)
    elif term == 'EDA':
        pattern = r'^EDA (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        if subject is None:
            raise Exception('Subject Id is required for EDA file.')
        # Set up path to the folder where EDA file is.
        path = OUTPUT_PATH.joinpath(subject)
    elif term == 'HRV Temperature Respiratory':
        pattern = r'^HRV_Temperature_Respiratory_At_Sleep (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        if subject is None:
            raise Exception('Subject Id is required for HRV Temperature Respiratory file.')
        # Set up path to the folder where HRV Temperature Respiratory file is.
        path = OUTPUT_PATH.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Heart Rate and Steps and Sleep Aggregated':
        pattern = r'^Heart Rate and Steps and Sleep Aggregated (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        if subject is None:
            raise Exception('Subject Id is required for Heart Rate and Steps and Sleep Aggregated file.')
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = OUTPUT_PATH.joinpath(subject)
    elif term == 'Sleep Daily Summary Full Week':
        pattern = r'^Sleep Daily Summary Full Week (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Full Week Summary of Heart Rate Metrics By Activity':
        pattern = r'^Full Week Summary of Heart Rate Metrics By Activity (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Summary Of HRV Temperature Respiratory At Sleep':
        pattern = r'^Summary Of HRV Temperature Respiratory At Sleep (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'EDA Summary':
        pattern = r'^EDA Summary (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'Final All Subjects Aggregation':
        pattern = r'^Final All Subjects Aggregation (\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}.csv'
        # Set up path to the folder where Sleep Daily Summary Full Week file is.
        path = AGGREGATED_OUTPUT_PATH
    elif term == 'EMA_valid':
        pattern = r'^EMA_raw_sub_\d{3} \d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.xlsx' # TODO: validate that it reads
        # Set up path to the folder where EMA file is.
        path = OUTPUT_PATH.joinpath(subject)
    if path == '':
        raise Exception(f'No path was found for term: {term}')
    # Get only the relevant files
    files = [f for f in os.listdir(path) if re.fullmatch(pattern, f)]
    # Sort the files by their last created time
    files_sorted_by_date = sorted(files, key=lambda f: os.path.getctime(path.joinpath(f)), reverse=True)
    # Check if there are any files in the sorted list, and return the file with the latest date
    if files_sorted_by_date:
        return path.joinpath(files_sorted_by_date[0])

    return Path("NOT_EXISTS_PATH")
