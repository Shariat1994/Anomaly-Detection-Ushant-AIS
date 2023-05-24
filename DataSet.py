import os
import urllib.request
import zipfile
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


class DatasetProcessor:
    def __init__(self, url, filename, folder_path, data_folder, test_size=0.2):
        """
        Initialize the DatasetProcessor class.

        Args:
            url (str): The download URL of the data.
            filename (str): The name of the downloaded file.
            data_folder (str): file name with all txt files.
            folder_path (str): The path to the folder where the data will be extracted.
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.
        """
        self.url = url
        self.filename = filename
        self.folder_path = folder_path
        self.data_folder = data_folder
        self.test_size = test_size
        # Download the dataset
        self.__download_data()
        # catch all .txt files names
        self.file_names = self.__read_data()

    def __download_data(self):
        """
        Download and extract the data from the provided URL if the file does not exist. 
        If the file exists, it removes the existing file.
        """
        if not os.path.exists(os.path.join(os.getcwd(), self.folder_path)):
            print("Downloading dataset...")
            # Download the data from the URL
            urllib.request.urlretrieve(self.url, self.filename)

            # Extract the data from the zip file
            with zipfile.ZipFile(self.filename, 'r') as zip_ref:
                zip_ref.extractall(self.folder_path)

            # Remove the zip file
            os.remove(self.filename)
        else:
            # If the file already exists, try remove the zip file
            print("Dataset already downloaded!")
            try:
                os.remove(self.filename)
            except:
                pass

    def __read_data(self):
        """
        Read the names of files from the specified data folder.

        Returns:
            list: A list of file names.
        """
        # Create a list of file names by iterating over the files in the data folder
        # and selecting the ones that end with '.txt'
        file_names = [os.path.join(self.data_folder, file) for file in os.listdir(self.data_folder) if file.endswith('.txt')]
        
        # Return the list of file names
        return file_names

    def create_dataframe(self):
        """
        Create a pandas DataFrame by reading and concatenating the data files.

        Returns:
            pandas.DataFrame: The combined DataFrame.
        """
        # Initialize an empty list to store the individual DataFrames
        data = []

        # Print status message
        print("Reading all files ...")

        # Loop through each file name and read the corresponding CSV file
        # using pandas read_csv function with delimiter=';'
        for file_name in tqdm(self.file_names):
            df = pd.read_csv(file_name, delimiter=';')

            # Append the DataFrame to the list
            data.append(df)

        # Concatenate the individual DataFrames into a single DataFrame
        data = pd.concat(data, ignore_index=True)

        # Return the combined DataFrame
        return data

    def __split_scale(self, df, dataset):
        """
        Split the data into train and test sets, and apply MinMaxScaler to scale the features.

        Args:
            df (pandas.DataFrame): The DataFrame to be split and scaled.
            dataset (str): determine name of dataset

        Returns:
            array: 2 array containing the scaled training and test sets.
        """
        # Split the data into train and test sets using train_test_split
        X_train, X_test = train_test_split(df, test_size=self.test_size, random_state=42)

        # Determine the appropriate scaler object based on the dataset name
        scaler_name = 'location_scaler.pkl' if dataset == 'location' else 'velocity_scaler.pkl'

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Scale the training data using fit_transform
        X_train_scaled = scaler.fit_transform(X_train)

        # Scale the test data using transform
        X_test_scaled = scaler.transform(X_test)

        # Save scalar
        joblib.dump(scaler, os.path.join(os.getcwd(), scaler_name))

        # Return the scaled training and test sets
        return X_train_scaled, X_test_scaled

    def create_location_dataset(self, df):
        """
        Create a location dataset by selecting the 'x' and 'y' columns and applying split and scale operations.

        Args:
            df (pandas.DataFrame): The data DataFrame.

        Returns:
            array: 2 array containing the scaled training and test sets of the location data.
        """
        # Select the 'x' and 'y' columns from the input DataFrame
        location_df = df[['x', 'y']]

        # Split and scale the location data using the __split_scale method
        X_train, X_test = self.__split_scale(location_df, 'location')

        # Return the scaled training and test sets of the location data
        return X_train, X_test

    def create_velocity_dataset(self, df):
        """
        Create a velocity dataset by selecting the 'vx' and 'vy' columns and applying split and scale operations.

        Args:
            df (pandas.DataFrame): The input DataFrame.

        Returns:
            tuple: A tuple containing the scaled training and test sets of the velocity data.
        """
        # Select the 'vx' and 'vy' columns from the input DataFrame
        velocity_df = df[['vx', 'vy']]

        # Split and scale the velocity data using the __split_scale method
        X_train, X_test = self.__split_scale(velocity_df, 'velocity')

        # Return the scaled training and test sets of the velocity data
        return X_train, X_test