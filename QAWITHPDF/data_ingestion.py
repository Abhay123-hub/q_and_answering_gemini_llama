from llama_index.core import SimpleDirectoryReader
import traceback

def load_data(directory_path):
    """
    Loads documents from a specified directory.

    Parameters:
    - directory_path (str): The path to the directory containing data files.

    Returns:
    - documents (list): A list of loaded documents, or None if loading fails.
    """
    try:
        # Initialize the directory reader and load documents
        loader = SimpleDirectoryReader(directory_path)
        documents = loader.load_data()
        print("Data loading completed successfully.")
        return documents

    except Exception as e:
        # Print traceback for debugging
        print("Failed to load data. See error details below:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None
