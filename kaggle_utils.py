from os import chdir
from IPython.display import FileLink, display

def download_file(file_path: str) -> None:
    """

    @param file_path: file name to download
    @return: None
    """
    os.chdir('/kaggle/working') # change work directory
    display(FileLink(file_path)) # static link for the download