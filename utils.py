import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """
    Ensure a directory exists
    
    Args:
        directory (str): Directory path
        
    Returns:
        str: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

def file_exists(filepath):
    """
    Check if a file exists
    
    Args:
        filepath (str): Path to file
        
    Returns:
        bool: True if file exists
    """
    return os.path.exists(filepath)

def get_file_size(filepath):
    """
    Get file size in MB
    
    Args:
        filepath (str): Path to file
        
    Returns:
        float: File size in MB
    """
    if file_exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def print_step_info(step_name, start=True):
    """
    Print step information
    
    Args:
        step_name (str): Name of the step
        start (bool): Whether this is the start of the step
        
    Returns:
        None
    """
    if start:
        logger.info(f"{'='*20} Starting: {step_name} {'='*20}")
    else:
        logger.info(f"{'='*20} Completed: {step_name} {'='*20}")
