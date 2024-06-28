import os

def print_directory_structure(folder_path, indent=''):
    """
    Print the structure of the directory recursively.

    :param folder_path: Path to the directory
    :param indent: Indentation string for subdirectories
    """
    # Get all items in the directory
    items = os.listdir(folder_path)
    
    for item in sorted(items):
        # Construct full path of the item
        item_path = os.path.join(folder_path, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Skip .git directory
            if item == '.git':
                continue
            # Print directory name
            print(f"{indent}+ {item}/")
            # Recursively print subdirectory structure
            print_directory_structure(item_path, indent + '  ')
        else:
            # Print file name
            print(f"{indent}- {item}")

# Replace 'path_to_your_directory' with the path to your directory
directory_path = './'

print(f"Directory Structure of {directory_path}:")
print_directory_structure(directory_path)
