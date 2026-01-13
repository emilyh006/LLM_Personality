import json
import os
from pathlib import Path

def merge_json_files(folder_path, output_file='merged_output.json'):
    """
    Scan a folder for JSON files and merge them into one file.
    
    Args:
        folder_path: Path to the folder containing JSON files
        output_file: Name of the output merged JSON file
    """
    merged_data = []
    json_files = []
    
    # Find all JSON files in the folder
    folder = Path(".")
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    json_files = list(folder.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    # Read and merge each JSON file
    for json_file in json_files:
        try:
            print(f"Reading: {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
                    
        except json.JSONDecodeError as e:
            print(f"Error decoding {json_file.name}: {e}")
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    # Write merged data to output file
    output_path = folder / output_file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully merged {len(json_files)} files into '{output_file}'")
        print(f"Output location: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    # Example usage - modify these values
    folder_to_scan = "."  # Current folder (or use "./json_files" for subfolder)
    output_filename = "merged_output.json"
    
    merge_json_files(folder_to_scan, output_filename)