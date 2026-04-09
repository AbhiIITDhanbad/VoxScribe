list_of_files = [
    f"Speech-to-Text/__init__.py",
    f"Speech-to-Text/cloud_storage/__init__.py",
    f"Speech-to-Text/components/__init__.py",
    f"Speech-to-Text/configuration/__init__.py",
    f"Speech-to-Text/constants/__init__.py",
    f"Speech-to-Text/entity/__init__.py",
    f"Speech-to-Text/exceptions/__init__.py",
    f"Speech-to-Text/logger/__init__.py",
    f"Speech-to-Text/models/__init__.py",
    f"Speech-to-Text/pipeline/__init__.py",
    f"Speech-to-Text/utils/__init__.py",
    f"setup.py",
    f"requirements.txt"
]
import os
from pathlib import Path

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir!= "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass

    else:
        print(f"file is already there: {filepath}")

        
