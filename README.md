# Optimization_2AMS50

### Developer Instructions
1. Create and activate virtual environment. For example, using `pip virtualenv`:
   `python3 -m venv venv; source venv/bin/activate`.
2. Install requirements. For example, using pip: `pip install -r requirements.txt`
3. Create file `settings/local_settings.py` and declare variable
   `DATA_PATH = "path/to/your/data"`. The data directory should contain
   one subdirectory per state and the `NumberofDistricts.txt` file describing
   the number of districts for each state.
4. Run `main.py`. For parameters run `python3 main.py --help` or check the code.
