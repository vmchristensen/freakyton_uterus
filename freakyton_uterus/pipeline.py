import subprocess
import os

# Define the order of scripts to run (relative to the project root)
SCRIPTS = [
    'data_cleaning/read_data.py',
    'data_cleaning/df_selection_translation.py',
    'data_cleaning/data_cleaning.py', # Ensure you create this file
    'model/modeling_binaryrecurrence_preop.py',
    'model/modeling_survival_preop.py',
    'model/modeling_survival_preop_plots.py',
    # 'model/app_preop.py' # App is usually run separately: python -m streamlit run app_preop.py
]

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

print("--- STARTING THE ENDOMETRIAL CANCER PIPELINE ---")
for script in SCRIPTS:
    full_path = os.path.join(PROJECT_ROOT, script)
    print(f"\n---> Running {script}...")
    try:
        # Use subprocess to execute the script
        result = subprocess.run(['python', full_path], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script} failed!")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        break

print("\n--- PIPELINE EXECUTION ATTEMPT COMPLETE ---")