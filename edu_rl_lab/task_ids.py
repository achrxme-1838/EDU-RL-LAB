"""Task identifiers safe to import before Isaac Sim starts.

Keep this module free of Gym/NumPy imports: Isaac Sim 4.5 adjusts the NumPy
search path during startup, so loading Gym first can mix NumPy installations.
"""

TASK_ID = "Edu-Go2-Velocity-v0"
A2_TASK_ID = "Edu-A2-Velocity-v0"
TASK_IDS = (TASK_ID, A2_TASK_ID)
