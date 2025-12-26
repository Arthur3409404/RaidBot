from modes import *
from utils import *
from datetime import datetime
import threading
import time

# --- Daily Routine Entry ---
def daily_routine(poll_interval=60):
    """
    Fully autonomous daily routine.
    Loops indefinitely, executes tasks based on conditions, handles errors.
    """
    access_logfile()
    
    # Start error-catching thread
    error_thread = threading.Thread(target=process_thread, args=(5,))
    error_thread.daemon = True
    error_thread.start()

    # Main routine loop
    while True:
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        if not log_has_entry_for(today_str):
            create_log_entry(today_str)

        try:
            control_daily_routine(today_str)
        except Exception as e:
            handle_error(f"Exception in main routine: {e}")
            log_error(f"Exception in main routine: {e}")

        # Wait before next iteration to avoid excessive CPU usage
        time.sleep(poll_interval)


# --- Control Function ---
def control_daily_routine(today_str):
    """
    Controls execution of daily tasks based on conditions.
    """
    switch_tosetup('arena')
    current_time = check_time()

    # --- Live Arena ---
    if 12.5 <= current_time < 15 and not log_entry_done(today_str, 'live_arena'):
        do_live_arena()
        update_log(today_str, 'live_arena', True)
    
    # --- Arena ---
    arena_coins = check_arena_coins()
    if arena_coins > 0:
        do_arena()
        update_log(today_str, 'arena', True)
        check_rewards()
    
    # --- Tag Team Arena ---
    tagteam_coins = check_rewards(tagteam=True)
    if tagteam_coins > 0:
        do_tagteam_arena()
        update_log(today_str, 'tagteam', True)
        check_rewards()
    
    # --- Demnlord ---
    if current_time >= 12 and not log_entry_done(today_str, 'demnlord'):
        switch_tosetup('demnlord')
        do_demnlord()
        update_log(today_str, 'demnlord', True)
        check_rewards()
    
    # --- Doom Tower ---
    if not log_entry_done(today_str, 'doomtower') and log_entry_done(today_str, 'demnlord'):
        switch_tosetup('doomtower')
        do_doomtower()
        update_log(today_str, 'doomtower', True)
        check_rewards()
    
    # --- Faction Wars ---
    if not log_entry_done(today_str, 'factionwars') and log_entry_done(today_str, 'tagteam'):
        switch_tosetup('factionwars')
        do_factionwars()
        update_log(today_str, 'factionwars', True)
        check_rewards()
    
    # --- Sintranos ---
    if not log_entry_done(today_str, 'sintranos'):
        switch_tosetup('sintranos')
        do_sintranos()
        update_log(today_str, 'sintranos', True)
        check_rewards()
    
    # --- Dragon Dungeon ---
    energy = check_energy()
    if energy > 40:
        switch_tosetup('dragon')
        run_dungeon('dragon')
        # Increment dragon counter in logfile
        current_dragon_count = get_log_status(today_str, 'dragon')
        if not isinstance(current_dragon_count, int):
            current_dragon_count = 0
        update_log(today_str, 'dragon', current_dragon_count + 1)
        
        check_rewards()


# --- Error-Catching Thread ---
def process_thread(poll_interval=5):
    """
    Continuously checks for errors from `new_process` and handles them.
    """
    while True:
        try:
            error_message = new_process()
            if error_message:
                handle_error(error_message)
                log_error(error_message)
        except Exception as e:
            handle_error(f"Exception in error thread: {e}")
            log_error(f"Exception in error thread: {e}")
        time.sleep(poll_interval)


# --- Logfile Helper Functions ---
def log_has_entry_for(day):
    return day in get_log_days()

def create_log_entry(day):
    log_structure = {
        'live_arena': False,
        'arena': False,
        'tagteam': False,
        'demnlord': False,
        'doomtower': False,
        'factionwars': False,
        'sintranos': False,
        'dragon': 0,
        'notes': ''
    }
    initialize_log_entry(day, log_structure)

def log_entry_done(day, task):
    return get_log_status(day, task)

def update_log(day, task, status=True):
    set_log_status(day, task, status)

def log_error(error_message):
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if log_has_entry_for(today_str):
        current_notes = get_log_notes(today_str)
        updated_notes = current_notes + f"\n{datetime.utcnow().isoformat()} - {error_message}"
        set_log_notes(today_str, updated_notes)
    else:
        create_log_entry(today_str)
        set_log_notes(today_str, f"{datetime.utcnow().isoformat()} - {error_message}")
        
