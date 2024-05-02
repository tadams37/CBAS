# controller.py - Run CBAS GUI
import logging
import os
import sys
import traceback
from datetime import datetime

import cbaerosurrogate as cbas

from gui import model, view
from gui.config import CBAS_RUN_PREFIX
from gui.log import log_handler

ctrl = sys.modules[__name__]

def start():
    """Begin running the app."""
    try:
        # Log to a file at DEBUG TODO Log to stdout, lab console? Incorporate cbaerosurrogate logging? How?
        logging.basicConfig(filename='cbas_gui.log',
                            format=f'%(levelname)s: %(message)s (%(filename)s:%(lineno)d, %(asctime)s)',
                            level=logging.DEBUG)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)  # Suppress debug entries for "...findfont..."

        # Build UI & file access objects
        model.start()
        paths = model.get_settings()
        view.start(paths)

        # Set methods to be called when buttons are pressed or settings widgets change
        view.run_btn.on_click(when_run)
        view.script_btn.on_click(when_script)
        view.cbaero_path.register_callback(when_settings)
        view.tables_path.register_callback(when_settings)
        view.run_path.register_callback(when_settings)

        logging.info('App running')
    except Exception:
        logging.error('start:\n'+traceback.format_exc())
        raise

def when_run(_):
    """React to user pressing run button."""
    try:
        logging.info('Run button was pressed')
        view.run_msg(None, clear=True)

        if valid_required_paths():
            cokrig_path, save_fname = valid_optional_paths()
            set_cbaero_env()
            orig_cwd = set_run_dir()
            log_handler.start_log_output(view.run_out)

            # Create cbas object
            logging.info('Creating CbaeroSurrogate object')
            cbas_obj = cbas.CbaeroSurrogate(view.model_path.selected,
                                            float(view.a_min_txt.value), float(view.a_max_txt.value),
                                            float(view.m_min_txt.value), float(view.m_max_txt.value),
                                            float(view.q_min_txt.value), float(view.q_max_txt.value),
                                            cokrig_path,
                                            save_fname)

            # Execute cbas run (does cbaero runs) TODO Redirect cbas' logging to notebook output widget?
            view.run_msg('Running...')
            cbas_obj.run(int(view.train_pts_start.value),
                         int(view.train_pts_stop.value),
                         int(view.train_pts_step.value),
                         cbas.SERIAL)

            # Finish up after run
            log_handler.stop_log_output()
            os.chdir(orig_cwd)
            view.run_msg('Run completed')

    except Exception:

        try:
            view.run_msg(traceback.format_exc())
        except Exception:
            logging.error(traceback.format_exc())

    view.run_msg('\nDone.')

def when_script(_):
    """React to user pressing generate-script button."""

    try:
        logging.info('Script button was pressed')
        view.script_lbl.value = 'Working...'
        model.generate_job_script()
        view.script_lbl.value = f'File "{model.SCRIPT_NAME}" generated successfully'
    except Exception:
        logging.error('start:\n'+traceback.format_exc())
        raise

def when_settings(_):
    model.save_settings()

def valid_required_paths():
    """Ensure required paths are correct."""
    valid = False

    if view.model_path.selected is None or not os.path.isfile(view.model_path.selected):
        view.run_msg('ERROR: Model file not found')
    elif view.run_path.selected is None or not os.path.exists(view.run_path.selected):
        view.run_msg('ERROR: Invalid run directory')
    elif view.tables_path.selected is None or not os.path.exists(os.path.join(view.tables_path.selected,'atmos')):
        view.run_msg('ERROR: Invalid tables directory')
    elif view.cbaero_path.selected is None or not os.path.isfile(os.path.join(view.cbaero_path.selected,'cbaero')):
        view.run_msg('ERROR: Invalid tables directory')
    else:
        logging.debug(f'Using model: "{view.model_path.selected}"')
        valid = True

    return valid

def valid_optional_paths():
    # Ensure remaining paths are either None or some text

    if view.cokrig_path.selected is None or view.cokrig_path.selected.strip() == '':
        cokrig_path = None
    else:
        cokrig_path = view.cokrig_path.selected.strip()
        view.run_msg(f'Using cokrig file "{cokrig_path}"')

    if view.save_fname.value is None or view.save_fname.value.strip() == '':
        save_fname = None
    else:
        save_fname = view.save_fname.value.strip()
        view.run_msg(f'Saving Kriging model to file "{save_fname}"')

    return cokrig_path, save_fname

def set_cbaero_env():
    """Set environment variables for calling CBAero."""
    os.environ['CBAERO_TABLES'] = view.tables_path.selected

    if not view.cbaero_path.selected in os.environ['PATH']:
        os.environ['PATH'] = '$PATH:' + view.cbaero_path.selected

def set_run_dir():
    """Change create run dir, change to it."""
    orig_cwd = os.getcwd()
    run_dir = os.path.join(view.run_path.selected,
                           CBAS_RUN_PREFIX+datetime.now().isoformat().replace(':', '-'))
    view.run_msg(f'Run directory is "{run_dir}"')

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    os.chdir(run_dir)
    return orig_cwd
