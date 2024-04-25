# controller.py - Run CBAS GUI
import logging
import os
import sys
import traceback

import cbaerosurrogate as cbas

from gui import model, view
from gui.log import log, log_handler

ctrl = sys.modules[__name__]

def start(debug=False):
    """Begin running the app."""
    try:
        if debug:
            log_handler.setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)

        # Build UI & data access objects
        model.start()
        cbaero_path, tables_path = model.get_settings()
        view.start(debug, cbaero_path, tables_path)

        # Setup callbacks
        view.run_btn.on_click(when_run)
        view.script_btn.on_click(when_script)
        view.cbaero_path.register_callback(when_settings)
        view.tables_path.register_callback(when_settings)

        log.info('App running')
    except Exception:
        log.error('start:\n'+traceback.format_exc())
        raise

def when_run(_):
    """React to user pressing run button."""
    try:
        log.info('Run button was pressed')
        view.run_msg(None, clear=True)

        # First, make sure model path leads to valid file

        log.debug(f'Using model: "{view.model_path.selected}"')
        model_path = None if view.model_path.selected is None else view.model_path.selected.strip()

        if model_path is None or not os.path.isfile(model_path):
            view.run_msg('Error: Invalid model path')
        else:
            # Ensure remaining paths are either None or some text

            if view.cokrig_path.selected is None or view.cokrig_path.selected.strip() == '':
                cokrig_path = None
            else:
                cokrig_path = view.cokrig_path.selected.strip()

            if view.save_fname.value is None or view.save_fname.value.strip() == '':
                save_fname = None
            else:
                save_fname = view.save_fname.value.strip()

            # Set environment variables for calling CBAero

            os.environ['CBAERO_TABLES'] = view.tables_path.selected

            if not view.cbaero_path.selected in os.environ['PATH']:
                os.environ['PATH'] = '$PATH:' + view.cbaero_path.selected

            # Create CbaeroSurrogate obj and run it TODO Redirect cbas' logging to notebook output widget?
            view.run_msg('Creating CbaeroSurrogate object')
            cbas_obj = cbas.CbaeroSurrogate(model_path,
                                            float(view.a_min_txt.value), float(view.a_max_txt.value),
                                            float(view.m_min_txt.value), float(view.m_max_txt.value),
                                            float(view.q_min_txt.value), float(view.q_max_txt.value),
                                            cokrig_path,
                                            save_fname)
            view.run_msg('Starting run')
            cbas_obj.run(int(view.train_pts_start.value),
                         int(view.train_pts_stop.value),
                         int(view.train_pts_step.value),
                         cbas.SERIAL)
            view.run_msg('Run completed')
    except Exception:

        try:
            view.run_msg(traceback.format_exc())
        except Exception:
            log.error(traceback.format_exc())

    view.run_msg('\nDone.')

def when_script(_):
    """React to user pressing generate-script button."""

    try:
        log.info('Script button was pressed')
        view.script_lbl.value = 'Working...'
        model.generate_job_script()
        view.script_lbl.value = f'File "{model.SCRIPT_NAME}" generated successfully'
    except Exception:
        log.error('start:\n'+traceback.format_exc())
        raise

def when_settings(_):
    model.save_settings()