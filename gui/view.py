# view.py - Build CBAS user interface
import sys
import logging

from IPython.display import display
from ipywidgets import HBox, IntText, Label, Layout, FloatText, \
                       Output, HTML, Image, Tab, Text, VBox, Button
from ipyfilechooser import FileChooser
from IPython.core.display import clear_output

from gui.config import CBAERO_FILES_FILTER, INIT_TRAIN_PTS_START, INIT_TRAIN_PTS_END, INIT_TRAIN_PTS_STEP, \
                       INIT_A_MIN, INIT_A_MAX, INIT_M_MIN, INIT_M_MAX, INIT_Q_MIN, INIT_Q_MAX, INIT_SAVE_FNAME

SM_FULL_WIDTH = '140px'
SM_DESC_WIDTH = '40px'
MED_FULL_WIDTH = '300px'
MED_DESC_WIDTH = '200px'
LG_FULL_WIDTH = '400px'
LG_DESC_WIDTH = '200px'
HALF_DESC_WIDTH = '100px'
PLUS_DESC_WIDTH = '150px'

view = sys.modules[__name__]


def start(paths):
    """Build the user interface."""
    display(HTML(filename='gui/custom.html'))  # Send CSS code down to browser TODO No worky-worky?

    # Color buttons
    display(HTML('<style>.lm-TabBar-tab.lm-mod-current::before {background: teal !important; }</style>'))
    display(HTML('<style>.jupyter-button { background-color: teal !important; }</style>'))
    display(HTML('<style>.jupyter-button { color: white !important; }</style>'))

    # Create header with logo

    with open('gui/logo.svg', "rb") as logo_file:
        logo = Image(value=logo_file.read(), format='png', layout={'max_height': '64px'})

    app_title = HTML('<h2 style="color: teal;">CBAS - CBAero Surrogate Modeling</h2>')
    header = HBox([app_title, Label(layout=Layout(width='500px')), logo])
    header.layout.min_width = '1000px'

    # Create tabs, fill with UI content (widgets), set titles

    tabs = Tab()
    tab_content = []
    tab_content.append(view.params_tab())
    tab_content.append(view.run_tab())
    tab_content.append(view.job_tab())
    tab_content.append(view.settings_tab(paths))
    tabs.children = tuple(tab_content)  # Fill tabs with content

    for i, tab_title in enumerate(['Parameters', 'Run', 'Job', 'Settings']):
        tabs.set_title(i, tab_title)

    display(VBox([header, tabs]))
    logging.info('UI build completed')


def set_width(widgets, width='auto', desc=False):
    """Set width for widgets' layouts or descriptions."""
    for widget in widgets:

        if desc:
            widget.style.description_width = width
        else:
            widget.layout = Layout(width=width)


def params_tab():
    """Create widgets for Parameters screen."""
    # Create widgets
    view.model_path = FileChooser(filter_pattern=CBAERO_FILES_FILTER, layout=Layout(width='auto'))
    view.train_pts_start = IntText(description='Training points range, start', value=INIT_TRAIN_PTS_START)
    view.train_pts_stop = IntText(description='stop', value=INIT_TRAIN_PTS_END)
    view.train_pts_step = IntText(description='step', value=INIT_TRAIN_PTS_STEP)
    view.a_min_txt = FloatText(description='Angle of attack (deg.), min', value=INIT_A_MIN)
    view.a_max_txt = FloatText(description='max', value=INIT_A_MAX)
    view.m_min_txt = FloatText(description='Mach number, min', value=INIT_M_MIN)
    view.m_max_txt = FloatText(description='max', value=INIT_M_MAX)
    view.q_min_txt = FloatText(description=' Dynamic pressure (bars): min', value=INIT_Q_MIN)
    view.q_max_txt = FloatText(description='max', value=INIT_Q_MAX)
    view.save_fname = Text(description='Save Kriging model as', value=INIT_SAVE_FNAME)
    view.cokrig_path = FileChooser(layout=Layout(width='auto'))

    # Set widths

    for widget in [view.train_pts_start, view.a_min_txt, view.m_min_txt, view.q_min_txt]:
        set_width([widget], width=MED_FULL_WIDTH)
        set_width([widget], width=MED_DESC_WIDTH, desc=True)

    for widget in [view.a_max_txt, view.m_max_txt, view.q_max_txt]:
        set_width([widget], width='140px')
        set_width([widget], width='40px', desc=True)

    for widget in [view.train_pts_stop, view.train_pts_step]:
        set_width([widget], width=SM_FULL_WIDTH)
        set_width([widget], width=SM_DESC_WIDTH, desc=True)

    set_width([view.save_fname], width=LG_FULL_WIDTH)
    set_width([view.save_fname], width=LG_DESC_WIDTH, desc=True)

    # Lay out widgets
    return VBox([HBox([view.fc_label('Model file'), view.model_path]),
                 HBox([view.train_pts_start, view.train_pts_stop, view.train_pts_step]),
                 HBox([view.a_min_txt, view.a_max_txt]),
                 HBox([view.m_min_txt, view.m_max_txt]),
                 HBox([view.q_min_txt, view.q_max_txt]),
                 view.save_fname,
                 HBox([view.fc_label('Cokrig file'), view.cokrig_path])
                 ])


def fc_label(txt, width=MED_DESC_WIDTH):
    """Produce a label for a file chooser widget."""
    return Label(txt, layout=Layout(width=width,
                                    display="flex",
                                    justify_content="flex-end",
                                    margin="0px 8px 0px 0px"))


def run_tab():
    view.run_btn = Button(description='Run',
                          button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                          tooltip='Run immediatly',
                          icon='play')  # (FontAwesome names without the `fa-` prefix)
    view.run_out = Output(layout={'border': '1px solid black', 'height': '400px', 'width': 'auto', 'overflow': 'auto'})
    # TODO Get run_out to automatically scroll to bottom as output is added

    with view.run_out:
        print('Ready...')

    return VBox([view.run_btn,
                 HTML('<hr style="visibility: hidden;">'),
                 view.run_out])


def job_tab():
    # --cpus-per-task=4
    # --account=sos
    # --time=0-00:10:00  (days-hours:minutes)

    view.job_days = IntText(description='Walltime, days', min=0, value=0)
    view.job_hrs = IntText(description='hrs.', min=0, value=0)
    view.job_mins = IntText(description='min.', min=0, value=0)
    view.job_cpus = IntText(description='CPUs per task', min=1, value=4)
    view.job_queue = Text(description='Queue name', value='')
    view.script_btn = Button(description='Create Job Script',
                             button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                             tooltip='Save job script file',
                             icon='file')  # (FontAwesome names without the `fa-` prefix)
    view.script_lbl = Label(value='', layout=Layout(margin='0 0 0 25px'))

    for widget in [view.job_days, view.job_cpus]:
        set_width([widget], width=MED_FULL_WIDTH)
        set_width([widget], width=HALF_DESC_WIDTH, desc=True)

    for widget in [view.job_hrs, view.job_mins]:
        set_width([widget], width=SM_FULL_WIDTH)
        set_width([widget], width=SM_DESC_WIDTH, desc=True)

    set_width([view.job_queue], width=LG_FULL_WIDTH)
    set_width([view.job_queue], width=HALF_DESC_WIDTH, desc=True)

    return VBox([HBox([view.job_days, view.job_hrs, view.job_mins]),
                 view.job_cpus,
                 view.job_queue,
                 HTML('<hr style="visibility: hidden;">'),
                 HBox([view.script_btn, view.script_lbl])])


def settings_tab(paths):
    """Create widgets for Settings screen."""
    cbaero_path, tables_path, run_path = paths

    # Create widgets
    if (cbaero_path is not None):
        view.cbaero_path = FileChooser(cbaero_path, select_default=True, show_only_dirs=True,
                                       layout=Layout(width='auto'))
    else:
        view.cbaero_path = FileChooser(show_only_dirs=True, layout=Layout(width='auto'))

    if (tables_path is not None):
        view.tables_path = FileChooser(tables_path, select_default=True, show_only_dirs=True,
                                       layout=Layout(width='auto'))
    else:
        view.tables_path = FileChooser(show_only_dirs=True, layout=Layout(width='auto'))

    if (run_path is not None):
        view.run_path = FileChooser(run_path, select_default=True, show_only_dirs=True, layout=Layout(width='auto'))
    else:
        view.run_path = FileChooser(show_only_dirs=True, layout=Layout(width='auto'))

    # Lay out widgets
    return VBox([HBox([view.fc_label('CBAero exec. dir.', width=PLUS_DESC_WIDTH), view.cbaero_path]),
                 HBox([view.fc_label('Tables directory', width=PLUS_DESC_WIDTH), view.tables_path]),
                 HBox([view.fc_label('Run directory', width=PLUS_DESC_WIDTH), view.run_path])
                 ])


def run_msg(txt, clear=False):
    """Display text in run output widget and log it."""
    with view.run_out:

        if clear:
            clear_output(wait=True)

        if txt is not None and not txt.strip() == '':
            print(txt)

        sys.stdout.flush()
