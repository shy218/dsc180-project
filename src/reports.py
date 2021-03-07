import os
import nbformat
from nbconvert import HTMLExporter


def generate_report_from_notebook(config):

    print(' => Generating EDA...')
    os.system('mkdir -p ' + config['data_dir'] + config['report_dir'])

    curdir = os.path.abspath(os.getcwd())
    # indir, _ = os.path.split(report_in_path)
    # outdir, _ = os.path.split(report_out_path)
    # os.makedirs(outdir, exist_ok=True)

    html_config = {
        "ExecutePreprocessor": {"enabled": True},
        "TemplateExporter": {"exclude_output_prompt": True, "exclude_input": True, "exclude_input_prompt": True},
    }

    # notebook file path
    nb_fp = config['notebook_dir'] + config['notebook_file']
    nb = nbformat.read(open(nb_fp), as_version=4)
    html_exporter = HTMLExporter(config=html_config)

    # change dir to notebook dir, to execute notebook
    os.chdir(config['notebook_dir'])
    body, resources = (
        html_exporter
        .from_notebook_node(nb)
    )

    # change back to original directory
    os.chdir(curdir)

    report_out_path = config['data_dir'] + config['report_dir'] + config['report_file']
    with open(report_out_path, 'w') as fh:
        fh.write(body)

    print(' => Done! See the result HTML file in ' + report_out_path)
