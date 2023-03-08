"""
cli.py: command line interface
------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0


"""

import argparse
import os
import sys
import shutil
import json
from loguru import logger
from datetime import date
import glob

from mlproject.exp_launcher import exp_launcher


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument(
        "command",
        action='store',
        type=str,
        choices=[
            'init',
            'new-project',
            'new-file',
            'modify-metadata',
            'launch-exp',
        ],
        help="primary command to run mlproject"
    )

    # options
    parser.add_argument(
        "--company",
        action='store',
        type=str,
        default=None,
        help="company that owns this project. Option for 'mlproject init' and 'mlproject new-project'"
    )
    parser.add_argument(
        "--authors",
        action='store',
        type=str,
        default=None,
        help=''.join([
            "name and emails of authors. Format: name1,email1;name2,email2 and so on",
            'For example: --authors "First Name Last Name,email1@gmail.com; First Name Last Name,email2@gmail.com"'
        ])
    )
    parser.add_argument(
        "--disable-warning",
        action='store',
        type=str,
        default=None,
        help='whether to disable warnings'
    )
    parser.add_argument(
        "--nb-parallel-jobs",
        action='store',
        type=int,
        default=None,
        help='number of parallel jobs to run in joblib'
    )
    parser.add_argument(
        "--log-level",
        action='store',
        type=str,
        default=None,
        help='logging level'
    )
    parser.add_argument(
        "--license",
        action='store',
        type=str,
        default=None,
        choices=['MIT', 'proprietary', 'apache'],
        help='license'
    )
    parser.add_argument(
        "--project-name",
        action='store',
        type=str,
        default=None,
        help='name of the project'
    )
    parser.add_argument(
        "--path",
        action='store',
        type=str,
        default=None,
        help='target directory to create the project or file'
    )
    parser.add_argument(
        "--template",
        action='store',
        type=str,
        default='generic',
        choices=['generic',],
        help='the type of template to create'
    )
    parser.add_argument(
        "--filename",
        action='store',
        type=str,
        default=None,
        help='name of the file to create'
    )
    parser.add_argument(
        "--desc",
        action='store',
        type=str,
        default=None,
        help='description of the file to create'
    )
    # parameters for launch-exp
    parser.add_argument(
        "--config-path",
        action='store',
        type=str,
        default=None,
        help='path to config [used in launch-exp command]'
    )
    parser.add_argument(
        "--entry-script",
        action='store',
        type=str,
        default=None,
        help='path to entry script [used in launch-exp command]'
    )

    return parser.parse_known_args()


def parse_authors_string(text):
    authors_ = text.split(';')
    authors = []
    for author in authors_:
        name = author.split(',')[0]
        emails = author.split(',')[1:]
        authors.append({'name': name, 'emails': emails})
    return authors

def modify_metadata(**kwargs):
    from mlproject.constants import (
        PACKAGE_DIR,
        COMPANY,
        AUTHORS,
        LICENSES,
        DEFAULT_LICENSE,
    )
    logger.warning('Modifying project metadata involves modifying source code')
    logger.warning('Potentiall failure could lead to loss of source code')
    response = input('Do you still want to perform this operation? (Y/N): ')
    if response != 'Y':
        return

    path = os.path.abspath('.')
    config_file = os.path.join(path, '.mlproject.json')
    if not os.path.exists(config_file):
        msg = (
            f'Cannot find the config file in: {config_file}; ',
            '"mlproject modify" must be called at the root of a project created by mlproject init'
        )
        raise RuntimeError(''.join(msg))

    with open(config_file, 'r') as fid:
        config = json.loads(fid.read())

    if kwargs['authors'] is not None:
        try:
            kwargs['authors'] = parse_authors_string(kwargs['authors'])
        except Exception:
            raise RuntimeError('Cannot parse authors with the provided value: "{}"'.format(kwargs['authors']))

    py_files = glob.glob(os.path.join(path, '**', '*.py'), recursive=True)
    for file in py_files:
        # backup
        shutil.copy(file, file + '.bk')

        with open(file, 'r') as fid:
            content = fid.read().split('\n')

        for i in range(len(content)):
            row = content[i]
            # modify company
            modify_company = row.startswith('* Copyright') and kwargs['company'] is not None
            if modify_company:
                year = config['year']
                content[i] = f'* Copyright: {year} {kwargs["company"]}'

            modify_authors = row.startswith('* Authors') and kwargs['authors'] is not None
            if modify_authors:
                authors_string = []
                for author in kwargs['authors']:
                    authors_string.append(author['name'] + ' (' + ', '.join(author['emails']) + ')')
                content[i] = f'* Authors: ' + ', '.join(authors_string)

            modify_name = row.startswith('This is part of') and kwargs['project_name'] is not None
            if modify_name:
                content[i] = 'This is part of the {} project'.format(kwargs['project_name'])

            modify_license = row.startswith('License') and kwargs['license'] is not None
            if modify_license:
                title = LICENSES[kwargs['license']]['title']
                content[i+2] = f'{title}'

        content = '\n'.join(content)
        with open(file, 'w') as fid:
            fid.write(content)
        os.remove(file + '.bk')

    # update config file
    if kwargs['project_name'] is not None:
        config['project_name_raw'] = kwargs['project_name']
        config['project_name'] = kwargs['project_name'].replace(' ', '_')

    if kwargs['authors'] is not None:
        config['authors'] = kwargs['authors']

    if kwargs['company'] is not None:
        config['company'] = kwargs['company']

    if kwargs['license'] is not None:
        config['license_title'] = LICENSES[kwargs['license']]['title']
        # copy the new license
        with open(LICENSES[kwargs['license']]['path'], 'r') as fid:
            license_content = fid.read()
            license_content = license_content.replace('<COPYRIGHT HOLDER>', config['company'])
            license_content = license_content.replace('<YEAR>', config['year'])

        with open(os.path.join(path, 'LICENSE.txt'), 'w') as fid:
            fid.write(license_content)


def create_file(**kwargs):
    from mlproject.constants import (
        PACKAGE_DIR,
        COMPANY,
        AUTHORS,
        LICENSES,
        DEFAULT_LICENSE,
    )

    # recursively find the mlproject configuration file
    if kwargs['path'] is None:
        path = os.path.abspath('.')
    else:
        path = kwargs['path']

    if not os.path.exists(path):
        raise RuntimeError(f'the given --path ({path}) doesnt exist')

    if not os.path.isdir(path):
        raise RuntimeError(f'the given --path ({path}) is not a directory')

    if kwargs['filename'] is None:
        raise RuntimeError('missing filename, please specify via --filename')

    if kwargs['description'] is None:
        raise RuntimeError('missing description for this file, please specify via --desc')

    filename = kwargs['filename']
    description = kwargs['description']

    config_file = None
    tmp_path = path
    while True:
        if os.path.exists(os.path.join(tmp_path, '.mlproject.json')):
            config_file = os.path.join(tmp_path, '.mlproject.json')
            break
        else:
            if tmp_path == '/':
                break
            else:
                tmp_path = os.path.dirname(tmp_path)

    if config_file is not None:
        logger.info(f'found mlproject configuration file in: {config_file}')
        with open(config_file, 'r') as fid:
            config = json.loads(fid.read())
    else:
        config = None

    if kwargs['authors'] is None:
        if config is None:
            authors = AUTHORS
        else:
            authors = config['authors']
    else:
        author_list = kwargs['authors'].split(';')
        authors = []
        for item in author_list:
            name = item.split(',')[0]
            emails = item.split(',')[1:]
            authors.append({'name': name, 'emails': emails})

    if kwargs['company'] is None:
        if config is None:
            company = COMPANY
        else:
            company = config['company']
    else:
        company = kwargs['company']

    if kwargs['project_name'] is None:
        if config is None:
            raise RuntimeError('Found no setting for project name. Please specify via --project-name')
        else:
            project_name = config['project_name_raw']
    else:
        project_name = kwargs['project_name']

    if kwargs['license'] is None:
        if config is None:
            license_title = DEFAULT_LICENSE['title']
        else:
            license_title = config['license_title']
    else:
        license_title = LICENSES[kwargs['license']]['title']

    # get current time
    today = date.today()
    year = today.year
    day = today.day
    month = today.month

    # 1st line
    file_headers = cut_string(f'{filename}: {description}', max_length=100)

    header = [
        '"""',
        *file_headers,
        '',
        '',
        '* Copyright: {} {}'.format(year, company),
    ]
    # author strings
    author_strings = []
    for author in authors:
        name = author['name']
        emails = author['emails']
        author_strings.append('{} ({})'.format(name, ', '.join(emails)))

    author_strings = ', '.join(author_strings)
    header.append(f'* Authors: {author_strings}')
    header.append('* Date: {}-{:02d}-{:02d}'.format(year, month, day))
    header.append('* Version: 0.0.1')
    header.append('')
    header.append(f'This is part of the {project_name} project')
    header.append('')
    header.append('License')
    header.append('-------')
    header.append(license_title)
    header.append('')
    header.append('"""')

    dst_file = os.path.join(path, filename)
    content = '\n'.join(header)
    with open(dst_file, 'w') as fid:
        fid.write(content)

    logger.info(f'complete creating {dst_file}')


def create_project(**kwargs):
    """
    create a new project using the template provided under templates
    """
    from mlproject.constants import (
        PACKAGE_DIR,
        COMPANY,
        AUTHORS,
        LICENSES,
        DEFAULT_LICENSE,
    )

    logger.info(f'creating a project from {kwargs["template"]} template')

    if kwargs['project_name'] is None:
        raise RuntimeError('--project-name is missing when calling "mlproject new-project"')

    if kwargs['company'] is None:
        company = COMPANY
    else:
        company = kwargs['company']

    if kwargs['authors'] is None:
        authors = AUTHORS
        print(f'AUTHORS: {AUTHORS}')
    else:
        # process authors into dictionary
        authors_lst = kwargs['authors'].split(';')
        authors = []
        for item in authors_lst:
            name = item.split(',')[0]
            emails = item.split(',')[1:]
            authors.append({'name': name, 'emails': emails})

    if kwargs['license'] is None:
        license = DEFAULT_LICENSE
    else:
        if kwargs['license'] not in LICENSES:
            logger.warning(f'Unknown license type: {kwargs["license"]}')
            logger.warning(f'The following licenses are supported: {LICENSES}')
            sys.exit(2)
        else:
            license = LICENSES[kwargs['license']]

    if kwargs['path'] is None:
        kwargs['path'] = '.'
    else:
        if not os.path.exists(kwargs['path']):
            logger.warning(f'the specified project path (--path) does not exist: {kwargs["path"]}')
            sys.exit(2)

    # create a new dir
    project_name_raw = kwargs['project_name']

    # replace space with underscore
    project_name = project_name_raw.replace(' ', '_')
    proj_dir = os.path.join(kwargs['path'], project_name)

    # handle proj path
    if os.path.exists(proj_dir) and os.path.isdir(proj_dir):
        logger.warning(f'there is already a directory in the given path {proj_dir}')
        response = input(f'do you want to remove existing directory ({proj_dir}) (Y/N)?: ')
        if response == 'Y':
            shutil.rmtree(proj_dir)
            os.mkdir(proj_dir)
            os.mkdir(os.path.join(proj_dir, 'configs'))
        else:
            sys.exit(2)
    else:
        os.mkdir(proj_dir)
        os.mkdir(os.path.join(proj_dir, 'configs'))

    template_dir = os.path.join(PACKAGE_DIR, 'templates', kwargs['template'])

    # handle python files
    py_files = [
        'data.py',
        'entry.py',
        'models.py',
        'trainer.py',
        'configs/config.py',
    ]

    for file in py_files:
        src_file = os.path.join(template_dir, file)
        dst_file = os.path.join(proj_dir, file)
        copy_py_file(
            src_file,
            dst_file,
            company,
            authors,
            license['title'],
            project_name_raw
        )

    # ----------- handle README.md -------------------------------
    with open(os.path.join(template_dir, 'README.md'), 'r') as fid:
        readme_lines = fid.read().split('\n')

    # change the title
    readme_lines[0] = f'# {project_name_raw}'

    # add authors
    for author in authors:
        name = author['name']
        emails = ', '.join(author['emails'])
        readme_lines.append(f'{name} ({emails})')

    readme_content = '\n'.join(readme_lines)
    with open(os.path.join(proj_dir, 'README.md'), 'w') as fid:
        fid.write(readme_content)

    # --------- copy license --------------------------------

    # open the license and put company as the copyright holder
    current_year = date.today().year
    with open(license['path'], 'r') as fid:
        license_content = fid.read()
    license_content = license_content.replace('<COPYRIGHT HOLDER>', company)
    license_content = license_content.replace('<YEAR>', str(current_year))

    # write license
    with open(os.path.join(proj_dir, 'LICENSE.txt'), 'w') as fid:
        fid.write(license_content)

    # --------- copy dependencies ---------------------------
    shutil.copy(os.path.join(template_dir, 'requirements.txt'), os.path.join(proj_dir, 'requirements.txt'))

    # --------- copy ignore file ----------------------------
    shutil.copy(os.path.join(template_dir, '.gitignore'), os.path.join(proj_dir, '.gitignore'))

    # --------- write mlconfig file -------------------------
    mlconfig = {
        'project_name_raw': project_name_raw,
        'project_name': project_name,
        'company': company,
        'authors': authors,
        'license_title': license['title'],
        'year': str(current_year),
    }
    with open(os.path.join(proj_dir, '.mlproject.json'), 'w') as fid:
        fid.write(json.dumps(mlconfig, indent=2))

    logger.info(f'successfully created a new project at {proj_dir}')

def cut_string(text, max_length=100):
    line = ''
    outputs = []
    words = text.split(' ')
    for word in words:
        if len(line + ' ' + word) < max_length:
            if len(line) > 0:
                line = line + ' ' + word
            else:
                line = line + word
        else:
            outputs.append(line)
            line = word

    outputs.append(line)

    return outputs

def copy_py_file(src_file, dst_file, company, authors, license_title, project_name):
    """
    Header for each file has the following format
    file.py: short description for the file
    ---------------------------------------


    * Copyright:
    * Authors:
        Name: First Last, Emails: email1, email2
    * Date: some date
    * Version: 0.0.1

    This is part of {project name}

    License
    -------
    Title-of-the-license


    """


    with open(src_file, 'r') as fid:
        lines = fid.read().split('\n')

    file_header = lines[1]
    header_end_idx = None
    for i in range(1, len(lines)):
        if lines[i] == '"""':
            header_end_idx = i + 1
            break

    today = date.today()
    year = today.year
    day = today.day
    month = today.month
    header = [
        '"""',
        file_header,
        '',
        '',
        '* Copyright: {} {}'.format(year, company),
    ]
    author_strings = []
    for author in authors:
        name = author['name']
        emails = author['emails']
        author_strings.append('{} ({})'.format(name, ', '.join(emails)))
    author_strings = ', '.join(author_strings)
    header.append(f'* Authors: {author_strings}')
    header.append('* Date: {}-{:02d}-{:02d}'.format(year, month, day))
    header.append('* Version: 0.0.1')
    header.append('')
    header.append(f'This is part of the {project_name} project')
    header.append('')
    header.append('License')
    header.append('-------')
    header.append(license_title)
    header.append('')
    header.append('"""')

    content = '\n'.join([str(line) for line in (header + lines[header_end_idx:])])
    with open(dst_file, 'w') as fid:
        fid.write(content)


def initialize(**kwargs):
    """
    init the configurations for mlproject tool
    """
    logger.info('initializing mlproject ...')

    # load default configs
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(package_dir, '.configuration.json'), 'r') as fid:
        default_config = json.loads(fid.read())

    if kwargs['authors'] is None:
        logger.warning('--authors must be provided for command "mlproject init"')
        sys.exit(2)
    else:
        # authors are separated by semicolon
        author_list = kwargs['authors'].split(';')
        authors = []
        for item in author_list:
            name = item.split(',')[0]
            emails = item.split(',')[1:]
            authors.append({'name': name, 'emails': emails})
        default_config['AUTHORS'] = authors

    if kwargs['company'] is None:
        logger.warning('--company must be provided for command "mlproject init"')
        sys.exit(2)
    else:
        default_config['COMPANY'] = kwargs['company']

    # overwrite values provided from users
    if kwargs['nb_parallel_jobs'] is not None:
        default_config['NB_PARALLEL_JOBS'] = kwargs['nb_parallel_jobs']

    if kwargs['disable_warning'] is not None:
        default_config['DISABLE_WARNING'] = kwargs['disable_warning']

    if kwargs['log_level'] is not None:
        default_config['LOG_LEVEL'] = kwargs['log_level']

    # handle license
    if kwargs['license'] is not None:
        if kwargs['license'] not in default_config['LICENSES']:
            logger.warning(f'the specified license ({kwargs["license"]}) is not supported')
            logger.warning('setting proprietary license as the default now')
            default_config['DEFAULT_LICENSE'] = default_config['LICENSES']['proprietary']
        else:
            default_config['DEFAULT_LICENSE'] = default_config['LICENSES'][kwargs['license']]
    else:
        default_config['DEFAULT_LICENSE'] = default_config['LICENSES']['proprietary']

    # write config
    root_dir = os.path.join(os.path.expanduser('~'), '.mlproject')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(os.path.join(root_dir, 'licenses')):
        os.mkdir(os.path.join(root_dir, 'licenses'))

    for _, license in default_config['LICENSES'].items():
        src_file = os.path.join(package_dir, license['path'])
        dst_file = os.path.join(root_dir, license['path'])
        shutil.copy(src_file, dst_file)
        license['path'] = dst_file

    config_file = os.path.join(root_dir, 'configurations.json')
    with open(config_file, 'w') as fid:
        fid.write(json.dumps(default_config, indent=2))

    logger.info(f'complete writing default configuration to {config_file}')


@logger.catch
def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        logger.warning(f'unknown command or argument: {arg}')
        sys.exit(2)

    # handle different primaryc commands
    if known_args.command == 'init':
        # initialization
        args = {
            'company': known_args.company,
            'authors': known_args.authors,
            'nb_parallel_jobs': known_args.nb_parallel_jobs,
            'disable_warning': known_args.disable_warning,
            'log_level': known_args.log_level,
            'license': known_args.license,
        }
        initialize(**args)
    elif known_args.command == 'new-project':
        args = {
            'company': known_args.company,
            'authors': known_args.authors,
            'project_name': known_args.project_name,
            'path': known_args.path,
            'license': known_args.license,
            'template': known_args.template,
        }
        create_project(**args)
    elif known_args.command == 'new-file':
        args = {
            'company': known_args.company,
            'authors': known_args.authors,
            'project_name': known_args.project_name,
            'path': known_args.path,
            'license': known_args.license,
            'filename': known_args.filename,
            'description': known_args.desc,
        }
        create_file(**args)

    elif known_args.command == 'modify-metadata':
        # modify project metadata
        args = {
            'company': known_args.company,
            'authors': known_args.authors,
            'project_name': known_args.project_name,
            'license': known_args.license,
        }
        modify_metadata(**args)
    elif known_args.command == 'launch-exp':
        exp_launcher(
            known_args.entry_file,
            known_args.config_file,
            args.device,
            args.gpu_indices,
            args.gpu_per_exp,
            args.nb_parallel_exp
        )

if (__name__ == "__main__"):
    main()
