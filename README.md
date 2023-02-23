# MLProject - Toolkit to build Machine Learning projects

## Installation 

(1) Install custom dependencies: [dataset_server](https://github.com/viebboy/dataset_server) and [cvinfer](https://github.com/viebboy/cvinfer)

(2) Install dependencies in `requirements.txt`

(3) Install this package in development mode: `pip3 install -e .` from the root of this package

(4) Initialize this tool by running:

```bash
mlproject init \
    --authors ${AUTHOR1_NAME,AUTHOR1_EMAIL1,AUTHOR1_EMAIL2;AUTHOR2_name,AUTHOR2_EMAIL..} \
    --company ${YOUR_DEFAULT_COMPANY} \
    --license ${YOUR_DEFAULT_LICENSE}
```

where:

- `--authors`: specifies the default authors for future projects created by the command `mlproject new-project`
               if each author is separated by colon (;), name and email is separated by comma
               for example `--authors "First1 Last1,abcxyz@gmail.com,mnp@gmail.com;First2 Last2,aaa@gmail.com,bbb@gmail.com"`
- `--company`: specifies the name of the default company of the authors
- `--license`: specifies default license for future projects created by `mlproject new-project`.
                available options: proprietary, apache, mit


## Usage

To create a new project template, we can run:

```bash
mlproject new-project \
    --project-name "name of this project" \
    --path "parent directory to create project under" \
    --authors "authors in this project" \
    --company "company that owns this project" \
    --license "license to use, current support proprietary/apache/mit"
    --template "template to use [currently support only generic]"
```

In addition, `mlproject` also provides convenient datastructures and classes that can be used to quickly implement new ML ideas.


When inside a project created by `mlproject`, we could create an empty file by using the following command:

```python
mlproject new-file \
    --filename ${NAME_OF_THE_FILE} \
    --path ${PATH_TO_CREATE_FILE_UNDER} \
    --desc ${SHORT_DESCRIPTION_OF_THIS_FILE}
```

with `--path` is defaulted to `.`

The new file when created will have the same header structure as other files in the project with proper author names, emails, licenses, etc.


## Composing a project

Overall,

- `mlproject.data` provides abstraction for data processing. Take a look at [mlproject.data](./docs/mlproject.data.md) for description.
- `mlproject.trainer` provides trainer class in pytorch. Take a look at [mlproject.trainer](./docs/mlproject.trainer.md) for description.
- `mlproject.metric` provides abstraction for metrics, which are used in `mlproject.trainer`. Take a look at [mlproject.metric](./docs/mlproject.metric.md) for description.
- `mlproject.loss` provides abstraction for losses, which are used in `mlproject.loss`. Take a look at [mlproject.loss](./docs/mlproject.loss.md) for description.

It's also a good idea to take a look at the source code to get an idea about the working mechanism of different abstractions.
In addition, examples under `examples` also serve as a good starting point

## Authors
Dat Tran (viebboy@gmail.com)
