# MLProject - Toolkit to build Machine Learning projects

## Installation 

(optional) Install custom dependencies: [swift-loader](https://github.com/viebboy/swift-loader)

(1) Install dependencies in `requirements.txt`

(2) Install this package in development mode: `pip3 install -e .` from the root of this package

(3) Initialize this tool by running:

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

### Create a new project from template

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


### Create new file inside a project

When inside a project created by `mlproject`, we could create an empty file by using the following command:

```bash
mlproject new-file \
    --filename ${NAME_OF_THE_FILE} \
    --path ${PATH_TO_CREATE_FILE_UNDER} \
    --desc ${SHORT_DESCRIPTION_OF_THIS_FILE}
```

with `--path` is defaulted to `.`

The new file when created will have the same header structure as other files in the project with proper date, author names, emails, licenses, etc.


### Modify project's metadata

When we want to modify the metadata of a project such as the list of authors, the license or the company, we could run the following command in the main directory of a project created by ˋmlproject new-projectˋ as follows:


```
mlproject modify-metadata \
    --project-name "new name for this project" \
    --authors "new author list" \
    --company "new company that owns this project" \
    --license "new license"

```

The above command will change content of the metadata file and all the headers of all python files. 
The convention for `--authors` is similar to the project creation command above


### Launch experiments in parallel

If you're using the project template created by `mlproject new-project`, you'll run a single experiment configuration via the `entry.py` script.

What if your machine has many GPUs or CPUs and you would like to run many experiment configurations in parallel? 

`mlproject launch-exp` is the command to use:


```bash
mlproject launch-exp \
    --entry-script "path to entry script" \
    --config-path "path to configuration file" \
    --device "either cpu or cuda" \
    --gpu-indices "the list of GPUs to use, comma separated. Default to all GPUs if device is cuda" \
    --gpu-per-exp "the number of GPUs to use for one experiment configuration" \
    --log-prefix "the prefix to dump logs from workers" \
    --nb-parallel-exp "number of parallel experiments to run. Only needed when device is cpu"

```


### Create a summary of experiment results

If you're using the project template created by `mlproject new-project` and you've run a lot of experiments, you could create a table that summarizes

the results by running the following command:

```bash
mlproject summarize-exp \
    --entry-script "path to entry script" \
    --config-path "path to configuration file" \
    --metrics "the list of metrics you want to include in the report. Comma separated"

```


The last switch `--metrics` is especially helpful if you only want to take a look at a subset of metrics.


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
