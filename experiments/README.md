# Study 3: Benchmarking Security, Safety, and Utility of Agents with Symbolic Guardrails

## Artifacts

All execution logs of experiments can be found under folder `logs_to_save`.

All statistical tests can be found in the jupyter notebooks under folder `stats`.

## Installation

`pip install -e ./agent`

`pip install -e ./mcp-server`

Note the pipeline may not execute normally due to a known bug in the MCP Python SDK. Please refer to this GitHub issue to see if the current version has fixed it, or to resolve it locally: <https://github.com/modelcontextprotocol/python-sdk/issues/1933>.

## Notation

Due to internal terminology inconsistency, `golden` in this repo refers to `baseline` group reported in the paper. `full` in this repo refers to `guardrail` group in the paper.

## Execution

Before execution, please navigate to the corresponding yaml configuration to substitute the local path to the actual path in your environment. Please ensure you have a valid $OPENAI_API_KEY as environment variable.

### tau2-airline

command:

`run_dataset --config agent/src/config/configs/tau2/{group}_config.yml `

Where {group} can be golden (stands for baseline group) or full (stands for gaurdrail group).

### CAR-bench

`run_dataset --config agent/src/config/configs/CarBench/{group}_config_base.yml`

Where {group} can be baseline or gaurdrail.

### MedAgentBench

command:

`run_dataset --config agent/src/config/configs/MedAgentBench/{group}_config_{dataset}_dataset.yml `

Where {group} can be raw (stands for raw group), golden (stands for baseline group) or full (stands for gaurdrail group), and {dataset} can be ori (stands for the original benchmark data) and safety (stands for the adversarial data).

## Data generation

The code used to generate the adversarial data for MedAgentBench can be found in `data/MedAgentBench/safety_task_generation`.
