# APEN TFT Monitoring and Fault-Warning Model

Wind-turbine SCADA data processing, offline TFT training, online learning, and EWMA fault warning.

Windows is the default platform. macOS users must adapt the PyTorch build, replace CUDA with MPS/CPU, and adjust paths, filename case, and data-loader workers as needed. macOS setup instructions are not provided.

## Project Structure

```text
Program/
├── Archive/                       Data documentation, metadata, and event summary
├── Model/
│   ├── main.py                    Entry point
│   ├── global_config.py           Configuration module
│   ├── global_param.yaml          Adjustable parameters
│   ├── sl1_load/                  Loading and preprocessing
│   ├── sl2_learn/                 Feature grouping and TFT learning
│   └── sl3_alarm/                 Prediction and fault warning
├── .gitattributes
├── .gitignore
├── environment.yml
├── windows-report.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Data

Data source: [Wind Turbine SCADA Data For Early Fault Detection — Zenodo](https://zenodo.org/records/14006163).

`[GitHub]` indicates files included with the repository; `[Download]` indicates data to download and place manually. Raw datasets are excluded from GitHub.

```text
Archive/
├── archive_info.txt                       [GitHub]
├── dataframe.csv                          [GitHub]
├── feature_info.csv                       [GitHub]
├── sensor_info.csv                        [GitHub]
├── Wind Farm A/
│   ├── comma_event_info.csv               [GitHub]
│   ├── comma_feature_description.csv      [GitHub]
│   └── datasets/                          [Download]
│       ├── comma_<event_id>.csv
│       └── ...
├── Wind Farm B/
│   ├── comma_event_info.csv               [GitHub]
│   ├── comma_feature_description.csv      [GitHub]
│   └── datasets/                          [Download]
│       ├── comma_<event_id>.csv
│       └── ...
└── Wind Farm C/
    ├── comma_event_info.csv               [GitHub]
    ├── comma_feature_description.csv      [GitHub]
    └── datasets/                          [Download]
        ├── comma_<event_id>.csv
        └── ...
```

Event files must be comma-separated CSVs with headers, named `comma_<event_id>.csv`, such as `comma_0.csv`. Place them as shown above, converting filenames and delimiters if necessary. Keep download archives outside the repository.

## Environment

Requirements: Windows, Conda, and Python 3.10. The reference GPU setup uses PyTorch 2.7.1+cu128 (CUDA 12.8) and a compatible NVIDIA driver, with Lightning 2.2.1 and PyTorch Forecasting 1.4.0. Other dependencies are listed in `environment.yml`.

The recorded Windows environment, including the full Conda package list and GPU details, is available in [windows-report.txt](windows-report.txt).

## Workflow

All adjustable program parameters are configured in `Model/global_param.yaml`.

Run from the repository root:

```powershell
python Model/main.py
```

For the first run, execute the full sequence: data preparation → hyperparameter preparation → offline training → online learning → prediction and warning. Later runs can reuse files in `Transit/` as listed below; comment out calls that are not needed in `main.py`.

| Stage | Prerequisites and reuse |
|---|---|
| Data preparation | Load, clean, split, and group data to produce `df_list` and `var_list`; training and warning also require `tsset`. These objects are not cached on disk and must be rebuilt as needed |
| Hyperparameters | With `automatic_conf: true`, a valid saved JSON is reused; optimization otherwise requires `tsset`. An existing JSON also allows the hyperparameter call to be skipped |
| Offline training | Requires `tsset` and the hyperparameter JSON. Existing `tft_net/*.ckpt` files allow training to be skipped; leaving the training call enabled retrains the models |
| Online learning | Requires `online_df`, `var_list`, and the main variable's offline checkpoint. Hyperparameter preparation, offline training, and `tsset` construction can be skipped. Saved online `.pth` files do not provide automatic resume |
| Prediction and warning | Requires `tsset`, `df_list[1]`, and offline checkpoints for all monitored variables. Hyperparameter preparation, offline training, and online learning can be skipped |

Build `online_df` by concatenating `df_list` and sorting by time. Online learning updates only `main_mv`; warning prediction still loads offline `tft_net/` checkpoints rather than `tft_ol/` online weights.

## Outputs

Runtime artifacts are saved under the automatically created, Git-ignored `Transit/` directory.

| Path | Content |
|---|---|
| `Transit/model_hyperparams.json` | Model hyperparameters |
| `Transit/tft_net/*.ckpt` | Offline models for monitored variables |
| `Transit/tft_ol/tft_*.pth` | Initial and selected online model weights |
| `Transit/tft_alarm_results.xlsx` | One sheet per monitored variable, with prediction, actual value, EWMA error, threshold, and alarm flag for each quantile |
| `Transit/logs/` | Logs when enabled |

Online predictions are returned as the `online` DataFrame in `main.py` and are not automatically exported.

## License

The original project code and documentation are licensed under the [MIT License](LICENSE), copyright (c) 2026 Zhen Wang. Third-party datasets, metadata, and documentation retain their original licenses and attribution requirements.

## Citation

If you use this software in your research, please cite it using [CITATION.cff](CITATION.cff).
