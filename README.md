Link to datafolder: https://drive.google.com/drive/folders/1AsDZbTzGzYI74DHcfHV5nnn2-DtmJlDy?usp=share_link

Link to pickle files: https://drive.google.com/drive/folders/1U7wcYFw-u7VVdnxQLLyhAXJi3fFYazDk?usp=share_link

[DepDetect Dataflow](https://www.figma.com/file/EcJ9jFfj6ejyPxTtE2kcuH/eegDep?type=whiteboard&t=COwI8jHtaG9aVf41-1)

Example of info.json file:

```python
{"dbname": "modma",
 "dbpath": "D:/amand/Documentos/Datasets/modma",
 "selected_channels": [21, 8, 23, 123, 35, 103, 51, 91, 69, 82, 32, 121, 44, 107, 57, 95],
 "ch_names": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"],
 "ch_types": "eeg",
 "ch_refs": ["Fp2", "T4", "O1"],
 "sub_refs": [27, 49, 4],
 "artifacts_labels": ["eyes_mov", "jaw_clench", "h_mov"],
 "tRec": 300,
 "bands": ["delta","theta","alpha","beta","gamma"],
 "window_time": 1000,
 "trProp": [0.81, 0.09, 0.1],
 "trName": "withallfreqsplitautozscore",
 "fs": 250, "f0": 50, "fa": 0.5, "fb": 50
}
