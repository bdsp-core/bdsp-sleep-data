# bdsp-sleep-data

## Human Sleep Project Data Processing
This repository contains scripts and documentation for processing sleep study data from the Human Sleep Project (HSP) dataset. The HSP dataset is a growing collection of clinical polysomnography (PSG) recordings, initially comprising data from ~19K patients evaluated at Massachusetts General Hospital and expected to grow significantly.
See: bdsp.io/content/hsp/

## High-Level Summary
This script is designed to process sleep study data, including signal data from .edf files and annotations from .csv files. It loads the signal data, preprocesses the annotations, and then vectorizes various types of events (respiratory events, sleep stages, arousals, and limb movements) to facilitate further analysis.
