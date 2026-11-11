# AI-Based Smart Parking Analytics

A computer vision benchmark for detecting **empty and occupied parking bays** in fixed-camera images. The study compares four object detectors on a prepared PKLot-derived dataset. It also proposes a way to use detections for parking availability and utilization analytics.

**Author:** Niddena Kiran

## Benchmark results

| Model | Completed epochs | Test mAP@0.50 | Test mAP@0.50–0.95 | Processing time per image | Stage FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s | 30/30 | 99.50% | 98.70% | 30.60 ms | 32.7 |
| YOLOv5su | 30/30 | 99.50% | 98.30% | 31.00 ms | 32.3 |
| **YOLO11s** | **30/30** | **99.50%** | **98.80%** | **20.00 ms** | **50.0** |
| RT-DETR-L | 6/30 | 97.58% | 69.21% | 54.77 ms | 18.3 |

YOLO11s had the highest reported mAP@0.50–0.95 and the lowest recorded processing time on this test split. Its 0.1 percentage-point mAP lead over YOLOv8s comes from a single run per model.

RT-DETR-L training stopped during epoch seven. The available checkpoint was evaluated after GPU memory cleanup, using explicit half precision. Its results describe that checkpoint under those settings. Stage FPS is calculated as 1,000 divided by recorded milliseconds per image and excludes camera capture, streaming and dashboard updates.

## Dataset and evaluation

The notebook downloads the [PKLot-derived YOLO dataset](https://www.kaggle.com/datasets/duythanhng/parking-lot-database-for-yolo) when it cannot find a local copy. It audits images and labels, maps class `0` to **empty** and class `1` to **occupied**, then prepares YOLO-format splits with seed 42.

| Split | Images |
| --- | ---: |
| Training | 4,306 |
| Validation | 861 |
| Test | 575 |
| **Total retained** | **5,742** |

The test set contains **34,400 bay annotations**: 19,053 empty and 15,347 occupied. Training used Google Colab with a Tesla T4 GPU, a 640-pixel image-size setting and AdamW. The three YOLO models used batch size 32. RT-DETR-L used batch size 8. The notebook selects checkpoints using validation and evaluates the available checkpoints on the held-out image split.

## Run in Google Colab

1. Open `00_MASTER_PARKING_BENCHMARK_COLAB.ipynb` in Google Colab and select a GPU runtime. The notebook installs its dependencies, including `ultralytics==8.3.40`.
2. Run the setup and dataset preparation cells. Inspect the dataset audit and ground-truth previews before training.
3. Run the training and test-evaluation sections. For the completed models, use the validation-selected `best.pt` checkpoints. Test results and checkpoints are written under `benchmark_results/`.
4. If a GPU job fails, preserve any available checkpoints and restart the runtime or stop only your own training job. The notebook contains a recovery cell with a machine-specific process ID. Review that cell before use.
5. Use the saved-image demonstration to inspect empty and occupied detections. Review the optional packaging cell before running it because it can upload an archive to a third-party file service and copy it to Google Drive.

## Example and deployment scope

On one saved test image, the detector display reported **39 empty bays and 1 occupied bay out of 40 detections**, giving **2.5% detection-based occupancy**. The ratio uses detected bays as its denominator. A production system would need a registered bay inventory and a way to handle missed detections and stale observations.

The manuscript proposes bay association, temporal stabilization, event logging, an availability dashboard and demand forecasting. The notebook evaluates the detector and demonstrates the display on a saved image. A camera- or date-grouped evaluation, a live video test and repeated runs would help establish transfer and reliability.

## Repository files

- `README.md` — project overview, benchmark results and run instructions
- `00_MASTER_PARKING_BENCHMARK_COLAB.ipynb` — dataset preparation, training, evaluation and saved outputs, when added to the repository
- `benchmark_results/` — generated during a run, including checkpoints and evaluation artifacts

## Reference

de Almeida, P. R. L., et al. (2015). *PKLot – A robust dataset for parking lot classification*. Expert Systems with Applications, 42(11), 4937–4949. [DOI](https://doi.org/10.1016/j.eswa.2015.02.009)
