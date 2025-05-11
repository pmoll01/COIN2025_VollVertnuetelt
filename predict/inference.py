#!/usr/bin/env python3
"""
Inference script for a trained Temporal Fusion Transformer (TFT) model.

Usage:
    python inference.py --features data/twitter_data/musk_twitter_sentiment.csv \
                       --targets data/finance_data/financeData_target_variables.csv \
                       --checkpoint outputs/checkpoints/tft_model.ckpt \
                       --output outputs/predictions.csv \
                       --target_column bitcoin_close \
                       --encoder_len 30 \
                       --prediction_len 1
"""
import os
import argparse
import pandas as pd
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from model.tft_model import create_dataset


def run_inference(
    features_path: str,
    targets_path: str,
    checkpoint_path: str,
    output_path: str,
    target_column: str,
    encoder_len: int,
    prediction_len: int,
):
    # 1. Load raw data
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)

    # 2. Build a template dataset and merged DataFrame
    template_dataset, merged_df = create_dataset(
        features_df,
        targets_df,
        target_column=target_column,
        encoder_len=encoder_len,
        prediction_len=prediction_len,
    )

    # Ensure 'date' is datetime
    merged_df["date"] = pd.to_datetime(merged_df["date"])

    # 3. Prepare inference window: last `encoder_len` observations
    inference_df = merged_df.tail(encoder_len)

    # 4. Create inference TimeSeriesDataSet (no randomization, prediction mode)
    inference_dataset = TimeSeriesDataSet.from_dataset(
        template_dataset,
        inference_df,
        predict=True,
        stop_randomization=True,
    )
    inference_loader = inference_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    # 5. Load trained TFT model from checkpoint
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # 6. Create a Trainer and run prediction
    trainer = Trainer(accelerator="auto", devices=1)
    raw_predictions = trainer.predict(
        model,
        dataloaders=inference_loader,
    )

    # 7. Extract and concatenate prediction tensors
    preds = torch.cat([x["prediction"] for x in raw_predictions]).detach().cpu().numpy()

    # 8. Build output DataFrame with forecasted dates
    last_date = merged_df["date"].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=prediction_len,
    )
    output_df = pd.DataFrame({
        "date": forecast_dates,
        "prediction": preds.flatten(),
    })

    # 9. Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(preds)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TFT Inference Script")
    parser.add_argument(
        "--features", type=str,
        default="data/twitter_data/musk_twitter_sentiment.csv",
        help="Path to CSV with feature data"
    )
    parser.add_argument(
        "--targets", type=str,
        default="data/finance_data/financeData_target_variables.csv",
        help="Path to CSV with target variables"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/checkpoints/tft_model.ckpt",
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/predictions.csv",
        help="Path where to save the forecast results"
    )
    parser.add_argument(
        "--target_column", type=str, default="bitcoin_close",
        help="Name of the target column in the targets CSV"
    )
    parser.add_argument(
        "--encoder_len", type=int, default=30,
        help="Number of timesteps used by the encoder"
    )
    parser.add_argument(
        "--prediction_len", type=int, default=1,
        help="Number of timesteps to forecast"
    )
    args = parser.parse_args()

    run_inference(
        features_path=args.features,
        targets_path=args.targets,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        target_column=args.target_column,
        encoder_len=args.encoder_len,
        prediction_len=args.prediction_len,
    )


if __name__ == "__main__":
    main()
