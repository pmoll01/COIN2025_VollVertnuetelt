from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE


def create_dataset(df, encoder_len=30, prediction_len=1):
    time_varying_unknown_reals = [
        "sentiment_avg", "tweet_volume", "emotion_anger", "emotion_joy", "target"
    ]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=encoder_len,
        max_prediction_length=prediction_len,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=[],
    )
    return dataset


def build_tft_model(train_dataset, learning_rate=0.01, hidden_size=16):
    return TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=1,
        dropout=0.1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
