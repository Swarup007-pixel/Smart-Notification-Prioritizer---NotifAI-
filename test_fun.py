import pandas as pd
import pickle

from processing import weekday_and_weekend

with open('feature_order_model2_for_retrain.pkl','rb') as f:
    feature_order_model2_for_retrain = pickle.load(f)

with open('oer_app.pkl','rb') as f:
    oer_app = pickle.load(f)

with open('onehot_encoder_model1_priority_label.pkl','rb') as f:
    oe_model1_priority_label = pickle.load(f)

with open('onehot_encoder_time.pkl','rb') as f:
    oe_time = pickle.load(f)

# ==========================================================
# FEATURE PIPELINE TEST
# ----------------------------------------------------------
# Purpose:
# Verify that the retraining preprocessing pipeline produces
# exactly the same feature order and feature count that the
# original model was trained on.
#
# Why this matters:
# If even one column is missing, extra, or out of order,
# retraining may succeed but predictions will become wrong.
#
# Run this test whenever:
# - A new feature is added
# - An encoder is changed
# - Retraining logic is modified
# - A preprocessing function is updated
#
# Expected Output:
#
# ===== FEATURE CHECK =====
# Expected feature count: 26
# Actual feature count: 26
# ✅ Feature order matches perfectly
#
# If a mismatch exists, the test will print:
# - Position of mismatch
# - Expected column name
# - Actual column name
# - Missing columns
# - Extra columns

# ==========================================================

def test(data_test):

    new_data = pd.DataFrame(data_test)

    # Remove MongoDB _id column if present
    new_data.drop(['_id'], axis=1, inplace=True, errors='ignore')

    # Convert weekday names into weekday/weekend format
    new_data['weekday'] = new_data['weekday'].apply(
        weekday_and_weekend
    )

    # Encode model1 priority label
    model1_label = oe_model1_priority_label.transform(
        new_data[['model1_priority_label']]
    )

    new_data = pd.concat(
        [
            new_data,
            pd.DataFrame(
                model1_label.toarray(),
                columns=oe_model1_priority_label.get_feature_names_out(),
                index=new_data.index
            )
        ],
        axis=1
    )

    new_data.drop(
        ['model1_priority_label'],
        axis=1,
        inplace=True
    )

    # Encode app name
    ecapp = oer_app.transform(new_data[['app_name']])

    new_data = pd.concat(
        [
            new_data,
            pd.DataFrame(
                ecapp.toarray(),
                columns=oer_app.get_feature_names_out(),
                index=new_data.index
            )
        ],
        axis=1
    )

    new_data.drop(
        ['app_name'],
        axis=1,
        inplace=True
    )

    # Encode time of day
    etime = oe_time.transform(
        new_data[['time_of_day']]
    )

    new_data = pd.concat(
        [
            new_data,
            pd.DataFrame(
                etime.toarray(),
                columns=oe_time.get_feature_names_out(),
                index=new_data.index
            )
        ],
        axis=1
    )

    new_data.drop(
        ['time_of_day'],
        axis=1,
        inplace=True
    )

    # Match exact training feature order
    new_data = new_data[
        feature_order_model2_for_retrain
    ]

    # Training features
    x = new_data.drop(
        ['priority_score'],
        axis=1
    )

    # Expected feature order for model input
    expected_x = feature_order_model2_for_retrain.copy()
    expected_x.remove('priority_score')

    print("\n===== FEATURE CHECK =====")
    print("Expected feature count:", len(expected_x))
    print("Actual feature count:", len(x.columns))

    if expected_x == list(x.columns):
        print("✅ Feature order matches perfectly")

    else:
        print("❌ Feature order mismatch")

        for i, (exp, act) in enumerate(
            zip(expected_x, x.columns)
        ):
            if exp != act:
                print(f"\nMismatch at position {i}")
                print(f"Expected: {exp}")
                print(f"Actual:   {act}")
                break

        missing = [
            col for col in expected_x
            if col not in x.columns
        ]

        extra = [
            col for col in x.columns
            if col not in expected_x
        ]

        print("\nMissing columns:")
        print(missing)

        print("\nExtra columns:")
        print(extra)

    return x


# ==========================================================
# SAMPLE TEST DATA
# ----------------------------------------------------------
# Used only to verify preprocessing consistency.
# Not used for actual model training.
# ==========================================================

sample_data = [
    {
        "model1_priority_label": "High",
        "app_name": "WhatsApp",
        "battery_level": 80,
        "screen_on": True,
        "headset_connected": False,
        "time_of_day": "Morning",
        "weekday": "Monday",
        "notification_frequency_today": 15,
        "app_usage_minutes_today": 45,
        "app_open_frequency": 12,
        "notification_clicked": True,
        "notification_dismissed": False,
        "priority_score": 0.92
    },
    {
        "model1_priority_label": "Low",
        "app_name": "Instagram",
        "battery_level": 30,
        "screen_on": False,
        "headset_connected": True,
        "time_of_day": "Night",
        "weekday": "Sunday",
        "notification_frequency_today": 50,
        "app_usage_minutes_today": 120,
        "app_open_frequency": 25,
        "notification_clicked": False,
        "notification_dismissed": True,
        "priority_score": 0.35
    }
]

# Run the test
test(sample_data)