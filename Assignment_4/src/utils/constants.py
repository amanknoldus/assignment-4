from pathlib import Path

path = Path(__file__).resolve().parent.parent
data_dir = path / "dataset"
file_path = data_dir / "credit_card_data.csv"
