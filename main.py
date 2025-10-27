import argparse

def run_pipeline(stage: str):
    if stage == "prep":
        from src.data_prep import main as m
        m()
    elif stage == "eda":
        print("Open notebooks/eda.ipynb and run there.")
    elif stage == "train":
        from src.train import main as m
        m()
    elif stage == "eval":
        from src.evaluate import main as m
        m()
    elif stage == "attend":
        from src.attendance import main as m
        m()
    else:
        raise SystemExit(f"Unknown stage: {stage}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["prep","train","eval","attend","eda"])
    args = p.parse_args()
    run_pipeline(args.stage)
