from pathlib import Path


def download_data(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    req_files = [
        "prompts_train.csv",
        "summaries_train.csv",
        "prompts_test.csv",
        "summaries_test.csv",
    ]
    missed_files = [
        file_name for file_name in req_files if not (data / file_name).exists()
    ]
    if missed_files:
        raise FileNotFoundError(
            f"Отсутствуют файлы {missed_files}.Скачайте данные с https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data и положите их в data/raw"
        )
