import argparse
import logging
import logging.config
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

from label_studio_ml.api import init_app
from model import RiverSAMSegmenter


backend_root = Path(__file__).resolve().parent

local_files_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
if local_files_root and not Path(local_files_root).is_absolute():
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(
        (backend_root / local_files_root).resolve()
    )

model_dir = Path(os.getenv("ML_BACKEND_MODEL_DIR") or (backend_root / "model_data"))
model_dir.mkdir(parents=True, exist_ok=True)
os.environ["ML_BACKEND_MODEL_DIR"] = str(model_dir)

app = init_app(model_class=RiverSAMSegmenter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)
