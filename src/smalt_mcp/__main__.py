"""Entrypoint: `python -m smalt_mcp` -> uvicorn."""

import logging

import uvicorn

from smalt_mcp.config import HOST, PORT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("smalt_mcp.server:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
