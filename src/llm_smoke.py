"""One real chat call, used by the Step 1 check in the README."""

from src.llm import chat


def main() -> None:
    reply = chat("Say hello")
    print(reply)


if __name__ == "__main__":
    main()
