from pathlib import Path
from blocked_orchestrator import detect_blocked_nozzles, InvalidInputImageError

BASE_DIR = Path(__file__).parent
weights = BASE_DIR / "best.pt"
picture = BASE_DIR / "pictures" / "1.jpg"

def main():
    try:
        blocked_list = detect_blocked_nozzles(str(picture), str(weights))
        if blocked_list:
            print("Blocked nozzles detected:", blocked_list)
        else:
            print("No blocked nozzles detected.")  # ตอนนี้คาดว่า [] เพราะ isBlockedHole เป็นสตับ
    except FileNotFoundError as e:
        print("File error:", e)
    except InvalidInputImageError as e:
        print("Detection error:", e)

if __name__ == "__main__":
    main()
