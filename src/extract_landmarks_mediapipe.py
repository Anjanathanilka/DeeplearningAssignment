import os, csv, cv2, mediapipe as mp

# ============ CHANGE THIS WHEN NEEDED ============
IMAGES_DIR = "train"   # change to "test" later for your test folder
# =================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, f"landmarks_{os.path.basename(IMAGES_DIR)}.csv")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

header = ["image_path", "width", "height", "num_points"]
header += [f"{c}{i}" for i in range(468) for c in ("x", "y", "z")]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if not result.multi_face_landmarks:
                    writer.writerow([path, w, h, 0])
                    continue

                lms = result.multi_face_landmarks[0].landmark
                flat = []
                for lm in lms:
                    flat += [lm.x, lm.y, lm.z]
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

                writer.writerow([path, w, h, len(lms)] + flat)

                cv2.imshow("Face Landmarks", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

cv2.destroyAllWindows()
print(f"✅ Done! Landmarks saved to: {CSV_PATH}")
